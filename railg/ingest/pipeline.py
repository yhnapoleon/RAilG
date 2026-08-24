"""入库流水线:source → extract → chunk → embed → index → 登记。

固定流水线,不引 DAG 引擎。但阶段划分与中间产物形态是清晰分离的,
将来要换成声明式编排不用重写业务逻辑。

两处增量:
    delta —— 内容哈希未变则跳过,变了先删旧 chunk 再写新的
    登记 —— 文档级信息写进 SQLite,支撑"列出 / 删除 / 重建"这类管理操作
             (OpenSearch 里只有 chunk,没有"文档"这一层)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path

from railg.config import Settings, get_settings
from railg.db import Database, get_db
from railg.ingest.chunker import Chunker, build_chunker
from railg.ingest.extractors import extract
from railg.ingest.sources import LocalSource, Source, SourceItem, UrlSource
from railg.providers import EmbeddingProvider, get_embedding_provider
from railg.schema.document import DocumentMeta, IndexDoc, IngestResult, IngestStatus
from railg.store import Store, get_store

logger = logging.getLogger(__name__)

ProgressHook = Callable[[IngestResult], None]


@dataclass(slots=True)
class IngestSummary:
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    total_chunks: int = 0
    results: list[IngestResult] = field(default_factory=list)

    def add(self, result: IngestResult) -> None:
        self.results.append(result)
        if result.status is IngestStatus.INDEXED:
            self.indexed += 1
            self.total_chunks += result.n_chunks
        elif result.status is IngestStatus.SKIPPED_UNCHANGED:
            self.skipped += 1
        else:
            self.failed += 1

    @property
    def failures(self) -> list[IngestResult]:
        return [r for r in self.results if r.error]


class IngestPipeline:
    def __init__(
        self,
        store: Store | None = None,
        chunker: Chunker | None = None,
        embedder: EmbeddingProvider | None = None,
        db: Database | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_store()
        self.chunker = chunker or build_chunker(self.settings.chunk)
        # 显式注入优先 —— 测试和离线跑批要能换掉真实 provider
        self.embedder = embedder or get_embedding_provider()
        self.db = db or get_db()

    # ----------------------------------------------------------------- #
    async def run(
        self,
        source: Source,
        force: bool = False,
        concurrency: int = 4,
        on_progress: ProgressHook | None = None,
    ) -> IngestSummary:
        await self.store.ensure_index()
        await self.db.init()

        summary = IngestSummary()
        sem = asyncio.Semaphore(max(1, concurrency))

        async def one(item: SourceItem) -> IngestResult:
            async with sem:
                return await self.ingest_item(item, force=force)

        items = list(source.iter_items())
        if not items:
            logger.warning("数据源 %s 没有产出任何可处理的文件", source.name)
            return summary

        logger.info("待处理 %d 个文件(来源 %s)", len(items), source.name)
        for coro in asyncio.as_completed([one(i) for i in items]):
            result = await coro
            summary.add(result)
            if on_progress:
                on_progress(result)

        await self.store.refresh()
        return summary

    async def run_paths(
        self,
        root: Path | str,
        acl_principals: list[str] | None = None,
        force: bool = False,
        concurrency: int = 4,
        on_progress: ProgressHook | None = None,
    ) -> IngestSummary:
        return await self.run(
            LocalSource(root, acl_principals=acl_principals),
            force=force, concurrency=concurrency, on_progress=on_progress,
        )

    async def run_urls(
        self,
        urls: list[str],
        acl_principals: list[str] | None = None,
        force: bool = False,
        concurrency: int = 2,
        on_progress: ProgressHook | None = None,
    ) -> IngestSummary:
        return await self.run(
            UrlSource(urls, acl_principals=acl_principals),
            force=force, concurrency=concurrency, on_progress=on_progress,
        )

    # ----------------------------------------------------------------- #
    async def ingest_item(self, item: SourceItem, force: bool = False) -> IngestResult:
        meta = item.meta
        try:
            payload = await asyncio.to_thread(item.read_bytes)
            meta.content_hash = DocumentMeta.make_content_hash(payload)

            # --- delta:内容没变就不重做 ---
            if not force:
                existing = await self.store.get_content_hash(meta.doc_id)
                if existing == meta.content_hash:
                    return IngestResult(
                        file_path=meta.file_path,
                        status=IngestStatus.SKIPPED_UNCHANGED,
                        doc_id=meta.doc_id,
                    )

            # --- extract:统一产出 markdown ---
            doc = await asyncio.to_thread(extract, item.path)
            if doc.is_empty:
                return await self._fail(
                    item, "提取结果为空(可能是扫描件,需要注册 OCR 后端)"
                )
            if doc.meta:
                meta.extras.update(doc.meta)
            if doc.ocr_pages:
                meta.extras["ocr_pages"] = doc.ocr_pages

            # --- chunk ---
            chunks = await asyncio.to_thread(
                self.chunker.chunk, doc.file_markdown, doc.page_markdowns, meta.file_name
            )
            if not chunks:
                return await self._fail(item, "切块结果为空")

            # --- embed ---
            vectors = await self.embedder.embed([c.page_content for c in chunks])
            if len(vectors) != len(chunks):
                raise RuntimeError(f"向量数({len(vectors)})与块数({len(chunks)})不一致")

            # --- index:先删旧块,避免文件变短后残留 ---
            await self.store.delete_doc(meta.doc_id)
            docs = [IndexDoc.from_chunk(c, meta, v) for c, v in zip(chunks, vectors)]
            ok, errors = await self.store.index_docs(docs)
            if errors:
                return await self._fail(item, errors[0], n_chunks=ok)

            await self._register(item, n_chunks=ok, ocr_pages=doc.ocr_pages)
            return IngestResult(
                file_path=meta.file_path,
                status=IngestStatus.INDEXED,
                doc_id=meta.doc_id,
                n_chunks=ok,
            )

        except Exception as exc:
            logger.exception("处理 %s 失败", meta.file_path)
            return await self._fail(item, f"{type(exc).__name__}: {exc}")

    # ----------------------------------------------------------------- #
    async def _register(
        self, item: SourceItem, n_chunks: int, ocr_pages: list[int] | None = None
    ) -> None:
        meta = item.meta
        await self.db.upsert_document({
            "doc_id": meta.doc_id,
            "file_name": meta.file_name,
            "file_path": meta.file_path,
            "file_url": meta.file_url,
            "document_type": meta.document_type,
            "content_hash": meta.content_hash,
            "n_chunks": n_chunks,
            "acl": meta.acl_principals,
            "source_kind": item.source_kind,
            "ocr_pages": ocr_pages or [],
            "status": "indexed",
            "error": "",
        })

    async def _fail(self, item: SourceItem, error: str, n_chunks: int = 0) -> IngestResult:
        meta = item.meta
        # 失败也要登记,否则管理界面上"这个文件为什么没进去"无从查起
        try:
            await self.db.upsert_document({
                "doc_id": meta.doc_id,
                "file_name": meta.file_name,
                "file_path": meta.file_path,
                "file_url": meta.file_url,
                "document_type": meta.document_type,
                "content_hash": meta.content_hash,
                "n_chunks": n_chunks,
                "acl": meta.acl_principals,
                "source_kind": item.source_kind,
                "status": "failed",
                "error": error,
            })
        except Exception:
            logger.debug("登记失败状态时出错", exc_info=True)
        return IngestResult(
            file_path=meta.file_path, status=IngestStatus.FAILED,
            doc_id=meta.doc_id, n_chunks=n_chunks, error=error,
        )

    # ----------------------------------------------------------------- #
    async def delete_document(self, doc_id: str) -> int:
        """从索引和登记表里一起删掉。"""
        deleted = await self.store.delete_doc(doc_id)
        await self.db.delete_document(doc_id)
        await self.store.refresh()
        return deleted

    async def reindex_document(self, doc_id: str) -> IngestResult:
        """按登记的路径重新入库(强制,忽略哈希)。"""
        record = await self.db.get_document(doc_id)
        if not record:
            raise KeyError(f"没有这个文档: {doc_id}")

        if record["source_kind"] == "url":
            source: Source = UrlSource([record["file_path"]], acl_principals=record["acl"])
        else:
            path = Path(record["file_path"])
            if not path.exists():
                raise FileNotFoundError(f"源文件已不存在: {path}")
            source = LocalSource(path, acl_principals=record["acl"])

        items = list(source.iter_items())
        if not items:
            raise RuntimeError(f"源不再产出该文档: {record['file_path']}")

        result = await self.ingest_item(items[0], force=True)
        await self.store.refresh()
        return result


async def iter_ingest(
    root: Path | str,
    acl_principals: list[str] | None = None,
    force: bool = False,
) -> AsyncIterator[IngestResult]:
    """流式版本,供 API 的进度推送使用。"""
    pipeline = IngestPipeline()
    await pipeline.store.ensure_index()
    await pipeline.db.init()
    for item in LocalSource(root, acl_principals=acl_principals).iter_items():
        yield await pipeline.ingest_item(item, force=force)
    await pipeline.store.refresh()
