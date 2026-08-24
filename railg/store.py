"""OpenSearch 客户端。

索引结构完全由 schema/mapping.py 生成 —— 这里不允许出现字段名字面量,
否则又会回到"两个地方各写一套字段"的老路。
"""

from __future__ import annotations

import logging
from typing import Any

from opensearchpy import AsyncOpenSearch, NotFoundError
from opensearchpy.helpers import async_bulk

from railg.config import Settings, get_settings
from railg.schema.document import IndexDoc
from railg.schema.mapping import build_index_body, verify_contract

logger = logging.getLogger(__name__)


class Store:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.index = self.settings.store.index
        self.dims = self.settings.embedding.dims
        self._client = AsyncOpenSearch(
            hosts=[self.settings.store.url],
            http_compress=True,
            timeout=60,
            max_retries=3,
            retry_on_timeout=True,
        )

    @property
    def client(self) -> AsyncOpenSearch:
        return self._client

    async def aclose(self) -> None:
        await self._client.close()

    # ----------------------------------------------------------------- #
    # 索引生命周期
    # ----------------------------------------------------------------- #
    async def ping(self, timeout: float = 3.0) -> bool:
        """连通性探测。

        用短超时且不重试:服务没起是最常见的情况,让它快速失败,
        而不是把默认的 60s + 3 次重试耗完(测试里跳过集成用例会明显变慢)。

        另外 opensearch-py 连不上时会把整段 aiohttp 栈打到日志里,对这种情况
        毫无价值,反而盖住我们自己的提示,所以临时压掉。
        """
        transport_log = logging.getLogger("opensearch")
        previous = transport_log.level
        transport_log.setLevel(logging.CRITICAL)
        try:
            return await self._client.ping(
                request_timeout=timeout, params={"error_trace": "false"}
            )
        except TypeError:
            # 老版本客户端不认 request_timeout,退回默认行为
            try:
                return await self._client.ping()
            except Exception:
                return False
        except Exception as exc:
            logger.debug("连接 OpenSearch 失败 (%s): %s", self.settings.store.url, exc)
            return False
        finally:
            transport_log.setLevel(previous)

    async def ensure_index(self, synonyms: list[str] | None = None) -> bool:
        """确保索引存在。返回 True 表示本次新建。"""
        verify_contract(self.dims)  # 启动即校验契约,不等到检索时才炸

        if await self._client.indices.exists(index=self.index):
            await self._check_dims()
            return False

        body = build_index_body(self.dims, synonyms)
        await self._client.indices.create(index=self.index, body=body)
        logger.info("已创建索引 %s(向量维度 %d)", self.index, self.dims)
        return True

    async def _check_dims(self) -> None:
        """已存在的索引维度必须与当前 embedding 模型一致,否则写入会静默出错。"""
        mapping = await self._client.indices.get_mapping(index=self.index)
        props = next(iter(mapping.values()))["mappings"].get("properties", {})
        actual = props.get("semantic_vector", {}).get("dimension")
        if actual and actual != self.dims:
            raise RuntimeError(
                f"索引 {self.index} 的向量维度是 {actual},当前配置是 {self.dims}。"
                "换了 embedding 模型就必须重建索引:railg reset 之后重新 ingest。"
            )

    async def drop_index(self) -> None:
        try:
            await self._client.indices.delete(index=self.index)
            logger.info("已删除索引 %s", self.index)
        except NotFoundError:
            pass

    async def refresh(self) -> None:
        await self._client.indices.refresh(index=self.index)

    async def count(self) -> int:
        try:
            resp = await self._client.count(index=self.index)
            return int(resp.get("count", 0))
        except NotFoundError:
            return 0

    async def stats(self) -> dict[str, Any]:
        """索引概览:块数、文档数、各类型分布。"""
        if not await self._client.indices.exists(index=self.index):
            return {"exists": False, "n_chunks": 0, "n_docs": 0, "by_type": {}}
        resp = await self._client.search(
            index=self.index,
            body={
                "size": 0,
                "aggs": {
                    "docs": {"cardinality": {"field": "doc_id"}},
                    "types": {"terms": {"field": "document_type", "size": 20}},
                },
            },
        )
        aggs = resp.get("aggregations", {})
        return {
            "exists": True,
            "n_chunks": resp["hits"]["total"]["value"],
            "n_docs": aggs.get("docs", {}).get("value", 0),
            "by_type": {
                b["key"]: b["doc_count"] for b in aggs.get("types", {}).get("buckets", [])
            },
        }

    # ----------------------------------------------------------------- #
    # 写入
    # ----------------------------------------------------------------- #
    async def index_docs(self, docs: list[IndexDoc]) -> tuple[int, list[str]]:
        if not docs:
            return 0, []
        actions = [
            {"_op_type": "index", "_index": self.index,
             "_id": d.chunk_uid, "_source": d.to_source()}
            for d in docs
        ]
        ok, errors = await async_bulk(
            self._client, actions,
            chunk_size=self.settings.store.bulk_size,
            raise_on_error=False,
            request_timeout=120,
        )
        messages = [str(e)[:300] for e in (errors or [])]
        if messages:
            logger.error("bulk 写入有 %d 条失败,首条: %s", len(messages), messages[0])
        return ok, messages

    async def delete_doc(self, doc_id: str) -> int:
        """删除一个源文档的全部 chunk。重建同一文件时先调它,避免残留旧块。"""
        try:
            resp = await self._client.delete_by_query(
                index=self.index,
                body={"query": {"term": {"doc_id": doc_id}}},
                refresh=True,
                conflicts="proceed",
            )
            return int(resp.get("deleted", 0))
        except NotFoundError:
            return 0

    async def get_content_hash(self, doc_id: str) -> str | None:
        """取已入库文档的内容哈希 —— delta 增量的依据。"""
        try:
            resp = await self._client.search(
                index=self.index,
                body={
                    "size": 1,
                    "query": {"term": {"doc_id": doc_id}},
                    "_source": ["content_hash"],
                },
            )
        except NotFoundError:
            return None
        hits = resp["hits"]["hits"]
        return hits[0]["_source"].get("content_hash") if hits else None

    # ----------------------------------------------------------------- #
    # 查询
    # ----------------------------------------------------------------- #
    async def search(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._client.search(index=self.index, body=body)

    async def fetch_siblings(
        self, doc_id: str, section_id: int, context_id: int, size: int = 50
    ) -> list[dict[str, Any]]:
        """取同一上下文块下的全部兄弟 chunk —— 父块还原用。"""
        resp = await self._client.search(
            index=self.index,
            body={
                "size": size,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"doc_id": doc_id}},
                            {"term": {"section_id": section_id}},
                            {"term": {"context_id": context_id}},
                        ]
                    }
                },
                "sort": [{"chunk_index": "asc"}],
                "_source": {"excludes": ["semantic_vector"]},
            },
        )
        return resp["hits"]["hits"]


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


async def close_store() -> None:
    global _store
    if _store is not None:
        await _store.aclose()
        _store = None
