"""数据源连接器。

    LocalSource   本地文件 / 目录
    UrlSource     网页或在线文件(个人知识库最常见的第二类来源)

接 S3 / Confluence / Notion 时,实现 `Source` 并产出同样的 `SourceItem` 即可,
下游 extract → chunk → embed → index 零改动。

★ ACL 在这一层产出。`acl_principals` 是 Source 的固有职责而非可选步骤 ——
  权限字段一旦漏写,检索侧的过滤就会恒返回空结果,且不报错。
"""

from __future__ import annotations

import hashlib
import logging
import re
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

from railg.ingest.extractors import supported_extensions
from railg.schema.document import PUBLIC, DocumentMeta, Principal

logger = logging.getLogger(__name__)

SKIP_DIRS = {
    ".git", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".idea", ".vscode", ".railg", ".pytest_cache",
}


@dataclass(slots=True)
class SourceItem:
    """一个待入库的源文件。"""

    path: Path
    meta: DocumentMeta
    source_kind: str = "local"
    # 提取时才读内容,避免大目录一次性占满内存
    _payload: bytes | None = field(default=None, repr=False)

    def read_bytes(self) -> bytes:
        if self._payload is None:
            self._payload = self.path.read_bytes()
        return self._payload


class Source(ABC):
    name: str = "source"

    @abstractmethod
    def iter_items(self) -> Iterator[SourceItem]:
        ...


# --------------------------------------------------------------------------- #
class LocalSource(Source):
    name = "local"

    def __init__(
        self,
        root: Path | str,
        acl_principals: list[Principal] | None = None,
        patterns: list[str] | None = None,
        recursive: bool = True,
        max_file_mb: float = 100.0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.acl_principals = acl_principals or [PUBLIC]
        self.patterns = patterns
        self.recursive = recursive
        self.max_bytes = int(max_file_mb * 1024 * 1024)

    def iter_items(self) -> Iterator[SourceItem]:
        if self.root.is_file():
            item = self._make_item(self.root)
            if item:
                yield item
            return

        if not self.root.is_dir():
            raise FileNotFoundError(f"路径不存在: {self.root}")

        supported = supported_extensions()
        globber = self.root.rglob if self.recursive else self.root.glob
        seen: set[Path] = set()

        for pattern in self.patterns or ["*"]:
            for path in sorted(globber(pattern)):
                if path in seen or not path.is_file():
                    continue
                if any(part in SKIP_DIRS for part in path.parts):
                    continue
                if path.name.startswith("~$") or path.name.startswith("."):
                    continue
                if path.suffix.lower() not in supported:
                    continue
                seen.add(path)
                item = self._make_item(path)
                if item:
                    yield item

    def _make_item(self, path: Path) -> SourceItem | None:
        try:
            stat = path.stat()
        except OSError as exc:
            logger.warning("无法读取 %s: %s", path, exc)
            return None

        if stat.st_size > self.max_bytes:
            logger.warning("跳过超大文件 %s (%.1f MB)", path.name, stat.st_size / 1024 / 1024)
            return None
        if stat.st_size == 0:
            return None

        try:
            rel = path.relative_to(self.root)
        except ValueError:
            rel = Path(path.name)

        meta = DocumentMeta(
            doc_id=DocumentMeta.make_doc_id(str(path)),
            file_name=path.name,
            file_path=str(path),
            file_url=path.as_uri(),
            document_type=path.suffix.lower().lstrip("."),
            last_modified_date=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            acl_principals=list(self.acl_principals),
            extras={"relative_path": str(rel)},
        )
        return SourceItem(path=path, meta=meta, source_kind="local")


# --------------------------------------------------------------------------- #
_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._\-一-鿿]+")


class UrlSource(Source):
    """抓取网页或在线文件。

    下载到本地缓存目录后复用同一套提取器 —— 网页走 HtmlExtractor,
    在线 PDF 走 PdfExtractor,不需要为 URL 单独写一条解析路径。

    doc_id 由 URL 决定,所以同一个 URL 重复抓取会走 delta 增量:
    内容没变就跳过,变了就重建。
    """

    name = "url"

    def __init__(
        self,
        urls: list[str],
        acl_principals: list[Principal] | None = None,
        cache_dir: Path | str | None = None,
        timeout: float = 30.0,
        max_file_mb: float = 100.0,
    ) -> None:
        self.urls = [u.strip() for u in urls if u.strip()]
        self.acl_principals = acl_principals or [PUBLIC]
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "railg-url"
        self.timeout = timeout
        self.max_bytes = int(max_file_mb * 1024 * 1024)

    def iter_items(self) -> Iterator[SourceItem]:
        import httpx

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with httpx.Client(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": "railg/0.2 (+https://github.com/yhnapoleon/RAilG)"},
        ) as client:
            for url in self.urls:
                item = self._fetch(client, url)
                if item:
                    yield item

    def _fetch(self, client, url: str) -> SourceItem | None:
        try:
            resp = client.get(url)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("抓取失败 %s: %s", url, exc)
            return None

        payload = resp.content
        if len(payload) > self.max_bytes:
            logger.warning("跳过超大响应 %s (%.1f MB)", url, len(payload) / 1024 / 1024)
            return None
        if not payload:
            return None

        suffix = self._guess_suffix(url, resp.headers.get("content-type", ""))
        if suffix not in supported_extensions():
            logger.warning("不支持的内容类型 %s (%s)", url, suffix)
            return None

        path = self.cache_dir / f"{hashlib.sha1(url.encode()).hexdigest()[:16]}{suffix}"
        path.write_bytes(payload)

        meta = DocumentMeta(
            doc_id=DocumentMeta.make_doc_id(url),
            file_name=self._guess_name(url, suffix),
            file_path=url,
            file_url=url,
            document_type=suffix.lstrip("."),
            last_modified_date=datetime.now(timezone.utc),
            acl_principals=list(self.acl_principals),
            extras={"source_url": url},
        )
        return SourceItem(path=path, meta=meta, source_kind="url", _payload=payload)

    @staticmethod
    def _guess_suffix(url: str, content_type: str) -> str:
        ctype = content_type.split(";")[0].strip().lower()
        by_type = {
            "text/html": ".html",
            "application/xhtml+xml": ".html",
            "application/pdf": ".pdf",
            "text/markdown": ".md",
            "text/plain": ".txt",
            "text/csv": ".csv",
            "application/json": ".json",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        }
        if ctype in by_type:
            return by_type[ctype]
        suffix = Path(unquote(urlparse(url).path)).suffix.lower()
        return suffix or ".html"

    @staticmethod
    def _guess_name(url: str, suffix: str) -> str:
        parsed = urlparse(url)
        stem = Path(unquote(parsed.path)).stem or parsed.netloc or "page"
        stem = _FILENAME_SAFE.sub("-", stem).strip("-")[:80] or "page"
        return f"{stem}{suffix}"
