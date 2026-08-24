"""★ 唯一真相源。

索引侧写什么、检索侧读什么、OpenSearch mapping 长什么样,全部由这里推导。

写入侧和读取侧各维护一份字段清单,是这类系统最容易出的问题:漏写一个
过滤字段,查询会静默返回空结果,不报错也没日志。一份模型加一个契约测试
(tests/test_contract.py)可以根除这一类。
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# 权限主体
# --------------------------------------------------------------------------- #
#   "public"           所有人可见
#   "user:<sub>"       指定用户
#   "group:<gid>"      指定用户组
#   "role:<name>"      指定角色
Principal = str

PUBLIC: Principal = "public"


class Identity(BaseModel):
    """当前请求者。auth.enabled=false 时是一个匿名 public 身份。"""

    sub: str = "anonymous"
    display_name: str = "anonymous"
    groups: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)

    def principals(self) -> list[Principal]:
        """展开为可用于 terms filter 的主体集合。"""
        out = [PUBLIC]
        if self.sub and self.sub != "anonymous":
            out.append(f"user:{self.sub}")
        out.extend(f"group:{g}" for g in self.groups)
        out.extend(f"role:{r}" for r in self.roles)
        return out


ANONYMOUS = Identity()


# --------------------------------------------------------------------------- #
# 切块产物
# --------------------------------------------------------------------------- #
ChunkFormat = Literal["text", "table", "heading", "appendix"]


class ChunkMeta(BaseModel):
    """chunker 产出的结构化元信息。

    section_id / context_id / chunk_index 三者共同支撑父块还原:
    同一 (doc_id, section_id, context_id) 下按 chunk_index 排序即可无损拼回。
    """

    source: str = ""
    header1: str = ""
    header2: str = ""
    page_no: int = 1
    section_id: int = 0
    context_id: int = 0
    format: ChunkFormat = "text"


class Chunk(BaseModel):
    page_content: str
    metadata: ChunkMeta = Field(default_factory=ChunkMeta)
    chunk_index: int = 0


# --------------------------------------------------------------------------- #
# 文档级元信息
# --------------------------------------------------------------------------- #
class DocumentMeta(BaseModel):
    doc_id: str = ""
    file_name: str = ""
    file_path: str = ""
    file_url: str = ""
    document_type: str = ""
    last_modified_date: datetime | None = None
    content_hash: str = ""
    #  ★ 权限:由 source connector 产出,默认公开
    acl_principals: list[Principal] = Field(default_factory=lambda: [PUBLIC])
    extras: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def make_doc_id(file_path: str) -> str:
        return hashlib.sha1(file_path.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def make_content_hash(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- #
# 落库文档 —— OpenSearch 里一条 _source 就长这样
# --------------------------------------------------------------------------- #
class IndexDoc(BaseModel):
    doc_id: str
    chunk_uid: str
    chunk_index: int

    file_name: str = ""
    file_path: str = ""
    file_url: str = ""
    document_type: str = ""
    last_modified_date: datetime | None = None
    content_hash: str = ""

    # 检索字段 / 展示字段分离:前者可被分析器改写,后者保持原文
    text_to_index: str = ""
    snippet: str = ""
    semantic_vector: list[float] = Field(default_factory=list)

    page_no: int = 1
    section_id: int = 0
    context_id: int = 0
    header1: str = ""
    header2: str = ""
    format: ChunkFormat = "text"

    acl_principals: list[Principal] = Field(default_factory=lambda: [PUBLIC])
    metadata: dict[str, Any] = Field(default_factory=dict)
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_chunk(cls, chunk: Chunk, meta: DocumentMeta, vector: list[float]) -> "IndexDoc":
        return cls(
            doc_id=meta.doc_id,
            chunk_uid=f"{meta.doc_id}:{chunk.chunk_index}",
            chunk_index=chunk.chunk_index,
            file_name=meta.file_name,
            file_path=meta.file_path,
            file_url=meta.file_url,
            document_type=meta.document_type,
            last_modified_date=meta.last_modified_date,
            content_hash=meta.content_hash,
            text_to_index=chunk.page_content,
            snippet=chunk.page_content,
            semantic_vector=vector,
            page_no=chunk.metadata.page_no,
            section_id=chunk.metadata.section_id,
            context_id=chunk.metadata.context_id,
            header1=chunk.metadata.header1,
            header2=chunk.metadata.header2,
            format=chunk.metadata.format,
            acl_principals=meta.acl_principals or [PUBLIC],
            metadata=meta.extras,
        )

    def to_source(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        return {k: v for k, v in d.items() if v not in (None, "", [], {})}


# --------------------------------------------------------------------------- #
# 检索产物
# --------------------------------------------------------------------------- #
class Candidate(BaseModel):
    """一条检索结果。父块还原后 snippet 会被替换为拼接后的完整上下文。"""

    chunk_uid: str = ""
    doc_id: str = ""
    file_name: str = ""
    file_url: str = ""
    file_path: str = ""
    snippet: str = ""
    page_no: int = 1
    section_id: int = 0
    context_id: int = 0
    chunk_index: int = 0
    header1: str = ""
    header2: str = ""
    format: ChunkFormat = "text"
    document_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    score: float = 0.0            # OpenSearch 原始分
    normalized_score: float = 0.0  # 归一化到 [0,1]
    rerank_score: float | None = None
    is_parent: bool = False        # 是否由父块还原产生

    @classmethod
    def from_hit(cls, hit: dict[str, Any]) -> "Candidate":
        src = hit.get("_source", {})
        return cls(
            chunk_uid=src.get("chunk_uid", hit.get("_id", "")),
            doc_id=src.get("doc_id", ""),
            file_name=src.get("file_name", ""),
            file_url=src.get("file_url", ""),
            file_path=src.get("file_path", ""),
            snippet=src.get("snippet", "") or src.get("text_to_index", ""),
            page_no=src.get("page_no", 1),
            section_id=src.get("section_id", 0),
            context_id=src.get("context_id", 0),
            chunk_index=src.get("chunk_index", 0),
            header1=src.get("header1", ""),
            header2=src.get("header2", ""),
            format=src.get("format", "text"),
            document_type=src.get("document_type", ""),
            metadata=src.get("metadata", {}) or {},
            score=hit.get("_score", 0.0) or 0.0,
        )

    def parent_key(self) -> tuple[str, int, int]:
        """父块归并键 —— 与 chunker 的三级结构一一对应。"""
        return (self.doc_id, self.section_id, self.context_id)

    def citation_label(self) -> str:
        loc = f"p.{self.page_no}" if self.page_no else ""
        head = self.header2 or self.header1
        parts = [p for p in (self.file_name, head, loc) if p]
        return " · ".join(parts)


class IngestStatus(str, Enum):
    INDEXED = "indexed"
    SKIPPED_UNCHANGED = "skipped_unchanged"
    FAILED = "failed"


class IngestResult(BaseModel):
    file_path: str
    status: IngestStatus
    doc_id: str = ""
    n_chunks: int = 0
    error: str = ""
