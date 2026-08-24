"""查询处理器链。

一串 `(term, builder, ctx) -> term` 的处理器,依次改写查询词、
往 QueryBuilder 上挂子句,任一环出错即短路。加过滤条件不用动主流程。

★ 权限过滤(process_acl)用的 `acl_principals` 由 schema 统一定义,
  写入侧(sources.py)与这里共用同一份模型,并由契约测试守卫 ——
  过滤字段没人写入会导致查询恒空,这种错误不报错也没日志。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

from railg.config import RetrievalConfig
from railg.retrieval.builder import QueryBuilder
from railg.schema.document import Identity

logger = logging.getLogger(__name__)

# 字段名集中在这里,与 schema/mapping.py 的 RETRIEVAL_READ_FIELDS 对应
TEXT_FIELD = "text_to_index"
NAME_FIELD = "file_name"
VECTOR_FIELD = "semantic_vector"
DATE_FIELD = "last_modified_date"
TYPE_FIELD = "document_type"
ACL_FIELD = "acl_principals"

# 查询里的内联过滤语法
_FILTER_RE = re.compile(
    r"\b(?:filetype|type|ext)\s*:\s*([A-Za-z0-9,]+)|"
    r"\b(?:after|since)\s*:\s*(\d{4}-\d{2}-\d{2})|"
    r"\b(?:before|until)\s*:\s*(\d{4}-\d{2}-\d{2})",
    flags=re.I,
)

# 文件类型别名 → 实际后缀
FILE_TYPE_ALIASES: dict[str, list[str]] = {
    "excel": ["xlsx", "xls", "xlsm", "csv"],
    "word": ["docx", "doc"],
    "doc": ["docx", "doc"],
    "web": ["html", "htm"],
    "ppt": ["pptx", "ppt"],
    "text": ["txt", "md", "markdown"],
    "markdown": ["md", "markdown"],
}

# 时间意图关键词(中英)。命中则加大时间衰减权重。
_RECENCY_WORDS = (
    "latest", "newest", "recent", "recently", "current", "up to date", "this year",
    "最新", "最近", "近期", "当前", "今年", "本月", "本周", "目前",
)


@dataclass
class QueryContext:
    """一次检索的全部输入。"""

    # dataclass 不接受可变对象作默认值,用工厂;语义等同 ANONYMOUS
    identity: Identity = field(default_factory=Identity)
    config: RetrievalConfig = field(default_factory=RetrievalConfig)
    query_vector: list[float] | None = None
    file_types: list[str] = field(default_factory=list)
    exclude_types: list[str] = field(default_factory=list)
    date_from: str = ""
    date_to: str = ""
    time_decay: bool = True
    error: str = ""

    def fail(self, msg: str) -> str:
        self.error = msg
        logger.error("检索处理器短路: %s", msg)
        return ""


Processor = Callable[[str, QueryBuilder, QueryContext], str]


# --------------------------------------------------------------------------- #
# 处理器
# --------------------------------------------------------------------------- #
def process_acl(term: str, builder: QueryBuilder, ctx: QueryContext) -> str:
    """★ 文档级权限过滤。

    放进 filter 而不是 must:权限不参与打分,且 filter 可被 OpenSearch 缓存。
    principals 恒定包含 "public",所以公开文档对任何人可见。
    """
    principals = ctx.identity.principals()
    if not principals:
        return ctx.fail("身份未解析出任何权限主体")
    builder.add_filter({"terms": {ACL_FIELD: principals}})
    return term


def detect_inline_filters(term: str, ctx: QueryContext) -> str:
    """从查询里剥离 `filetype:pdf` / `after:2024-01-01` 这类内联过滤。"""
    found = False
    for m in _FILTER_RE.finditer(term):
        ftype, after, before = m.groups()
        if ftype:
            for token in ftype.lower().split(","):
                ctx.file_types.extend(FILE_TYPE_ALIASES.get(token, [token]))
            found = True
        elif after:
            ctx.date_from, found = after, True
        elif before:
            ctx.date_to, found = before, True
    if found:
        term = _FILTER_RE.sub(" ", term)
        term = re.sub(r"\s{2,}", " ", term).strip()
    return term


def process_file_type(term: str, builder: QueryBuilder, ctx: QueryContext) -> str:
    term = detect_inline_filters(term, ctx)
    if ctx.file_types:
        builder.add_filter({"terms": {TYPE_FIELD: sorted(set(ctx.file_types))}})
    if ctx.exclude_types:
        builder.add_exclusion({"terms": {TYPE_FIELD: sorted(set(ctx.exclude_types))}})
    return term


def process_date_filter(term: str, builder: QueryBuilder, ctx: QueryContext) -> str:
    rng: dict[str, str] = {}
    if ctx.date_from:
        rng["gte"] = ctx.date_from
    if ctx.date_to:
        rng["lte"] = ctx.date_to
    if rng:
        builder.add_filter({"range": {DATE_FIELD: rng}})
    return term


def process_temporal(term: str, builder: QueryBuilder, ctx: QueryContext) -> str:
    """时间衰减。查询里出现"最新/latest"这类词时,权重翻倍。

    做法是高斯衰减 + 关键词加权,关键词表中英双语。
    """
    if not ctx.time_decay:
        return term

    lowered = term.lower()
    has_recency = any(w in lowered for w in _RECENCY_WORDS)
    origin = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    builder.add_function_query({
        "gauss": {
            DATE_FIELD: {"origin": origin, "scale": "180d", "decay": 0.5},
        },
        "weight": 1.0 * (2 if has_recency else 1),
    })
    return term


def process_embeddings(term: str, builder: QueryBuilder, ctx: QueryContext) -> str:
    """向量召回。向量由 service 层预先算好,处理器只负责挂子句。"""
    if not ctx.query_vector:
        return term
    cfg = ctx.config
    builder.add_knn({
        VECTOR_FIELD: {
            "vector": ctx.query_vector,
            "k": cfg.top_k,
            "boost": cfg.vector_weight,
        }
    })
    return term


def process_file_content(term: str, builder: QueryBuilder, ctx: QueryContext) -> str:
    """正文 BM25:精确匹配 + 模糊匹配 + 短语匹配,三路并行。

    三路都保留是有意义的 —— 精确管准确率,模糊管错别字,短语管固定搭配。
    """
    if not term.strip():
        return term
    w = ctx.config.bm25_weight
    (builder
        .add_query({"match": {TEXT_FIELD: {"query": "", "boost": w}}})
        .add_query({"match": {TEXT_FIELD: {"query": "", "fuzziness": "AUTO", "boost": w * 0.5}}})
        .add_query({"match_phrase": {TEXT_FIELD: {"query": "", "slop": 2, "boost": w * 1.5}}}))
    return term


def process_file_name(term: str, builder: QueryBuilder, ctx: QueryContext) -> str:
    """文件名匹配 —— 用户经常直接搜文件名。"""
    if not term.strip():
        return term
    w = ctx.config.bm25_weight
    (builder
        .add_query({"match": {NAME_FIELD: {"query": "", "boost": w * 2}}})
        .add_query({"match": {NAME_FIELD: {"query": "", "fuzziness": "AUTO", "boost": w}}}))
    return term


# --------------------------------------------------------------------------- #
# 链
# --------------------------------------------------------------------------- #
class QueryProcessor:
    def __init__(self, processors: list[Processor] | None = None) -> None:
        self.processors: list[Processor] = processors or []
        self.builder = QueryBuilder()

    def add(self, processor: Processor) -> "QueryProcessor":
        self.processors.append(processor)
        return self

    def run(self, query: str, ctx: QueryContext, size: int) -> dict | None:
        """跑完整条链。任一处理器置了 error 就短路返回 None。"""
        term = query
        for processor in self.processors:
            term = processor(term, self.builder, ctx)
            if ctx.error:
                return None
        return self.builder.build_body(term, size=size)


def compose(semantic_only: bool = False, keyword_only: bool = False) -> QueryProcessor:
    """默认链。顺序有讲究:先过滤后打分,能让 kNN 的前置过滤拿到完整条件。"""
    processor = QueryProcessor()
    processor.add(process_acl)          # ★ 权限最先,且必定执行
    processor.add(process_file_type)    # 剥离内联过滤,改写 term
    processor.add(process_date_filter)
    processor.add(process_temporal)

    if not keyword_only:
        processor.add(process_embeddings)
    if not semantic_only:
        processor.add(process_file_content)
        processor.add(process_file_name)
    return processor
