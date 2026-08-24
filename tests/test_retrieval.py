"""检索链路测试:QueryBuilder 结构、处理器链、权限过滤。"""

from __future__ import annotations

from railg.config import RetrievalConfig
from railg.retrieval.builder import QueryBuilder
from railg.retrieval.processors import (
    ACL_FIELD,
    TYPE_FIELD,
    VECTOR_FIELD,
    QueryContext,
    compose,
    detect_inline_filters,
    process_acl,
)
from railg.schema.document import ANONYMOUS, Identity


def _ctx(identity: Identity = ANONYMOUS, **kw) -> QueryContext:
    return QueryContext(identity=identity, config=RetrievalConfig(), **kw)


# --------------------------------------------------------------------------- #
# QueryBuilder
# --------------------------------------------------------------------------- #
def test_builder_produces_bool_skeleton():
    b = QueryBuilder()
    b.add_query({"match": {"text_to_index": {"query": ""}}})
    b.add_filter({"terms": {ACL_FIELD: ["public"]}})
    query = b.build_query("年假")

    assert "bool" in query
    assert query["bool"]["should"][0]["match"]["text_to_index"]["query"] == "年假"
    assert query["bool"]["filter"] == [{"terms": {ACL_FIELD: ["public"]}}]


def test_knn_clause_inherits_filters():
    """kNN 必须拿到与外层一致的过滤条件,否则先取 top-k 再过滤会损失召回。"""
    b = QueryBuilder()
    b.add_filter({"terms": {ACL_FIELD: ["public", "user:alice"]}})
    b.add_knn({VECTOR_FIELD: {"vector": [0.1] * 8, "k": 10}})
    query = b.build_query("测试")

    knn = next(c for c in query["bool"]["should"] if "knn" in c)
    knn_filter = knn["knn"][VECTOR_FIELD]["filter"]
    assert knn_filter["bool"]["filter"] == [{"terms": {ACL_FIELD: ["public", "user:alice"]}}]


def test_builder_falls_back_to_match_all():
    """只有过滤条件时应退化为纯过滤,而不是产出空 bool。"""
    b = QueryBuilder()
    b.add_filter({"term": {"document_type": "pdf"}})
    query = b.build_query("")
    assert query["bool"]["must"] == [{"match_all": {}}]


def test_body_excludes_vector():
    b = QueryBuilder()
    b.add_query({"match": {"text_to_index": {"query": ""}}})
    body = b.build_body("x", size=7)
    assert body["size"] == 7
    assert VECTOR_FIELD in body["_source"]["excludes"]


# --------------------------------------------------------------------------- #
# 权限
# --------------------------------------------------------------------------- #
def test_acl_filter_always_present_for_anonymous():
    b = QueryBuilder()
    process_acl("q", b, _ctx())
    assert b.filters == [{"terms": {ACL_FIELD: ["public"]}}]


def test_acl_expands_user_groups_roles():
    identity = Identity(sub="alice", groups=["finance", "hr"], roles=["admin"])
    b = QueryBuilder()
    process_acl("q", b, _ctx(identity))

    principals = b.filters[0]["terms"][ACL_FIELD]
    assert principals == [
        "public", "user:alice", "group:finance", "group:hr", "role:admin",
    ]


def test_acl_is_first_in_chain():
    """权限必须最先执行,且不可被后续处理器绕过。"""
    processor = compose()
    assert processor.processors[0] is process_acl


def test_full_chain_always_emits_acl_filter():
    processor = compose()
    ctx = _ctx(Identity(sub="bob"))
    body = processor.run("年假 有几天", ctx, size=10)

    filters = body["query"]["bool"].get("filter", [])
    assert any(ACL_FIELD in f.get("terms", {}) for f in filters), "权限过滤丢失"


# --------------------------------------------------------------------------- #
# 内联过滤
# --------------------------------------------------------------------------- #
def test_inline_filetype_filter():
    ctx = _ctx()
    term = detect_inline_filters("预算表 filetype:excel", ctx)
    assert "filetype" not in term
    assert term.strip() == "预算表"
    assert set(ctx.file_types) >= {"xlsx", "xls"}


def test_inline_date_filters():
    ctx = _ctx()
    term = detect_inline_filters("报告 after:2024-01-01 before:2024-12-31", ctx)
    assert ctx.date_from == "2024-01-01"
    assert ctx.date_to == "2024-12-31"
    assert term.strip() == "报告"


def test_filetype_becomes_terms_filter():
    processor = compose()
    ctx = _ctx()
    body = processor.run("预算 filetype:pdf", ctx, size=5)
    filters = body["query"]["bool"]["filter"]
    assert {"terms": {TYPE_FIELD: ["pdf"]}} in filters


# --------------------------------------------------------------------------- #
# 模式开关
# --------------------------------------------------------------------------- #
def test_keyword_only_has_no_knn():
    processor = compose(keyword_only=True)
    ctx = _ctx(query_vector=[0.1] * 8)
    body = processor.run("测试", ctx, size=5)
    assert not any("knn" in c for c in body["query"]["bool"].get("should", []))


def test_semantic_only_has_no_bm25_match():
    processor = compose(semantic_only=True)
    ctx = _ctx(query_vector=[0.1] * 8)
    body = processor.run("测试", ctx, size=5)
    should = body["query"]["bool"].get("should", [])
    assert any("knn" in c for c in should)
    assert not any("match" in c for c in should)


def test_hybrid_has_both():
    processor = compose()
    ctx = _ctx(query_vector=[0.1] * 8)
    body = processor.run("测试", ctx, size=5)
    should = body["query"]["bool"]["should"]
    assert any("knn" in c for c in should)
    assert any("match" in c for c in should)
