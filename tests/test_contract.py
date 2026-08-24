"""★ schema 契约测试。

守的是"写入侧和读取侧看到的是同一份字段清单"。没有这道闸时的典型故障:
  · 检索层读某个向量字段,但没有任何地方生产它  → 那条召回路径是死的
  · 按某个字段过滤,但入库时压根不写            → 过滤条件一开,结果恒空
  · 字段靠动态映射,而非显式定义                → 行为随第一条数据而变

三种都不报错、不打日志,只是结果悄悄变差。这里让它们在 CI 就挂掉。
"""

from __future__ import annotations

import pytest

from railg.retrieval import processors as P
from railg.schema.document import IndexDoc
from railg.schema.mapping import (
    RETRIEVAL_READ_FIELDS,
    build_index_body,
    build_properties,
    verify_contract,
)

DIMS = 1024


def test_every_retrieval_field_is_mapped():
    """检索层读的每个字段,mapping 里都必须有定义。"""
    props = build_properties(DIMS)
    missing = [f for f in RETRIEVAL_READ_FIELDS if f not in props]
    assert not missing, f"检索层读取但未映射的字段: {missing}"


def test_every_model_field_is_mapped():
    """IndexDoc 写的每个字段,mapping 里都必须有定义。"""
    props = build_properties(DIMS)
    missing = [f for f in IndexDoc.model_fields if f not in props]
    assert not missing, f"IndexDoc 定义但未映射的字段: {missing}"


def test_processor_field_constants_are_declared():
    """processors.py 里用到的字段常量必须登记在 RETRIEVAL_READ_FIELDS。

    这条防的是"加了个新过滤器却忘了登记字段"。
    """
    used = {
        P.TEXT_FIELD, P.NAME_FIELD, P.VECTOR_FIELD,
        P.DATE_FIELD, P.TYPE_FIELD, P.ACL_FIELD,
    }
    undeclared = used - set(RETRIEVAL_READ_FIELDS)
    assert not undeclared, f"处理器用到但未登记的字段: {undeclared}"


def test_acl_field_is_written_and_read():
    """★ 权限字段必须同时存在于写入模型与读取声明。

    漏了任何一边,权限过滤都会恒返回空结果,且不报错。
    """
    assert P.ACL_FIELD in IndexDoc.model_fields, "写入模型缺少权限字段"
    assert P.ACL_FIELD in RETRIEVAL_READ_FIELDS, "检索层未声明权限字段"
    assert P.ACL_FIELD in build_properties(DIMS), "mapping 缺少权限字段"


@pytest.mark.parametrize("dims", [384, 768, 1024, 1536])
def test_vector_dimension_follows_config(dims):
    """mapping 的维度永远跟着配置走,不能有第二处硬编码。"""
    assert build_properties(dims)["semantic_vector"]["dimension"] == dims
    verify_contract(dims)


def test_missing_read_field_is_detected(monkeypatch):
    """人为制造"读了但没映射",契约测试必须能抓到。"""
    import railg.schema.mapping as m

    monkeypatch.setattr(m, "RETRIEVAL_READ_FIELDS", (*RETRIEVAL_READ_FIELDS, "ghost_field"))
    with pytest.raises(RuntimeError, match="ghost_field"):
        m.verify_contract(DIMS)


def test_verify_contract_passes():
    verify_contract(DIMS)


def test_index_body_is_wellformed():
    body = build_index_body(DIMS)
    assert body["settings"]["index"]["knn"] is True
    assert "indexing_analyzer" in body["settings"]["analysis"]["analyzer"]
    assert "query_analyzer" in body["settings"]["analysis"]["analyzer"]
    assert body["mappings"]["properties"]["semantic_vector"]["dimension"] == DIMS


def test_snippet_is_explicitly_mapped():
    """展示字段必须显式映射,不能靠动态映射,否则行为随数据而变。"""
    props = build_properties(DIMS)
    assert "snippet" in props
    assert props["snippet"]["type"] == "text"
