"""对真实 OpenSearch 的集成测试。

OpenSearch 没起时自动跳过,所以可以放心留在默认测试集里:
    docker compose up -d && pytest

用假向量,不依赖任何模型 API —— 这里要验的是 mapping 能不能被引擎接受、
查询 DSL 能不能跑、权限过滤是否真的生效、父块还原能不能取到兄弟块。
这几件事纯单测覆盖不到,而它们出错时通常不报错,只是结果悄悄变空。
"""

from __future__ import annotations

import math
import random

import pytest
import pytest_asyncio

from railg.config import RetrievalConfig, load_settings
from railg.retrieval.parents import construct_parents, normalize_scores
from railg.retrieval.processors import QueryContext, compose
from railg.schema.document import (
    Candidate,
    Chunk,
    ChunkMeta,
    DocumentMeta,
    Identity,
    IndexDoc,
)
from railg.store import Store

DIMS = 8
INDEX = "railg_itest"


def _vec(seed: int, dims: int = DIMS) -> list[float]:
    rng = random.Random(seed)
    raw = [rng.uniform(-1, 1) for _ in range(dims)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


@pytest_asyncio.fixture
async def store():
    settings = load_settings()
    settings.store.index = INDEX
    settings.embedding.dims = DIMS

    s = Store(settings)
    if not await s.ping():
        await s.aclose()
        pytest.skip("OpenSearch 未运行 —— docker compose up -d")

    await s.drop_index()
    await s.ensure_index()
    yield s
    await s.drop_index()
    await s.aclose()


def _make_docs() -> list[IndexDoc]:
    """两篇文档:一篇公开,一篇仅 finance 组可见。

    公开那篇的 section 0 / context 0 下有 3 个连续子块,用来测父块还原。
    """
    docs: list[IndexDoc] = []

    public_meta = DocumentMeta(
        doc_id="pub", file_name="公开手册.md", file_path="/tmp/pub.md",
        file_url="file:///tmp/pub.md", document_type="md",
        content_hash="h1", acl_principals=["public"],
    )
    pieces = [
        "年假制度规定员工每年享有十四天带薪年假。",
        "年假需要提前三个工作日提交申请并由主管审批。",
        "未休完的年假最多结转五天到次年。",
    ]
    for i, text in enumerate(pieces):
        docs.append(IndexDoc.from_chunk(
            Chunk(page_content=text, chunk_index=i,
                  metadata=ChunkMeta(section_id=0, context_id=0, page_no=1,
                                     header2="请假制度")),
            public_meta, _vec(i),
        ))

    secret_meta = DocumentMeta(
        doc_id="sec", file_name="薪酬机密.md", file_path="/tmp/sec.md",
        file_url="file:///tmp/sec.md", document_type="md",
        content_hash="h2", acl_principals=["group:finance"],
    )
    docs.append(IndexDoc.from_chunk(
        Chunk(page_content="年假折算工资的计算方式属于机密信息。", chunk_index=0,
              metadata=ChunkMeta(section_id=0, context_id=0, page_no=1)),
        secret_meta, _vec(99),
    ))
    return docs


@pytest_asyncio.fixture
async def seeded(store: Store):
    ok, errors = await store.index_docs(_make_docs())
    assert not errors, errors
    assert ok == 4
    await store.refresh()
    return store


# --------------------------------------------------------------------------- #
# mapping
# --------------------------------------------------------------------------- #
async def test_index_is_created_with_expected_mapping(store: Store):
    """mapping 能被真实引擎接受 —— knn_vector / 自定义分析器都得过。"""
    mapping = await store.client.indices.get_mapping(index=INDEX)
    props = next(iter(mapping.values()))["mappings"]["properties"]

    assert props["semantic_vector"]["type"] == "knn_vector"
    assert props["semantic_vector"]["dimension"] == DIMS
    assert props["acl_principals"]["type"] == "keyword"
    assert props["snippet"]["type"] == "text"


async def test_dimension_mismatch_is_rejected(store: Store):
    """换了 embedding 模型却没重建索引,必须报错而不是静默写坏。"""
    store.dims = DIMS + 1
    with pytest.raises(RuntimeError, match="维度"):
        await store.ensure_index()


async def test_custom_analyzer_handles_cjk(store: Store):
    resp = await store.client.indices.analyze(
        index=INDEX, body={"analyzer": "indexing_analyzer", "text": "年假制度"}
    )
    tokens = [t["token"] for t in resp["tokens"]]
    assert tokens, "中文没有被切出任何 token"


# --------------------------------------------------------------------------- #
# 检索
# --------------------------------------------------------------------------- #
async def test_bm25_finds_document(seeded: Store):
    ctx = QueryContext(identity=Identity(), config=RetrievalConfig(top_k=10))
    body = compose(keyword_only=True).run("年假 结转", ctx, size=10)
    resp = await seeded.search(body)
    assert resp["hits"]["total"]["value"] > 0


async def test_hybrid_query_runs_on_real_engine(seeded: Store):
    """★ kNN 作为 bool.should 子句 + filter 注入,必须被引擎接受。"""
    ctx = QueryContext(
        identity=Identity(), config=RetrievalConfig(top_k=10), query_vector=_vec(0)
    )
    body = compose().run("年假", ctx, size=10)

    should = body["query"]["bool"]["should"]
    assert any("knn" in c for c in should)
    assert any("match" in c for c in should)

    resp = await seeded.search(body)
    assert resp["hits"]["hits"], "混合召回没有命中"


# --------------------------------------------------------------------------- #
# ★ 权限
# --------------------------------------------------------------------------- #
async def test_anonymous_cannot_see_restricted_doc(seeded: Store):
    ctx = QueryContext(identity=Identity(), config=RetrievalConfig(top_k=20))
    body = compose(keyword_only=True).run("年假", ctx, size=20)
    resp = await seeded.search(body)

    names = {h["_source"]["file_name"] for h in resp["hits"]["hits"]}
    assert "公开手册.md" in names
    assert "薪酬机密.md" not in names, "匿名用户看到了受限文档"


async def test_group_member_sees_restricted_doc(seeded: Store):
    ctx = QueryContext(
        identity=Identity(sub="alice", groups=["finance"]),
        config=RetrievalConfig(top_k=20),
    )
    body = compose(keyword_only=True).run("年假", ctx, size=20)
    resp = await seeded.search(body)

    names = {h["_source"]["file_name"] for h in resp["hits"]["hits"]}
    assert "薪酬机密.md" in names, "finance 组成员看不到本该可见的文档"


async def test_acl_also_constrains_knn_recall(seeded: Store):
    """kNN 子句自带 filter,受限文档不该出现在向量召回里。"""
    ctx = QueryContext(
        identity=Identity(), config=RetrievalConfig(top_k=20), query_vector=_vec(99)
    )
    body = compose(semantic_only=True).run("机密", ctx, size=20)
    resp = await seeded.search(body)

    names = {h["_source"]["file_name"] for h in resp["hits"]["hits"]}
    assert "薪酬机密.md" not in names


# --------------------------------------------------------------------------- #
# ★ 父块还原
# --------------------------------------------------------------------------- #
async def test_fetch_siblings_returns_ordered_chunks(seeded: Store):
    hits = await seeded.fetch_siblings("pub", 0, 0)
    assert len(hits) == 3
    indices = [h["_source"]["chunk_index"] for h in hits]
    assert indices == [0, 1, 2], "兄弟块没有按 chunk_index 排序"


async def test_parent_reconstruction_merges_siblings(seeded: Store):
    """命中一个子块,应还原出整个上下文块。"""
    ctx = QueryContext(identity=Identity(), config=RetrievalConfig(top_k=10))
    body = compose(keyword_only=True).run("提前三个工作日", ctx, size=10)
    resp = await seeded.search(body)

    candidates = [Candidate.from_hit(h) for h in resp["hits"]["hits"]]
    normalize_scores(candidates)
    assert candidates

    parents = await construct_parents(seeded, candidates, window=5, max_parents=5)
    assert parents
    merged = parents[0]
    assert merged.is_parent
    # 三个兄弟块的内容都应出现在父块里
    for piece in ("十四天带薪年假", "提前三个工作日", "结转五天"):
        assert piece in merged.snippet, f"父块缺少 {piece}"


async def test_parents_dedup_same_context(seeded: Store):
    """同一上下文的多个子块命中,只应产出一个父块。"""
    ctx = QueryContext(identity=Identity(), config=RetrievalConfig(top_k=20))
    body = compose(keyword_only=True).run("年假", ctx, size=20)
    resp = await seeded.search(body)

    candidates = [Candidate.from_hit(h) for h in resp["hits"]["hits"]]
    normalize_scores(candidates)
    public = [c for c in candidates if c.doc_id == "pub"]
    assert len(public) > 1, "前提不成立:公开文档应有多个子块命中"

    parents = await construct_parents(seeded, public, window=5, max_parents=10)
    assert len(parents) == 1, "同一上下文块产生了重复父块"


# --------------------------------------------------------------------------- #
# delta 增量
# --------------------------------------------------------------------------- #
async def test_content_hash_supports_delta(seeded: Store):
    assert await seeded.get_content_hash("pub") == "h1"
    assert await seeded.get_content_hash("不存在") is None


async def test_delete_doc_removes_all_chunks(seeded: Store):
    deleted = await seeded.delete_doc("pub")
    assert deleted == 3
    await seeded.refresh()
    assert await seeded.get_content_hash("pub") is None
    assert await seeded.count() == 1
