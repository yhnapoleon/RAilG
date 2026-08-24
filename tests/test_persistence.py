"""持久化层测试:会话、消息、反馈、文档登记、评测集。

用临时文件建库,不碰真实数据。
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from railg.db import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    d = Database(tmp_path / "test.db")
    await d.init()
    return d


# --------------------------------------------------------------------------- #
# 会话与消息
# --------------------------------------------------------------------------- #
async def test_session_lifecycle(db: Database):
    sid = await db.create_session("年假问题", owner="alice")
    session = await db.get_session(sid)
    assert session["title"] == "年假问题"
    assert session["owner"] == "alice"

    await db.rename_session(sid, "改了个名")
    assert (await db.get_session(sid))["title"] == "改了个名"

    await db.delete_session(sid)
    assert await db.get_session(sid) is None


async def test_sessions_are_scoped_by_owner(db: Database):
    await db.create_session("alice 的", owner="alice")
    await db.create_session("bob 的", owner="bob")

    assert len(await db.list_sessions(owner="alice")) == 1
    assert len(await db.list_sessions(owner="bob")) == 1


async def test_messages_round_trip_sources_and_trace(db: Database):
    sid = await db.create_session()
    sources = [{"n": 1, "file_name": "a.pdf", "label": "a.pdf · p.3"}]
    trace = {"n_raw": 42, "timings_ms": {"search": 30}}

    mid = await db.add_message(sid, "assistant", "答案 [1]", sources=sources, trace=trace)
    messages = await db.list_messages(sid)

    assert len(messages) == 1
    assert messages[0]["id"] == mid
    # JSON 字段必须还原成对象,而不是字符串
    assert messages[0]["sources"] == sources
    assert messages[0]["trace"]["n_raw"] == 42


async def test_history_for_llm_is_chronological_and_bounded(db: Database):
    sid = await db.create_session()
    for i in range(10):
        await db.add_message(sid, "user", f"问题{i}")
        await db.add_message(sid, "assistant", f"回答{i}")

    history = await db.history_for_llm(sid, turns=3)
    assert len(history) == 6
    # 取的是最近三轮,且按时间正序
    assert history[0]["content"] == "问题7"
    assert history[-1]["content"] == "回答9"
    assert all(set(m) == {"role", "content"} for m in history)


async def test_deleting_session_cascades_messages(db: Database):
    sid = await db.create_session()
    await db.add_message(sid, "user", "会被级联删掉")
    await db.delete_session(sid)
    assert await db.list_messages(sid) == []


# --------------------------------------------------------------------------- #
# 反馈
# --------------------------------------------------------------------------- #
async def test_feedback_records_and_summarises(db: Database):
    sid = await db.create_session()
    mid = await db.add_message(sid, "assistant", "答案")

    await db.add_feedback(mid, sid, "down", comment="召回不对", query="年假")
    await db.add_feedback(mid, sid, "up")

    assert await db.feedback_summary() == {"up": 1, "down": 1}
    only_down = await db.list_feedback(rating="down")
    assert len(only_down) == 1
    assert only_down[0]["comment"] == "召回不对"


async def test_feedback_rejects_bad_rating(db: Database):
    sid = await db.create_session()
    mid = await db.add_message(sid, "assistant", "答案")
    with pytest.raises(Exception):
        await db.add_feedback(mid, sid, "maybe")


# --------------------------------------------------------------------------- #
# 文档登记
# --------------------------------------------------------------------------- #
def _doc(doc_id="d1", **kw):
    base = {
        "doc_id": doc_id, "file_name": "手册.pdf", "file_path": "/tmp/手册.pdf",
        "document_type": "pdf", "content_hash": "h1", "n_chunks": 12,
        "acl": ["public"], "source_kind": "local", "status": "indexed",
    }
    base.update(kw)
    return base


async def test_document_upsert_is_idempotent(db: Database):
    await db.upsert_document(_doc())
    await db.upsert_document(_doc(n_chunks=20, content_hash="h2"))

    docs = await db.list_documents()
    assert len(docs) == 1
    assert docs[0]["n_chunks"] == 20
    assert docs[0]["content_hash"] == "h2"
    assert docs[0]["acl"] == ["public"]


async def test_document_stats_counts_failures(db: Database):
    await db.upsert_document(_doc("ok1"))
    await db.upsert_document(_doc("bad1", status="failed", error="提取为空", n_chunks=0))

    stats = await db.document_stats()
    assert stats["n_docs"] == 2
    assert stats["n_chunks"] == 12
    assert stats["n_failed"] == 1


async def test_document_keyword_filter(db: Database):
    await db.upsert_document(_doc("a", file_name="年假制度.pdf"))
    await db.upsert_document(_doc("b", file_name="报销标准.xlsx"))

    hits = await db.list_documents(keyword="年假")
    assert [d["doc_id"] for d in hits] == ["a"]


# --------------------------------------------------------------------------- #
# 评测集
# --------------------------------------------------------------------------- #
async def test_eval_case_round_trip(db: Database):
    cid = await db.add_eval_case("年假几天", ["手册.pdf"], note="基础问题", tags=["hr"])
    cases = await db.list_eval_cases()

    assert len(cases) == 1
    assert cases[0]["id"] == cid
    assert cases[0]["expected_docs"] == ["手册.pdf"]
    assert cases[0]["tags"] == ["hr"]

    await db.delete_eval_case(cid)
    assert await db.list_eval_cases() == []


async def test_eval_run_is_stored_with_config(db: Database):
    rid = await db.save_eval_run(
        label="baseline",
        config={"top_k": 50, "rerank_enabled": True},
        metrics={"recall@5": 0.8, "mrr": 0.65},
        details=[{"query": "x", "metrics": {}}],
        n_cases=1,
    )
    run = await db.get_eval_run(rid)
    # 配置必须一起存 —— 否则历史跑分无法解释
    assert run["config"]["top_k"] == 50
    assert run["metrics"]["recall@5"] == 0.8
    assert len(run["details"]) == 1

    runs = await db.list_eval_runs()
    assert runs[0]["label"] == "baseline"


# --------------------------------------------------------------------------- #
# 请求日志
# --------------------------------------------------------------------------- #
async def test_request_metrics_aggregate(db: Database):
    await db.log_request("chat", "q1", ok=True, latency_ms=100)
    await db.log_request("chat", "q2", ok=True, latency_ms=300)
    await db.log_request("chat", "q3", ok=False, latency_ms=50)

    m = await db.request_metrics(hours=24)
    assert m["requests"] == 3
    assert m["success"] == 2
    assert m["avg_latency_ms"] == 150
    assert m["max_latency_ms"] == 300
    assert m["error_rate"] == pytest.approx(1 / 3, abs=1e-3)


async def test_request_metrics_empty_window(db: Database):
    m = await db.request_metrics(hours=24)
    assert m["requests"] == 0
    assert m["error_rate"] == 0.0
