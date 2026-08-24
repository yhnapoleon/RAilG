"""引用归因测试。

断言的核心行为:只有回答里真正标注了 [n] 的候选才会进来源列表。
把检索到的候选一律列成来源,等于给出模型没用过的依据 —— 那是错误归因。
"""

from __future__ import annotations

from railg.generation.attribution import build_sources, parse_citations, split_sentences
from railg.generation.packer import pack
from railg.generation.prompt import build_context_block, build_messages
from railg.schema.document import Candidate


def _cand(n: int, text: str = "内容", score: float = 0.9) -> Candidate:
    return Candidate(
        chunk_uid=f"doc{n}:0", doc_id=f"doc{n}", file_name=f"文件{n}.pdf",
        file_url=f"file:///doc{n}.pdf", snippet=text, page_no=n,
        normalized_score=score,
    )


# --------------------------------------------------------------------------- #
def test_parse_single_and_multi_citations():
    assert parse_citations("这句出自 [1]。", 3) == [1]
    assert parse_citations("这句出自 [1][3]。", 3) == [1, 3]
    assert parse_citations("这句出自 [2, 3]。", 3) == [2, 3]
    assert parse_citations("这句出自 [1，2]。", 3) == [1, 2]


def test_out_of_range_citations_ignored():
    assert parse_citations("[9] 不存在", 3) == []
    assert parse_citations("[0] 也不合法", 3) == []


def test_duplicate_citations_deduped_in_order():
    assert parse_citations("[2] 然后 [1] 再 [2]", 3) == [2, 1]


# --------------------------------------------------------------------------- #
def test_only_cited_candidates_become_sources():
    """★ 核心断言:没被引用的候选不进来源列表。"""
    candidates = [_cand(1), _cand(2), _cand(3)]
    answer = "年假是 14 天 [1]。报销上限见规定 [3]。"

    sources = build_sources(answer, candidates)
    assert [s["n"] for s in sources] == [1, 3]
    assert all(s["file_name"] != "文件2.pdf" for s in sources), "未引用的候选混进来了"


def test_no_citation_means_no_sources():
    """模型没标注就不给来源 —— 宁可没有,也不给错误归因。"""
    candidates = [_cand(1), _cand(2)]
    assert build_sources("我不知道，资料里没有提到。", candidates) == []


def test_source_carries_locator():
    sources = build_sources("见 [1]。", [_cand(1, "年假规定")])
    assert sources[0]["page_no"] == 1
    assert "文件1.pdf" in sources[0]["label"]
    assert sources[0]["file_url"].endswith("doc1.pdf")


# --------------------------------------------------------------------------- #
def test_prompt_numbers_candidates():
    block = build_context_block([_cand(1), _cand(2)], max_chars=100)
    assert "[1]" in block and "[2]" in block


def test_prompt_handles_empty_candidates():
    block = build_context_block([], max_chars=100)
    assert "没有检索到" in block


def test_messages_include_system_and_history():
    history = [
        {"role": "user", "content": "上一个问题"},
        {"role": "assistant", "content": "上一个回答"},
    ]
    messages = build_messages("新问题", [_cand(1)], history, max_candidate_chars=200)
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "上一个问题"
    assert "新问题" in messages[-1]["content"]


def test_sentence_split_handles_cjk_punctuation():
    sentences = split_sentences("第一句。第二句！第三句？")
    assert len(sentences) == 3


# --------------------------------------------------------------------------- #
def test_packer_respects_budget():
    big = [_cand(i, "很长的内容" * 500) for i in range(1, 11)]
    packed, used = pack(big, budget_tokens=2000, max_candidate_chars=4000)
    assert 0 < len(packed) < len(big)
    assert used > 0


def test_packer_always_returns_at_least_one():
    huge = [_cand(1, "字" * 100_000)]
    packed, _ = pack(huge, budget_tokens=500, max_candidate_chars=4000)
    assert len(packed) == 1


def test_packer_preserves_order():
    cands = [_cand(i) for i in range(1, 6)]
    packed, _ = pack(cands, budget_tokens=100_000, max_candidate_chars=4000)
    assert [c.doc_id for c in packed] == [c.doc_id for c in cands]
