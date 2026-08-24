"""评测指标测试。

指标算错比没有指标更糟 —— 会把人往错误方向带。所以这些纯函数逐个钉死。
"""

from __future__ import annotations

import math

import pytest

from railg.evaluation.metrics import (
    aggregate,
    compare,
    evaluate_one,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# --------------------------------------------------------------------------- #
# Recall
# --------------------------------------------------------------------------- #
def test_recall_perfect_and_zero():
    assert recall_at_k(["a", "b"], {"a", "b"}, 5) == 1.0
    assert recall_at_k(["x", "y"], {"a", "b"}, 5) == 0.0


def test_recall_partial():
    assert recall_at_k(["a", "x"], {"a", "b"}, 5) == 0.5


def test_recall_respects_k():
    # 正确答案排在第 3 位,k=2 时够不着
    assert recall_at_k(["x", "y", "a"], {"a"}, 2) == 0.0
    assert recall_at_k(["x", "y", "a"], {"a"}, 3) == 1.0


def test_recall_with_no_expectation_is_zero():
    assert recall_at_k(["a"], set(), 5) == 0.0


# --------------------------------------------------------------------------- #
# Precision
# --------------------------------------------------------------------------- #
def test_precision_counts_noise():
    assert precision_at_k(["a", "x", "y", "z"], {"a"}, 4) == 0.25


def test_precision_on_empty_retrieval():
    assert precision_at_k([], {"a"}, 5) == 0.0


# --------------------------------------------------------------------------- #
# MRR
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("retrieved,expected_score", [
    (["a", "x", "y"], 1.0),
    (["x", "a", "y"], 0.5),
    (["x", "y", "a"], 1 / 3),
    (["x", "y", "z"], 0.0),
])
def test_mrr_uses_first_hit(retrieved, expected_score):
    assert mrr(retrieved, {"a"}) == pytest.approx(expected_score)


# --------------------------------------------------------------------------- #
# nDCG
# --------------------------------------------------------------------------- #
def test_ndcg_is_one_when_perfectly_ordered():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 5) == pytest.approx(1.0)


def test_ndcg_penalises_late_hits():
    early = ndcg_at_k(["a", "x", "y"], {"a"}, 5)
    late = ndcg_at_k(["x", "y", "a"], {"a"}, 5)
    assert early > late > 0


def test_ndcg_matches_manual_calculation():
    # 命中在第 2 位:DCG = 1/log2(3);IDCG = 1/log2(2) = 1
    got = ndcg_at_k(["x", "a"], {"a"}, 5)
    assert got == pytest.approx(1 / math.log2(3), abs=1e-4)


def test_ndcg_zero_without_expectation():
    assert ndcg_at_k(["a"], set(), 5) == 0.0


# --------------------------------------------------------------------------- #
# 组合与汇总
# --------------------------------------------------------------------------- #
def test_evaluate_one_returns_all_metrics():
    m = evaluate_one(["a", "x"], ["a"], ks=(1, 3))
    assert set(m) == {"mrr", "recall@1", "recall@3", "precision@1",
                      "precision@3", "ndcg@1", "ndcg@3"}
    assert m["recall@1"] == 1.0
    assert m["mrr"] == 1.0


def test_aggregate_is_macro_average():
    per_case = [{"recall@5": 1.0, "mrr": 1.0}, {"recall@5": 0.0, "mrr": 0.0}]
    assert aggregate(per_case) == {"recall@5": 0.5, "mrr": 0.5}


def test_aggregate_on_empty():
    assert aggregate([]) == {}


# --------------------------------------------------------------------------- #
# 回归检测
# --------------------------------------------------------------------------- #
def test_compare_flags_regression():
    diff = compare({"recall@5": 0.80}, {"recall@5": 0.70})
    assert diff["recall@5"]["regressed"] is True
    assert diff["recall@5"]["delta"] == pytest.approx(-0.10)


def test_compare_flags_improvement():
    diff = compare({"recall@5": 0.70}, {"recall@5": 0.85})
    assert diff["recall@5"]["regressed"] is False
    assert diff["recall@5"]["delta"] == pytest.approx(0.15)


def test_compare_handles_new_metric():
    diff = compare({}, {"ndcg@10": 0.5})
    assert diff["ndcg@10"]["before"] == 0.0
    assert diff["ndcg@10"]["after"] == 0.5


# --------------------------------------------------------------------------- #
# 标识空间归一 —— 回归测试
# --------------------------------------------------------------------------- #
def test_normalise_maps_filenames_to_doc_ids():
    """期望值写文件名、召回是 doc_id,必须先归一到同一空间再算指标。

    若把 doc_id 和 file_name 都塞进 ranked list,一个候选会占两个位置,
    recall@1 会被系统性压低。这条测试钉死归一行为。
    """
    from railg.evaluation.runner import Evaluator

    aliases = {"abc123": "abc123", "手册.pdf": "abc123", "/tmp/手册.pdf": "abc123"}

    # 三种写法都归一到同一个 doc_id,且只占一个位置
    assert Evaluator._normalise(["手册.pdf"], aliases) == ["abc123"]
    assert Evaluator._normalise(["/tmp/手册.pdf"], aliases) == ["abc123"]
    assert Evaluator._normalise(["abc123", "手册.pdf"], aliases) == ["abc123"]


def test_normalise_preserves_rank_order():
    from railg.evaluation.runner import Evaluator

    aliases = {"a.pdf": "d1", "b.pdf": "d2", "c.pdf": "d3"}
    assert Evaluator._normalise(["b.pdf", "a.pdf", "c.pdf"], aliases) == ["d2", "d1", "d3"]


def test_normalise_keeps_unknown_keys():
    """语料里没有的期望值保持原样 —— 自然匹配不上,这是正确行为。"""
    from railg.evaluation.runner import Evaluator

    assert Evaluator._normalise(["不存在.pdf"], {"a.pdf": "d1"}) == ["不存在.pdf"]


def test_one_candidate_occupies_one_rank():
    """归一后,recall@1 对"第一个候选就命中"必须给满分。"""
    from railg.evaluation.runner import Evaluator

    aliases = {"手册.pdf": "d1", "d1": "d1"}
    expected = Evaluator._normalise(["手册.pdf"], aliases)
    retrieved = Evaluator._normalise(["d1", "d2"], aliases)
    assert evaluate_one(retrieved, expected, ks=(1,))["recall@1"] == 1.0
