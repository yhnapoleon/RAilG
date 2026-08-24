"""检索评测指标。

没有指标,调 top_k、改召回权重、开关 rerank 就全靠感觉,而感觉在检索上
经常是反的。这里实现四个最常用的:
    Recall@k   期望文档有多少被召回了 —— 召回层的天花板
    Precision@k 召回的里面有多少是对的 —— 噪音水平
    MRR        第一个正确结果排在多靠前 —— 用户看第一眼的体验
    nDCG@k     兼顾位置和多个正确答案 —— 综合排序质量

全部是纯函数,无依赖,可单测。
"""

from __future__ import annotations

import math

__all__ = ["recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "evaluate_one", "aggregate"]


def _hits(retrieved: list[str], expected: set[str], k: int) -> list[bool]:
    return [doc in expected for doc in retrieved[:k]]


def recall_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """前 k 个里覆盖了多少期望文档。"""
    if not expected:
        return 0.0
    found = len(set(retrieved[:k]) & expected)
    return found / len(expected)


def precision_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """前 k 个里有多少是期望文档。"""
    if not retrieved or k <= 0:
        return 0.0
    window = retrieved[:k]
    return sum(1 for d in window if d in expected) / len(window)


def mrr(retrieved: list[str], expected: set[str]) -> float:
    """第一个命中的倒数排名。全没命中记 0。"""
    for i, doc in enumerate(retrieved, 1):
        if doc in expected:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """归一化折损累计增益。二元相关性(命中=1,未命中=0)。"""
    if not expected:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, hit in enumerate(_hits(retrieved, expected, k), 1)
        if hit
    )
    # 理想情况:所有期望文档都排在最前面
    ideal_n = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_one(
    retrieved: list[str], expected: list[str], ks: tuple[int, ...] = (1, 3, 5, 10)
) -> dict[str, float]:
    """单条 query 的全部指标。"""
    expected_set = set(expected)
    out: dict[str, float] = {"mrr": round(mrr(retrieved, expected_set), 4)}
    for k in ks:
        out[f"recall@{k}"] = round(recall_at_k(retrieved, expected_set, k), 4)
        out[f"precision@{k}"] = round(precision_at_k(retrieved, expected_set, k), 4)
        out[f"ndcg@{k}"] = round(ndcg_at_k(retrieved, expected_set, k), 4)
    return out


def aggregate(per_case: list[dict[str, float]]) -> dict[str, float]:
    """对全部 case 求宏平均(每条 query 权重相同)。"""
    if not per_case:
        return {}
    keys = per_case[0].keys()
    return {
        key: round(sum(c.get(key, 0.0) for c in per_case) / len(per_case), 4)
        for key in keys
    }


def compare(baseline: dict[str, float], current: dict[str, float]) -> dict[str, dict]:
    """两次评测的逐指标对比。回归检测用。"""
    out: dict[str, dict] = {}
    for key in sorted(set(baseline) | set(current)):
        before = baseline.get(key, 0.0)
        after = current.get(key, 0.0)
        out[key] = {
            "before": before,
            "after": after,
            "delta": round(after - before, 4),
            "regressed": after < before - 1e-9,
        }
    return out
