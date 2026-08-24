"""上下文装配:在 token 预算内塞进尽可能多的高分候选。

只做"截断到 N 条 + 每条截断到 M 字符"是不够的:没有全局预算,
上下文长度会随候选长度波动,长文档场景容易顶爆窗口。
"""

from __future__ import annotations

import logging

from railg.providers.tokens import heuristic_tokens
from railg.schema.document import Candidate

logger = logging.getLogger(__name__)

# 给系统提示词、历史、问题本身留的余量
RESERVED_TOKENS = 800


def pack(
    candidates: list[Candidate],
    budget_tokens: int,
    max_candidate_chars: int,
    min_candidates: int = 1,
) -> tuple[list[Candidate], int]:
    """按分数顺序装填,超预算即停。

    返回 (选中的候选, 估算 token 数)。至少保留 min_candidates 条,
    否则一条超长候选会导致整个上下文为空。
    """
    if not candidates:
        return [], 0

    available = max(budget_tokens - RESERVED_TOKENS, 500)
    packed: list[Candidate] = []
    used = 0

    for cand in candidates:
        text = (cand.snippet or "")[:max_candidate_chars]
        cost = heuristic_tokens(text) + 30  # 编号、标题、位置行的开销
        if packed and used + cost > available:
            break
        if len(text) < len(cand.snippet or ""):
            cand = cand.model_copy(update={"snippet": text})
        packed.append(cand)
        used += cost

    if not packed:
        head = candidates[0].model_copy()
        head.snippet = (head.snippet or "")[: max_candidate_chars // 2]
        packed = [head]
        used = heuristic_tokens(head.snippet)

    if len(packed) < len(candidates):
        logger.debug("上下文预算 %d,装入 %d/%d 条(约 %d token)",
                     available, len(packed), len(candidates), used)
    return packed[: max(len(packed), min_candidates)], used
