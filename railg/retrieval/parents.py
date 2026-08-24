"""父块还原(small-to-big)。

★ 这是与 chunker 共生的另一半算法。

思路:用小块召回(向量更聚焦、BM25 更准),再把命中块所在的整个上下文块
拼回来送给 LLM(上下文更完整)。两者靠 chunker 写入的
(doc_id, section_id, context_id, chunk_index) 四元组对齐。

★★ 前提:切块时零重叠。有重叠的话这里拼出来的文本会有重复段落,
   而且不报错。config.ChunkConfig 和 tests/test_chunker.py 一起守这条约束。

表格的特殊处理:split_table_chunks 会把表头 `<th>…</th>` 传播到每个子块,
拼回时必须去掉重复表头,只保留一份,并还原成正常的 markdown 表头行。
"""

from __future__ import annotations

import logging
import re

from railg.schema.document import Candidate
from railg.store import Store

logger = logging.getLogger(__name__)

_TH_RE = re.compile(r"^<th>(.*?)</th>\s*\n?", flags=re.S)


def strip_propagated_header(text: str) -> tuple[str, str]:
    """剥掉传播上去的表头,返回 (表头行, 剩余内容)。"""
    m = _TH_RE.match(text)
    if not m:
        return "", text
    return m.group(1).strip(), text[m.end():]


def rejoin_table(pieces: list[str]) -> str:
    """把表格子块拼回一张完整表:表头只留一份。"""
    header = ""
    rows: list[str] = []
    for piece in pieces:
        found, body = strip_propagated_header(piece)
        if found and not header:
            header = found
        body = body.strip()
        if body:
            rows.append(body)
    if not header:
        return "\n".join(rows)
    # 还原成标准 markdown 表格:表头 + 分隔行 + 数据行
    n_cols = max(1, header.count("|") - 1)
    sep = "|" + "|".join([" --- "] * n_cols) + "|"
    body = "\n".join(rows)
    # 子块里若已带分隔行就不再补
    if body.lstrip().startswith("|") and re.match(r"^\|[\s:-]+\|", body.lstrip()):
        return f"{header}\n{body}"
    return f"{header}\n{sep}\n{body}"


def rejoin_text(pieces: list[str]) -> str:
    out: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        # 相邻子块本就是同一段被切开的,用换行拼即可
        out.append(piece)
    return "\n".join(out)


def _window(siblings: list[dict], hit_index: int, window: int) -> list[dict]:
    """取命中块前后各 window 个兄弟。上下文块本身不大时等于全取。"""
    if window <= 0:
        return siblings
    positions = [i for i, s in enumerate(siblings)
                 if s["_source"].get("chunk_index") == hit_index]
    center = positions[0] if positions else len(siblings) // 2
    lo = max(0, center - window)
    hi = min(len(siblings), center + window + 1)
    return siblings[lo:hi]


async def construct_parents(
    store: Store,
    candidates: list[Candidate],
    window: int = 3,
    max_parents: int = 10,
) -> list[Candidate]:
    """候选 → 父块。同一上下文块只保留分最高的那一个。"""
    if not candidates:
        return []

    # 1. 按 (doc_id, section_id, context_id) 去重,保留最高分
    best: dict[tuple[str, int, int], Candidate] = {}
    order: list[tuple[str, int, int]] = []
    for cand in candidates:
        key = cand.parent_key()
        if key not in best:
            best[key] = cand
            order.append(key)
        else:
            current = best[key]
            if (cand.rerank_score or cand.score) > (current.rerank_score or current.score):
                best[key] = cand

    parents: list[Candidate] = []
    for key in order[:max_parents]:
        cand = best[key]
        doc_id, section_id, context_id = key
        try:
            siblings = await store.fetch_siblings(doc_id, section_id, context_id)
        except Exception as exc:
            logger.warning("取兄弟块失败(%s),退回原始块: %s", key, exc)
            parents.append(cand)
            continue

        if len(siblings) <= 1:
            parents.append(cand)
            continue

        selected = _window(siblings, cand.chunk_index, window)
        pieces = [s["_source"].get("snippet") or s["_source"].get("text_to_index", "")
                  for s in selected]

        parent = cand.model_copy()
        parent.snippet = (
            rejoin_table(pieces) if cand.format == "table" else rejoin_text(pieces)
        )
        parent.is_parent = True
        # 页码取窗口内最小的,引用时指向父块起始处更符合直觉
        pages = [s["_source"].get("page_no", 1) for s in selected]
        if pages:
            parent.page_no = min(pages)
        parents.append(parent)

    return parents


def normalize_scores(candidates: list[Candidate]) -> list[Candidate]:
    """把 OpenSearch 原始分归一化到 [0,1]。

    BM25 与 kNN 的分数量纲不同,原始分只在同一次查询内可比,
    所以归一化只做 max 缩放 —— 跨查询比较本来就没有意义。
    """
    if not candidates:
        return candidates
    top = max(c.score for c in candidates) or 1.0
    for c in candidates:
        c.normalized_score = round(c.score / top, 6)
    return candidates
