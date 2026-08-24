"""引用归因。

两件事:

1. **只输出真正被引用的来源。**
   解析回答里出现的 [n] 标记,回查候选表。模型没标注的候选不会进 Sources。

2. **句子级支撑校验(可选)。**
   把回答拆成句子,只看带标注的句子,算它与所引候选的向量相似度;
   低于阈值的标出来。这不是严格的 NLI,但能抓住"标了个不相干的编号"
   这类最常见的错误归因,且只需一次 embedding 调用。

关掉校验(generation.verify_attribution=false)时,第 1 条依然生效 ——
它本身就已经比"把所有候选都列成来源"正确得多。
"""

from __future__ import annotations

import logging
import re

from railg.providers import EmbeddingProvider
from railg.schema.document import Candidate

logger = logging.getLogger(__name__)

# 匹配 [1] / [1][3] / [1,3] / [1, 3]
_CITE_RE = re.compile(r"\[(\d+(?:\s*[,，]\s*\d+)*)\]")
_SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?]?|\n+")


def parse_citations(answer: str, n_candidates: int) -> list[int]:
    """按首次出现顺序返回被引用的候选序号(1-based)。"""
    seen: list[int] = []
    for m in _CITE_RE.finditer(answer):
        for part in re.split(r"[,，]", m.group(1)):
            try:
                idx = int(part.strip())
            except ValueError:
                continue
            if 1 <= idx <= n_candidates and idx not in seen:
                seen.append(idx)
    return seen


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]


class Source(dict):
    """输出给前端的来源条目。"""


def build_sources(answer: str, candidates: list[Candidate]) -> list[Source]:
    """只保留被实际引用的候选。"""
    cited = parse_citations(answer, len(candidates))
    out: list[Source] = []
    for idx in cited:
        cand = candidates[idx - 1]
        out.append(Source(
            n=idx,
            file_name=cand.file_name,
            file_url=cand.file_url,
            file_path=cand.file_path,
            page_no=cand.page_no,
            label=cand.citation_label(),
            snippet=(cand.snippet or "")[:500],
            score=round(cand.rerank_score if cand.rerank_score is not None
                        else cand.normalized_score, 4),
        ))
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


async def verify_attribution(
    answer: str,
    candidates: list[Candidate],
    embedder: EmbeddingProvider,
    threshold: float = 0.45,
) -> list[dict]:
    """返回支撑不足的句子列表。出错时返回空表 —— 校验不该阻断回答。"""
    sentences = [s for s in split_sentences(answer) if _CITE_RE.search(s)]
    if not sentences:
        return []

    pairs: list[tuple[str, list[int]]] = []
    for sentence in sentences:
        cited = parse_citations(sentence, len(candidates))
        if cited:
            pairs.append((_CITE_RE.sub("", sentence).strip(), cited))
    if not pairs:
        return []

    try:
        texts = [s for s, _ in pairs] + [(c.snippet or "")[:2000] for c in candidates]
        vectors = await embedder.embed(texts)
    except Exception as exc:
        logger.warning("归因校验失败,跳过: %s", exc)
        return []

    sent_vecs = vectors[: len(pairs)]
    cand_vecs = vectors[len(pairs):]

    weak: list[dict] = []
    for (sentence, cited), vec in zip(pairs, sent_vecs):
        best = max(
            (_cosine(vec, cand_vecs[i - 1]) for i in cited if i - 1 < len(cand_vecs)),
            default=0.0,
        )
        if best < threshold:
            weak.append({
                "sentence": sentence[:200],
                "cited": cited,
                "similarity": round(best, 4),
            })

    if weak:
        logger.info("发现 %d 句支撑不足的引用", len(weak))
    return weak
