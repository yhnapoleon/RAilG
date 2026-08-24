"""检索服务:把 processors / rerank / parents 串成一次完整召回。

    查询改写 → 向量化 → 混合召回 → 归一化 → 重排 → 父块还原
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from railg.config import Settings, get_settings
from railg.providers import (
    EmbeddingProvider,
    LLMProvider,
    NoopRerankProvider,
    RerankProvider,
    get_embedding_provider,
    get_llm_provider,
    get_rerank_provider,
)
from railg.retrieval.parents import construct_parents, normalize_scores
from railg.retrieval.processors import QueryContext, compose
from railg.retrieval.understand import rewrite_query
from railg.schema.document import ANONYMOUS, Candidate, Identity
from railg.store import Store, get_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievalTrace:
    """一次检索的过程记录 —— 前端 debug 面板和后续评测都靠它。"""

    original_query: str = ""
    rewritten_query: str = ""
    n_raw: int = 0
    n_reranked: int = 0
    n_parents: int = 0
    timings_ms: dict[str, int] = field(default_factory=dict)
    error: str = ""


@dataclass
class RetrievalResult:
    candidates: list[Candidate]
    trace: RetrievalTrace


class RetrievalService:
    def __init__(
        self,
        store: Store | None = None,
        embedder: EmbeddingProvider | None = None,
        reranker: RerankProvider | None = None,
        llm: LLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_store()
        self.embedder = embedder or get_embedding_provider()
        # rerank 与 llm 惰性解析:
        #   · rerank 关掉时不该要 rerank 的 key
        #   · 评测走 rewrite=False,那时根本不需要 LLM,不该因为没配 key 就跑不了
        # 直接在 __init__ 里构造会让"只配了 embedding key"的场景直接崩掉。
        self._reranker = reranker
        self._llm = llm

    @property
    def reranker(self) -> RerankProvider:
        if self._reranker is None:
            self._reranker = (
                get_rerank_provider() if self.settings.rerank.enabled
                else NoopRerankProvider()
            )
        return self._reranker

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = get_llm_provider()
        return self._llm

    async def retrieve(
        self,
        query: str,
        identity: Identity = ANONYMOUS,
        history: list[dict[str, str]] | None = None,
        rewrite: bool = True,
        semantic_only: bool = False,
        keyword_only: bool = False,
    ) -> RetrievalResult:
        cfg = self.settings.retrieval
        trace = RetrievalTrace(original_query=query)
        clock = _Clock(trace.timings_ms)

        # 1. 查询改写(多轮才触发)。关闭时不解析 LLM provider,
        #    这样只配 embedding key 也能跑纯检索和评测。
        with clock("rewrite"):
            search_query = (
                await rewrite_query(self.llm, query, history, enabled=True)
                if rewrite and history else query
            )
        trace.rewritten_query = search_query

        # 2. 向量化
        query_vector = None
        if not keyword_only:
            with clock("embed"):
                try:
                    query_vector = await self.embedder.embed_one(search_query)
                except Exception as exc:
                    logger.error("查询向量化失败,降级为纯关键词检索: %s", exc)

        # 3. 组装并执行查询
        ctx = QueryContext(identity=identity, config=cfg, query_vector=query_vector)
        processor = compose(semantic_only=semantic_only, keyword_only=keyword_only)
        body = processor.run(search_query, ctx, size=cfg.top_k)
        if body is None:
            trace.error = ctx.error
            return RetrievalResult([], trace)

        with clock("search"):
            try:
                resp = await self.store.search(body)
            except Exception as exc:
                logger.exception("OpenSearch 查询失败")
                trace.error = f"检索失败: {exc}"
                return RetrievalResult([], trace)

        candidates = [Candidate.from_hit(h) for h in resp["hits"]["hits"]]
        trace.n_raw = len(candidates)
        if not candidates:
            return RetrievalResult([], trace)
        normalize_scores(candidates)

        # 4. 重排
        with clock("rerank"):
            candidates = await self._rerank(search_query, candidates, cfg.rerank_top_n)
        trace.n_reranked = len(candidates)

        # 5. 父块还原
        if cfg.return_parent:
            with clock("parents"):
                candidates = await construct_parents(
                    self.store, candidates,
                    window=cfg.parent_window,
                    max_parents=cfg.max_context_docs,
                )
        else:
            candidates = candidates[: cfg.max_context_docs]
        trace.n_parents = len(candidates)

        return RetrievalResult(candidates, trace)

    async def _rerank(
        self, query: str, candidates: list[Candidate], top_n: int
    ) -> list[Candidate]:
        if not self.settings.rerank.enabled or len(candidates) <= 1:
            return candidates[:top_n]
        try:
            hits = await self.reranker.rerank(
                query, [c.snippet for c in candidates], top_n=top_n
            )
        except Exception as exc:
            logger.warning("重排失败,退回原始排序: %s", exc)
            return candidates[:top_n]

        out: list[Candidate] = []
        for hit in hits:
            if 0 <= hit.index < len(candidates):
                cand = candidates[hit.index]
                cand.rerank_score = hit.score
                out.append(cand)
        return out or candidates[:top_n]


class _Clock:
    """极简计时上下文。"""

    def __init__(self, sink: dict[str, int]) -> None:
        self.sink = sink
        self._label = ""
        self._start = 0.0

    def __call__(self, label: str) -> "_Clock":
        self._label = label
        return self

    def __enter__(self) -> "_Clock":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.sink[self._label] = int((time.perf_counter() - self._start) * 1000)


_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    global _service
    if _service is None:
        _service = RetrievalService()
    return _service
