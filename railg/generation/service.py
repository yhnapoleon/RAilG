"""对话服务:检索 → 装配 → 生成 → 归因 → 落库。

流式事件协议(SSE 的 data 字段):
    {"type": "session", "session_id": "...", "title": "..."}  会话就绪(可能是新建的)
    {"type": "trace",   "data": {...}}      检索过程,前端 debug 面板用
    {"type": "delta",   "text": "..."}      增量正文
    {"type": "sources", "data": [...]}      实际被引用的来源
    {"type": "warning", "data": [...]}      支撑不足的句子
    {"type": "message", "message_id": "..."} 落库后的消息 id,前端据此提交反馈
    {"type": "done"}
    {"type": "error",   "message": "..."}

会话持久化在这一层,而不是 API 层 —— 因为 trace 和 sources 只有这里拿得到,
而它们正是反馈闭环里最有价值的部分:用户点"没帮助"时,你要能回看当时召回了什么。
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field

from railg.config import Settings, get_settings
from railg.db import Database, get_db
from railg.generation.attribution import build_sources, verify_attribution
from railg.generation.packer import pack
from railg.generation.prompt import build_messages
from railg.providers import (
    EmbeddingProvider,
    LLMProvider,
    get_embedding_provider,
    get_llm_provider,
)
from railg.retrieval.service import RetrievalService, get_retrieval_service
from railg.schema.document import ANONYMOUS, Candidate, Identity

logger = logging.getLogger(__name__)

TITLE_MAX_CHARS = 30


@dataclass
class ChatAnswer:
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    session_id: str = ""
    message_id: str = ""


class ChatService:
    def __init__(
        self,
        retrieval: RetrievalService | None = None,
        llm: LLMProvider | None = None,
        embedder: EmbeddingProvider | None = None,
        db: Database | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retrieval = retrieval or get_retrieval_service()
        self.llm = llm or get_llm_provider()
        self.embedder = embedder or get_embedding_provider()
        self.db = db or get_db()

    # ----------------------------------------------------------------- #
    async def ensure_session(
        self, session_id: str | None, first_query: str, owner: str = "anonymous"
    ) -> tuple[str, str, bool]:
        """返回 (session_id, title, 是否新建)。"""
        if session_id:
            existing = await self.db.get_session(session_id)
            if existing:
                return session_id, existing["title"], False

        title = first_query.strip().replace("\n", " ")[:TITLE_MAX_CHARS] or "新会话"
        new_id = await self.db.create_session(title=title, owner=owner)
        return new_id, title, True

    async def _prepare(
        self,
        query: str,
        identity: Identity,
        history: list[dict[str, str]] | None,
        rewrite: bool,
    ) -> tuple[list[Candidate], list[dict[str, str]], dict]:
        gen = self.settings.generation
        result = await self.retrieval.retrieve(
            query, identity=identity, history=history, rewrite=rewrite
        )
        packed, used = pack(
            result.candidates,
            budget_tokens=gen.context_budget_tokens,
            max_candidate_chars=gen.max_candidate_chars,
        )
        trace = asdict(result.trace)
        trace["n_packed"] = len(packed)
        trace["context_tokens"] = used

        messages = build_messages(
            query, packed, history, max_candidate_chars=gen.max_candidate_chars
        )
        return packed, messages, trace

    # ----------------------------------------------------------------- #
    async def stream(
        self,
        query: str,
        identity: Identity = ANONYMOUS,
        history: list[dict[str, str]] | None = None,
        rewrite: bool = True,
        session_id: str | None = None,
        persist: bool = True,
    ) -> AsyncIterator[dict]:
        started = time.perf_counter()
        title = ""
        created = False

        if persist:
            session_id, title, created = await self.ensure_session(
                session_id, query, owner=identity.sub
            )
            # 历史以库里的为准,前端传的只是缓存
            if history is None:
                history = await self.db.history_for_llm(session_id)
            yield {"type": "session", "session_id": session_id,
                   "title": title, "created": created}

        try:
            packed, messages, trace = await self._prepare(query, identity, history, rewrite)
        except Exception as exc:
            logger.exception("检索阶段失败")
            await self._log(started, query, ok=False)
            yield {"type": "error", "message": f"检索失败: {exc}"}
            return

        yield {"type": "trace", "data": trace}

        parts: list[str] = []
        try:
            async for piece in self.llm.stream(messages):
                parts.append(piece)
                yield {"type": "delta", "text": piece}
        except Exception as exc:
            logger.exception("生成阶段失败")
            await self._log(started, query, ok=False, trace=trace)
            yield {"type": "error", "message": f"生成失败: {exc}"}
            return

        answer = "".join(parts)
        sources = build_sources(answer, packed)
        yield {"type": "sources", "data": sources}

        warnings: list[dict] = []
        if self.settings.generation.verify_attribution and packed:
            warnings = await verify_attribution(
                answer, packed, self.embedder,
                threshold=self.settings.generation.attribution_threshold,
            )
            if warnings:
                yield {"type": "warning", "data": warnings}

        if persist and session_id:
            try:
                await self.db.add_message(session_id, "user", query)
                message_id = await self.db.add_message(
                    session_id, "assistant", answer, sources=sources, trace=trace
                )
                yield {"type": "message", "message_id": message_id}
            except Exception:
                logger.exception("消息落库失败(不影响本次回答)")

        await self._log(started, query, ok=True, trace=trace)
        yield {"type": "done"}

    # ----------------------------------------------------------------- #
    async def answer(
        self,
        query: str,
        identity: Identity = ANONYMOUS,
        history: list[dict[str, str]] | None = None,
        rewrite: bool = True,
        session_id: str | None = None,
        persist: bool = False,
    ) -> ChatAnswer:
        """非流式。CLI 和评测用,默认不落库。"""
        started = time.perf_counter()
        title_created = False
        if persist:
            session_id, _, title_created = await self.ensure_session(
                session_id, query, owner=identity.sub
            )
            if history is None:
                history = await self.db.history_for_llm(session_id)

        packed, messages, trace = await self._prepare(query, identity, history, rewrite)
        text = await self.llm.complete(messages)

        out = ChatAnswer(
            answer=text,
            sources=build_sources(text, packed),
            trace=trace,
            session_id=session_id or "",
        )
        if self.settings.generation.verify_attribution and packed:
            out.warnings = await verify_attribution(
                text, packed, self.embedder,
                threshold=self.settings.generation.attribution_threshold,
            )
        if persist and session_id:
            await self.db.add_message(session_id, "user", query)
            out.message_id = await self.db.add_message(
                session_id, "assistant", text, sources=out.sources, trace=trace
            )
        del title_created
        await self._log(started, query, ok=True, trace=trace)
        return out

    # ----------------------------------------------------------------- #
    async def _log(
        self, started: float, query: str, ok: bool, trace: dict | None = None
    ) -> None:
        try:
            await self.db.log_request(
                kind="chat",
                query=query,
                ok=ok,
                latency_ms=int((time.perf_counter() - started) * 1000),
                stages=(trace or {}).get("timings_ms", {}),
                n_candidates=(trace or {}).get("n_parents", 0),
            )
        except Exception:
            logger.debug("请求日志写入失败", exc_info=True)


_chat: ChatService | None = None


def get_chat_service() -> ChatService:
    global _chat
    if _chat is None:
        _chat = ChatService()
    return _chat
