"""provider 工厂。所有实例按配置构造一次,全局复用(内含 httpx 连接池)。"""

from __future__ import annotations

from functools import lru_cache

from railg.config import Settings, get_settings
from railg.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    NoopRerankProvider,
    RerankHit,
    RerankProvider,
)
from railg.providers.openai_compat import OpenAIChat, OpenAIEmbedding, OpenAIRerank
from railg.providers.tokens import get_size_function, heuristic_tokens

__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "RerankHit",
    "RerankProvider",
    "get_embedding_provider",
    "get_llm_provider",
    "get_rerank_provider",
    "get_size_function",
    "heuristic_tokens",
    "close_providers",
]


def _require_key(settings: Settings, section: str) -> str:
    key = getattr(settings, section).api_key
    if not key:
        raise RuntimeError(
            f"{section} 缺少 API key。请在 .env 里设置 RAILG_API_KEY,"
            f"或单独设置 RAILG_{section.upper()}_API_KEY。"
        )
    return key


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    s = get_settings()
    return OpenAIEmbedding(
        base_url=s.embedding.base_url,
        api_key=_require_key(s, "embedding"),
        model=s.embedding.model,
        dims=s.embedding.dims,
        batch_size=s.embedding.batch_size,
        max_input_chars=s.embedding.max_input_chars,
    )


@lru_cache(maxsize=1)
def get_rerank_provider() -> RerankProvider:
    s = get_settings()
    if not s.rerank.enabled:
        return NoopRerankProvider()
    return OpenAIRerank(
        base_url=s.rerank.base_url,
        api_key=_require_key(s, "rerank"),
        model=s.rerank.model,
    )


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    s = get_settings()
    return OpenAIChat(
        base_url=s.llm.base_url,
        api_key=_require_key(s, "llm"),
        model=s.llm.model,
        temperature=s.llm.temperature,
        max_tokens=s.llm.max_tokens,
    )


async def close_providers() -> None:
    for factory in (get_embedding_provider, get_rerank_provider, get_llm_provider):
        if factory.cache_info().currsize:
            await factory().aclose()
        factory.cache_clear()
