"""模型 provider 抽象。

云 API 与本地部署的差别被挡在这三个接口后面 —— 换 provider 只改 config.yaml,
业务代码不感知。这也是后续把 embedding/rerank 换成本地 TEI、
把 LLM 换成本地 vLLM 时唯一需要新增实现的地方。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(slots=True)
class RerankHit:
    index: int
    score: float


class EmbeddingProvider(ABC):
    """把文本编码为稠密向量。"""

    dims: int

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量编码。返回顺序必须与入参一致。"""

    async def embed_one(self, text: str) -> list[float]:
        out = await self.embed([text])
        return out[0]

    async def aclose(self) -> None:  # pragma: no cover - 默认无资源
        return


class RerankProvider(ABC):
    """交叉编码重排。"""

    @abstractmethod
    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankHit]:
        """返回按相关性降序的 (原始下标, 分数)。"""

    async def aclose(self) -> None:  # pragma: no cover
        return


class LLMProvider(ABC):
    """对话生成。"""

    @abstractmethod
    async def stream(self, messages: list[dict[str, str]], **kw) -> AsyncIterator[str]:
        """流式产出增量文本。"""
        raise NotImplementedError
        yield ""  # pragma: no cover - 使其成为 async generator

    @abstractmethod
    async def complete(self, messages: list[dict[str, str]], **kw) -> str:
        """非流式,一次拿全文。查询改写等内部调用用这个。"""

    async def aclose(self) -> None:  # pragma: no cover
        return


class NoopRerankProvider(RerankProvider):
    """rerank.enabled=false 时的占位实现:保持原顺序。"""

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankHit]:
        return [RerankHit(index=i, score=0.0) for i in range(min(top_n, len(documents)))]
