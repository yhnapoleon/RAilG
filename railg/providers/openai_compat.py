"""OpenAI 兼容端点的 provider 实现。

适用于:硅基流动、阿里云百炼(兼容模式)、DeepSeek、OpenAI、
以及本地的 vLLM / Ollama / TEI —— 它们都提供 /embeddings 与 /chat/completions。

rerank 走 /rerank(Jina/Cohere 风格),这是目前事实标准;
不提供该端点的服务商把 rerank.enabled 置 false 即可。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from railg.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    RerankHit,
    RerankProvider,
)

logger = logging.getLogger(__name__)

_RETRYABLE = (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)

_retry = retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True,
)


def _raise_for_status(resp: httpx.Response) -> None:
    """4xx 里只有 429 值得重试,其余直接失败,避免拿错误 key 重试四次。"""
    if resp.status_code < 400:
        return
    if 400 <= resp.status_code < 500 and resp.status_code != 429:
        body = resp.text[:500]
        raise RuntimeError(f"{resp.request.url} 返回 {resp.status_code}: {body}")
    resp.raise_for_status()


class _Base:
    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class OpenAIEmbedding(_Base, EmbeddingProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dims: int,
        batch_size: int = 32,
        max_input_chars: int = 8000,
    ) -> None:
        super().__init__(base_url, api_key)
        self.model = model
        self.dims = dims
        self.batch_size = batch_size
        self.max_input_chars = max_input_chars

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t[: self.max_input_chars] or " " for t in texts[i : i + self.batch_size]]
            out.extend(await self._embed_batch(batch))
        return out

    @_retry
    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        resp = await self._client.post(
            "/embeddings", json={"model": self.model, "input": batch}
        )
        _raise_for_status(resp)
        data = resp.json()["data"]
        # 有的服务商不保证返回顺序,按 index 归位
        data.sort(key=lambda d: d.get("index", 0))
        vectors = [d["embedding"] for d in data]

        if vectors and len(vectors[0]) != self.dims:
            raise RuntimeError(
                f"模型 {self.model} 实际输出 {len(vectors[0])} 维,"
                f"但 config 里 embedding.dims={self.dims}。"
                "两者必须一致,否则 mapping 建错、检索全废。"
            )
        return vectors


class OpenAIRerank(_Base, RerankProvider):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        super().__init__(base_url, api_key)
        self.model = model

    @_retry
    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankHit]:
        if not documents:
            return []
        resp = await self._client.post(
            "/rerank",
            json={
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
            },
        )
        _raise_for_status(resp)
        results = resp.json().get("results", [])
        return [
            RerankHit(index=r["index"], score=float(r.get("relevance_score", 0.0)))
            for r in results
        ]


class OpenAIChat(_Base, LLMProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        super().__init__(base_url, api_key, timeout=180.0)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _payload(self, messages: list[dict[str, str]], stream: bool, **kw) -> dict:
        return {
            "model": kw.get("model", self.model),
            "messages": messages,
            "temperature": kw.get("temperature", self.temperature),
            "max_tokens": kw.get("max_tokens", self.max_tokens),
            "stream": stream,
        }

    async def stream(self, messages: list[dict[str, str]], **kw) -> AsyncIterator[str]:
        payload = self._payload(messages, stream=True, **kw)
        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "replace")[:500]
                raise RuntimeError(f"LLM 返回 {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                # 推理模型(如 Qwen3 thinking)会先吐 reasoning_content,这里只取正文
                piece = delta.get("content")
                if piece:
                    yield piece

    @_retry
    async def complete(self, messages: list[dict[str, str]], **kw) -> str:
        resp = await self._client.post(
            "/chat/completions", json=self._payload(messages, stream=False, **kw)
        )
        _raise_for_status(resp)
        choices = resp.json().get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "") or ""
