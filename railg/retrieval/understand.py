"""查询理解。

只拿当前这一句去检索,多轮对话里"那第二个呢?""它的上限是多少?"
这类指代必然检索出噪音。

这里做最小可用的一件事:把历史 + 当前问题改写成一个自包含查询。
只在有历史时触发,单轮零开销。
"""

from __future__ import annotations

import logging

from railg.providers import LLMProvider

logger = logging.getLogger(__name__)

REWRITE_SYSTEM = (
    "你是查询改写器。根据对话历史,把用户最新的问题改写成一个**自包含**的检索查询:\n"
    "1. 把代词、省略成分补全为具体实体\n"
    "2. 保留原问题的全部关键词,不要概括、不要发挥\n"
    "3. 如果原问题本身已经自包含,原样返回\n"
    "4. 只输出改写后的查询本身,不要解释、不要引号、不要前缀"
)

MAX_HISTORY_TURNS = 6
MAX_QUERY_CHARS = 300


async def rewrite_query(
    llm: LLMProvider,
    query: str,
    history: list[dict[str, str]] | None,
    enabled: bool = True,
) -> str:
    """返回自包含查询。失败时回退原查询 —— 改写不该成为单点故障。"""
    if not enabled or not history:
        return query

    turns = [m for m in history if m.get("role") in ("user", "assistant")][-MAX_HISTORY_TURNS:]
    if not turns:
        return query

    transcript = "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}: {(m.get('content') or '')[:500]}"
        for m in turns
    )
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": f"对话历史:\n{transcript}\n\n最新问题: {query}\n\n改写后的查询:"},
    ]

    try:
        rewritten = (await llm.complete(messages, temperature=0.0, max_tokens=200)).strip()
    except Exception as exc:
        logger.warning("查询改写失败,使用原查询: %s", exc)
        return query

    rewritten = rewritten.strip().strip('"').strip("'").split("\n")[0]
    if not rewritten or len(rewritten) > MAX_QUERY_CHARS:
        return query
    if rewritten != query:
        logger.info("查询改写: %r → %r", query, rewritten)
    return rewritten
