"""token 计数。

用 transformers.AutoTokenizer 加载真实 tokenizer 会为了准确计数拖进
几百 MB 依赖。这里抽象成可替换的计数器:

    默认  HeuristicCounter   零依赖,中英文分别估算,误差约 ±15%
    可选  TiktokenCounter    pip install railg[tokenizer]
    可选  HFCounter          需要精确对齐某个模型时用

切块尺寸本来就是个经验参数,±15% 的计数误差不会改变检索质量;
真正要紧的是同一份语料前后用同一个计数器 —— 换计数器要重建索引。
"""

from __future__ import annotations

from typing import Callable, Protocol

TokenCounter = Callable[[str], int]


class _Tokenizer(Protocol):  # pragma: no cover
    def encode(self, text: str) -> list: ...


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF      # 中日韩统一表意
        or 0x3040 <= cp <= 0x30FF   # 平假名/片假名
        or 0xAC00 <= cp <= 0xD7AF   # 谚文
        or 0x3400 <= cp <= 0x4DBF   # 扩展 A
    )


def heuristic_tokens(text: str) -> int:
    """CJK 约 1 字 1 token,拉丁文约 3.5 字符 1 token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return max(1, int(cjk + other / 3.5))


def n_words(text: str) -> int:
    return len(text.split())


def n_newlines(text: str) -> int:
    return len([ln for ln in text.split("\n") if ln])


def n_chars(text: str) -> int:
    return len(text)


def make_tiktoken_counter(encoding: str = "cl100k_base") -> TokenCounter:  # pragma: no cover
    import tiktoken

    enc = tiktoken.get_encoding(encoding)
    return lambda s: len(enc.encode(s, disallowed_special=()))


def make_hf_counter(model_name: str) -> TokenCounter:  # pragma: no cover
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    return lambda s: len(tok.encode(s))


SIZE_FUNCTIONS: dict[str, TokenCounter] = {
    "n_tokens": heuristic_tokens,
    "n_words": n_words,
    "n_newlines": n_newlines,
    "len": n_chars,
}


def get_size_function(name: str) -> TokenCounter:
    try:
        return SIZE_FUNCTIONS[name]
    except KeyError:
        raise ValueError(
            f"未知的 size_function: {name!r},可选 {sorted(SIZE_FUNCTIONS)}"
        ) from None
