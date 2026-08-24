"""Markdown 块级结构。

用 mistletoe 做块级解析(纯 Python、无二级依赖、几十 KB)。切块器靠它区分
表格 / 标题 / 段落,从而做到"一张表是一个上下文块"这种结构感知的切分。

注意:mistletoe 的渲染器会在 __enter__/__exit__ 里增删 token 类型,
模块级全局单例在并发下不安全。所以每次解析新建一个渲染器,
解析与渲染都在同一个上下文内完成。
"""

from __future__ import annotations

import re

import mistletoe
from mistletoe.block_token import (
    BlockCode,
    BlockToken,
    CodeFence,
    Footnote,
    Heading,
    HtmlBlock,
    List,
    Paragraph,
    Quote,
    Table,
)
from mistletoe.markdown_renderer import MarkdownRenderer

PAGE_ELEMENTS = (Heading, Quote, Paragraph, BlockCode, CodeFence, List, Table, Footnote, HtmlBlock)


def compress_table_markdown(md: str) -> str:
    """压掉表格里连续的横线与空格,省 token。

    副作用:让"字符数 → 页码"的估算更准。
    """
    md = re.sub(r"-{2,}", "-", md)
    md = re.sub(r" {2,}", " ", md)
    return md.strip()


class Element:
    """一个 markdown 块。"""

    __slots__ = ("block", "markdown")

    def __init__(self, block: BlockToken, markdown: str) -> None:
        self.block = block
        self.markdown = markdown

    def istable(self) -> bool:
        return isinstance(self.block, Table)

    def isheading(self) -> bool:
        return isinstance(self.block, Heading)

    def ispara(self) -> bool:
        return isinstance(self.block, Paragraph)

    def islist(self) -> bool:
        return isinstance(self.block, List)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Element({type(self.block).__name__}, {self.markdown[:40]!r})"


class Page:
    __slots__ = ("page_no", "children")

    def __init__(self, page_no: int = 0, children: list[Element] | None = None) -> None:
        self.page_no = page_no
        self.children: list[Element] = children or []

    @classmethod
    def from_markdown(cls, md: str = "", page_no: int = 0, compress_table: bool = False) -> "Page":
        page = cls(page_no=page_no)
        if not md or not md.strip():
            return page

        with MarkdownRenderer() as renderer:
            blocks = mistletoe.Document(md).children or []
            prev_is_heading = False
            for block in blocks:
                if not isinstance(block, PAGE_ELEMENTS):
                    continue
                text = renderer.render(block).strip()
                if isinstance(block, Table) and compress_table:
                    text = compress_table_markdown(text)
                element = Element(block, text)
                # 连续标题:把前一个提升为一级,避免产生空章节
                if prev_is_heading and element.isheading() and page.children:
                    page.children[-1].markdown = page.children[-1].markdown.replace("##", "#", 1)
                prev_is_heading = element.isheading()
                page.children.append(element)
        return page

    def markdown(self, sep: str = "\n\n") -> str:
        return sep.join(e.markdown for e in self.children)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Page(page_no={self.page_no}, n_elements={len(self.children)})"
