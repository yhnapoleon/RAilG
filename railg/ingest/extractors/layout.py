"""版式块 → markdown。

之所以按这套 label 来:PP-StructureV3 / PaddleX 的版式检测输出就是
(doc_title / paragraph_title / text / table / formula / figure_title),
所以将来接 PaddleOCR 时,后端只需吐 OcrBlock 列表,这里直接转 markdown。

HTML 表格转 markdown 的逻辑也一并保留 —— OCR 出来的表格都是 HTML。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_TITLE_LABELS = {"doc_title", "title"}
_PARA_TITLE_LABELS = {"paragraph_title", "para_title", "section_title"}
_TABLE_LABELS = {"table"}
_FORMULA_LABELS = {"formula", "equation"}
_SKIP_LABELS = {"header", "footer", "page_number", "aside_text", "footnote"}


@dataclass(slots=True)
class OcrBlock:
    label: str
    content: str
    bbox: list[int] = field(default_factory=list)

    @property
    def top(self) -> int:
        return self.bbox[1] if len(self.bbox) >= 2 else 0

    @property
    def left(self) -> int:
        return self.bbox[0] if self.bbox else 0


def _html_table_to_markdown(html: str) -> str:
    """把 OCR 输出的 HTML 表格转成 markdown 表格。

    只处理 OCR 会产出的简单结构(tr/td/th),不追求通用 HTML 解析。
    合并单元格(rowspan/colspan)按内容重复展开,保证列数对齐。
    """
    if not html or "<tr" not in html.lower():
        return html.strip()

    rows: list[list[str]] = []
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I):
        cells: list[str] = []
        for attrs, cell in re.findall(r"<t[dh]([^>]*)>(.*?)</t[dh]>", row_html, flags=re.S | re.I):
            text = re.sub(r"<[^>]+>", " ", cell)
            text = re.sub(r"\s+", " ", text).strip()
            text = text.replace("|", "\\|")
            span = re.search(r"colspan\s*=\s*['\"]?(\d+)", attrs, flags=re.I)
            cells.extend([text] * (int(span.group(1)) if span else 1))
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    head, *body = rows
    out = ["| " + " | ".join(head) + " |",
           "| " + " | ".join(["---"] * width) + " |"]
    out.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(out)


def block_to_markdown(block: OcrBlock) -> str:
    label = (block.label or "").lower()
    content = (block.content or "").strip()
    if not content or label in _SKIP_LABELS:
        return ""

    if label in _TITLE_LABELS:
        return f"# {content}"
    if label in _PARA_TITLE_LABELS:
        # ★ 用 ## 是有原因的:chunker 的章节切分正则就是 r'\n?## .+\n'
        return f"## {content}"
    if label in _TABLE_LABELS:
        return _html_table_to_markdown(content)
    if label in _FORMULA_LABELS:
        return f"$$\n{content}\n$$"
    return content


def blocks_to_markdown(blocks: list[OcrBlock], sort_by_position: bool = True) -> str:
    """整页版式块拼成 markdown。"""
    if not blocks:
        return ""
    ordered = sorted(blocks, key=lambda b: (b.top, b.left)) if sort_by_position else blocks
    parts = [md for b in ordered if (md := block_to_markdown(b))]
    return "\n\n".join(parts)


def blocks_from_paddle(result: list[dict[str, Any]]) -> list[OcrBlock]:
    """PaddleX 版式检测结果 → OcrBlock。

    兼容 PP-StructureV3 的几种字段命名。接入真实后端时可能需要按版本微调。
    """
    blocks: list[OcrBlock] = []
    for item in result or []:
        label = item.get("label") or item.get("type") or "text"
        content = (
            item.get("content")
            or item.get("text")
            or item.get("res", {}).get("html", "")
            if isinstance(item.get("res"), dict)
            else item.get("content") or item.get("text") or ""
        )
        bbox = item.get("bbox") or item.get("box") or []
        blocks.append(OcrBlock(label=str(label), content=str(content), bbox=list(bbox)))
    return blocks
