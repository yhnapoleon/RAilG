"""提取器注册表。按扩展名路由到具体提取器。"""

from __future__ import annotations

import logging
from pathlib import Path

from railg.ingest.extractors.base import (
    ExtractedDocument,
    Extractor,
    OcrBackend,
    get_ocr_backend,
    ocr_available,
    register_ocr_backend,
)
from railg.ingest.extractors.layout import (
    OcrBlock,
    blocks_from_paddle,
    blocks_to_markdown,
)
from railg.ingest.extractors.office import DocxExtractor, XlsxExtractor
from railg.ingest.extractors.pdf import PdfExtractor
from railg.ingest.extractors.text_formats import (
    HtmlExtractor,
    MarkdownExtractor,
    TextExtractor,
)

logger = logging.getLogger(__name__)

_EXTRACTORS: list[Extractor] = [
    PdfExtractor(),
    DocxExtractor(),
    XlsxExtractor(),
    MarkdownExtractor(),
    HtmlExtractor(),
    TextExtractor(),
]


def register_extractor(extractor: Extractor, front: bool = True) -> None:
    """注册自定义提取器。front=True 时优先于内置的同后缀提取器。"""
    _EXTRACTORS.insert(0, extractor) if front else _EXTRACTORS.append(extractor)


def supported_extensions() -> set[str]:
    return {ext for e in _EXTRACTORS for ext in e.extensions}


def resolve_extractor(path: Path) -> Extractor | None:
    for extractor in _EXTRACTORS:
        if extractor.supports(path):
            return extractor
    return None


def extract(path: Path) -> ExtractedDocument:
    """提取单个文件。不支持的类型抛 ValueError。"""
    extractor = resolve_extractor(path)
    if extractor is None:
        raise ValueError(
            f"不支持的文件类型: {path.suffix}(已支持 {sorted(supported_extensions())})"
        )
    doc = extractor.extract(path)
    logger.debug("%s 由 %s 提取,%d 页", path.name, extractor.name, len(doc.page_markdowns))
    return doc


__all__ = [
    "DocxExtractor",
    "ExtractedDocument",
    "Extractor",
    "HtmlExtractor",
    "MarkdownExtractor",
    "OcrBackend",
    "OcrBlock",
    "PdfExtractor",
    "TextExtractor",
    "XlsxExtractor",
    "blocks_from_paddle",
    "blocks_to_markdown",
    "extract",
    "get_ocr_backend",
    "ocr_available",
    "register_extractor",
    "register_ocr_backend",
    "resolve_extractor",
    "supported_extensions",
]
