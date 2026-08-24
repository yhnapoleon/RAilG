"""PDF 提取器 —— 含 OCR 路由。

分两条路:
    有文本层的页  →  pypdf 直接抽,零成本
    无文本层的页  →  路由到 OcrBackend(若已注册)

★ 路由逻辑现在就是完整的。缺的只是后端实现。这就是"为 OCR/VLM 留余量"
  的具体含义:接 PaddleOCR 时只需写一个 OcrBackend 并注册,本文件不用改,
  下游 chunk/embed/index/retrieval 更不用改。

渲染页面为图像需要 pymupdf(pip install railg[ocr])。未安装时,
低文本页会被如实标记并跳过,而不是静默产出空内容。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from railg.ingest.extractors.base import (
    ExtractedDocument,
    Extractor,
    get_ocr_backend,
)

logger = logging.getLogger(__name__)

# 一页少于这么多个非空白字符,就认为它没有可用文本层
MIN_TEXT_CHARS = 40
# 渲染倍率:2.0 ≈ 144dpi,OCR 精度与体积的常用折中
RENDER_ZOOM = 2.0


class PdfExtractor(Extractor):
    name = "pdf"
    extensions = (".pdf",)

    def __init__(self, min_text_chars: int = MIN_TEXT_CHARS) -> None:
        self.min_text_chars = min_text_chars

    def extract(self, path: Path) -> ExtractedDocument:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages: list[str] = []
        low_text_idx: list[int] = []

        for i, page in enumerate(reader.pages):
            try:
                raw = page.extract_text() or ""
            except Exception as exc:  # 单页损坏不该毁掉整个文件
                logger.warning("%s 第 %d 页文本抽取失败: %s", path.name, i + 1, exc)
                raw = ""
            text = _normalize(raw)
            pages.append(text)
            if len(text.strip()) < self.min_text_chars:
                low_text_idx.append(i)

        ocr_pages: list[int] = []
        if low_text_idx:
            ocr_pages = self._try_ocr(path, pages, low_text_idx)

        meta = {"n_pages": len(pages)}
        if reader.metadata:
            for key, field in (("title", "/Title"), ("author", "/Author")):
                value = reader.metadata.get(field)
                if value:
                    meta[key] = str(value)
        if low_text_idx:
            meta["low_text_pages"] = [i + 1 for i in low_text_idx]

        return ExtractedDocument(page_markdowns=pages, meta=meta, ocr_pages=ocr_pages)

    # ----------------------------------------------------------------- #
    # OCR 路由
    # ----------------------------------------------------------------- #
    def _try_ocr(self, path: Path, pages: list[str], targets: list[int]) -> list[int]:
        backend = get_ocr_backend()
        if backend is None:
            logger.warning(
                "%s 有 %d 页缺少文本层(第 %s 页),但未注册 OCR 后端,这些页将为空。"
                "安装 railg[ocr] 并注册后端即可处理。",
                path.name, len(targets), ", ".join(str(i + 1) for i in targets[:10]),
            )
            return []

        images = _render_pages(path, targets)
        if not images:
            return []

        try:
            markdowns = backend.ocr_images(images, hint={"source": str(path)})
        except Exception as exc:
            logger.error("%s OCR 失败: %s", path.name, exc)
            return []

        done: list[int] = []
        for idx, md in zip(targets, markdowns):
            if md and md.strip():
                pages[idx] = md.strip()
                done.append(idx + 1)
        logger.info("%s 通过 %s 补齐 %d 页", path.name, backend.name, len(done))
        return done


def _render_pages(path: Path, indices: list[int]) -> list[bytes]:
    """把指定页渲染为 PNG 字节。需要 pymupdf。"""
    try:
        import fitz  # pymupdf
    except ImportError:
        logger.warning(
            "需要 pymupdf 才能把 PDF 页渲染为图像:pip install railg[ocr]"
        )
        return []

    out: list[bytes] = []
    with fitz.open(str(path)) as doc:
        matrix = fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM)
        for i in indices:
            if i >= doc.page_count:
                continue
            pix = doc.load_page(i).get_pixmap(matrix=matrix)
            out.append(pix.tobytes("png"))
    return out


def _normalize(text: str) -> str:
    """pypdf 抽出来的文本常有断行和多余空白。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    # 连字符断行拼回:  exam-\nple  ->  example
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
