"""文档提取层。

★ 全局契约:任何提取器都输出 markdown,形态为 (file_markdown, page_markdowns)。

不管前面是原生文本抽取、OCR 还是 VLM,出口形态都一样,下游因此只需要
认识 markdown 这一种输入。保住这个契约,就意味着:
    今天  PDF/DOCX/XLSX/MD/HTML  →  markdown  →  chunker
    以后  扫描件 → OCR           →  markdown  →  chunker      (下游零改动)
    以后  复杂版式 → VLM         →  markdown  →  chunker      (下游零改动)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExtractedDocument:
    """提取结果。

    Attributes:
        file_markdown: 整篇 markdown。留空时由 page_markdowns 拼接。
        page_markdowns: 逐页 markdown。用于 chunker 定位页码;
            无分页概念的格式(md/txt/html)给单元素列表即可。
        meta: 提取器附带的元信息(标题、作者、页数……),并入 DocumentMeta.extras。
        ocr_pages: 实际走了 OCR 的页码,用于观测与质量归因。
    """

    file_markdown: str = ""
    page_markdowns: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    ocr_pages: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.page_markdowns and self.file_markdown:
            self.page_markdowns = [self.file_markdown]
        if not self.file_markdown and self.page_markdowns:
            self.file_markdown = "\n\n".join(self.page_markdowns)

    @property
    def is_empty(self) -> bool:
        return not self.file_markdown.strip()


class Extractor(ABC):
    """按文件类型提取 markdown。"""

    name: str = "base"
    extensions: tuple[str, ...] = ()

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    @abstractmethod
    def extract(self, path: Path) -> ExtractedDocument:
        """同步提取。IO 密集,由 ingest 层放进线程池调用。"""


# --------------------------------------------------------------------------- #
# ✚ OCR / VLM 扩展点 —— 接口就位,实现按需接
# --------------------------------------------------------------------------- #
class OcrBackend(ABC):
    """把图像转成 markdown 的后端。

    这是整个项目为 OCR/VLM 预留的**唯一**扩展点。实现它并调用
    `register_ocr_backend()`,PDF 提取器就会自动把无文本层的页路由过来,
    其余代码一行不用改。

    可能的实现:
        PaddleOcrBackend   PP-StructureV3,版式块 → blocks_to_markdown()
        VlmBackend         Qwen2.5-VL 等,走 OpenAI 兼容的 vision 端点
        MineruBackend      重版式文档
    """

    name: str = "ocr"

    @abstractmethod
    def ocr_images(self, images: list[bytes], hint: dict[str, Any] | None = None) -> list[str]:
        """输入若干页图像(PNG/JPEG 字节),返回等长的 markdown 列表。"""

    def available(self) -> bool:
        """依赖是否就绪。未就绪的后端不会被注册进路由。"""
        return True


_ocr_backend: OcrBackend | None = None


def register_ocr_backend(backend: OcrBackend | None) -> None:
    global _ocr_backend
    if backend is not None and not backend.available():
        logger.warning("OCR 后端 %s 依赖未就绪,跳过注册", backend.name)
        return
    _ocr_backend = backend
    if backend:
        logger.info("已注册 OCR 后端: %s", backend.name)


def get_ocr_backend() -> OcrBackend | None:
    return _ocr_backend


def ocr_available() -> bool:
    return _ocr_backend is not None
