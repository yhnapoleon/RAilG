"""✚ OCR / VLM 后端 —— 预留实现。

两个后端都已写全,默认**不注册**,因为它们需要额外依赖或额外费用:

    PaddleOcrBackend   pip install railg[ocr]     本地跑,零 API 费用
    VlmBackend         走 OpenAI 兼容的 vision 端点,按量计费

启用方式(在 CLI 或应用启动时调一次):

    from railg.ingest.extractors.base import register_ocr_backend
    from railg.ingest.extractors.ocr_backends import PaddleOcrBackend, VlmBackend

    register_ocr_backend(PaddleOcrBackend())          # 本地 OCR
    register_ocr_backend(VlmBackend(model="Qwen/Qwen2.5-VL-7B-Instruct"))  # VLM

注册后,PdfExtractor 会自动把无文本层的页路由过来,其余代码零改动。
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

from railg.ingest.extractors.base import OcrBackend
from railg.ingest.extractors.layout import blocks_from_paddle, blocks_to_markdown

logger = logging.getLogger(__name__)


class PaddleOcrBackend(OcrBackend):
    """PP-StructureV3 版式识别 → markdown。

    走的是 layout.py 里那套 block→markdown 转换,输出形态与其它提取器一致。
    """

    name = "paddle-ocr"

    def __init__(self, lang: str = "ch", use_layout: bool = True) -> None:
        self.lang = lang
        self.use_layout = use_layout
        self._engine: Any = None

    def available(self) -> bool:
        try:
            import paddleocr  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_engine(self) -> Any:
        if self._engine is None:
            from paddleocr import PPStructure

            self._engine = PPStructure(show_log=False, lang=self.lang)
        return self._engine

    def ocr_images(self, images: list[bytes], hint: dict[str, Any] | None = None) -> list[str]:
        import numpy as np
        from PIL import Image
        import io

        engine = self._get_engine()
        out: list[str] = []
        for payload in images:
            try:
                img = np.array(Image.open(io.BytesIO(payload)).convert("RGB"))
                result = engine(img)
                blocks = blocks_from_paddle(result)
                out.append(blocks_to_markdown(blocks))
            except Exception as exc:
                logger.error("PaddleOCR 单页失败: %s", exc)
                out.append("")
        return out


VLM_PROMPT = (
    "把这一页文档完整转成 Markdown。要求:\n"
    "1. 标题用 ## 标记(这是下游切块的分节依据,务必遵守)\n"
    "2. 表格转成 Markdown 表格,保留表头\n"
    "3. 公式用 $$ 包裹\n"
    "4. 忽略页眉、页脚、页码\n"
    "5. 只输出 Markdown 正文,不要任何解释或代码块围栏"
)


class VlmBackend(OcrBackend):
    """视觉语言模型 → markdown。

    走 OpenAI 兼容的多模态 chat 端点,因此云 API 和本地 vLLM 都能用。
    默认复用 llm 段的 base_url / api_key,可单独指定。
    """

    name = "vlm"

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        base_url: str | None = None,
        api_key: str | None = None,
        prompt: str = VLM_PROMPT,
        max_concurrency: int = 4,
    ) -> None:
        from railg.config import get_settings

        s = get_settings()
        self.model = model
        self.base_url = (base_url or s.llm.base_url).rstrip("/")
        self.api_key = api_key or s.llm.api_key
        self.prompt = prompt
        self.max_concurrency = max_concurrency

    def available(self) -> bool:
        return bool(self.api_key)

    def ocr_images(self, images: list[bytes], hint: dict[str, Any] | None = None) -> list[str]:
        # 提取器是同步接口(跑在线程池里),这里自建事件循环做并发页处理
        return asyncio.run(self._ocr_async(images))

    async def _ocr_async(self, images: list[bytes]) -> list[str]:
        import httpx

        sem = asyncio.Semaphore(self.max_concurrency)
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(180.0, connect=10.0),
        ) as client:

            async def one(payload: bytes) -> str:
                b64 = base64.b64encode(payload).decode()
                body = {
                    "model": self.model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            {"type": "text", "text": self.prompt},
                        ],
                    }],
                    "temperature": 0.0,
                    "max_tokens": 4096,
                }
                async with sem:
                    try:
                        resp = await client.post("/chat/completions", json=body)
                        if resp.status_code >= 400:
                            logger.error("VLM 返回 %s: %s", resp.status_code, resp.text[:300])
                            return ""
                        choices = resp.json().get("choices") or []
                        return choices[0]["message"]["content"].strip() if choices else ""
                    except Exception as exc:
                        logger.error("VLM 单页失败: %s", exc)
                        return ""

            return list(await asyncio.gather(*(one(img) for img in images)))
