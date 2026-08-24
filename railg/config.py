"""配置加载。

来源优先级(后者覆盖前者):
    config.yaml  →  环境变量 RAILG__<SECTION>__<KEY>  →  API key 专用变量

API key 永远只从环境变量读,不进 YAML。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
class StoreConfig(BaseModel):
    url: str = "http://localhost:9200"
    index: str = "railg"
    bulk_size: int = 200
    # 会话/反馈/文档登记/评测集都放这一个文件里,备份=拷贝
    sqlite_path: str = "data/railg.db"


class EmbeddingConfig(BaseModel):
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "BAAI/bge-m3"
    dims: int = 1024
    batch_size: int = 32
    max_input_chars: int = 8000
    api_key: str = ""


class RerankConfig(BaseModel):
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "BAAI/bge-reranker-v2-m3"
    enabled: bool = True
    api_key: str = ""


class LLMConfig(BaseModel):
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "Qwen/Qwen3-8B"
    temperature: float = 0.3
    max_tokens: int = 2048
    api_key: str = ""


class ChunkConfig(BaseModel):
    """★ 这些参数与父块还原强耦合,改动必须重建索引。"""

    chunk_size: int = 100
    context_size: int = 20
    chunk_overlap: int = 0
    context_overlap: int = 0
    enable_page_numbers: bool = True
    table_header_propagation: bool = True
    ignore_sections: list[str] = Field(default_factory=list)

    @field_validator("chunk_overlap", "context_overlap")
    @classmethod
    def _no_overlap(cls, v: int, info) -> int:
        # 见 railg/ingest/chunker.py 顶部说明:零重叠是父块无损还原的前提。
        # 这里硬性拦截,而不是留一行注释等人踩坑。
        if v != 0:
            raise ValueError(
                f"chunk.{info.field_name} 必须为 0。"
                "非零重叠会让 retrieval/parents.py 的父块还原产生重复文本。"
            )
        return v


class RetrievalConfig(BaseModel):
    top_k: int = 50
    rerank_top_n: int = 10
    max_context_docs: int = 6
    bm25_weight: float = 1.0
    vector_weight: float = 1.0
    knn_num_candidates: int = 200
    knn_min_similarity: float = 0.3
    return_parent: bool = True
    parent_window: int = 3


class GenerationConfig(BaseModel):
    context_budget_tokens: int = 6000
    max_candidate_chars: int = 4000
    verify_attribution: bool = True
    attribution_threshold: float = 0.45


class AuthConfig(BaseModel):
    enabled: bool = False
    admin_user: str = "admin"
    admin_password: str = ""
    jwt_secret: str = ""
    jwt_expire_minutes: int = 10080


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class Settings(BaseModel):
    store: StoreConfig = Field(default_factory=StoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @model_validator(mode="after")
    def _check_ready(self) -> "Settings":
        if self.rerank.enabled and not self.rerank.base_url:
            raise ValueError("rerank.enabled=true 但未配置 rerank.base_url")
        return self


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _load_dotenv(path: Path) -> None:
    """极简 .env 加载,不引入额外依赖。已存在的环境变量不覆盖。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """RAILG__SECTION__KEY=value 覆盖对应字段。"""
    prefix = "RAILG__"
    for env_key, raw in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = [p.lower() for p in env_key[len(prefix):].split("__") if p]
        if len(parts) < 2:
            continue
        section, key = parts[0], parts[-1]
        data.setdefault(section, {})
        if isinstance(data[section], dict):
            data[section][key] = _coerce(raw)
    return data


def _coerce(raw: str) -> Any:
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _inject_api_keys(data: dict[str, Any]) -> dict[str, Any]:
    shared = os.getenv("RAILG_API_KEY", "")
    for section, env_name in (
        ("embedding", "RAILG_EMBEDDING_API_KEY"),
        ("rerank", "RAILG_RERANK_API_KEY"),
        ("llm", "RAILG_LLM_API_KEY"),
    ):
        data.setdefault(section, {})
        data[section]["api_key"] = os.getenv(env_name) or shared

    data.setdefault("auth", {})
    data["auth"]["admin_password"] = os.getenv("RAILG_ADMIN_PASSWORD", "")
    data["auth"]["jwt_secret"] = os.getenv("RAILG_JWT_SECRET", "")
    return data


def load_settings(config_path: Path | str | None = None) -> Settings:
    path = Path(config_path) if config_path else Path(os.getenv("RAILG_CONFIG", DEFAULT_CONFIG))
    _load_dotenv(PROJECT_ROOT / ".env")

    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    data = _apply_env_overrides(data)
    data = _inject_api_keys(data)
    return Settings(**data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
