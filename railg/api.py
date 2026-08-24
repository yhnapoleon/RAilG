"""FastAPI 应用 —— 全流程的 HTTP 面。

    GET    /                          聊天界面
    GET    /admin                     管理界面(文档 / 反馈 / 评测 / 指标)

    GET    /api/health                健康检查
    GET    /api/stats                 索引 + 文档 + 反馈概览
    GET    /api/metrics               请求指标
    POST   /api/login

    POST   /api/chat                  SSE 流式问答(自动建/续会话)
    GET    /api/sessions              会话列表
    POST   /api/sessions              新建会话
    GET    /api/sessions/{id}         会话消息
    PATCH  /api/sessions/{id}         重命名
    DELETE /api/sessions/{id}         删除

    POST   /api/feedback              提交 👍/👎
    GET    /api/feedback              查看反馈

    POST   /api/search                只检索不生成(调参用)

    GET    /api/documents             文档列表
    DELETE /api/documents/{doc_id}    删除文档(索引 + 登记一起删)
    POST   /api/documents/{doc_id}/reindex
    POST   /api/ingest                入库(本地路径)
    POST   /api/ingest/urls           入库(URL)

    GET    /api/eval/cases            评测集
    POST   /api/eval/cases            新增 case
    DELETE /api/eval/cases/{id}
    POST   /api/eval/run              跑评测
    GET    /api/eval/runs             历史跑分
    GET    /api/eval/runs/{id}
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from railg.auth import get_auth_manager
from railg.config import get_settings
from railg.db import get_db
from railg.evaluation.runner import Evaluator
from railg.generation.service import get_chat_service
from railg.ingest.pipeline import IngestPipeline
from railg.providers import close_providers
from railg.retrieval.service import get_retrieval_service
from railg.schema.document import Identity
from railg.store import close_store, get_store

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


# --------------------------------------------------------------------------- #
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    history: list[Message] | None = None
    rewrite: bool = True
    persist: bool = True


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    semantic_only: bool = False
    keyword_only: bool = False


class IngestRequest(BaseModel):
    path: str
    acl_principals: list[str] = Field(default_factory=lambda: ["public"])
    force: bool = False


class IngestUrlRequest(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=50)
    acl_principals: list[str] = Field(default_factory=lambda: ["public"])
    force: bool = False


class FeedbackRequest(BaseModel):
    message_id: str
    session_id: str
    rating: str = Field(pattern="^(up|down)$")
    comment: str = ""
    query: str = ""
    answer: str = ""


class SessionCreate(BaseModel):
    title: str = "新会话"


class SessionRename(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class EvalCaseRequest(BaseModel):
    query: str = Field(min_length=1)
    expected_docs: list[str] = Field(default_factory=list)
    note: str = ""
    tags: list[str] = Field(default_factory=list)


class EvalRunRequest(BaseModel):
    label: str = ""
    compare_with: str = ""


# --------------------------------------------------------------------------- #
def current_identity(authorization: str | None = Header(default=None)) -> Identity:
    return get_auth_manager().identity_from_token(authorization)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await get_db().init()
    store = get_store()
    if await store.ping():
        try:
            await store.ensure_index()
        except Exception as exc:
            logger.error("索引初始化失败: %s", exc)
    else:
        logger.warning(
            "连不上 OpenSearch (%s) —— 先跑 docker compose up -d", settings.store.url
        )
    yield
    await close_store()
    await close_providers()


app = FastAPI(title="RAilG", version="0.2.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# 概览
# --------------------------------------------------------------------------- #
@app.get("/api/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "ok": await get_store().ping(),
        "store_url": settings.store.url,
        "index": settings.store.index,
        "auth_enabled": settings.auth.enabled,
        "models": {
            "embedding": settings.embedding.model,
            "rerank": settings.rerank.model if settings.rerank.enabled else None,
            "llm": settings.llm.model,
        },
    }


@app.get("/api/stats")
async def stats() -> dict:
    db = get_db()
    index_stats = await get_store().stats()
    return {
        "index": index_stats,
        "documents": await db.document_stats(),
        "feedback": await db.feedback_summary(),
    }


@app.get("/api/metrics")
async def metrics(hours: int = Query(default=24, ge=1, le=720)) -> dict:
    return await get_db().request_metrics(hours)


@app.post("/api/login")
async def login(req: dict) -> dict:
    manager = get_auth_manager()
    if not manager.enabled:
        return {"token": "", "detail": "未启用鉴权,无需登录"}
    token = manager.login(req.get("username", ""), req.get("password", ""))
    if not token:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return {"token": token}


# --------------------------------------------------------------------------- #
# 对话
# --------------------------------------------------------------------------- #
@app.post("/api/chat")
async def chat(req: ChatRequest, identity: Identity = Depends(current_identity)):
    service = get_chat_service()
    history = [m.model_dump() for m in req.history] if req.history is not None else None

    async def event_stream():
        try:
            async for event in service.stream(
                req.query,
                identity=identity,
                history=history,
                rewrite=req.rewrite,
                session_id=req.session_id,
                persist=req.persist,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # 兜底,保证前端一定收到终止事件
            logger.exception("chat 流异常")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/sessions")
async def list_sessions(identity: Identity = Depends(current_identity)) -> dict:
    return {"sessions": await get_db().list_sessions(owner=identity.sub)}


@app.post("/api/sessions")
async def create_session(
    req: SessionCreate, identity: Identity = Depends(current_identity)
) -> dict:
    sid = await get_db().create_session(title=req.title, owner=identity.sub)
    return {"session_id": sid}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    db = get_db()
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"session": session, "messages": await db.list_messages(session_id)}


@app.patch("/api/sessions/{session_id}")
async def rename_session(session_id: str, req: SessionRename) -> dict:
    await get_db().rename_session(session_id, req.title)
    return {"ok": True}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    await get_db().delete_session(session_id)
    return {"ok": True}


# --------------------------------------------------------------------------- #
# 反馈
# --------------------------------------------------------------------------- #
@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest) -> dict:
    fid = await get_db().add_feedback(
        message_id=req.message_id,
        session_id=req.session_id,
        rating=req.rating,
        comment=req.comment,
        query=req.query,
        answer=req.answer,
    )
    return {"feedback_id": fid}


@app.get("/api/feedback")
async def list_feedback(
    rating: str | None = Query(default=None, pattern="^(up|down)$"),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict:
    db = get_db()
    return {
        "summary": await db.feedback_summary(),
        "items": await db.list_feedback(rating=rating, limit=limit),
    }


# --------------------------------------------------------------------------- #
# 检索(调参)
# --------------------------------------------------------------------------- #
@app.post("/api/search")
async def search(req: SearchRequest, identity: Identity = Depends(current_identity)) -> dict:
    result = await get_retrieval_service().retrieve(
        req.query,
        identity=identity,
        rewrite=False,
        semantic_only=req.semantic_only,
        keyword_only=req.keyword_only,
    )
    return {
        "trace": result.trace.__dict__,
        "candidates": [c.model_dump() for c in result.candidates],
    }


# --------------------------------------------------------------------------- #
# 文档管理
# --------------------------------------------------------------------------- #
@app.get("/api/documents")
async def list_documents(
    keyword: str = "",
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    db = get_db()
    return {
        "stats": await db.document_stats(),
        "documents": await db.list_documents(limit=limit, offset=offset, keyword=keyword),
    }


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    deleted = await IngestPipeline().delete_document(doc_id)
    return {"ok": True, "deleted_chunks": deleted}


@app.post("/api/documents/{doc_id}/reindex")
async def reindex_document(doc_id: str) -> dict:
    try:
        result = await IngestPipeline().reindex_document(doc_id)
    except (KeyError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": result.status.value,
        "n_chunks": result.n_chunks,
        "error": result.error,
    }


@app.post("/api/ingest")
async def ingest(req: IngestRequest) -> dict:
    path = Path(req.path).expanduser()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"路径不存在: {path}")
    summary = await IngestPipeline().run_paths(
        path, acl_principals=req.acl_principals, force=req.force
    )
    return _summary_payload(summary)


@app.post("/api/ingest/urls")
async def ingest_urls(req: IngestUrlRequest) -> dict:
    summary = await IngestPipeline().run_urls(
        req.urls, acl_principals=req.acl_principals, force=req.force
    )
    return _summary_payload(summary)


def _summary_payload(summary) -> dict:
    return {
        "indexed": summary.indexed,
        "skipped": summary.skipped,
        "failed": summary.failed,
        "total_chunks": summary.total_chunks,
        "failures": [
            {"file": r.file_path, "error": r.error} for r in summary.failures
        ][:20],
    }


# --------------------------------------------------------------------------- #
# 评测
# --------------------------------------------------------------------------- #
@app.get("/api/eval/cases")
async def list_eval_cases() -> dict:
    return {"cases": await get_db().list_eval_cases()}


@app.post("/api/eval/cases")
async def add_eval_case(req: EvalCaseRequest) -> dict:
    cid = await get_db().add_eval_case(
        query=req.query, expected_docs=req.expected_docs, note=req.note, tags=req.tags
    )
    return {"case_id": cid}


@app.delete("/api/eval/cases/{case_id}")
async def delete_eval_case(case_id: str) -> dict:
    await get_db().delete_eval_case(case_id)
    return {"ok": True}


@app.post("/api/eval/run")
async def run_eval(req: EvalRunRequest, identity: Identity = Depends(current_identity)) -> dict:
    evaluator = Evaluator()
    try:
        report = await evaluator.run(label=req.label, identity=identity)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = {
        "run_id": report.run_id,
        "label": report.label,
        "n_cases": report.n_cases,
        "n_failed": report.n_failed,
        "metrics": report.metrics,
        "config": report.config,
        "worst": [c.to_dict() for c in report.worst(5)],
    }
    if req.compare_with:
        payload["comparison"] = await evaluator.compare_with(report, req.compare_with)
    return payload


@app.get("/api/eval/runs")
async def list_eval_runs(limit: int = Query(default=30, ge=1, le=200)) -> dict:
    return {"runs": await get_db().list_eval_runs(limit=limit)}


@app.get("/api/eval/runs/{run_id}")
async def get_eval_run(run_id: str) -> dict:
    run = await get_db().get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="没有这次评测记录")
    return run


# --------------------------------------------------------------------------- #
# 静态页
# --------------------------------------------------------------------------- #
def _page(name: str):
    page = WEB_DIR / name
    if not page.exists():
        raise HTTPException(status_code=404, detail=f"web/{name} 缺失")
    return FileResponse(page)


@app.get("/")
async def index():
    return _page("index.html")


@app.get("/admin")
async def admin():
    return _page("admin.html")
