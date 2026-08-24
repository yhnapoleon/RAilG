"""SQLite 持久化:会话、消息、反馈、文档登记、评测集、请求日志。

为什么用 SQLite 而不是 PostgreSQL:个人项目单机跑,一个文件就是全部状态,
备份等于拷贝文件。要上多副本时换掉 `_connect` 即可,表结构和 SQL 都是标准的。

为什么不用 ORM:表只有六张,SQL 直白可读,省一个重依赖。

线程模型:sqlite3 连接不跨线程共享,所以每次调用新建连接并走 asyncio.to_thread。
个人规模下这点开销可以忽略,换来的是完全不用操心并发。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '新会话',
    owner       TEXT NOT NULL DEFAULT 'anonymous',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_owner ON sessions(owner, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    sources     TEXT NOT NULL DEFAULT '[]',
    trace       TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);

-- 反馈是 RAG 产品迭代的唯一真实输入:
-- 记录用户说哪次答得不好,以及当时召回了什么(trace 在 messages 里)。
CREATE TABLE IF NOT EXISTS feedback (
    id          TEXT PRIMARY KEY,
    message_id  TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    session_id  TEXT NOT NULL,
    rating      TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment     TEXT NOT NULL DEFAULT '',
    query       TEXT NOT NULL DEFAULT '',
    answer      TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating, created_at DESC);

-- OpenSearch 里存的是 chunk,这里存"文档"这一层,支撑管理界面的增删查改。
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    file_name     TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    file_url      TEXT NOT NULL DEFAULT '',
    document_type TEXT NOT NULL DEFAULT '',
    content_hash  TEXT NOT NULL DEFAULT '',
    n_chunks      INTEGER NOT NULL DEFAULT 0,
    acl           TEXT NOT NULL DEFAULT '["public"]',
    source_kind   TEXT NOT NULL DEFAULT 'local',
    ocr_pages     TEXT NOT NULL DEFAULT '[]',
    status        TEXT NOT NULL DEFAULT 'indexed',
    error         TEXT NOT NULL DEFAULT '',
    indexed_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_indexed ON documents(indexed_at DESC);

CREATE TABLE IF NOT EXISTS eval_cases (
    id            TEXT PRIMARY KEY,
    query         TEXT NOT NULL,
    expected_docs TEXT NOT NULL DEFAULT '[]',
    note          TEXT NOT NULL DEFAULT '',
    tags          TEXT NOT NULL DEFAULT '[]',
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id          TEXT PRIMARY KEY,
    label       TEXT NOT NULL DEFAULT '',
    config      TEXT NOT NULL DEFAULT '{}',
    metrics     TEXT NOT NULL DEFAULT '{}',
    details     TEXT NOT NULL DEFAULT '[]',
    n_cases     INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_runs_time ON eval_runs(created_at DESC);

CREATE TABLE IF NOT EXISTS request_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT NOT NULL,
    query        TEXT NOT NULL DEFAULT '',
    ok           INTEGER NOT NULL DEFAULT 1,
    latency_ms   INTEGER NOT NULL DEFAULT 0,
    stages       TEXT NOT NULL DEFAULT '{}',
    n_candidates INTEGER NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_request_log_time ON request_log(created_at DESC);
"""

PUBLIC_ACL_JSON = '["public"]'


def _now() -> float:
    return time.time()


def _uid() -> str:
    return uuid.uuid4().hex[:16]


@dataclass(slots=True)
class Database:
    path: Path

    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _exec(self, sql: str, params: tuple = ()) -> None:
        with self._connect() as conn:
            conn.execute(sql, params)

    def _query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def _one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        rows = self._query(sql, params)
        return rows[0] if rows else None

    # ------------------------------------------------------------------ #
    def init_sync(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
        logger.debug("SQLite 就绪: %s", self.path)

    async def init(self) -> None:
        await asyncio.to_thread(self.init_sync)

    # --- 会话 ---------------------------------------------------------- #
    async def create_session(self, title: str = "新会话", owner: str = "anonymous") -> str:
        sid = _uid()
        now = _now()
        await asyncio.to_thread(
            self._exec,
            "INSERT INTO sessions (id, title, owner, created_at, updated_at) VALUES (?,?,?,?,?)",
            (sid, title[:120], owner, now, now),
        )
        return sid

    async def list_sessions(self, owner: str = "anonymous", limit: int = 50) -> list[dict]:
        return await asyncio.to_thread(
            self._query,
            "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) "
            "AS n_messages FROM sessions s WHERE s.owner = ? "
            "ORDER BY s.updated_at DESC LIMIT ?",
            (owner, limit),
        )

    async def get_session(self, session_id: str) -> dict | None:
        return await asyncio.to_thread(
            self._one, "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )

    async def rename_session(self, session_id: str, title: str) -> None:
        await asyncio.to_thread(
            self._exec,
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title[:120], _now(), session_id),
        )

    async def delete_session(self, session_id: str) -> None:
        await asyncio.to_thread(self._exec, "DELETE FROM sessions WHERE id = ?", (session_id,))

    async def touch_session(self, session_id: str) -> None:
        await asyncio.to_thread(
            self._exec, "UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id)
        )

    # --- 消息 ---------------------------------------------------------- #
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list | None = None,
        trace: dict | None = None,
    ) -> str:
        mid = _uid()
        await asyncio.to_thread(
            self._exec,
            "INSERT INTO messages (id, session_id, role, content, sources, trace, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                mid,
                session_id,
                role,
                content,
                json.dumps(sources or [], ensure_ascii=False),
                json.dumps(trace or {}, ensure_ascii=False),
                _now(),
            ),
        )
        await self.touch_session(session_id)
        return mid

    async def list_messages(self, session_id: str, limit: int = 200) -> list[dict]:
        rows = await asyncio.to_thread(
            self._query,
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at LIMIT ?",
            (session_id, limit),
        )
        for r in rows:
            r["sources"] = json.loads(r["sources"] or "[]")
            r["trace"] = json.loads(r["trace"] or "{}")
        return rows

    async def history_for_llm(self, session_id: str, turns: int = 6) -> list[dict[str, str]]:
        """取最近若干轮喂给查询改写与生成。只保留 role/content。"""
        rows = await asyncio.to_thread(
            self._query,
            "SELECT role, content FROM messages WHERE session_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, turns * 2),
        )
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    # --- 反馈 ---------------------------------------------------------- #
    async def add_feedback(
        self,
        message_id: str,
        session_id: str,
        rating: str,
        comment: str = "",
        query: str = "",
        answer: str = "",
    ) -> str:
        fid = _uid()
        await asyncio.to_thread(
            self._exec,
            "INSERT INTO feedback "
            "(id, message_id, session_id, rating, comment, query, answer, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                fid, message_id, session_id, rating,
                comment[:2000], query[:2000], answer[:8000], _now(),
            ),
        )
        return fid

    async def list_feedback(self, rating: str | None = None, limit: int = 100) -> list[dict]:
        if rating:
            return await asyncio.to_thread(
                self._query,
                "SELECT * FROM feedback WHERE rating = ? ORDER BY created_at DESC LIMIT ?",
                (rating, limit),
            )
        return await asyncio.to_thread(
            self._query, "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,)
        )

    async def feedback_summary(self) -> dict[str, int]:
        rows = await asyncio.to_thread(
            self._query, "SELECT rating, COUNT(*) AS n FROM feedback GROUP BY rating"
        )
        out = {"up": 0, "down": 0}
        for r in rows:
            out[r["rating"]] = r["n"]
        return out

    # --- 文档登记 ------------------------------------------------------ #
    async def upsert_document(self, doc: dict) -> None:
        await asyncio.to_thread(
            self._exec,
            "INSERT INTO documents "
            "(doc_id, file_name, file_path, file_url, document_type, content_hash, "
            " n_chunks, acl, source_kind, ocr_pages, status, error, indexed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(doc_id) DO UPDATE SET "
            "  file_name=excluded.file_name, file_path=excluded.file_path, "
            "  file_url=excluded.file_url, document_type=excluded.document_type, "
            "  content_hash=excluded.content_hash, n_chunks=excluded.n_chunks, "
            "  acl=excluded.acl, source_kind=excluded.source_kind, "
            "  ocr_pages=excluded.ocr_pages, status=excluded.status, "
            "  error=excluded.error, indexed_at=excluded.indexed_at",
            (
                doc["doc_id"],
                doc.get("file_name", ""),
                doc.get("file_path", ""),
                doc.get("file_url", ""),
                doc.get("document_type", ""),
                doc.get("content_hash", ""),
                int(doc.get("n_chunks", 0)),
                json.dumps(doc.get("acl", ["public"]), ensure_ascii=False),
                doc.get("source_kind", "local"),
                json.dumps(doc.get("ocr_pages", []), ensure_ascii=False),
                doc.get("status", "indexed"),
                (doc.get("error") or "")[:1000],
                _now(),
            ),
        )

    async def list_documents(
        self, limit: int = 200, offset: int = 0, keyword: str = ""
    ) -> list[dict]:
        if keyword:
            like = f"%{keyword}%"
            rows = await asyncio.to_thread(
                self._query,
                "SELECT * FROM documents WHERE file_name LIKE ? OR file_path LIKE ? "
                "ORDER BY indexed_at DESC LIMIT ? OFFSET ?",
                (like, like, limit, offset),
            )
        else:
            rows = await asyncio.to_thread(
                self._query,
                "SELECT * FROM documents ORDER BY indexed_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        for r in rows:
            r["acl"] = json.loads(r["acl"] or PUBLIC_ACL_JSON)
            r["ocr_pages"] = json.loads(r["ocr_pages"] or "[]")
        return rows

    async def get_document(self, doc_id: str) -> dict | None:
        row = await asyncio.to_thread(
            self._one, "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        )
        if row:
            row["acl"] = json.loads(row["acl"] or PUBLIC_ACL_JSON)
            row["ocr_pages"] = json.loads(row["ocr_pages"] or "[]")
        return row

    async def delete_document(self, doc_id: str) -> None:
        await asyncio.to_thread(self._exec, "DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    async def document_stats(self) -> dict:
        rows = await asyncio.to_thread(
            self._query,
            "SELECT COUNT(*) AS n_docs, COALESCE(SUM(n_chunks), 0) AS n_chunks, "
            "COALESCE(SUM(CASE WHEN status != 'indexed' THEN 1 ELSE 0 END), 0) AS n_failed "
            "FROM documents",
        )
        return rows[0] if rows else {"n_docs": 0, "n_chunks": 0, "n_failed": 0}

    # --- 评测 ---------------------------------------------------------- #
    async def add_eval_case(
        self,
        query: str,
        expected_docs: list[str],
        note: str = "",
        tags: list[str] | None = None,
    ) -> str:
        cid = _uid()
        await asyncio.to_thread(
            self._exec,
            "INSERT INTO eval_cases (id, query, expected_docs, note, tags, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (
                cid, query,
                json.dumps(expected_docs, ensure_ascii=False),
                note,
                json.dumps(tags or [], ensure_ascii=False),
                _now(),
            ),
        )
        return cid

    async def list_eval_cases(self, limit: int = 1000) -> list[dict]:
        rows = await asyncio.to_thread(
            self._query, "SELECT * FROM eval_cases ORDER BY created_at LIMIT ?", (limit,)
        )
        for r in rows:
            r["expected_docs"] = json.loads(r["expected_docs"] or "[]")
            r["tags"] = json.loads(r["tags"] or "[]")
        return rows

    async def delete_eval_case(self, case_id: str) -> None:
        await asyncio.to_thread(self._exec, "DELETE FROM eval_cases WHERE id = ?", (case_id,))

    async def save_eval_run(
        self, label: str, config: dict, metrics: dict, details: list, n_cases: int
    ) -> str:
        rid = _uid()
        await asyncio.to_thread(
            self._exec,
            "INSERT INTO eval_runs (id, label, config, metrics, details, n_cases, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                rid, label,
                json.dumps(config, ensure_ascii=False),
                json.dumps(metrics, ensure_ascii=False),
                json.dumps(details, ensure_ascii=False),
                n_cases, _now(),
            ),
        )
        return rid

    async def list_eval_runs(self, limit: int = 30) -> list[dict]:
        rows = await asyncio.to_thread(
            self._query,
            "SELECT id, label, metrics, n_cases, created_at FROM eval_runs "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        for r in rows:
            r["metrics"] = json.loads(r["metrics"] or "{}")
        return rows

    async def get_eval_run(self, run_id: str) -> dict | None:
        row = await asyncio.to_thread(self._one, "SELECT * FROM eval_runs WHERE id = ?", (run_id,))
        if row:
            row["config"] = json.loads(row["config"] or "{}")
            row["metrics"] = json.loads(row["metrics"] or "{}")
            row["details"] = json.loads(row["details"] or "[]")
        return row

    # --- 请求日志 ------------------------------------------------------ #
    async def log_request(
        self,
        kind: str,
        query: str,
        ok: bool,
        latency_ms: int,
        stages: dict | None = None,
        n_candidates: int = 0,
    ) -> None:
        await asyncio.to_thread(
            self._exec,
            "INSERT INTO request_log "
            "(kind, query, ok, latency_ms, stages, n_candidates, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                kind, query[:500], int(ok), latency_ms,
                json.dumps(stages or {}, ensure_ascii=False),
                n_candidates, _now(),
            ),
        )

    async def request_metrics(self, hours: int = 24) -> dict:
        since = _now() - hours * 3600
        rows = await asyncio.to_thread(
            self._query,
            "SELECT COUNT(*) AS n, COALESCE(SUM(ok), 0) AS n_ok, "
            "COALESCE(AVG(latency_ms), 0) AS avg_ms, COALESCE(MAX(latency_ms), 0) AS max_ms "
            "FROM request_log WHERE created_at >= ?",
            (since,),
        )
        row = rows[0] if rows else {}
        n = row.get("n") or 0
        n_ok = row.get("n_ok") or 0
        return {
            "window_hours": hours,
            "requests": n,
            "success": n_ok,
            "error_rate": round(1 - n_ok / n, 4) if n else 0.0,
            "avg_latency_ms": int(row.get("avg_ms") or 0),
            "max_latency_ms": int(row.get("max_ms") or 0),
        }


_db: Database | None = None


def get_db() -> Database:
    global _db
    if _db is None:
        from railg.config import get_settings

        _db = Database(Path(get_settings().store.sqlite_path))
        _db.init_sync()
    return _db


def reset_db() -> None:
    """测试用:丢弃全局实例。"""
    global _db
    _db = None
