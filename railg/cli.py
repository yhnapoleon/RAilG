"""命令行入口。

    railg check                     体检:配置、契约、OpenSearch、模型连通性
    railg ingest <路径>             本地入库
    railg ingest-url <URL...>       抓网页 / 在线文件入库
    railg ask "问题"                命令行问答
    railg search "关键词"           只看检索结果(调参用)
    railg docs list|delete|reindex  文档管理
    railg eval add|import|run|list  评测
    railg feedback                  查看反馈
    railg stats                     总览
    railg serve                     起 Web 服务
    railg reset --yes               删索引
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from railg.config import get_settings
from railg.schema.document import IngestStatus

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"

G, R, Y, DIM, B, X = "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[1m", "\033[0m"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "opensearch", "elastic_transport", "aiohttp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _ok(m: str) -> None:
    print(f"  {G}✓{X} {m}")


def _bad(m: str) -> None:
    print(f"  {R}✗{X} {m}")


def _ts(value: float) -> str:
    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M")


def _progress(result) -> None:
    name = Path(result.file_path).name
    if result.status is IngestStatus.INDEXED:
        print(f"  {G}✓{X} {name}  ({result.n_chunks} 块)")
    elif result.status is IngestStatus.SKIPPED_UNCHANGED:
        print(f"  {DIM}·{X} {name}  (未变更)")
    else:
        print(f"  {R}✗{X} {name}  {result.error}")


def _register_ocr(kind: str | None) -> None:
    if not kind:
        return
    from railg.ingest.extractors.base import register_ocr_backend
    from railg.ingest.extractors.ocr_backends import PaddleOcrBackend, VlmBackend

    register_ocr_backend(VlmBackend() if kind == "vlm" else PaddleOcrBackend())


# --------------------------------------------------------------------------- #
async def cmd_check(args) -> int:
    from railg.db import get_db
    from railg.providers import get_embedding_provider, get_llm_provider
    from railg.schema.mapping import verify_contract
    from railg.store import get_store

    settings = get_settings()
    failed = 0

    print("\n配置")
    _ok(f"OpenSearch  {settings.store.url}  索引={settings.store.index}")
    _ok(f"SQLite      {settings.store.sqlite_path}")
    _ok(f"Embedding   {settings.embedding.model}  dims={settings.embedding.dims}")
    _ok(f"Rerank      {settings.rerank.model if settings.rerank.enabled else '未启用'}")
    _ok(f"LLM         {settings.llm.model}")
    if not settings.embedding.api_key:
        _bad("未设置 API key,请在 .env 里填 RAILG_API_KEY")
        failed += 1

    print("\nschema 契约")
    try:
        verify_contract(settings.embedding.dims)
        _ok("检索读取字段与 mapping 定义一致")
    except Exception as exc:
        _bad(f"契约校验失败: {exc}")
        failed += 1

    print("\nSQLite")
    try:
        db = get_db()
        await db.init()
        _ok(f"就绪,{(await db.document_stats())['n_docs']} 篇文档登记在案")
    except Exception as exc:
        _bad(f"SQLite 初始化失败: {exc}")
        failed += 1

    print("\nOpenSearch")
    store = get_store()
    if await store.ping():
        _ok("连接正常")
        try:
            created = await store.ensure_index()
            _ok("已创建索引" if created else "索引已存在")
            info = await store.stats()
            _ok(f"当前 {info['n_docs']} 篇文档 / {info['n_chunks']} 个块")
        except Exception as exc:
            _bad(f"索引检查失败: {exc}")
            failed += 1
    else:
        _bad(f"连不上 {settings.store.url} —— 先跑 docker compose up -d")
        failed += 1

    if settings.embedding.api_key:
        print("\n模型连通性")
        try:
            vec = await get_embedding_provider().embed_one("连通性测试")
            _ok(f"embedding 正常,返回 {len(vec)} 维")
        except Exception as exc:
            _bad(f"embedding 调用失败: {exc}")
            failed += 1
        try:
            reply = await get_llm_provider().complete(
                [{"role": "user", "content": "只回复:ok"}], max_tokens=10
            )
            _ok(f"LLM 正常,返回 {reply.strip()[:40]!r}")
        except Exception as exc:
            _bad(f"LLM 调用失败: {exc}")
            failed += 1

    print(f"\n{G}全部通过{X}\n" if not failed else f"\n{R}{failed} 项未通过{X}\n")
    return 1 if failed else 0


async def cmd_ingest(args) -> int:
    from railg.ingest.pipeline import IngestPipeline

    path = Path(args.path).expanduser()
    if not path.exists():
        print(f"路径不存在: {path}")
        return 1

    _register_ocr(args.ocr)
    print(f"\n开始入库: {path}\n")
    summary = await IngestPipeline().run_paths(
        path, acl_principals=args.acl or ["public"],
        force=args.force, concurrency=args.concurrency, on_progress=_progress,
    )
    print(
        f"\n完成: 入库 {summary.indexed} / 跳过 {summary.skipped} / "
        f"失败 {summary.failed},共 {summary.total_chunks} 个块\n"
    )
    return 0 if summary.failed == 0 else 1


async def cmd_ingest_url(args) -> int:
    from railg.ingest.pipeline import IngestPipeline

    print(f"\n开始抓取 {len(args.urls)} 个 URL\n")
    summary = await IngestPipeline().run_urls(
        args.urls, acl_principals=args.acl or ["public"],
        force=args.force, on_progress=_progress,
    )
    print(
        f"\n完成: 入库 {summary.indexed} / 跳过 {summary.skipped} / "
        f"失败 {summary.failed},共 {summary.total_chunks} 个块\n"
    )
    return 0 if summary.failed == 0 else 1


async def cmd_ask(args) -> int:
    from railg.generation.service import get_chat_service

    print()
    sources: list[dict] = []
    warnings: list[dict] = []

    async for event in get_chat_service().stream(
        args.query, rewrite=False, persist=not args.no_save
    ):
        kind = event["type"]
        if kind == "delta":
            print(event["text"], end="", flush=True)
        elif kind == "trace" and args.verbose:
            t = event["data"]
            print(f"{DIM}[召回 {t['n_raw']} → 重排 {t['n_reranked']} → "
                  f"父块 {t['n_parents']} | {t['timings_ms']}]{X}")
        elif kind == "sources":
            sources = event["data"]
        elif kind == "warning":
            warnings = event["data"]
        elif kind == "error":
            print(f"\n{R}{event['message']}{X}")
            return 1

    if sources:
        print(f"\n\n{DIM}来源:{X}")
        for s in sources:
            print(f"  {DIM}[{s['n']}] {s['label']}{X}")
    if warnings:
        print(f"\n{Y}支撑不足的引用:{X}")
        for w in warnings:
            print(f"  {Y}{w['sentence'][:80]}… (相似度 {w['similarity']}){X}")
    print()
    return 0


async def cmd_search(args) -> int:
    from railg.retrieval.service import get_retrieval_service

    result = await get_retrieval_service().retrieve(args.query, rewrite=False)
    t = result.trace
    print(f"\n召回 {t.n_raw} → 重排 {t.n_reranked} → 父块 {t.n_parents}   {t.timings_ms}\n")
    for i, c in enumerate(result.candidates, 1):
        score = c.rerank_score if c.rerank_score is not None else c.normalized_score
        print(f"{B}[{i}] {c.file_name}{X}  第 {c.page_no} 页  分数 {score:.4f}"
              f"{'  (父块)' if c.is_parent else ''}")
        head = c.header2 or c.header1
        if head:
            print(f"    {DIM}{head}{X}")
        print(f"    {(c.snippet or '')[:200].replace(chr(10), ' ')}…\n")
    return 0


# --- 文档管理 --------------------------------------------------------------- #
async def cmd_docs(args) -> int:
    from railg.db import get_db
    from railg.ingest.pipeline import IngestPipeline

    db = get_db()
    await db.init()

    if args.docs_action == "list":
        docs = await db.list_documents(limit=args.limit, keyword=args.keyword or "")
        if not docs:
            print("\n还没有文档,先跑 railg ingest\n")
            return 0
        stats = await db.document_stats()
        print(f"\n共 {stats['n_docs']} 篇 / {stats['n_chunks']} 块")
        if stats.get("n_failed"):
            print(f"{R}其中 {stats['n_failed']} 篇失败{X}")
        print()
        for d in docs:
            mark = f"{G}✓{X}" if d["status"] == "indexed" else f"{R}✗{X}"
            print(f"  {mark} {d['doc_id']}  {d['file_name']}  "
                  f"{d['n_chunks']} 块  {DIM}{_ts(d['indexed_at'])}{X}")
            if d["error"]:
                print(f"      {R}{d['error'][:100]}{X}")
        print()
        return 0

    if args.docs_action == "delete":
        n = await IngestPipeline().delete_document(args.doc_id)
        print(f"已删除 {args.doc_id},清理 {n} 个块")
        return 0

    if args.docs_action == "reindex":
        try:
            result = await IngestPipeline().reindex_document(args.doc_id)
        except Exception as exc:
            print(f"{R}{exc}{X}")
            return 1
        print(f"{result.status.value}: {result.n_chunks} 块 {result.error}")
        return 0

    return 1


# --- 评测 ------------------------------------------------------------------- #
async def cmd_eval(args) -> int:
    from railg.db import get_db
    from railg.evaluation.runner import Evaluator, export_cases, import_cases

    db = get_db()
    await db.init()

    if args.eval_action == "add":
        cid = await db.add_eval_case(args.query, args.doc or [], note=args.note or "")
        print(f"已添加 case {cid}")
        return 0

    if args.eval_action == "list":
        cases = await db.list_eval_cases()
        if not cases:
            print("\n评测集是空的。加一条:railg eval add \"问题\" --doc 文件名.pdf\n")
            return 0
        print(f"\n共 {len(cases)} 条\n")
        for c in cases:
            print(f"  {c['id']}  {c['query']}")
            print(f"      {DIM}期望: {', '.join(c['expected_docs']) or '(未标注)'}{X}")
        print()
        return 0

    if args.eval_action == "import":
        n = await import_cases(args.path, db=db)
        print(f"导入 {n} 条")
        return 0

    if args.eval_action == "export":
        n = await export_cases(args.path, db=db)
        print(f"导出 {n} 条到 {args.path}")
        return 0

    if args.eval_action == "runs":
        runs = await db.list_eval_runs()
        if not runs:
            print("\n还没跑过评测\n")
            return 0
        print()
        for r in runs:
            m = r["metrics"]
            print(f"  {r['id']}  {B}{r['label'] or '(无标签)'}{X}  {r['n_cases']} 条  "
                  f"{DIM}{_ts(r['created_at'])}{X}")
            print(f"      recall@5={m.get('recall@5', 0):.3f}  "
                  f"ndcg@10={m.get('ndcg@10', 0):.3f}  mrr={m.get('mrr', 0):.3f}")
        print()
        return 0

    if args.eval_action == "run":
        def on_case(i, total, result):
            mark = f"{G}✓{X}" if result.metrics.get("recall@5", 0) > 0 else f"{Y}·{X}"
            print(f"  [{i:>3}/{total}] {mark} {result.query[:50]}")

        evaluator = Evaluator()
        try:
            print()
            report = await evaluator.run(label=args.label or "", on_case=on_case)
        except RuntimeError as exc:
            print(f"\n{R}{exc}{X}\n")
            return 1

        print(f"\n{B}评测结果{X}  ({report.n_cases} 条"
              f"{f', {report.n_failed} 条出错' if report.n_failed else ''})\n")
        for key in ("recall@1", "recall@3", "recall@5", "recall@10",
                    "ndcg@5", "ndcg@10", "mrr"):
            if key in report.metrics:
                print(f"  {key:<12} {report.metrics[key]:.4f}")

        if args.compare:
            comparison = await evaluator.compare_with(report, args.compare)
            if comparison:
                print(f"\n{B}与基线 {args.compare} 对比{X}\n")
                for key, d in comparison["diff"].items():
                    if not key.startswith(("recall@", "ndcg@")) and key != "mrr":
                        continue
                    color = R if d["regressed"] else (G if d["delta"] > 0 else DIM)
                    sign = "+" if d["delta"] >= 0 else ""
                    print(f"  {key:<12} {d['before']:.4f} → {d['after']:.4f}  "
                          f"{color}{sign}{d['delta']:.4f}{X}")
            else:
                print(f"\n{Y}没找到标签为 {args.compare} 的历史跑分{X}")

        worst = [c for c in report.worst(5) if c.metrics.get("recall@5", 0) < 1]
        if worst:
            print(f"\n{B}最差的几条(下一步该看这里){X}\n")
            for c in worst:
                print(f"  recall@5={c.metrics.get('recall@5', 0):.2f}  {c.query[:60]}")
                print(f"      {DIM}期望 {c.expected} / 召回 {c.retrieved[:3]}{X}")

        print(f"\n{DIM}run_id={report.run_id}{X}\n")
        return 0

    return 1


async def cmd_feedback(args) -> int:
    from railg.db import get_db

    db = get_db()
    await db.init()
    summary = await db.feedback_summary()
    print(f"\n👍 {summary['up']}   👎 {summary['down']}\n")

    items = await db.list_feedback(rating=args.rating, limit=args.limit)
    for f in items:
        mark = f"{G}👍{X}" if f["rating"] == "up" else f"{R}👎{X}"
        print(f"  {mark} {DIM}{_ts(f['created_at'])}{X}  {f['query'][:60]}")
        if f["comment"]:
            print(f"      {Y}{f['comment'][:120]}{X}")
    print()
    return 0


async def cmd_stats(args) -> int:
    from railg.db import get_db
    from railg.store import get_store

    db = get_db()
    await db.init()
    index_info = await get_store().stats()
    doc_info = await db.document_stats()
    fb = await db.feedback_summary()
    metrics = await db.request_metrics(24)

    print(f"\n{B}索引{X}")
    if index_info["exists"]:
        print(f"  {index_info['n_docs']} 篇文档 / {index_info['n_chunks']} 个块")
        for k, v in sorted(index_info["by_type"].items(), key=lambda x: -x[1]):
            print(f"    {k:<10} {v}")
    else:
        print("  索引还不存在,先跑 railg ingest")

    print(f"\n{B}文档登记{X}\n  {doc_info['n_docs']} 篇,失败 {doc_info['n_failed']} 篇")
    print(f"\n{B}反馈{X}\n  👍 {fb['up']}   👎 {fb['down']}")
    print(f"\n{B}近 24 小时{X}\n  {metrics['requests']} 次请求,"
          f"平均 {metrics['avg_latency_ms']}ms,错误率 {metrics['error_rate']:.1%}\n")
    return 0


async def cmd_reset(args) -> int:
    from railg.store import get_store

    if not args.yes:
        print("这会删除整个索引。确认请加 --yes")
        return 1
    await get_store().drop_index()
    print("索引已删除(SQLite 里的会话/反馈/评测集保留)")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "railg.api:app",
        host=args.host or settings.server.host,
        port=args.port or settings.server.port,
        reload=args.reload,
        log_level="debug" if args.verbose else "info",
    )
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="railg", description="RAilG —— 端到端 RAG Chatbot")
    parser.add_argument("-v", "--verbose", action="store_true", help="打开调试日志")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="体检:配置 / 连通性 / 契约")

    p = sub.add_parser("ingest", help="把文件或目录入库")
    p.add_argument("path")
    p.add_argument("--acl", action="append", help="权限主体,可重复,如 --acl group:finance")
    p.add_argument("--force", action="store_true", help="忽略内容哈希强制重建")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--ocr", choices=["paddle", "vlm"], help="启用 OCR 后端处理扫描件")

    p = sub.add_parser("ingest-url", help="抓取网页或在线文件入库")
    p.add_argument("urls", nargs="+")
    p.add_argument("--acl", action="append")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("ask", help="提一个问题")
    p.add_argument("query")
    p.add_argument("--no-save", action="store_true", help="不写入会话历史")

    p = sub.add_parser("search", help="只做检索,不生成")
    p.add_argument("query")

    p = sub.add_parser("docs", help="文档管理")
    docs_sub = p.add_subparsers(dest="docs_action", required=True)
    q = docs_sub.add_parser("list")
    q.add_argument("--keyword")
    q.add_argument("--limit", type=int, default=100)
    q = docs_sub.add_parser("delete")
    q.add_argument("doc_id")
    q = docs_sub.add_parser("reindex")
    q.add_argument("doc_id")

    p = sub.add_parser("eval", help="检索评测")
    eval_sub = p.add_subparsers(dest="eval_action", required=True)
    q = eval_sub.add_parser("add", help="新增一条评测 case")
    q.add_argument("query")
    q.add_argument("--doc", action="append", help="期望命中的文件名或 doc_id,可重复")
    q.add_argument("--note")
    eval_sub.add_parser("list", help="查看评测集")
    q = eval_sub.add_parser("import", help="从 JSONL 导入")
    q.add_argument("path")
    q = eval_sub.add_parser("export", help="导出为 JSONL")
    q.add_argument("path")
    q = eval_sub.add_parser("run", help="跑一次评测")
    q.add_argument("--label", help="给这次跑分打个标签,便于后续对比")
    q.add_argument("--compare", help="与某个标签的历史跑分对比")
    eval_sub.add_parser("runs", help="历史跑分")

    p = sub.add_parser("feedback", help="查看用户反馈")
    p.add_argument("--rating", choices=["up", "down"])
    p.add_argument("--limit", type=int, default=50)

    sub.add_parser("stats", help="总览")

    p = sub.add_parser("reset", help="删除索引")
    p.add_argument("--yes", action="store_true")

    p = sub.add_parser("serve", help="启动 Web 服务")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--reload", action="store_true")

    return parser


ASYNC_COMMANDS = {
    "check": cmd_check,
    "ingest": cmd_ingest,
    "ingest-url": cmd_ingest_url,
    "ask": cmd_ask,
    "search": cmd_search,
    "docs": cmd_docs,
    "eval": cmd_eval,
    "feedback": cmd_feedback,
    "stats": cmd_stats,
    "reset": cmd_reset,
}


def main() -> int:
    args = build_parser().parse_args()
    _setup_logging(args.verbose)

    if args.command == "serve":
        return cmd_serve(args)

    try:
        return asyncio.run(_run(ASYNC_COMMANDS[args.command], args))
    except KeyboardInterrupt:
        print()
        return 130


async def _run(handler, args) -> int:
    from railg.providers import close_providers
    from railg.store import close_store

    try:
        return await handler(args)
    finally:
        await close_store()
        await close_providers()


if __name__ == "__main__":
    sys.exit(main())
