"""评测执行器:跑 golden set,出指标,存历史,做回归对比。

用法(CLI):
    railg eval add "年假有几天" --doc 员工手册.md      建 case
    railg eval import cases.jsonl                     批量导入
    railg eval run --label baseline                   跑一次并存为基线
    railg eval run --label 调大topk --compare baseline 跑一次并和基线比

匹配口径:期望值可以写 doc_id,也可以写文件名(更好写)。内部会借助文档
登记表把两边统一归一到 doc_id 再算指标 —— ranked list 里一个位置必须
对应一个候选,否则 recall@k / nDCG@k 的 k 就失去意义。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from railg.config import Settings, get_settings
from railg.db import Database, get_db
from railg.evaluation.metrics import aggregate, compare, evaluate_one
from railg.retrieval.service import RetrievalService, get_retrieval_service
from railg.schema.document import ANONYMOUS, Identity

logger = logging.getLogger(__name__)

DEFAULT_KS = (1, 3, 5, 10)


@dataclass
class CaseResult:
    case_id: str
    query: str
    expected: list[str]
    retrieved: list[str]
    metrics: dict[str, float]
    n_raw: int = 0
    timings_ms: dict[str, int] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "expected": self.expected,
            "retrieved": self.retrieved[:20],
            "metrics": self.metrics,
            "n_raw": self.n_raw,
            "timings_ms": self.timings_ms,
            "error": self.error,
        }


@dataclass
class EvalReport:
    label: str
    metrics: dict[str, float]
    cases: list[CaseResult]
    config: dict[str, Any]
    run_id: str = ""

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.cases if c.error)

    def worst(self, n: int = 5, key: str = "recall@5") -> list[CaseResult]:
        """最差的几条 —— 这才是下一步该看的东西。"""
        return sorted(self.cases, key=lambda c: c.metrics.get(key, 0.0))[:n]


class Evaluator:
    def __init__(
        self,
        retrieval: RetrievalService | None = None,
        db: Database | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retrieval = retrieval or get_retrieval_service()
        self.db = db or get_db()
        self._aliases: dict[str, str] | None = None

    # ------------------------------------------------------------------ #
    def _config_snapshot(self) -> dict[str, Any]:
        """把影响结果的参数一起存下来,否则历史记录没法解释。"""
        r = self.settings.retrieval
        return {
            "top_k": r.top_k,
            "rerank_top_n": r.rerank_top_n,
            "max_context_docs": r.max_context_docs,
            "bm25_weight": r.bm25_weight,
            "vector_weight": r.vector_weight,
            "return_parent": r.return_parent,
            "parent_window": r.parent_window,
            "rerank_enabled": self.settings.rerank.enabled,
            "embedding_model": self.settings.embedding.model,
            "rerank_model": self.settings.rerank.model if self.settings.rerank.enabled else None,
            "chunk_size": self.settings.chunk.chunk_size,
            "context_size": self.settings.chunk.context_size,
        }

    async def _alias_map(self) -> dict[str, str]:
        """file_name / file_path → doc_id。

        期望值让人手写 doc_id 太反人类,所以允许写文件名;但**指标必须在
        同一个标识空间里算**。这里把两边都归一到 doc_id ——
        早先的写法把 doc_id 和 file_name 一起塞进 ranked list,
        位置 k 就不再等于"第 k 个候选",recall@1 会被系统性压低。
        """
        if self._aliases is None:
            aliases: dict[str, str] = {}
            for doc in await self.db.list_documents(limit=10000):
                doc_id = doc["doc_id"]
                aliases[doc_id] = doc_id
                for key in (doc.get("file_name"), doc.get("file_path")):
                    if key:
                        aliases.setdefault(key, doc_id)
            self._aliases = aliases
        return self._aliases

    @staticmethod
    def _normalise(keys: list[str], aliases: dict[str, str]) -> list[str]:
        """归一到 doc_id 并按序去重。映射不到的保持原样(自然就匹配不上)。"""
        seen: set[str] = set()
        out: list[str] = []
        for key in keys:
            if not key:
                continue
            canonical = aliases.get(key, key)
            if canonical not in seen:
                seen.add(canonical)
                out.append(canonical)
        return out

    async def run_case(
        self,
        case: dict,
        identity: Identity = ANONYMOUS,
        ks: tuple[int, ...] = DEFAULT_KS,
    ) -> CaseResult:
        query = case["query"]
        expected_raw = case.get("expected_docs", [])
        aliases = await self._alias_map()
        expected = self._normalise(expected_raw, aliases)

        try:
            # 评测不做查询改写:要测的是检索本身,改写会引入 LLM 抖动
            result = await self.retrieval.retrieve(query, identity=identity, rewrite=False)
        except Exception as exc:
            logger.exception("评测 case 失败: %s", query)
            return CaseResult(
                case_id=case.get("id", ""), query=query, expected=expected_raw,
                retrieved=[], metrics=evaluate_one([], expected, ks),
                error=f"{type(exc).__name__}: {exc}",
            )

        # 一个候选 = ranked list 里的一个位置,位置 k 才有意义
        retrieved = self._normalise(
            [c.doc_id or c.file_name for c in result.candidates], aliases
        )

        return CaseResult(
            case_id=case.get("id", ""),
            query=query,
            expected=expected_raw,
            retrieved=retrieved,
            metrics=evaluate_one(retrieved, expected, ks),
            n_raw=result.trace.n_raw,
            timings_ms=result.trace.timings_ms,
        )

    async def run(
        self,
        label: str = "",
        identity: Identity = ANONYMOUS,
        ks: tuple[int, ...] = DEFAULT_KS,
        save: bool = True,
        on_case: Any = None,
    ) -> EvalReport:
        cases = await self.db.list_eval_cases()
        if not cases:
            raise RuntimeError(
                "评测集是空的。先加 case:railg eval add \"问题\" --doc 文件名.pdf"
            )

        results: list[CaseResult] = []
        for i, case in enumerate(cases, 1):
            result = await self.run_case(case, identity=identity, ks=ks)
            results.append(result)
            if on_case:
                on_case(i, len(cases), result)

        metrics = aggregate([r.metrics for r in results])
        config = self._config_snapshot()
        report = EvalReport(label=label, metrics=metrics, cases=results, config=config)

        if save:
            report.run_id = await self.db.save_eval_run(
                label=label, config=config, metrics=metrics,
                details=[r.to_dict() for r in results], n_cases=len(results),
            )
        return report

    # ------------------------------------------------------------------ #
    async def compare_with(self, report: EvalReport, baseline_label: str) -> dict | None:
        """和历史上某个 label 的最近一次跑分对比。"""
        runs = await self.db.list_eval_runs(limit=100)
        match = next(
            (r for r in runs if r["label"] == baseline_label and r["id"] != report.run_id),
            None,
        )
        if not match:
            return None
        return {
            "baseline_label": baseline_label,
            "baseline_run": match["id"],
            "diff": compare(match["metrics"], report.metrics),
        }


# --------------------------------------------------------------------------- #
# 评测集导入导出
# --------------------------------------------------------------------------- #
async def import_cases(path: Path | str, db: Database | None = None) -> int:
    """从 JSONL 导入。每行:{"query": "...", "expected_docs": ["a.pdf"], "note": ""}"""
    db = db or get_db()
    path = Path(path)
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("跳过无法解析的行: %s", line[:80])
            continue
        query = row.get("query", "").strip()
        if not query:
            continue
        await db.add_eval_case(
            query=query,
            expected_docs=row.get("expected_docs") or row.get("expected") or [],
            note=row.get("note", ""),
            tags=row.get("tags", []),
        )
        n += 1
    return n


async def export_cases(path: Path | str, db: Database | None = None) -> int:
    db = db or get_db()
    cases = await db.list_eval_cases()
    lines = [
        json.dumps(
            {
                "query": c["query"],
                "expected_docs": c["expected_docs"],
                "note": c["note"],
                "tags": c["tags"],
            },
            ensure_ascii=False,
        )
        for c in cases
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)
