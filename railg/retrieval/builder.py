"""Elastic/OpenSearch 查询构造器。

bool 骨架(should / must / must_not / filter)+ 链式 API。

kNN 作为 bool 的一个 should 子句下发,而不是顶层 `knn_query`:
这样 BM25 分与向量分由引擎直接相加,不用在应用层拼分数,混合召回更自然。

同时把过滤条件也注入 kNN 子句的 filter —— 否则 kNN 先取 top-k 再被外层
过滤,会白白损失召回(权限收紧时尤其明显)。
"""

from __future__ import annotations

from typing import Any


class QueryBuilder:
    def __init__(self) -> None:
        self.queries: list[dict[str, Any]] = []      # should
        self.must: list[dict[str, Any]] = []
        self.exclusions: list[dict[str, Any]] = []   # must_not
        self.filters: list[dict[str, Any]] = []
        self.knn_clauses: list[dict[str, Any]] = []
        self.script_query_index: int | None = None
        self.function_query_index: int | None = None

    # --- 链式添加 ------------------------------------------------------ #
    def add_query(self, query: dict[str, Any]) -> "QueryBuilder":
        self.queries.append(query)
        return self

    def add_must(self, query: dict[str, Any]) -> "QueryBuilder":
        self.must.append(query)
        return self

    def add_filter(self, query: dict[str, Any]) -> "QueryBuilder":
        self.filters.append(query)
        return self

    def add_exclusion(self, query: dict[str, Any]) -> "QueryBuilder":
        self.exclusions.append(query)
        return self

    def add_knn(self, clause: dict[str, Any]) -> "QueryBuilder":
        self.knn_clauses.append(clause)
        return self

    def add_function_query(self, fn: dict[str, Any]) -> "QueryBuilder":
        """function_score 必须挂在同一个父查询下,故记录其位置。"""
        if self.function_query_index is None:
            self.function_query_index = len(self.queries)
            self.queries.append({
                "function_score": {
                    "query": {"match_all": {}},
                    "functions": [],
                    "score_mode": "sum",
                    "boost_mode": "multiply",
                }
            })
        self.queries[self.function_query_index]["function_score"]["functions"].append(fn)
        return self

    def add_script_query(self, script: dict[str, Any]) -> "QueryBuilder":
        if self.script_query_index is None:
            self.script_query_index = len(self.queries)
            self.queries.append({
                "script_score": {"query": {"match_all": {}}, "script": {}}
            })
        self.queries[self.script_query_index]["script_score"]["script"].update(script)
        return self

    # --- 产出 ---------------------------------------------------------- #
    def _knn_filter(self) -> dict[str, Any] | None:
        """kNN 的前置过滤 —— 与外层保持一致,避免召回被外层过滤吃掉。"""
        clauses = self.filters + self.must
        if not clauses:
            return None
        body: dict[str, Any] = {"filter": clauses}
        if self.exclusions:
            body["must_not"] = self.exclusions
        return {"bool": body}

    def build_query(self, query_term: str = "") -> dict[str, Any]:
        """把查询词填进各 match 子句,再组装成完整 query。"""
        for query in self.queries:
            for kind in ("match", "match_phrase"):
                if kind in query:
                    field = next(iter(query[kind]))
                    query[kind][field]["query"] = query_term

        knn_filter = self._knn_filter()
        should = list(self.queries)
        for clause in self.knn_clauses:
            field = next(iter(clause))
            if knn_filter:
                clause[field]["filter"] = knn_filter
            should.append({"knn": clause})

        bool_body: dict[str, Any] = {}
        if should:
            bool_body["should"] = should
        if self.must:
            bool_body["must"] = self.must
        if self.exclusions:
            bool_body["must_not"] = self.exclusions
        if self.filters:
            bool_body["filter"] = self.filters
        # 只有过滤条件、没有任何打分子句时,退化为纯过滤
        if not should and not self.must:
            bool_body.setdefault("must", [{"match_all": {}}])

        return {"bool": bool_body}

    def build_body(self, query_term: str = "", size: int = 50) -> dict[str, Any]:
        return {
            "size": size,
            "query": self.build_query(query_term),
            # 向量体积大且前端用不到,一律排除
            "_source": {"excludes": ["semantic_vector"]},
        }

    def size(self) -> int:
        return len(self.queries) + len(self.must) + len(self.filters) + len(self.knn_clauses)
