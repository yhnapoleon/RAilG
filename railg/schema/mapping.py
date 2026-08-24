"""由 IndexDoc 推导 OpenSearch mapping,并声明检索层读取的字段。

`tests/test_contract.py` 会校验三件事:
    1. IndexDoc 的每个字段都在 mapping 里有定义(没有"写了但没映射")
    2. RETRIEVAL_READ_FIELDS 的每个字段都在 mapping 里有定义(没有"读了但没人写")
    3. 向量维度与 embedding.dims 一致

"检索层读了、但没有任何地方写入"的字段过不了第 2 条 —— 那种字段会让
过滤器静默失效,查询返回空结果却不报错。
"""

from __future__ import annotations

from typing import Any

from railg.schema.document import IndexDoc

# --------------------------------------------------------------------------- #
# 检索层实际读取/过滤/排序用到的字段。新增读取必须登记在这里。
# --------------------------------------------------------------------------- #
RETRIEVAL_READ_FIELDS: tuple[str, ...] = (
    "text_to_index",      # BM25 主字段
    "file_name",          # 文件名匹配
    "semantic_vector",    # kNN
    "snippet",            # 展示
    "acl_principals",     # ★ 权限过滤
    "document_type",      # 类型过滤
    "last_modified_date",  # 时间过滤/排序
    "doc_id",             # 父块归并
    "section_id",         # 父块归并
    "context_id",         # 父块归并
    "chunk_index",        # 父块排序
    "chunk_uid",          # 去重
    "page_no",            # 引用定位
    "header1",
    "header2",
    "format",
    "file_url",
    "file_path",
    "content_hash",       # delta 增量
    "metadata",
)

# 不参与检索、仅回写的字段(允许存在于 mapping 而不在上面列表里)
_WRITE_ONLY_FIELDS = frozenset({"indexed_at"})


def _text_field(analyzer: bool = True) -> dict[str, Any]:
    f: dict[str, Any] = {
        "type": "text",
        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
    }
    if analyzer:
        f["analyzer"] = "indexing_analyzer"
        f["search_analyzer"] = "query_analyzer"
    return f


def build_properties(dims: int) -> dict[str, Any]:
    return {
        "doc_id": {"type": "keyword"},
        "chunk_uid": {"type": "keyword"},
        "chunk_index": {"type": "integer"},

        "file_name": _text_field(),
        "file_path": {"type": "keyword"},
        "file_url": {"type": "keyword"},
        "document_type": {"type": "keyword"},
        "last_modified_date": {"type": "date"},
        "content_hash": {"type": "keyword"},

        "text_to_index": _text_field(),
        # snippet 是展示字段,不参与打分,但必须显式映射
        # (靠动态映射会让行为随第一条写入的数据而变,不可预期)
        "snippet": {"type": "text", "index": False},
        "semantic_vector": {
            "type": "knn_vector",
            "dimension": dims,
            "method": {
                "name": "hnsw",
                "space_type": "cosinesimil",
                "engine": "lucene",
                "parameters": {"ef_construction": 128, "m": 16},
            },
        },

        "page_no": {"type": "integer"},
        "section_id": {"type": "integer"},
        "context_id": {"type": "integer"},
        "header1": _text_field(),
        "header2": _text_field(),
        "format": {"type": "keyword"},

        "acl_principals": {"type": "keyword"},
        "metadata": {"type": "object", "dynamic": True},
        "indexed_at": {"type": "date"},
    }


def build_settings(synonyms: list[str] | None = None) -> dict[str, Any]:
    """分析器。零插件方案:standard 分词 + cjk_bigram,中英文都能用。

    需要更好的中文效果时装 IK 插件,把 indexing_analyzer 的 tokenizer
    换成 ik_max_word 即可,其余不动。
    """
    query_filters = ["cjk_width", "lowercase", "cjk_bigram_filter", "stopwords_filter", "kstem"]
    filters: dict[str, Any] = {
        "cjk_bigram_filter": {
            "type": "cjk_bigram",
            # 保留单字,牺牲一点精度换召回 —— 个人语料通常偏小
            "output_unigrams": True,
        },
        "stopwords_filter": {
            "type": "stop",
            "stopwords": "_english_",
            "ignore_case": True,
        },
    }
    if synonyms:
        filters["synonym_filter"] = {"type": "synonym_graph", "synonyms": synonyms}
        query_filters.append("synonym_filter")

    return {
        "index": {
            "knn": True,
            "number_of_shards": 1,
            "number_of_replicas": 0,  # 单节点,副本会让集群变 yellow
        },
        "analysis": {
            "filter": filters,
            "analyzer": {
                "indexing_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["cjk_width", "lowercase", "cjk_bigram_filter",
                               "stopwords_filter", "kstem"],
                },
                "query_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": query_filters,
                },
            },
        },
    }


def build_index_body(dims: int, synonyms: list[str] | None = None) -> dict[str, Any]:
    return {
        "settings": build_settings(synonyms),
        "mappings": {"properties": build_properties(dims)},
    }


# --------------------------------------------------------------------------- #
# 契约校验 —— 也可在运行时调用(indexer 启动时会跑一次)
# --------------------------------------------------------------------------- #
def verify_contract(dims: int) -> None:
    props = build_properties(dims)

    missing_read = [f for f in RETRIEVAL_READ_FIELDS if f not in props]
    if missing_read:
        raise RuntimeError(
            f"检索层声明读取但 mapping 未定义的字段: {missing_read}。"
            "这类不一致会让过滤器静默失效:查询恒返回空结果,却不报错。"
        )

    model_fields = set(IndexDoc.model_fields) - _WRITE_ONLY_FIELDS
    missing_write = [f for f in model_fields if f not in props]
    if missing_write:
        raise RuntimeError(f"IndexDoc 定义但 mapping 未映射的字段: {missing_write}")

    vec = props["semantic_vector"]
    if vec["dimension"] != dims:
        raise RuntimeError(f"向量维度不一致: mapping={vec['dimension']} config={dims}")
