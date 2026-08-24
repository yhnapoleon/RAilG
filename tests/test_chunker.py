"""★ 切块 ↔ 父块还原 round-trip 测试。

它守的是整个项目最脆弱的一条隐式契约:

    chunker 切出来的块,必须能被 parents.py 无损拼回。

这条约束光写在注释里是守不住的:切块参数一改,父块还原就会静默出错 ——
拼出来的文本多一段或少一段,没有任何报错。这里把它变成 CI 能拦下来的断言。
"""

from __future__ import annotations

import re

import pytest

from railg.ingest.chunker import Chunker, ChunkerConfig, split_markdown_to_sections
from railg.retrieval.parents import rejoin_table, rejoin_text, strip_propagated_header

SAMPLE = """# 员工手册

本手册适用于全体员工。

## 请假制度

员工每年享有 14 天带薪年假。年假需提前三个工作日申请，由直属主管审批。
病假需提供医疗机构证明。连续病假超过三天的，须提交三甲医院诊断书。
未休完的年假最多可结转 5 天至次年，逾期作废。

## 报销标准

差旅报销按下表执行。

| 职级 | 城市等级 | 住宿标准 | 餐饮标准 |
| --- | --- | --- | --- |
| P5 | 一线 | 500 | 150 |
| P5 | 二线 | 350 | 120 |
| P6 | 一线 | 700 | 200 |
| P6 | 二线 | 500 | 150 |
| P7 | 一线 | 900 | 250 |
| P7 | 二线 | 700 | 200 |

报销单需在费用发生后 30 天内提交，逾期不予受理。

## 保密义务

员工在职期间及离职后两年内，不得向第三方披露公司商业秘密。
"""


def _norm(text: str) -> str:
    """比较时忽略空白差异 —— 切分器会在分隔符处调整空白。"""
    return re.sub(r"\s+", "", text)


@pytest.fixture
def chunker() -> Chunker:
    return Chunker(ChunkerConfig(chunk_size=60, context_size=12))


# --------------------------------------------------------------------------- #
# 基本行为
# --------------------------------------------------------------------------- #
def test_sections_are_split_by_headers():
    sections = split_markdown_to_sections(SAMPLE)
    headers = [s.header2 for s in sections if s.header2]
    assert "请假制度" in headers
    assert "报销标准" in headers
    assert "保密义务" in headers


def test_header1_is_populated():
    """一级标题必须真的写进块元信息,否则父块还原后定位不到出处。"""
    sections = split_markdown_to_sections(SAMPLE)
    assert any(s.header1 == "员工手册" for s in sections)


def test_chunks_carry_full_provenance(chunker):
    chunks = chunker.chunk(file_markdown=SAMPLE, source="handbook.md")
    assert chunks
    for c in chunks:
        assert c.metadata.source == "handbook.md"
        assert c.metadata.section_id >= 0
        assert c.metadata.context_id >= 0
    # chunk_index 必须严格递增,父块还原靠它排序
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_table_becomes_its_own_context(chunker):
    chunks = chunker.chunk(file_markdown=SAMPLE)
    table_chunks = [c for c in chunks if c.metadata.format == "table"]
    assert table_chunks, "表格应当被识别为独立的上下文块"
    # 一张表只属于一个 context
    assert len({c.metadata.context_id for c in table_chunks}) == 1


def test_table_header_is_propagated(chunker):
    """表被切碎后,每个子块都应带上表头,否则单行数据检索不出来。"""
    chunks = chunker.chunk(file_markdown=SAMPLE)
    table_chunks = [c for c in chunks if c.metadata.format == "table"]
    if len(table_chunks) > 1:
        assert any("<th>" in c.page_content for c in table_chunks)


# --------------------------------------------------------------------------- #
# ★ round-trip:核心断言
# --------------------------------------------------------------------------- #
def test_text_chunks_rejoin_without_loss_or_duplication(chunker):
    """同一 context 内的文本块拼回后,字符集必须与切分前完全一致。

    多一个字 = 有重叠(父块会出现重复段落);少一个字 = 丢内容。
    """
    chunks = chunker.chunk(file_markdown=SAMPLE)

    groups: dict[tuple[int, int], list] = {}
    for c in chunks:
        groups.setdefault((c.metadata.section_id, c.metadata.context_id), []).append(c)

    for key, group in groups.items():
        group.sort(key=lambda c: c.chunk_index)
        if group[0].metadata.format == "table":
            continue
        rejoined = rejoin_text([c.page_content for c in group])
        concat = "".join(c.page_content for c in group)
        assert _norm(rejoined) == _norm(concat), f"上下文 {key} 拼接后内容变了"


def test_no_content_is_duplicated_across_chunks(chunker):
    """零重叠的直接体现:所有块拼起来的长度 == 各块长度之和。"""
    chunks = chunker.chunk(file_markdown=SAMPLE)
    text_chunks = [c for c in chunks if c.metadata.format == "text"]

    total = sum(len(_norm(c.page_content)) for c in text_chunks)
    joined = len(_norm("".join(c.page_content for c in text_chunks)))
    assert total == joined


def test_source_content_survives_chunking(chunker):
    """正文里的关键句子必须能在某个块里找到 —— 防止切块吃掉内容。"""
    chunks = chunker.chunk(file_markdown=SAMPLE)
    haystack = _norm(" ".join(c.page_content for c in chunks))
    for needle in ("员工每年享有14天带薪年假", "逾期不予受理", "不得向第三方披露公司商业秘密"):
        assert _norm(needle) in haystack, f"内容丢失: {needle}"


def test_table_rejoin_keeps_single_header():
    """表格子块拼回时,传播的表头只能留一份。"""
    pieces = [
        "<th>| 职级 | 城市等级 |</th>\n| P5 | 一线 |",
        "<th>| 职级 | 城市等级 |</th>\n| P6 | 一线 |",
        "<th>| 职级 | 城市等级 |</th>\n| P7 | 一线 |",
    ]
    rejoined = rejoin_table(pieces)
    assert rejoined.count("职级") == 1, "表头重复了"
    for level in ("P5", "P6", "P7"):
        assert level in rejoined
    assert "<th>" not in rejoined


def test_strip_propagated_header():
    header, body = strip_propagated_header("<th>| a | b |</th>\n| 1 | 2 |")
    assert header == "| a | b |"
    assert body.strip() == "| 1 | 2 |"

    header, body = strip_propagated_header("普通文本")
    assert header == ""
    assert body == "普通文本"


# --------------------------------------------------------------------------- #
# 约束强制
# --------------------------------------------------------------------------- #
def test_nonzero_overlap_is_rejected():
    """★ 零重叠是父块还原的前提,配置层就该拦下。"""
    with pytest.raises(ValueError, match="overlap"):
        ChunkerConfig(chunk_overlap=20)
    with pytest.raises(ValueError, match="overlap"):
        ChunkerConfig(context_overlap=5)


def test_config_validator_rejects_overlap():
    from railg.config import ChunkConfig

    with pytest.raises(ValueError):
        ChunkConfig(chunk_overlap=10)


# --------------------------------------------------------------------------- #
# 页码
# --------------------------------------------------------------------------- #
def test_page_numbers_are_monotonic(chunker):
    pages = [
        "# 第一页\n\n这是第一页的内容，讲的是入职流程和材料准备。" * 3,
        "## 第二页\n\n这是第二页的内容，讲的是转正考核与评分细则。" * 3,
        "## 第三页\n\n这是第三页的内容，讲的是离职交接与结算安排。" * 3,
    ]
    chunks = chunker.chunk(page_markdowns=pages)
    numbers = [c.metadata.page_no for c in chunks]
    assert numbers == sorted(numbers), "页码不能倒退"
    assert min(numbers) >= 1
    assert max(numbers) <= len(pages)


def test_empty_input_returns_empty(chunker):
    assert chunker.chunk() == []
    assert chunker.chunk(file_markdown="   ") == []
