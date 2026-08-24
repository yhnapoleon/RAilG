"""三级切块 —— 本项目最核心的算法。

    第 1 级  section   按 markdown 标题切,携带 header1 / header2
    第 2 级  context   章节内按结构切:一张表 = 一个上下文块,连续文本 = 一个上下文块
    第 3 级  chunk     上下文块内按 token 切,表格块用双倍尺寸

★★★ 为什么 chunk_overlap 必须是 0 ★★★

    检索时 `retrieval/parents.py` 会把同一 (doc_id, section_id, context_id) 下的
    兄弟块按 chunk_index 顺序拼回父块,直接送进 LLM。若切块时有重叠,拼接结果
    就会出现重复文本 —— 而且是**静默**出现,没有任何报错。

    这类约束光靠注释是守不住的,所以它被固化成两道闸:
        · config.ChunkConfig 的字段校验(启动即拦截)
        · tests/test_chunker.py 的 round-trip 测试(CI 守卫)

章节切分认 `^#{1,3} ` 而不是只认 `##`:本项目要吃 DOCX / HTML / MD,
标题层级是任意的,只认某一级会导致大量文档切不出章节。
OCR 路径不受影响 —— layout.py 仍然输出 `##`。
"""

from __future__ import annotations

import bisect
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rapidfuzz.fuzz import partial_ratio

from railg.ingest.page import Page, compress_table_markdown
from railg.providers.tokens import get_size_function, heuristic_tokens
from railg.schema.document import Chunk, ChunkFormat, ChunkMeta

logger = logging.getLogger(__name__)

MIN_PAGE_NO = 1
MAX_CONTEXT_LENGTH = 512
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", ". ", ",", ", ", ";", " "]
TABLE_SEPARATORS = ["\n\n", "\n", "。", ". ", ",", ", ", ";"]
#  短于该 token 数的块会尝试与邻居合并
MERGE_THRESHOLD_TOKENS = 20

_HEADER_RE = re.compile(r"(?m)^(#{1,3})[ \t]+(.+?)[ \t]*$")


# --------------------------------------------------------------------------- #
# 中间结构
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Section:
    text: str
    header1: str = ""
    header2: str = ""


@dataclass(slots=True)
class ContextBlock:
    text: str
    fmt: ChunkFormat = "text"
    header1: str = ""
    header2: str = ""
    section_id: int = 0
    context_id: int = 0


@dataclass(slots=True)
class ChunkerConfig:
    chunk_size: int = 100
    context_size: int = 20
    chunk_overlap: int = 0
    context_overlap: int = 0
    chunk_size_function: str = "n_tokens"
    context_size_function: str = "n_newlines"
    enable_page_numbers: bool = True
    table_header_propagation: bool = True
    ignore_sections: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.chunk_overlap or self.context_overlap:
            raise ValueError(
                "chunk_overlap / context_overlap 必须为 0 —— 父块还原依赖零重叠。"
                "详见 railg/ingest/chunker.py 顶部说明。"
            )


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def make_splitter(
    chunk_size: int,
    chunk_overlap: int = 0,
    size_function: str | Callable[[str], int] = "n_tokens",
    separators: list[str] | None = None,
) -> RecursiveCharacterTextSplitter:
    length_function = (
        size_function if callable(size_function) else get_size_function(size_function)
    )
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        is_separator_regex=False,
        separators=separators or DEFAULT_SEPARATORS,
        keep_separator=True,
    )


def process_markdowns(
    file_markdown: str, page_markdowns: list[str], page_sep: str = "\n\n"
) -> tuple[str, list[str]]:
    """统一压缩表格,让整篇与逐页的字符计数口径一致(页码估算依赖这一点)。"""
    if not page_markdowns:
        page_markdowns = [file_markdown]
    page_markdowns = [compress_table_markdown(p) for p in page_markdowns]

    if not file_markdown:
        file_markdown = page_sep.join(page_markdowns)
    else:
        file_markdown = compress_table_markdown(file_markdown)
    return file_markdown, page_markdowns


# --------------------------------------------------------------------------- #
# 第 1 级:切章节
# --------------------------------------------------------------------------- #
def split_markdown_to_sections(
    markdown: str, ignore_sections: list[str] | None = None
) -> list[Section]:
    """按 markdown 标题切章节,标题文本保留在正文里(检索时是有用信号)。"""
    matches = list(_HEADER_RE.finditer(markdown))
    if not matches:
        text = markdown.strip()
        return [Section(text=text)] if text else []

    sections: list[Section] = []
    header1 = header2 = ""

    # 首个标题之前的引言部分
    preamble = markdown[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(text=preamble))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[m.start() : end].strip()

        if level == 1:
            header1, header2 = title, ""
        else:
            header2 = title

        if body:
            sections.append(Section(text=body, header1=header1, header2=header2))

    if ignore_sections:
        ignored = {s.lower() for s in ignore_sections}
        sections = [
            s for s in sections
            if s.header2.lower() not in ignored and s.header1.lower() not in ignored
        ]
    return sections


# --------------------------------------------------------------------------- #
# 第 2 级:切上下文块
# --------------------------------------------------------------------------- #
def split_section_to_contexts(
    section: Section, context_splitter: RecursiveCharacterTextSplitter
) -> list[ContextBlock]:
    """章节 → 上下文块。一张表独占一个块,连续文本合并后再按尺寸切。"""
    page = Page.from_markdown(section.text, compress_table=True)
    contexts: list[ContextBlock] = []
    buffer: list[str] = []

    def flush_text() -> None:
        if not buffer:
            return
        merged = "\n".join(buffer)
        buffer.clear()
        for piece in context_splitter.split_text(merged):
            if piece.strip():
                contexts.append(
                    ContextBlock(text=piece, fmt="text",
                                 header1=section.header1, header2=section.header2)
                )

    for element in page.children:
        if element.istable():
            flush_text()
            contexts.append(
                ContextBlock(text=element.markdown, fmt="table",
                             header1=section.header1, header2=section.header2)
            )
        elif element.isheading():
            # 标题不单独成块,避免出现「标题块 + 表格块」这种割裂;
            # 它会在下面被拼到本章节第一个文本块的开头。
            continue
        else:
            buffer.append(element.markdown)
    flush_text()

    # 章节标题拼到首个文本块开头,让该块自带上下文
    if contexts and contexts[0].fmt == "text":
        header_str = _header_prefix(section)
        if header_str:
            contexts[0].text = header_str + contexts[0].text
    elif not contexts:
        # 只有标题、没有正文的章节
        text = section.text.strip()
        if text:
            contexts = [ContextBlock(text=text, fmt="heading",
                                     header1=section.header1, header2=section.header2)]
    return contexts


def _header_prefix(section: Section) -> str:
    out = ""
    if section.header1:
        out += f"# {section.header1}\n"
    if section.header2:
        out += f"## {section.header2}\n"
    return out


# --------------------------------------------------------------------------- #
# 第 3 级:切子块
# --------------------------------------------------------------------------- #
def split_table_chunks(
    context: ContextBlock,
    table_splitter: RecursiveCharacterTextSplitter,
    header_propagation: bool = True,
) -> list[str]:
    """表格块切分,并把表头行传播到每个子块。

    表格被切碎后,单独一行数据是检索不出来的(没有列名)。这里用 `<th>` 标记
    把表头拼回每个子块开头 —— 它对表格问答的召回影响很大。
    """
    table_md = context.text
    header = ""
    splits = table_md.split("\n", maxsplit=2)
    if len(splits) == 3:
        head, sep, rest = splits
        if rest and _is_separator_row(sep) and _is_header_row(head):
            table_md = rest
            header = f"<th>{head.strip()}</th>\n"

    pieces = table_splitter.split_text(table_md)
    if not (header and header_propagation):
        return [p for p in pieces if p.strip()]

    out: list[str] = []
    for piece in pieces:
        if not piece.strip():
            continue
        if piece.startswith("|"):
            candidate = header + piece
            if heuristic_tokens(candidate) < MAX_CONTEXT_LENGTH - 50:
                piece = candidate
        out.append(piece)
    return out


def _is_header_row(row: str, max_tokens: int = 100) -> bool:
    row = row.strip()
    return row.startswith("|") and row.endswith("|") and heuristic_tokens(row) < max_tokens


def _is_separator_row(row: str) -> bool:
    return not (set(row) - set("-| :"))


# --------------------------------------------------------------------------- #
# 短块合并
# --------------------------------------------------------------------------- #
def merge_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """把过短的块合并到邻居,避免碎片污染召回。

    合并只在同 section 且同 context 内进行 —— 跨上下文合并会破坏父块还原。
    """

    def is_table(c: Chunk) -> bool:
        return c.metadata.format == "table"

    def too_short(c: Chunk) -> bool:
        return not is_table(c) and heuristic_tokens(c.page_content) <= MERGE_THRESHOLD_TOKENS

    def can_merge(a: Chunk, b: Chunk) -> bool:
        if is_table(a) or is_table(b):
            return False
        return (
            a.metadata.section_id == b.metadata.section_id
            and a.metadata.context_id == b.metadata.context_id
        )

    def join(left: Chunk, right: Chunk) -> Chunk:
        sep = "" if right.page_content[:1] and not right.page_content[0].isalnum() else " "
        meta = left.metadata.model_copy()
        meta.format = "text"
        return Chunk(page_content=left.page_content + sep + right.page_content, metadata=meta)

    out: list[Chunk] = []
    pending: Chunk | None = None

    for chunk in chunks:
        if pending is not None:
            if can_merge(pending, chunk):
                out.append(join(pending, chunk))
                pending = None
                continue
            if out and can_merge(out[-1], pending):
                out[-1] = join(out[-1], pending)
                pending = None
            else:
                out.append(pending)
                pending = None

        if too_short(chunk):
            pending = chunk
        else:
            out.append(chunk)

    if pending is not None:
        out.append(pending)
    return out


# --------------------------------------------------------------------------- #
# 页码定位
# --------------------------------------------------------------------------- #
def get_page_locs(page_markdowns: list[str]) -> list[int]:
    """各页在"去换行后字符流"中的累计结束位置。"""
    locs = [0]
    for md in page_markdowns:
        md = re.sub(r"\n\|", "||", md)
        md = re.sub(r"\n", "", md)
        locs.append(locs[-1] + len(md))
    return locs[1:]


def add_page_no_to_chunks(
    chunks: list[Chunk],
    page_markdowns: list[str],
    starts_from: int = MIN_PAGE_NO,
    min_page_len: int = 10,
) -> list[Chunk]:
    """三步定位页码:字符位置估算 → 邻页模糊匹配 → 单调不回退。

    这段代码看着朴素,但它是"引用能精确到页"的全部依据,
    而页码错了用户是会当场发现的。
    """
    n = len(page_markdowns)
    if not chunks or not n:
        return chunks

    page_locs = get_page_locs(page_markdowns)
    cur = 0
    n_chars = 0

    for i, chunk in enumerate(chunks):
        if i > 0:
            # 1. 按累计字符数估算
            loc_cur = min(bisect.bisect_right(page_locs, n_chars), n - 1)
            # 2. 在估算值 ±1 和上一页之间做候选
            candidates = sorted({c for c in (cur, loc_cur - 1, loc_cur, loc_cur + 1) if 0 <= c < n})
            # 3. 模糊匹配挑最像的一页,跳过过短的页
            best_page, best_score = loc_cur, -1.0
            for c in candidates:
                if len(page_markdowns[c]) < min_page_len:
                    continue
                score = partial_ratio(chunk.page_content, page_markdowns[c])
                if score > best_score:
                    best_score, best_page = score, c
            # 4. 页码只能前进
            cur = max(cur, best_page)

        chunk.metadata.page_no = cur + starts_from
        n_chars += len(chunk.page_content)

    return chunks


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
class Chunker:
    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk(
        self,
        file_markdown: str = "",
        page_markdowns: list[str] | None = None,
        source: str = "",
    ) -> list[Chunk]:
        cfg = self.config
        page_markdowns = page_markdowns or []
        if not file_markdown and not page_markdowns:
            return []

        # 0. 预处理
        file_markdown, page_markdowns = process_markdowns(file_markdown, page_markdowns)

        # 1. 切章节
        sections = split_markdown_to_sections(file_markdown, cfg.ignore_sections)
        if not sections:
            return []

        # 2. 切上下文块
        context_splitter = make_splitter(
            chunk_size=cfg.context_size,
            chunk_overlap=cfg.context_overlap,
            size_function=cfg.context_size_function,
        )
        contexts: list[ContextBlock] = []
        for section_id, section in enumerate(sections):
            for block in split_section_to_contexts(section, context_splitter):
                block.section_id = section_id
                contexts.append(block)

        # 3. 切子块
        text_splitter = make_splitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            size_function=cfg.chunk_size_function,
        )
        table_splitter = make_splitter(
            chunk_size=cfg.chunk_size * 2,
            chunk_overlap=cfg.chunk_overlap,
            size_function=cfg.chunk_size_function,
            separators=TABLE_SEPARATORS,
        )

        chunks: list[Chunk] = []
        for context_id, context in enumerate(contexts):
            context.context_id = context_id
            if context.fmt == "table":
                pieces = split_table_chunks(
                    context, table_splitter, cfg.table_header_propagation
                )
            else:
                pieces = [p for p in text_splitter.split_text(context.text) if p.strip()]

            for piece in pieces:
                chunks.append(
                    Chunk(
                        page_content=piece,
                        metadata=ChunkMeta(
                            source=source,
                            header1=context.header1,
                            header2=context.header2,
                            section_id=context.section_id,
                            context_id=context_id,
                            format=context.fmt,
                        ),
                    )
                )

        # 3.1 合并短块
        chunks = merge_chunks(chunks)

        # 4. 定位页码
        if cfg.enable_page_numbers and len(page_markdowns) > 1:
            chunks = add_page_no_to_chunks(chunks, page_markdowns)

        # 5. 编号 —— chunk_index 是父块还原的排序依据
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        logger.debug("切块完成: %d 章节 → %d 上下文 → %d 块",
                     len(sections), len(contexts), len(chunks))
        return chunks


def build_chunker(chunk_config) -> Chunker:
    """从 config.ChunkConfig 构造。"""
    return Chunker(
        ChunkerConfig(
            chunk_size=chunk_config.chunk_size,
            context_size=chunk_config.context_size,
            chunk_overlap=chunk_config.chunk_overlap,
            context_overlap=chunk_config.context_overlap,
            enable_page_numbers=chunk_config.enable_page_numbers,
            table_header_propagation=chunk_config.table_header_propagation,
            ignore_sections=list(chunk_config.ignore_sections),
        )
    )
