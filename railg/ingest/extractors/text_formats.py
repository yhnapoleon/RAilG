"""纯文本类提取器:Markdown / TXT / HTML。"""

from __future__ import annotations

import re
from pathlib import Path

from railg.ingest.extractors.base import ExtractedDocument, Extractor


def _read(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


class MarkdownExtractor(Extractor):
    name = "markdown"
    extensions = (".md", ".markdown", ".mdx")

    def extract(self, path: Path) -> ExtractedDocument:
        return ExtractedDocument(file_markdown=_read(path).strip())


class TextExtractor(Extractor):
    name = "text"
    extensions = (".txt", ".log", ".rst", ".csv", ".json", ".yaml", ".yml")

    def extract(self, path: Path) -> ExtractedDocument:
        text = _read(path).strip()
        if path.suffix.lower() == ".csv":
            text = self._csv_to_markdown(text)
        return ExtractedDocument(file_markdown=text)

    @staticmethod
    def _csv_to_markdown(text: str, max_rows: int = 5000) -> str:
        import csv
        import io

        rows = list(csv.reader(io.StringIO(text)))[:max_rows]
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        head, *body = rows
        out = ["| " + " | ".join(c.replace("|", "\\|") for c in head) + " |",
               "| " + " | ".join(["---"] * width) + " |"]
        out += ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in body]
        return "\n".join(out)


class HtmlExtractor(Extractor):
    name = "html"
    extensions = (".html", ".htm", ".xhtml")

    def extract(self, path: Path) -> ExtractedDocument:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(_read(path), "lxml")
        for tag in soup(["script", "style", "nav", "footer", "noscript"]):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else ""
        body = soup.body or soup
        parts: list[str] = []
        if title:
            parts.append(f"# {title}")

        for el in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "table", "pre"]):
            if el.name == "table":
                parts.append(self._table_to_markdown(el))
            elif el.name.startswith("h"):
                level = min(int(el.name[1]), 3)
                text = el.get_text(" ", strip=True)
                if text:
                    parts.append(f"{'#' * level} {text}")
            elif el.name == "pre":
                text = el.get_text("\n", strip=True)
                if text:
                    parts.append(f"```\n{text}\n```")
            else:
                text = el.get_text(" ", strip=True)
                if text:
                    parts.append(f"- {text}" if el.name == "li" else text)

        md = "\n\n".join(p for p in parts if p.strip())
        md = re.sub(r"\n{3,}", "\n\n", md)
        return ExtractedDocument(file_markdown=md, meta={"title": title} if title else {})

    @staticmethod
    def _table_to_markdown(table) -> str:
        rows = []
        for tr in table.find_all("tr"):
            cells = [
                td.get_text(" ", strip=True).replace("|", "\\|")
                for td in tr.find_all(["td", "th"])
            ]
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        head, *body = rows
        out = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * width) + " |"]
        out += ["| " + " | ".join(r) + " |" for r in body]
        return "\n".join(out)
