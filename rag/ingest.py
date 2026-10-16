"""PDF -> section chunks.

Generic logic: one chunk per numbered section ("N. Title" on its own line,
numbers strictly monotonic from 1). Tables found by PyMuPDF are rendered as
markdown and inserted at their vertical position inside the section text.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

if hasattr(pymupdf, "no_recommend_layout"):  # silence the stdout layout hint
    pymupdf.no_recommend_layout()

HEADING_RE = re.compile(r"^(\d{1,2})\.\s+(\S.{0,80})$")
PARAGRAPH_GAP_FACTOR = 1.5
_IGNORED_META_TITLES = {"", "(anonymous)", "untitled", "anonymous"}


@dataclass
class Chunk:
    chunk_id: str
    doc_file: str
    doc_title: str
    series_title: str
    section_num: int
    section_title: str
    page_start: int
    page_end: int
    text: str
    embed_text: str = field(default="")

    def __post_init__(self) -> None:
        if not self.embed_text:
            self.embed_text = make_embed_text(
                self.doc_title, self.section_num, self.section_title, self.text
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        names = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in names})


def make_embed_text(doc_title: str, num: int, title: str, text: str) -> str:
    return f"{doc_title} › §{num} {title}\n{text}"


def match_heading(text: str, last_num: int) -> tuple[int, str] | None:
    """Return (num, title) if `text` is the next numbered heading, else None."""
    text = text.strip()
    m = HEADING_RE.match(text)
    if not m or text.endswith("."):
        return None
    num = int(m.group(1))
    if num != last_num + 1:
        return None
    return num, m.group(2).strip()


def table_to_markdown(rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).replace("|", "\\|")

    rows = [r for r in rows if r and any(c not in (None, "") for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [[cell(c) for c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(norm[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in norm[1:]]
    return "\n".join(lines)


def _page_items(page: "pymupdf.Page") -> list[dict[str, Any]]:
    """Lines and tables of a page, sorted top to bottom."""
    tables = []
    try:
        found = page.find_tables()
        for t in found.tables:
            md = table_to_markdown(t.extract())
            if md:
                tables.append((pymupdf.Rect(t.bbox), md))
    except Exception:  # table detection is best-effort
        tables = []

    items: list[dict[str, Any]] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            centre = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
            if any(rect.contains(centre) for rect, _ in tables):
                continue
            sizes = [s.get("size", 0) for s in line.get("spans", [])]
            items.append(
                {
                    "kind": "line",
                    "text": text,
                    "x0": x0,
                    "y0": y0,
                    "y1": y1,
                    "size": round(max(sizes) if sizes else 0, 1),
                }
            )
    for rect, md in tables:
        items.append({"kind": "table", "text": md, "x0": rect.x0, "y0": rect.y0, "y1": rect.y1})
    items.sort(key=lambda it: (it["y0"], it["x0"]))
    return items


class _Section:
    def __init__(self, num: int, title: str, page: int) -> None:
        self.num = num
        self.title = title
        self.page_start = page
        self.page_end = page
        self.blocks: list[str] = []
        self._para: list[str] = []
        self._prev: dict[str, Any] | None = None

    def add(self, item: dict[str, Any], page: int) -> None:
        self.page_end = page
        if item["kind"] == "table":
            self._flush()
            self.blocks.append(item["text"])
            self._prev = None
            return
        prev = self._prev
        if prev is not None and self._para:
            height = max(prev["y1"] - prev["y0"], 1.0)
            if item["y0"] - prev["y0"] > PARAGRAPH_GAP_FACTOR * height:
                self._flush()
        self._para.append(item["text"])
        self._prev = item

    def _flush(self) -> None:
        if self._para:
            self.blocks.append(" ".join(self._para))
            self._para = []

    def text(self) -> str:
        self._flush()
        return "\n\n".join(self.blocks)


def load_pdf_sections(path: str | Path) -> list[Chunk]:
    path = Path(path)
    stem = path.stem
    doc = pymupdf.open(path)
    try:
        meta_title = (doc.metadata or {}).get("title") or ""
        sections: list[_Section] = []
        pre_lines: list[dict[str, Any]] = []  # lines before the first heading
        current: _Section | None = None
        last_num = 0
        for page_index, page in enumerate(doc):
            page_no = page_index + 1
            for item in _page_items(page):
                if item["kind"] == "line":
                    heading = match_heading(item["text"], last_num)
                    if heading:
                        last_num = heading[0]
                        current = _Section(heading[0], heading[1], page_no)
                        sections.append(current)
                        continue
                if current is None:
                    pre_lines.append({**item, "page": page_no})
                else:
                    current.add(item, page_no)
    finally:
        doc.close()

    series_title, doc_title, preamble = _split_titles(pre_lines)
    if not doc_title:
        doc_title = meta_title.strip() if meta_title.strip().lower() not in _IGNORED_META_TITLES else ""
    if not doc_title:
        doc_title = stem

    chunks: list[Chunk] = []
    if preamble:
        pre = _Section(0, "Preamble", preamble[0]["page"])
        for it in preamble:
            pre.add(it, it["page"])
        if pre.text().strip():
            sections.insert(0, pre)
    for s in sections:
        chunks.append(
            Chunk(
                chunk_id=f"{stem}#s{s.num}",
                doc_file=path.name,
                doc_title=doc_title,
                series_title=series_title,
                section_num=s.num,
                section_title=s.title,
                page_start=s.page_start,
                page_end=s.page_end,
                text=s.text(),
            )
        )
    return chunks


def _split_titles(pre_lines: list[dict[str, Any]]) -> tuple[str, str, list[dict[str, Any]]]:
    """First page-1 line = series title; following lines in the same font size
    as the second line = doc title; anything else before the first heading is
    preamble text."""
    first_page = [it for it in pre_lines if it["page"] == 1 and it["kind"] == "line"]
    if not first_page:
        return "", "", pre_lines
    if len(first_page) == 1:
        used = {id(first_page[0])}
        series, title = "", first_page[0]["text"]
    else:
        series = first_page[0]["text"]
        title_lines = [first_page[1]]
        for it in first_page[2:]:
            if it["size"] != first_page[1]["size"]:
                break
            title_lines.append(it)
        title = " ".join(it["text"] for it in title_lines)
        used = {id(first_page[0])} | {id(it) for it in title_lines}
    preamble = [it for it in pre_lines if id(it) not in used]
    return series, title, preamble


def load_corpus(data_dir: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf in sorted(Path(data_dir).glob("*.pdf")):
        chunks.extend(load_pdf_sections(pdf))
    return chunks
