"""Citation labels, HTML-safe rendering and digit normalisation."""

from __future__ import annotations

import html

from rag.ingest import Chunk

_DIGITS = {ord("٠") + i: str(i) for i in range(10)}
_DIGITS.update({ord("۰") + i: str(i) for i in range(10)})


def page_label(chunk: Chunk) -> str:
    if chunk.page_start == chunk.page_end:
        return f"p. {chunk.page_start}"
    return f"p. {chunk.page_start}–{chunk.page_end}"


def citation_label(chunk: Chunk) -> str:
    return f"{chunk.doc_title}, {page_label(chunk)}, §{chunk.section_num} {chunk.section_title}"


def safe_html(text: str, lang: str = "en") -> str:
    """Escape user/model text and wrap it with the right direction attributes."""
    lang = "ar" if lang == "ar" else "en"
    direction = "rtl" if lang == "ar" else "ltr"
    body = html.escape(text or "", quote=True).replace("\r\n", "\n").replace("\n", "<br>")
    return f'<div dir="{direction}" lang="{lang}" class="{direction}">{body}</div>'


def normalise_digits(text: str) -> str:
    """Arabic-Indic and Persian digits -> ASCII digits."""
    return (text or "").translate(_DIGITS)
