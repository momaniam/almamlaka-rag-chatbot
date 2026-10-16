"""Arabic / English detection by script ratio."""

from __future__ import annotations

_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
ARABIC_RATIO = 0.3


def is_arabic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _ARABIC_RANGES)


def detect_language(text: str, fallback: str | None = None) -> str:
    """'ar' if Arabic letters are >= 30% of all letters, else 'en'.

    With no letters at all (e.g. "?", "2027"), return `fallback` or 'en'.
    """
    letters = [ch for ch in text or "" if ch.isalpha()]
    if not letters:
        return fallback if fallback in ("en", "ar") else "en"
    arabic = sum(1 for ch in letters if is_arabic_char(ch))
    return "ar" if arabic / len(letters) >= ARABIC_RATIO else "en"
