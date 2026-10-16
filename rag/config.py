"""Paths, model names and tunable thresholds (all env-overridable)."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


DATA_DIR = _env_path("RAG_DATA_DIR", REPO_ROOT / "data")
INDEX_PATH = _env_path("RAG_INDEX_PATH", REPO_ROOT / "index" / "index.json")
CONFLICTS_PATH = _env_path("RAG_CONFLICTS_PATH", REPO_ROOT / "index" / "conflicts.json")
ASSETS_DIR = _env_path("RAG_ASSETS_DIR", REPO_ROOT / "assets")

EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o")
CONDENSE_MODEL = os.environ.get("OPENAI_CONDENSE_MODEL", "gpt-4o-mini")

# Tuned in eval/run_eval.py: unrelated queries scored <= 0.22, the lowest
# in-scope query 0.28-0.30 (see README "Evaluation").
RELEVANCE_THRESHOLD = _env_float("RAG_RELEVANCE_THRESHOLD", 0.25)
TOP_K = _env_int("RAG_TOP_K", 6)
MAX_CONTEXT_CHUNKS = _env_int("RAG_MAX_CONTEXT_CHUNKS", 10)
HISTORY_MESSAGES = _env_int("RAG_HISTORY_MESSAGES", 6)
MAX_INPUT_CHARS = _env_int("RAG_MAX_INPUT_CHARS", 1000)


def get_api_key() -> str | None:
    """Return the OpenAI API key from .env / environment / Streamlit secrets.

    The key is never logged or printed.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except Exception:  # dotenv missing or unreadable file: fall through to env
        pass
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:  # no secrets file / not running under streamlit
        key = None
    return key or None
