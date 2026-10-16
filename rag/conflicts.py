"""Ingestion-time conflict pre-scan over all chunks (one LLM call)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rag.config import CHAT_MODEL, CONFLICTS_PATH
from rag.ingest import Chunk
from rag.llm import LLM
from rag.prompts import CONFLICT_SCHEMA, build_conflict_messages

log = logging.getLogger(__name__)

_MIN_CLAIMS = {"contradiction": 2, "inconsistency": 1}


def postprocess(raw: dict[str, Any], valid_ids: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    """Drop facts/claims with unknown chunk ids and conflicts left too thin."""
    valid = set(valid_ids)
    facts = [f for f in raw.get("facts", []) if f.get("chunk_id") in valid]
    conflicts = []
    for c in raw.get("conflicts", []):
        kind = c.get("kind") if c.get("kind") in _MIN_CLAIMS else "contradiction"
        claims = [cl for cl in c.get("claims", []) if cl.get("chunk_id") in valid]
        if len(claims) < _MIN_CLAIMS[kind]:
            continue
        conflicts.append({**c, "kind": kind, "claims": claims})
    return {"facts": facts, "conflicts": conflicts}


def scan_conflicts(chunks: list[Chunk], llm: LLM, model: str = CHAT_MODEL) -> dict[str, Any]:
    raw = llm.chat_json(model, build_conflict_messages(chunks), "conflict_scan", CONFLICT_SCHEMA)
    result = postprocess(raw, (c.chunk_id for c in chunks))
    return {
        "model": model,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **result,
    }


def load_conflicts(path: str | Path = CONFLICTS_PATH, valid_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Conflicts list from conflicts.json; [] (with a warning) if unavailable."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("conflicts file %s unavailable; continuing without pre-scanned conflicts", path)
        return []
    if valid_ids is not None:
        return postprocess(data, valid_ids)["conflicts"]
    return list(data.get("conflicts", []))
