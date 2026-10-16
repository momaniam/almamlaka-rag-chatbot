"""Index build / load: normalised embeddings in JSON + BM25 over embed_text."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from rank_bm25 import BM25Okapi

from rag.config import DATA_DIR, EMBED_MODEL, INDEX_PATH
from rag.ingest import Chunk, load_corpus
from rag.llm import LLM

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[\w$%.,-]+")
_STRIP = ".,-"


class IndexStaleError(RuntimeError):
    """index.json is missing or does not match the PDFs in data/."""


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; keeps `$2,400,000`, `q2`, `2027`, `40%` intact."""
    tokens = []
    for raw in _TOKEN_RE.findall(text.lower()):
        tok = raw.strip(_STRIP)
        if tok:
            tokens.append(tok)
    return tokens


def normalise(vectors: Any) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim == 1:
        norm = np.linalg.norm(arr)
        return arr / norm if norm else arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def pdf_manifest(data_dir: str | Path) -> dict[str, str]:
    manifest = {}
    for pdf in sorted(Path(data_dir).glob("*.pdf")):
        manifest[pdf.name] = hashlib.sha256(pdf.read_bytes()).hexdigest()
    return manifest


@dataclass
class Index:
    chunks: list[Chunk]
    matrix: np.ndarray  # (n, d), rows L2-normalised
    embedding_model: str = EMBED_MODEL
    manifest: dict[str, str] = field(default_factory=dict)
    bm25: BM25Okapi | None = None

    def __post_init__(self) -> None:
        if self.bm25 is None:
            self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in self.chunks])
        self.by_id: dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}

    @property
    def doc_files(self) -> list[str]:
        seen: list[str] = []
        for c in self.chunks:
            if c.doc_file not in seen:
                seen.append(c.doc_file)
        return seen

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> "Index":
        chunks = [Chunk.from_dict(c) for c in data["chunks"]]
        matrix = normalise(data["embeddings"])
        return cls(
            chunks=chunks,
            matrix=matrix,
            embedding_model=data.get("embedding_model", EMBED_MODEL),
            manifest=data.get("manifest", {}),
        )


def build_index(
    data_dir: str | Path, llm: LLM, embedding_model: str = EMBED_MODEL
) -> dict[str, Any]:
    """Parse PDFs, embed every chunk in one batch and return the index dict."""
    chunks = load_corpus(data_dir)
    vectors = llm.embed([c.embed_text for c in chunks]) if chunks else []
    if len(vectors) != len(chunks):
        raise RuntimeError("embedding count does not match chunk count")
    matrix = normalise(vectors) if chunks else np.zeros((0, 0))
    return {
        "embedding_model": embedding_model,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": pdf_manifest(data_dir),
        "chunks": [c.to_dict() for c in chunks],
        "embeddings": [[round(float(x), 6) for x in row] for row in matrix],
    }


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def load_index(
    path: str | Path = INDEX_PATH,
    data_dir: str | Path = DATA_DIR,
    rebuild: Callable[[], dict[str, Any]] | None = None,
) -> Index:
    """Load index.json; if missing or stale vs data/, call `rebuild()` (if given)."""
    path = Path(path)
    data: dict[str, Any] | None = None
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("manifest") != pdf_manifest(data_dir):
            log.warning("index manifest does not match %s", data_dir)
            data = None
    else:
        log.warning("index file %s not found", path)
    if data is None:
        if rebuild is None:
            raise IndexStaleError(f"{path} is missing or stale")
        data = rebuild()
    return Index.from_data(data)
