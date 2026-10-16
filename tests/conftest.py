"""Offline fakes: FakeEmbedder (deterministic hashed bag-of-words vectors) and
FakeLLM (chat_json responses scripted per schema_name)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.index import Index, tokenize  # noqa: E402
from rag.ingest import Chunk  # noqa: E402

DATA_DIR = REPO_ROOT / "data"


class FakeEmbedder:
    """Hashes tokens into a fixed-size vector; overlapping words -> similar vectors."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def vector(self, text: str) -> list[float]:
        vec = np.zeros(self.dim)
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        return vec.tolist()

    def embed(self, texts):
        return [self.vector(t) for t in texts]


class FakeLLM:
    """`responses[schema_name]` is a dict, a list (consumed in order) or a
    callable(messages) -> dict. An Exception value is raised."""

    def __init__(self, responses: dict[str, Any] | None = None, embedder: FakeEmbedder | None = None):
        self.responses = dict(responses or {})
        self.embedder = embedder or FakeEmbedder()
        self.calls: list[dict[str, Any]] = []
        self.embed_calls: list[list[str]] = []

    def chat_json(self, model, messages, schema_name, schema):
        self.calls.append({"model": model, "messages": messages, "schema_name": schema_name})
        resp = self.responses[schema_name]
        if isinstance(resp, list):
            resp = resp.pop(0)
        if callable(resp):
            resp = resp(messages)
        if isinstance(resp, Exception):
            raise resp
        return resp

    def embed(self, texts):
        self.embed_calls.append(list(texts))
        return self.embedder.embed(texts)

    def schema_calls(self, name: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["schema_name"] == name]


def make_chunk(stem: str, num: int, title: str, text: str, page: int = 1, doc_title: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"{stem}#s{num}",
        doc_file=f"{stem}.pdf",
        doc_title=doc_title or stem.replace("_", " "),
        series_title="Series",
        section_num=num,
        section_title=title,
        page_start=page,
        page_end=page,
        text=text,
    )


def make_index(chunks: list[Chunk], embedder: FakeEmbedder | None = None) -> Index:
    embedder = embedder or FakeEmbedder()
    return Index.from_data(
        {
            "chunks": [c.to_dict() for c in chunks],
            "embeddings": embedder.embed([c.embed_text for c in chunks]),
            "embedding_model": "fake",
            "manifest": {},
        }
    )


@pytest.fixture
def toy_chunks() -> list[Chunk]:
    return [
        make_chunk("Doc_A", 1, "Key Details", "Total budget: $1,000,000. Launch date: May 1, 2030."),
        make_chunk("Doc_A", 2, "Goals", "Increase viewers by 10 percent."),
        make_chunk("Doc_B", 1, "Budget Notes", "The total budget was revised to $1,200,000."),
        make_chunk("Doc_B", 2, "Timeline", "Testing runs from April 1 to May 10, 2030.", page=2),
        make_chunk("Doc_C", 1, "Team", "Alice Example is the project lead in the Strategy department."),
        make_chunk("Doc_C", 2, "Revised Launch", "The launch date was moved to June 1, 2030."),
    ]


@pytest.fixture
def toy_index(toy_chunks) -> Index:
    return make_index(toy_chunks)


@pytest.fixture
def toy_conflicts() -> list[dict[str, Any]]:
    return [
        {
            "topic": "Total budget",
            "kind": "contradiction",
            "description": "Different totals.",
            "claims": [
                {"value": "$1,000,000", "chunk_id": "Doc_A#s1", "note": ""},
                {"value": "$1,200,000", "chunk_id": "Doc_B#s1", "note": "revision"},
            ],
        },
        {
            "topic": "Launch date",
            "kind": "contradiction",
            "description": "Different launch dates.",
            "claims": [
                {"value": "May 1, 2030", "chunk_id": "Doc_A#s1", "note": ""},
                {"value": "June 1, 2030", "chunk_id": "Doc_C#s2", "note": "revision"},
            ],
        },
    ]
