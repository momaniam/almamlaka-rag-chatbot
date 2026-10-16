"""Hybrid retrieval: dense cosine + BM25 fused with RRF, per-document
guarantee and conflict expansion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from rag.config import MAX_CONTEXT_CHUNKS, TOP_K
from rag.index import Index, normalise, tokenize
from rag.ingest import Chunk

RRF_K = 60


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    dense_max: float
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)


def _ranking(scores: np.ndarray) -> list[int]:
    # stable sort: ties keep corpus order
    return [int(i) for i in np.argsort(-scores, kind="stable")]


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> dict[int, float]:
    """score(i) = sum over rankings of 1 / (k + rank), rank 1-based."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


def retrieve(
    index: Index,
    query_en: str,
    query_vec: Sequence[float],
    conflicts: Sequence[dict[str, Any]] = (),
    k: int = TOP_K,
    max_chunks: int = MAX_CONTEXT_CHUNKS,
) -> RetrievalResult:
    n = len(index.chunks)
    if n == 0:
        return RetrievalResult(chunks=[], dense_max=0.0)

    dense = index.matrix @ normalise(query_vec)
    tokens = tokenize(query_en)
    bm25 = np.asarray(index.bm25.get_scores(tokens), dtype=float) if tokens else np.zeros(n)
    fused = rrf_fuse([_ranking(dense), _ranking(bm25)])
    order = sorted(range(n), key=lambda i: (-fused[i], i))

    selected: list[int] = order[:k]

    # Per-document guarantee: every document contributes its best chunk.
    present = {index.chunks[i].doc_file for i in selected}
    for i in order:
        doc = index.chunks[i].doc_file
        if doc not in present:
            selected.append(i)
            present.add(doc)

    # Conflict expansion: pull in the other side of any touched conflict.
    id_to_idx = {c.chunk_id: i for i, c in enumerate(index.chunks)}
    selected_ids = {index.chunks[i].chunk_id for i in selected}
    relevant = [
        c for c in conflicts if any(cl.get("chunk_id") in selected_ids for cl in c.get("claims", []))
    ]
    extra: set[int] = set()
    for c in relevant:
        for cl in c.get("claims", []):
            idx = id_to_idx.get(cl.get("chunk_id"))
            if idx is not None and idx not in selected:
                extra.add(idx)
    for i in sorted(extra, key=lambda i: (-fused[i], i)):
        if len(selected) >= max_chunks:
            break
        selected.append(i)
    selected = selected[: max(max_chunks, 0)]

    scores = {
        index.chunks[i].chunk_id: {
            "dense": round(float(dense[i]), 4),
            "bm25": round(float(bm25[i]), 4),
            "rrf": round(float(fused[i]), 6),
        }
        for i in range(n)
    }
    return RetrievalResult(
        chunks=[index.chunks[i] for i in selected],
        dense_max=float(np.max(dense)),
        scores=scores,
        conflicts=relevant,
    )
