import pytest

from conftest import FakeEmbedder, make_chunk, make_index
from rag.conflicts import postprocess
from rag.retrieve import RRF_K, retrieve, rrf_fuse


def test_rrf_math():
    fused = rrf_fuse([[0, 1, 2], [2, 0, 1]])
    assert fused[0] == pytest.approx(1 / (RRF_K + 1) + 1 / (RRF_K + 2))
    assert fused[1] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 3))
    assert fused[2] == pytest.approx(1 / (RRF_K + 3) + 1 / (RRF_K + 1))
    assert max(fused, key=fused.get) == 0


def test_top_hit_and_dense_max(toy_index):
    q = "what is the project lead in the strategy department"
    res = retrieve(toy_index, q, FakeEmbedder().vector(q), k=6)
    assert res.chunks[0].chunk_id == "Doc_C#s1"
    assert 0 < res.dense_max <= 1.0 + 1e-9
    assert set(res.scores) == set(toy_index.by_id)


def test_unrelated_query_has_low_dense_max(toy_index):
    q = "zzz qqq"
    res = retrieve(toy_index, q, FakeEmbedder().vector(q))
    assert res.dense_max < 0.30


def test_per_document_guarantee(toy_index):
    q = "total budget launch date"
    res = retrieve(toy_index, q, FakeEmbedder().vector(q), k=1)
    docs = [c.doc_file for c in res.chunks]
    assert set(docs) == {"Doc_A.pdf", "Doc_B.pdf", "Doc_C.pdf"}
    assert len(res.chunks) == 3


def test_conflict_expansion_appends_other_claims(toy_index, toy_conflicts):
    q = "increase viewers percent goals"
    res = retrieve(toy_index, q, FakeEmbedder().vector(q), toy_conflicts, k=1)
    ids = [c.chunk_id for c in res.chunks]
    assert ids[0] == "Doc_A#s2"
    # per-doc guarantee then expansion of whichever conflicts were touched
    touched = {cl["chunk_id"] for c in res.conflicts for cl in c["claims"]}
    assert touched and touched <= set(ids)


def test_conflict_expansion_respects_cap():
    chunks = [make_chunk("Doc_A", i, f"Section {i}", f"topic{i} common words") for i in range(1, 13)]
    index = make_index(chunks)
    conflict = {
        "topic": "t",
        "kind": "contradiction",
        "description": "",
        "claims": [{"value": str(i), "chunk_id": f"Doc_A#s{i}", "note": ""} for i in range(1, 13)],
    }
    q = "topic1"
    res = retrieve(index, q, FakeEmbedder().vector(q), [conflict], k=6, max_chunks=10)
    assert len(res.chunks) == 10
    assert res.chunks[0].chunk_id == "Doc_A#s1"
    assert res.conflicts == [conflict]


def test_conflict_postprocess_drops_invalid_ids():
    raw = {
        "facts": [{"attribute": "a", "value": "1", "chunk_id": "X#s1"}, {"attribute": "a", "value": "2", "chunk_id": "bad"}],
        "conflicts": [
            {"topic": "a", "kind": "contradiction", "description": "", "claims": [
                {"value": "1", "chunk_id": "X#s1", "note": ""}, {"value": "2", "chunk_id": "bad", "note": ""}]},
            {"topic": "b", "kind": "inconsistency", "description": "", "claims": [
                {"value": "1", "chunk_id": "X#s1", "note": ""}]},
            {"topic": "c", "kind": "contradiction", "description": "", "claims": [
                {"value": "1", "chunk_id": "X#s1", "note": ""}, {"value": "2", "chunk_id": "Y#s2", "note": ""}]},
        ],
    }
    out = postprocess(raw, ["X#s1", "Y#s2"])
    assert [f["chunk_id"] for f in out["facts"]] == ["X#s1"]
    assert [c["topic"] for c in out["conflicts"]] == ["b", "c"]
