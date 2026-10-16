import json

import pytest

from conftest import DATA_DIR, FakeLLM
from rag.index import IndexStaleError, build_index, load_index, tokenize, write_json
from rag.ingest import load_corpus, match_heading, table_to_markdown

EXPECTED = {
    "Almamlaka_Budget_Timeline": [
        (1, "Quarterly Budget Allocation", 1),
        (2, "Budget Notes", 1),
        (3, "Project Timeline", 2),
        (4, "Risk Notes", 2),
    ],
    "Almamlaka_Project_Overview": [
        (1, "Introduction", 1),
        (2, "Project Goals", 1),
        (3, "Key Details", 1),
        (4, "Platform Scope", 2),
        (5, "Stakeholders", 2),
        (6, "Success Metrics", 2),
    ],
    "Almamlaka_Team_Governance": [
        (1, "Project Team", 1),
        (2, "Governance Structure", 1),
        (3, "Revised Launch Date", 1),
        (4, "Data Privacy Policy", 2),
        (5, "Content Moderation Policy", 2),
        (6, "Escalation Contacts", 2),
    ],
}
DOC_TITLES = {
    "Almamlaka_Budget_Timeline": "Budget & Timeline Breakdown",
    "Almamlaka_Project_Overview": "Project Overview Document",
    "Almamlaka_Team_Governance": "Team & Governance Document",
}


@pytest.fixture(scope="module")
def corpus():
    return {c.chunk_id: c for c in load_corpus(DATA_DIR)}


def test_sixteen_chunks_with_ids_titles_pages(corpus):
    assert len(corpus) == 16
    for stem, sections in EXPECTED.items():
        for num, title, page in sections:
            c = corpus[f"{stem}#s{num}"]
            assert c.section_title == title
            assert (c.page_start, c.page_end) == (page, page)
            assert c.doc_title == DOC_TITLES[stem]
            assert c.series_title == "Almamlaka TV Digital Expansion Initiative"
            assert c.doc_file == f"{stem}.pdf"
            assert c.embed_text.startswith(f"{DOC_TITLES[stem]} › §{num} {title}\n")


def test_tables_rendered_as_markdown(corpus):
    budget = corpus["Almamlaka_Budget_Timeline#s1"].text
    assert "| Quarter | Development | Marketing | Operations | Total |" in budget
    assert "| Q2 2027 | $500,000 | $150,000 | $70,000 | $720,000 |" in budget
    assert "| Total | $1,700,000 | $460,000 | $240,000 | $2,400,000 |" in budget
    timeline = corpus["Almamlaka_Budget_Timeline#s3"].text
    assert "| Public launch | March 15, 2027 |" in timeline
    assert "| QA & testing | Mar 1 - Mar 20, 2027 |" in timeline
    team = corpus["Almamlaka_Team_Governance#s1"].text
    assert "| Omar Fayez | Lead Engineer | Engineering |" in team


def test_no_table_residue_outside_tables(corpus):
    for c in corpus.values():
        for line in c.text.splitlines():
            if line.startswith("|"):
                continue
            # one-cell-per-line residue would leave bare cell values as lines
            assert line.strip() not in {"Quarter", "Q2 2027", "$720,000", "Milestone", "Omar Fayez"}


def test_key_value_lines_are_separate_paragraphs(corpus):
    text = corpus["Almamlaka_Project_Overview#s3"].text
    assert "Project Lead: Sara Al-Rashid\n\nTotal Approved Budget: $2,400,000" in text


def test_heading_regex_requires_monotonic_numbers():
    assert match_heading("1. Introduction", 0) == (1, "Introduction")
    assert match_heading("3. Key Details", 2) == (3, "Key Details")
    assert match_heading("3. Key Details", 0) is None  # skips numbers
    assert match_heading("1. Introduction", 1) is None  # repeat
    assert match_heading("2. This is a sentence.", 1) is None  # terminal period
    assert match_heading("2027. Something", 2026) is None  # > 2 digits
    assert match_heading("Introduction", 0) is None


def test_table_to_markdown_handles_none_and_pipes():
    md = table_to_markdown([["A", "B"], ["x|y", None]])
    assert md.splitlines() == ["| A | B |", "|---|---|", "| x\\|y |  |"]


def test_tokenizer_keeps_money_and_quarters():
    assert tokenize("Q2 budget: $2,400,000.") == ["q2", "budget", "$2,400,000"]
    assert "40%" in tokenize("increase by 40% within")


def test_build_and_load_index_offline(tmp_path):
    llm = FakeLLM()
    data = build_index(DATA_DIR, llm, "fake-model")
    assert len(llm.embed_calls) == 1 and len(llm.embed_calls[0]) == 16
    assert len(data["chunks"]) == len(data["embeddings"]) == 16
    assert set(data["manifest"]) == {f"{s}.pdf" for s in EXPECTED}
    path = tmp_path / "index.json"
    write_json(path, data)
    index = load_index(path, DATA_DIR)
    assert index.matrix.shape[0] == 16
    assert index.by_id["Almamlaka_Budget_Timeline#s3"].page_start == 2


def test_load_index_stale_manifest_uses_rebuild_hook(tmp_path):
    data = build_index(DATA_DIR, FakeLLM(), "fake-model")
    stale = dict(data, manifest={"other.pdf": "0"})
    path = tmp_path / "index.json"
    path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(IndexStaleError):
        load_index(path, DATA_DIR)
    calls = []
    index = load_index(path, DATA_DIR, rebuild=lambda: calls.append(1) or data)
    assert calls == [1] and len(index.chunks) == 16
    with pytest.raises(IndexStaleError):
        load_index(tmp_path / "missing.json", DATA_DIR)
