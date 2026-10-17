"""Streamlit UI smoke tests (AppTest, no network: the engine answer is faked)."""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from rag.chat import ChatEngine, ChatResult
from rag.index import load_index

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(REPO_ROOT / "app.py")
PDFS = [
    "Almamlaka_Budget_Timeline.pdf",
    "Almamlaka_Project_Overview.pdf",
    "Almamlaka_Team_Governance.pdf",
]


@pytest.fixture(autouse=True)
def _dummy_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-dummy-key")
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()


def _all_markdown(elements) -> str:
    return "\n".join(m.value for m in elements)


def test_app_boots_with_branding_and_doc_list():
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception
    assert not at.error
    page = _all_markdown(at.markdown)
    assert "Almamlaka TV" in page
    assert 'alt="المملكة"' in page
    assert 'dir="rtl"' in page
    assert "كيف يمكنني مساعدتك اليوم؟" in page
    for pdf in PDFS:
        assert pdf in page
    labels = [b.label for b in at.button]
    assert "What is the total project budget?" in labels
    assert "متى موعد الإطلاق؟" in labels
    assert at.toggle(key="theme_toggle").value is True  # dark navy by default


def _fake_answer(lang: str, text: str):
    index = load_index()
    chunks = [index.by_id["Almamlaka_Budget_Timeline#s2"], index.by_id["Almamlaka_Project_Overview#s3"]]

    def answer(self, user_msg, history=()):
        return ChatResult(
            kind="answer",
            lang=lang,
            answer=text,
            citations=chunks,
            conflicts=[
                {
                    "topic": "Total budget",
                    "claims": [
                        {"value": "$2,400,000", "chunk_id": chunks[1].chunk_id, "label": "Overview"},
                        {"value": "$2,600,000", "chunk_id": chunks[0].chunk_id, "label": "Budget"},
                    ],
                }
            ],
        )

    return answer


def test_example_question_renders_answer_conflict_and_sources(monkeypatch):
    monkeypatch.setattr(ChatEngine, "answer", _fake_answer("en", "Budget <script>alert(1)</script>"))
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    at.button(key="ex_en_0").click().run(timeout=30)
    assert not at.exception
    assert len(at.chat_message) == 2
    page = _all_markdown(at.markdown)
    assert "&lt;script&gt;" in page and "<script>" not in page
    assert len(at.warning) == 1
    assert "Conflicting information in the documents" in at.warning[0].value
    assert len(at.expander) == 2
    assert "Budget & Timeline Breakdown" in at.expander[0].label

    at.button(key="clear_chat").click().run(timeout=30)
    assert len(at.chat_message) == 0


def test_arabic_answer_renders_rtl(monkeypatch):
    monkeypatch.setattr(ChatEngine, "answer", _fake_answer("ar", "الميزانية الإجمالية"))
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    at.button(key="ex_ar_0").click().run(timeout=30)
    assert not at.exception
    page = _all_markdown(at.markdown)
    assert "الميزانية الإجمالية" in page
    assert '<div dir="rtl" lang="ar" class="rtl">متى موعد الإطلاق؟</div>' in page
    # The assistant answer sits in the RTL container (CSS: [class*="st-key-rtl_"]).
    rtl_answer = at.get_by_key("rtl_answer_1")
    assert "الميزانية الإجمالية" in _all_markdown(rtl_answer.markdown)
    with pytest.raises(KeyError):
        at.get_by_key("ltr_answer_1")
    assert "معلومات متعارضة في الوثائق" in at.warning[0].value
    assert "المصادر" in page


def test_missing_index_without_key_shows_error(monkeypatch, tmp_path):
    monkeypatch.setattr("rag.config.INDEX_PATH", tmp_path / "missing.json")
    monkeypatch.setattr("rag.config.get_api_key", lambda: None)
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception
    assert any("index is missing or out of date" in e.value for e in at.error)


def test_missing_index_is_rebuilt_at_startup(monkeypatch, tmp_path):
    import json

    data = json.loads((REPO_ROOT / "index" / "index.json").read_text(encoding="utf-8"))
    calls = []

    def fake_build_index(data_dir, llm, embedding_model):
        calls.append(data_dir)
        return data

    monkeypatch.setattr("rag.config.INDEX_PATH", tmp_path / "missing.json")
    monkeypatch.setattr("rag.index.build_index", fake_build_index)
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception
    assert not at.error
    assert len(calls) == 1
    page = _all_markdown(at.markdown)
    for pdf in PDFS:
        assert pdf in page


def test_missing_key_shows_error_without_crash(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("rag.config.get_api_key", lambda: None)
    at = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not at.exception
    at.button(key="ex_en_1").click().run(timeout=30)
    assert not at.exception
    assert any("OPENAI_API_KEY is not configured" in e.value for e in at.error)
