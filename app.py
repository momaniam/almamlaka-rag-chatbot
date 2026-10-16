"""Almamlaka TV Digital Expansion Initiative assistant (Streamlit UI only).

All retrieval / generation logic lives in the `rag` package; this file only
renders the branded chat interface around `rag.chat.ChatEngine`.
"""

from __future__ import annotations

import base64
import logging
import re
from typing import Any

import streamlit as st

from rag import config
from rag.chat import ChatEngine, ChatResult
from rag.conflicts import load_conflicts
from rag.formatting import citation_label, safe_html
from rag.index import Index, IndexStaleError, load_index
from rag.language import detect_language
from rag.llm import OpenAIClient

log = logging.getLogger(__name__)

LOGO_PATH = config.ASSETS_DIR / "unnamed.jpg"
WORDMARK_PATH = config.ASSETS_DIR / "61yv40dMDTL.png"
CSS_PATH = config.ASSETS_DIR / "style.css"

EXAMPLES_EN = [
    "What is the total project budget?",
    "When is the public launch?",
    "Who is on the project team?",
]
EXAMPLES_AR = [
    "ما هي الميزانية الإجمالية للمشروع؟",
    "متى موعد الإطلاق؟",
    "من هي قائدة المشروع؟",
]

UI_TEXT = {
    "en": {
        "conflict": "Conflicting information in the documents: {topic}",
        "sources": "Sources",
        "thinking": "Searching the documents…",
    },
    "ar": {
        "conflict": "معلومات متعارضة في الوثائق: {topic}",
        "sources": "المصادر",
        "thinking": "جارٍ البحث في الوثائق…",
    },
}

MISSING_KEY_MSG = (
    "OPENAI_API_KEY is not configured. Add it to a local .env file, the environment, "
    "or the Hugging Face Space secrets, then restart the app."
)
INDEX_ERROR_MSG = (
    "The knowledge-base index is missing or out of date. "
    "Run `python scripts/build_index.py` and restart the app."
)

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$])")


class MissingAPIKey(RuntimeError):
    """Raised by get_engine() so a missing key is never cached."""


# ----------------------------------------------------------------- helpers


def md_escape(text: str) -> str:
    """Escape Markdown (and Streamlit's `$` LaTeX) in model text for st.warning."""
    return _MD_SPECIAL.sub(r"\\\1", text or "")


def trusted_markdown(text: str) -> str:
    """Chunk text is trusted repo content; only stop `$` from becoming LaTeX."""
    return (text or "").replace("$", "\\$")


@st.cache_data(show_spinner=False)
def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


@st.cache_resource(show_spinner=False)
def load_knowledge_base() -> tuple[Index, list[dict[str, Any]]]:
    index = load_index(config.INDEX_PATH, config.DATA_DIR)
    conflicts = load_conflicts(config.CONFLICTS_PATH, valid_ids=index.by_id.keys())
    return index, conflicts


@st.cache_resource(show_spinner=False)
def get_engine() -> ChatEngine:
    api_key = config.get_api_key()
    if not api_key:
        raise MissingAPIKey("OPENAI_API_KEY is not configured")
    index, conflicts = load_knowledge_base()
    llm = OpenAIClient(api_key=api_key, embed_model=config.EMBED_MODEL)
    return ChatEngine(index, conflicts, llm)


def doc_summaries(index: Index) -> list[dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for c in index.chunks:
        d = docs.setdefault(c.doc_file, {"file": c.doc_file, "title": c.doc_title, "pages": 0})
        d["pages"] = max(d["pages"], c.page_end)
    return list(docs.values())


def queue_question(question: str) -> None:
    st.session_state.pending_question = question


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.pop("pending_question", None)


# --------------------------------------------------------------- rendering


def render_css() -> None:
    try:
        css = CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header() -> None:
    logo = ""
    if LOGO_PATH.exists():
        logo = (
            f'<img src="data:image/jpeg;base64,{_b64(str(LOGO_PATH))}" '
            'alt="Almamlaka TV logo">'
        )
    st.markdown(
        f"""<header class="mm-header" role="banner">{logo}<div>
<h1 class="mm-title">Almamlaka TV</h1>
<p class="mm-subtitle">Digital Expansion Initiative Assistant /
<span dir="rtl" lang="ar">مساعد مبادرة التوسع الرقمي</span></p>
</div></header>""",
        unsafe_allow_html=True,
    )


def render_sidebar(index: Index | None) -> None:
    with st.sidebar:
        if WORDMARK_PATH.exists():
            st.image(str(WORDMARK_PATH), width="stretch")
        st.subheader("Knowledge base")
        if index is None:
            st.caption("Index not available.")
        else:
            lines = [
                f"- **{d['title']}**  \n  `{d['file']}` · {d['pages']} pages"
                for d in doc_summaries(index)
            ]
            st.markdown("\n".join(lines))
        st.subheader("Try asking")
        for i, q in enumerate(EXAMPLES_EN):
            st.button(q, key=f"ex_en_{i}", on_click=queue_question, args=(q,), width="stretch")
        for i, q in enumerate(EXAMPLES_AR):
            st.button(q, key=f"ex_ar_{i}", on_click=queue_question, args=(q,), width="stretch")
        st.divider()
        st.button("Clear chat", key="clear_chat", on_click=clear_chat, type="primary", width="stretch")
        st.markdown(
            '<p class="mm-note">Answers come only from the three Digital Expansion Initiative PDFs. '
            '<span dir="rtl" lang="ar">الإجابات مستندة فقط إلى الوثائق الثلاث.</span></p>',
            unsafe_allow_html=True,
        )


def render_conflicts(msg_idx: int, conflicts: list[dict[str, Any]], lang: str) -> None:
    text = UI_TEXT["ar" if lang == "ar" else "en"]
    for i, conflict in enumerate(conflicts):
        body = "**" + md_escape(text["conflict"].format(topic=conflict.get("topic", ""))) + "**"
        for claim in conflict.get("claims", []):
            body += f"\n- {md_escape(claim.get('value', ''))} ({md_escape(claim.get('label', ''))})"
        if lang == "ar":
            with st.container(key=f"rtl_conflict_{msg_idx}_{i}"):
                st.warning(body, icon="⚠️")
        else:
            st.warning(body, icon="⚠️")


def render_citations(citations: list[Any], lang: str) -> None:
    if not citations:
        return
    lang = "ar" if lang == "ar" else "en"
    direction = "rtl" if lang == "ar" else "ltr"
    st.markdown(
        f'<div class="mm-sources {direction}" dir="{direction}" lang="{lang}">'
        f'{UI_TEXT[lang]["sources"]}</div>',
        unsafe_allow_html=True,
    )
    for chunk in citations:
        with st.expander(citation_label(chunk)):
            st.markdown(trusted_markdown(chunk.text))


def render_message_body(msg_idx: int, msg: dict[str, Any]) -> None:
    lang = msg.get("lang") or "en"
    st.markdown(safe_html(msg.get("content", ""), lang), unsafe_allow_html=True)
    if msg.get("role") == "assistant":
        render_conflicts(msg_idx, msg.get("conflicts") or [], lang)
        render_citations(msg.get("citations") or [], lang)


def avatar_for(role: str) -> str | None:
    if role == "assistant" and LOGO_PATH.exists():
        return str(LOGO_PATH)
    return None


def assistant_message(result: ChatResult) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": result.answer,
        "lang": result.lang,
        "kind": result.kind,
        "citations": list(result.citations),
        "conflicts": list(result.conflicts),
    }


def previous_user_lang(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("lang") in ("en", "ar"):
            return m["lang"]
    return "en"


# -------------------------------------------------------------------- main


def handle_question(question: str, index: Index | None) -> None:
    messages: list[dict[str, Any]] = st.session_state.messages
    question = question.strip()[: config.MAX_INPUT_CHARS]
    if not question:
        return
    lang = detect_language(question, previous_user_lang(messages))
    user_msg = {"role": "user", "content": question, "lang": lang, "kind": "question"}
    with st.chat_message("user"):
        render_message_body(len(messages), user_msg)

    if index is None:
        st.error(INDEX_ERROR_MSG)
        return
    try:
        engine = get_engine()
    except MissingAPIKey:
        st.error(MISSING_KEY_MSG)
        return
    except Exception as exc:  # never surface details that could include secrets
        log.warning("engine initialisation failed: %s", type(exc).__name__)
        st.error("The assistant could not start. Please try again later.")
        return

    history = [{"role": m["role"], "content": m["content"], "lang": m.get("lang")} for m in messages]
    with st.chat_message("assistant", avatar=avatar_for("assistant")):
        with st.spinner(UI_TEXT["ar" if lang == "ar" else "en"]["thinking"]):
            result = engine.answer(question, history)
        user_msg["lang"] = result.lang
        reply = assistant_message(result)
        render_message_body(len(messages) + 1, reply)
    messages.extend([user_msg, reply])


def main() -> None:
    st.set_page_config(
        page_title="Almamlaka TV Assistant",
        page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "📺",
        layout="centered",
    )
    render_css()
    st.session_state.setdefault("messages", [])

    try:
        index, _ = load_knowledge_base()
    except (IndexStaleError, OSError, ValueError) as exc:
        log.warning("knowledge base unavailable: %s", type(exc).__name__)
        index = None

    render_header()
    render_sidebar(index)
    if index is None:
        st.error(INDEX_ERROR_MSG)

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar=avatar_for(msg["role"])):
            render_message_body(i, msg)

    typed = st.chat_input("Ask about the project… / اسأل عن المشروع…", max_chars=config.MAX_INPUT_CHARS)
    pending = st.session_state.pop("pending_question", None)
    question = typed or pending
    if question:
        handle_question(question, index)


main()
