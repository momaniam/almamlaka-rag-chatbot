"""Almamlaka TV Digital Expansion Initiative assistant (Streamlit UI only).

All retrieval / generation logic lives in the `rag` package; this file only
renders the branded chat interface around `rag.chat.ChatEngine`.
"""

from __future__ import annotations

import base64
import html
import logging
import re
from typing import Any

import streamlit as st

from rag import config
from rag.chat import ChatEngine, ChatResult
from rag.conflicts import load_conflicts
from rag.formatting import citation_label, safe_html
from rag.index import Index, IndexStaleError, build_index, load_index
from rag.language import detect_language
from rag.llm import OpenAIClient

log = logging.getLogger(__name__)

LOGO_SVG_PATH = config.ASSETS_DIR / "logo.svg"
LOGO_JPG_PATH = config.ASSETS_DIR / "unnamed.jpg"
LOGO_PATH = LOGO_SVG_PATH if LOGO_SVG_PATH.exists() else LOGO_JPG_PATH
CSS_PATH = config.ASSETS_DIR / "style.css"

# Suggestion chips on the empty screen: (widget key, question). They mix
# languages so reviewers can see bilingual answers, conflicts and follow-ups.
SUGGESTIONS = [
    ("ex_en_0", "What is the total project budget?"),
    ("ex_ar_0", "متى موعد الإطلاق؟"),
    ("ex_en_1", "Who leads the project?"),
    ("ex_ar_1", "من هي مسؤولة الامتثال؟"),
]

UI_TEXT = {
    "en": {
        "conflict": "Conflicting information in the documents: {topic}",
        "sources": "Sources",
        "thinking": "Searching the documents…",
        "out_of_scope": "Not covered by the documents",
        "injection": "Outside the assistant's scope",
    },
    "ar": {
        "conflict": "معلومات متعارضة في الوثائق: {topic}",
        "sources": "المصادر",
        "thinking": "جارٍ البحث في الوثائق…",
        "out_of_scope": "غير مذكور في الوثائق",
        "injection": "خارج نطاق المساعد",
    },
}

# Trailing LRM keeps the English ellipsis after "project" in the RTL placeholder.
COMPOSER_PLACEHOLDER = "اكتب رسالتك… / Ask about the project…\u200e"
DISCLAIMER = "قد يخطئ المساعد أحياناً. تحقق من المعلومات المهمة."

MISSING_KEY_MSG = (
    "OPENAI_API_KEY is not configured. Add it to a local .env file, the environment, "
    "or the Hugging Face Space secrets, then restart the app."
)
INDEX_ERROR_MSG = (
    "The knowledge-base index is missing or out of date. "
    "Run `python scripts/build_index.py` and restart the app."
)

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$])")

# Small script (constant, no user data) that marks the message list as a
# polite live region and keeps the newest message in view.
THREAD_SCRIPT = """<script>
(function () {
  var doc = window.parent && window.parent.document ? window.parent.document : document;
  var thread = doc.querySelector('.st-key-mm_thread');
  if (thread) {
    thread.setAttribute('aria-live', 'polite');
    thread.setAttribute('aria-relevant', 'additions');
  }
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var scroller = doc.querySelector('[data-testid="stMain"]');
  [scroller, doc.scrollingElement].forEach(function (el) {
    if (el) { el.scrollTo({ top: el.scrollHeight, behavior: reduce ? 'auto' : 'smooth' }); }
  });
})();
</script><span class="mm-thread-tick" data-count="%d" hidden></span>"""


# Gives Streamlit's send button an Arabic accessible name (constant script).
COMPOSER_SCRIPT = """<script>
(function () {
  if (window.__mmComposerA11y) { return; }
  window.__mmComposerA11y = true;
  var label = function () {
    document.querySelectorAll('[data-testid="stChatInputSubmitButton"]').forEach(function (b) {
      if (b.getAttribute('aria-label') !== 'إرسال') { b.setAttribute('aria-label', 'إرسال'); b.title = 'إرسال / Send'; }
    });
  };
  label();
  new MutationObserver(label).observe(document.body, { childList: true, subtree: true });
})();
</script><span class="mm-script-holder" hidden></span>"""


class MissingAPIKey(RuntimeError):
    """Raised by get_engine() so a missing key is never cached."""


# ----------------------------------------------------------------- helpers


def md_escape(text: str) -> str:
    """Escape Markdown (and Streamlit's `$` LaTeX) in model text for st.warning."""
    return _MD_SPECIAL.sub(r"\\\1", text or "")


def trusted_markdown(text: str) -> str:
    """Chunk text is trusted repo content; only stop `$` from becoming LaTeX."""
    return (text or "").replace("$", "\\$")


def answer_markdown(text: str) -> str:
    """Model text -> safe Markdown: HTML-escaped, no LaTeX, line breaks kept.

    The answer is plain text with optional "- " bullet lines, so it renders
    as Markdown (bullets become lists) without ever being treated as HTML.
    """
    body = html.escape(text or "", quote=False).replace("$", "\\$")
    lines = body.replace("\r\n", "\n").split("\n")
    return "  \n".join(line.rstrip() for line in lines)


@st.cache_data(show_spinner=False)
def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def logo_data_uri() -> str:
    if not LOGO_PATH.exists():
        return ""
    mime = "image/svg+xml" if LOGO_PATH.suffix == ".svg" else "image/jpeg"
    return f"data:{mime};base64,{_b64(str(LOGO_PATH))}"


def rebuild_index_data() -> dict[str, Any]:
    """Fallback when index.json is missing or stale: re-embed the PDFs in data/."""
    api_key = config.get_api_key()
    if not api_key:
        raise IndexStaleError("index is stale and OPENAI_API_KEY is not configured to rebuild it")
    log.warning("rebuilding the index from %s at startup", config.DATA_DIR)
    llm = OpenAIClient(api_key=api_key, embed_model=config.EMBED_MODEL)
    return build_index(config.DATA_DIR, llm, config.EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def load_knowledge_base() -> tuple[Index, list[dict[str, Any]]]:
    rebuilt = False

    def rebuild() -> dict[str, Any]:
        nonlocal rebuilt
        rebuilt = True
        return rebuild_index_data()

    index = load_index(config.INDEX_PATH, config.DATA_DIR, rebuild=rebuild)
    if rebuilt:
        # Pre-scanned conflicts belong to the old PDFs; the generator still
        # flags conflicts it sees in the retrieved excerpts.
        log.warning("index rebuilt at startup; continuing without pre-scanned conflicts")
        return index, []
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


def submit_composer() -> None:
    text = st.session_state.get("composer")
    if isinstance(text, str) and text.strip():
        queue_question(text)


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.pop("pending_question", None)


# ------------------------------------------------------------------- theme


def theme_mode() -> str:
    """'dark' (default, matches almamlakatv.com) or 'light' once toggled."""
    return "light" if st.session_state.get("theme_choice") == "light" else "dark"


def on_theme_toggle() -> None:
    st.session_state.theme_choice = "dark" if st.session_state.get("theme_toggle") else "light"


# --------------------------------------------------------------- rendering


def render_css() -> None:
    try:
        css = CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.html(COMPOSER_SCRIPT, unsafe_allow_javascript=True)


def render_topbar(index: Index | None, has_conversation: bool) -> None:
    mode = theme_mode()
    # Keep the switch in sync with the remembered choice (dark by default).
    st.session_state.theme_toggle = mode == "dark"
    with st.container(key="mm_topbar", horizontal=True, vertical_alignment="center", gap="small"):
        logo = logo_data_uri()
        logo_img = f'<img class="mm-logo" src="{logo}" alt="المملكة">' if logo else ""
        st.markdown(
            f'<div class="mm-brand mm-theme-{mode}">{logo_img}'
            '<span class="mm-brand-text"><span class="mm-brand-name">Almamlaka TV</span>'
            '<span class="mm-brand-sub" dir="rtl" lang="ar">مساعد مبادرة التوسع الرقمي</span>'
            "</span></div>",
            unsafe_allow_html=True,
        )
        with st.container(key="mm_actions", horizontal=True, vertical_alignment="center", gap="small"):
            with st.popover("3 PDFs", icon=":material/description:", help="المصادر / Sources"):
                st.markdown(
                    '<p class="mm-label">المصادر · Knowledge base</p>',
                    unsafe_allow_html=True,
                )
                if index is None:
                    st.caption("Index not available.")
                else:
                    lines = [
                        f"- **{d['title']}**  \n  `{d['file']}` · {d['pages']} pages"
                        for d in doc_summaries(index)
                    ]
                    st.markdown("\n".join(lines))
                st.markdown(
                    '<p class="mm-note">Answers come only from these three PDFs. '
                    '<span dir="rtl" lang="ar">الإجابات مستندة فقط إلى هذه الوثائق الثلاث.</span></p>',
                    unsafe_allow_html=True,
                )
            if has_conversation:
                st.button(
                    "محادثة جديدة",
                    key="clear_chat",
                    on_click=clear_chat,
                    icon=":material/add_comment:",
                    help="New chat",
                )
            st.toggle(
                "الوضع الداكن",
                key="theme_toggle",
                on_change=on_theme_toggle,
                label_visibility="collapsed",
            )


def render_composer() -> None:
    st.chat_input(
        COMPOSER_PLACEHOLDER,
        key="composer",
        max_chars=config.MAX_INPUT_CHARS,
        on_submit=submit_composer,
        submit_mode="disable",
    )


def render_empty_state() -> None:
    title_ar = "كيف يمكنني مساعدتك اليوم؟"
    words = title_ar.split(" ")
    spans = " ".join(
        f'<span class="mm-word" style="animation-delay:{i * 70}ms">{html.escape(w)}</span>'
        for i, w in enumerate(words)
    )
    after = len(words) * 70
    with st.container(key="mm_hero"):
        st.markdown(
            f"""<div class="mm-hero-head">
<span class="mm-pill" dir="rtl" lang="ar"><span class="mm-pulse" aria-hidden="true"></span>متصل</span>
<h1 class="mm-greeting" dir="rtl" lang="ar"><span class="mm-sr-only">{html.escape(title_ar)}</span><span aria-hidden="true">{spans}</span></h1>
<p class="mm-greeting-en" lang="en" style="animation-delay:{after}ms">How can I help you today?</p>
</div>""",
            unsafe_allow_html=True,
        )
        with st.container(key="mm_composer_inline"):
            render_composer()
        st.markdown(
            f'<p class="mm-label mm-suggest-label" style="animation-delay:{after + 200}ms">'
            "اقتراحات · Suggested</p>",
            unsafe_allow_html=True,
        )
        with st.container(key="mm_chips", horizontal=True, horizontal_alignment="center", gap="small"):
            for key, question in SUGGESTIONS:
                st.button(question, key=key, on_click=queue_question, args=(question,))


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
        with st.expander(citation_label(chunk), icon=":material/description:"):
            st.markdown(trusted_markdown(chunk.text))


def render_kind_badge(kind: str, lang: str) -> None:
    lang = "ar" if lang == "ar" else "en"
    label = UI_TEXT[lang].get(kind) if kind in ("out_of_scope", "injection") else None
    if not label:
        return
    direction = "rtl" if lang == "ar" else "ltr"
    st.markdown(
        f'<span class="mm-badge mm-badge-{kind}" dir="{direction}" lang="{lang}">{html.escape(label)}</span>',
        unsafe_allow_html=True,
    )


def render_message_body(msg_idx: int, msg: dict[str, Any]) -> None:
    lang = "ar" if msg.get("lang") == "ar" else "en"
    if msg.get("role") != "assistant":
        # User text: escaped by safe_html and wrapped in the bubble.
        st.markdown(
            f'<div class="mm-user-bubble">{safe_html(msg.get("content", ""), lang)}</div>',
            unsafe_allow_html=True,
        )
        return
    direction = "rtl" if lang == "ar" else "ltr"
    with st.container(key=f"{direction}_answer_{msg_idx}"):
        render_kind_badge(msg.get("kind") or "", lang)
        st.markdown(answer_markdown(msg.get("content", "")))
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


def typing_indicator(lang: str) -> str:
    label = html.escape(UI_TEXT["ar" if lang == "ar" else "en"]["thinking"])
    return (
        f'<div class="mm-typing" role="status" aria-label="{label}">'
        '<span></span><span></span><span></span>'
        f'<span class="mm-sr-only">{label}</span></div>'
    )


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
        typing = st.empty()
        typing.markdown(typing_indicator(lang), unsafe_allow_html=True)
        result = engine.answer(question, history)
        typing.empty()
        user_msg["lang"] = result.lang
        reply = assistant_message(result)
        render_message_body(len(messages) + 1, reply)
    messages.extend([user_msg, reply])


def main() -> None:
    st.set_page_config(
        page_title="Almamlaka TV Assistant",
        page_icon=str(LOGO_JPG_PATH) if LOGO_JPG_PATH.exists() else "📺",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    render_css()
    st.session_state.setdefault("messages", [])

    try:
        index, _ = load_knowledge_base()
    except Exception as exc:  # stale index without key, rebuild/API failure, bad JSON
        log.warning("knowledge base unavailable: %s", type(exc).__name__)
        index = None

    pending = st.session_state.pop("pending_question", None)
    has_conversation = bool(st.session_state.messages) or bool(pending)

    render_topbar(index, has_conversation)
    if index is None:
        st.error(INDEX_ERROR_MSG)

    if not has_conversation:
        render_empty_state()
        return

    # Pinned composer first, so it is on screen while the answer is generated.
    with st.bottom:
        render_composer()
        st.markdown(
            f'<p class="mm-disclaimer" dir="rtl" lang="ar">{html.escape(DISCLAIMER)}</p>',
            unsafe_allow_html=True,
        )

    with st.container(key="mm_thread"):
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"], avatar=avatar_for(msg["role"])):
                render_message_body(i, msg)
        if pending:
            handle_question(pending, index)
        st.html(THREAD_SCRIPT % len(st.session_state.messages), unsafe_allow_javascript=True)



main()
