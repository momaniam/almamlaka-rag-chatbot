"""Query pipeline: hygiene -> language -> condense/intent -> retrieval ->
relevance gate -> structured generation -> citation/conflict validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rag import config
from rag.formatting import citation_label
from rag.index import Index
from rag.ingest import Chunk
from rag.language import detect_language
from rag.llm import LLM
from rag.prompts import (
    CONDENSE_SCHEMA,
    GENERATION_SCHEMA,
    build_condense_messages,
    build_generation_messages,
    describe_corpus,
    template,
)
from rag.retrieve import retrieve

log = logging.getLogger(__name__)

LANGUAGE_NAMES = {"en": "English", "ar": "Arabic"}


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    lang: str | None = None


@dataclass
class ChatResult:
    kind: str  # answer | out_of_scope | injection | greeting | error
    lang: str  # en | ar
    answer: str
    citations: list[Chunk] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


def _as_turn(item: Turn | Mapping[str, Any]) -> Turn:
    if isinstance(item, Turn):
        return item
    return Turn(
        role=str(item.get("role", "user")),
        content=str(item.get("content", "") or ""),
        lang=item.get("lang"),
    )


class ChatEngine:
    def __init__(
        self,
        index: Index,
        conflicts: Sequence[dict[str, Any]],
        llm: LLM,
        chat_model: str = config.CHAT_MODEL,
        condense_model: str = config.CONDENSE_MODEL,
        threshold: float = config.RELEVANCE_THRESHOLD,
        top_k: int = config.TOP_K,
        max_context_chunks: int = config.MAX_CONTEXT_CHUNKS,
        history_messages: int = config.HISTORY_MESSAGES,
        max_input_chars: int = config.MAX_INPUT_CHARS,
    ) -> None:
        self.index = index
        self.conflicts = list(conflicts)
        self.llm = llm
        self.chat_model = chat_model
        self.condense_model = condense_model
        self.threshold = threshold
        self.top_k = top_k
        self.max_context_chunks = max_context_chunks
        self.history_messages = history_messages
        self.max_input_chars = max_input_chars
        self.corpus_description = describe_corpus(index.chunks)

    # ------------------------------------------------------------ helpers

    def _template(self, kind: str, lang: str, debug: dict[str, Any]) -> ChatResult:
        return ChatResult(kind=kind, lang=lang, answer=template(kind, lang), debug=debug)

    @staticmethod
    def _previous_user_lang(history: list[Turn]) -> str | None:
        for turn in reversed(history):
            if turn.role == "user":
                if turn.lang in ("en", "ar"):
                    return turn.lang
                return detect_language(turn.content, None)
        return None

    # ------------------------------------------------------------- answer

    def answer(self, user_msg: str, history: Sequence[Turn | Mapping[str, Any]] = ()) -> ChatResult:
        turns = [_as_turn(t) for t in history]
        # a. input hygiene + language
        msg = (user_msg or "").strip()[: self.max_input_chars]
        lang = detect_language(msg, self._previous_user_lang(turns) or "en")
        debug: dict[str, Any] = {
            "intent": None,
            "standalone_query": None,
            "dense_max": None,
            "retrieved_ids": [],
        }
        if not msg:
            debug["intent"] = "greeting"
            return self._template("greeting", lang, debug)

        window = turns[-self.history_messages :] if self.history_messages > 0 else []
        recent = [{"role": t.role, "content": t.content} for t in window]

        # b. condense + intent
        try:
            condensed = self.llm.chat_json(
                self.condense_model,
                build_condense_messages(recent, msg, self.corpus_description),
                "condense",
                CONDENSE_SCHEMA,
            )
        except Exception as exc:
            log.warning("condense step failed: %s", type(exc).__name__)
            return self._template("error", lang, debug)
        intent = condensed.get("intent")
        standalone = (condensed.get("standalone_query") or "").strip() or msg
        debug["intent"] = intent
        debug["standalone_query"] = standalone
        if intent == "greeting":
            return self._template("greeting", lang, debug)
        if intent != "in_scope_question":
            return self._template("injection", lang, debug)

        # c. retrieval
        try:
            query_vec = self.llm.embed([standalone])[0]
        except Exception as exc:
            log.warning("query embedding failed: %s", type(exc).__name__)
            return self._template("error", lang, debug)
        result = retrieve(
            self.index,
            standalone,
            query_vec,
            self.conflicts,
            k=self.top_k,
            max_chunks=self.max_context_chunks,
        )
        debug["dense_max"] = round(result.dense_max, 4)
        debug["retrieved_ids"] = [c.chunk_id for c in result.chunks]

        # d. relevance gate
        if result.dense_max < self.threshold:
            return self._template("out_of_scope", lang, debug)

        # e. generation
        labels = {c.chunk_id: citation_label(c) for c in result.chunks}
        messages = build_generation_messages(
            result.chunks,
            labels,
            result.conflicts,
            recent,
            standalone,
            msg,
            LANGUAGE_NAMES[lang],
        )
        try:
            gen = self.llm.chat_json(self.chat_model, messages, "grounded_answer", GENERATION_SCHEMA)
        except Exception as exc:
            log.warning("generation step failed: %s", type(exc).__name__)
            return self._template("error", lang, debug)

        # f. validation
        retrieved = {c.chunk_id: c for c in result.chunks}
        cited: list[str] = []
        for cid in gen.get("citations") or []:
            if cid in retrieved and cid not in cited:
                cited.append(cid)

        conflicts_out: list[dict[str, Any]] = []
        for conflict in gen.get("conflicts") or []:
            claims = [
                {
                    "value": str(cl.get("value", "")),
                    "chunk_id": cl.get("chunk_id"),
                    "label": labels[cl.get("chunk_id")],
                }
                for cl in conflict.get("claims") or []
                if cl.get("chunk_id") in retrieved
            ]
            if len(claims) < 2:
                continue
            conflicts_out.append({"topic": str(conflict.get("topic", "")), "claims": claims})
            for cl in claims:
                if cl["chunk_id"] not in cited:
                    cited.append(cl["chunk_id"])

        answer_text = (gen.get("answer") or "").strip()
        debug["model_in_scope"] = bool(gen.get("in_scope"))
        if not gen.get("in_scope") or not cited or not answer_text:
            return self._template("out_of_scope", lang, debug)

        return ChatResult(
            kind="answer",
            lang=lang,
            answer=answer_text,
            citations=[retrieved[cid] for cid in cited],
            conflicts=conflicts_out,
            debug=debug,
        )
