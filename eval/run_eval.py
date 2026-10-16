"""End-to-end evaluation against the real OpenAI API.

Usage: python eval/run_eval.py [--no-judge] [--only ID[,ID...]]

Runs every case in eval/questions.json through ChatEngine (built from the
committed index/ artifacts), applies deterministic checks to each turn plus an
LLM judge against `expected_behavior`, and writes eval/results.json and
eval/results.md. With --only, results are printed but the files are not
rewritten (they always describe a full run).

Requires OPENAI_API_KEY in the environment or .env (loaded via python-dotenv,
never printed). Exit code is always 0.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag import config  # noqa: E402
from rag.chat import ChatEngine, ChatResult  # noqa: E402
from rag.conflicts import load_conflicts  # noqa: E402
from rag.formatting import citation_label, normalise_digits  # noqa: E402
from rag.index import load_index  # noqa: E402
from rag.llm import OpenAIClient  # noqa: E402
from rag.retrieve import retrieve  # noqa: E402

EVAL_DIR = REPO_ROOT / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.json"
RESULTS_JSON = EVAL_DIR / "results.json"
RESULTS_MD = EVAL_DIR / "results.md"
# gpt-4o-mini (the planned judge) repeatedly contradicted its own analysis
# (e.g. failing a reply for an item it had just called optional), so the
# judge defaults to gpt-4o; override with EVAL_JUDGE_MODEL.
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o")

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    # analysis comes first so the verdict is produced after the reasoning
    "required": ["analysis", "pass", "reason"],
    "properties": {
        "analysis": {"type": "string"},
        "pass": {"type": "boolean"},
        "reason": {"type": "string"},
    },
}

JUDGE_SYSTEM = """You grade one reply of a document-grounded chatbot against the expected behaviour written by a test author.

You receive the conversation so far, the latest user message, the bot's reply (answer text, reply kind, cited sources and flagged conflicts shown to the user) and the expected behaviour.

Rules:
- The expected behaviour is the ground truth about the documents. Do not use your own knowledge to decide whether a fact is correct.
- Essential points are what the expected behaviour says the reply must do. Items introduced with "may", "e.g." or "for example", and details in parentheses, are optional.
- pass=true if the reply satisfies every essential point. Wording, order, paraphrase, extra correct detail, number formatting and transliteration of names do not matter.
- pass=false if an essential point is missing or contradicted, if the reply invents facts the expected behaviour forbids, or if it is in the wrong language when a language is required.
- Flagged conflicts and cited sources are displayed with the answer and count as part of the reply. Presenting all values with their sources (or saying one is a later revision) counts as flagging a conflict neutrally. "Cites X" means X is among the cited sources; extra sources are fine. Which or how many sources are cited is checked separately, so never fail a reply for that.
- A reply that states the required fact in other words (or in Arabic) satisfies the point; do not fail it for not using the exact phrasing.
- Reply kinds "out_of_scope", "injection" and "greeting" are fixed polite templates; judge whether such a template is an acceptable reply for the expected behaviour.
- The bot is not allowed to use outside knowledge; do not penalise it for not knowing things the expected behaviour says are not in the documents.
In "analysis", check each essential point against the reply, quoting the relevant words of the reply. Then set pass, and give a one-sentence reason. Return JSON only."""


# ------------------------------------------------------------------ checks


def norm(text: str) -> str:
    """Lowercase, ASCII digits, thousands separators removed."""
    text = normalise_digits(text or "").lower()
    return text.replace(",", "").replace("٬", "")


def _contains_term(haystack: str, term: str) -> bool:
    """Term match that does not split words at alphanumeric edges
    (so 'lg' does not match 'algorithm' and 'law no' does not match 'law not')."""
    t = norm(term)
    left = r"(?<!\w)" if t[:1].isalnum() else ""
    right = r"(?!\w)" if t[-1:].isalnum() else ""
    return re.search(left + re.escape(t) + right, haystack) is not None


def result_text(res: ChatResult) -> str:
    """All text shown to the user: answer + conflict claims."""
    parts = [res.answer]
    for c in res.conflicts:
        parts.append(c.get("topic", ""))
        parts.extend(str(cl.get("value", "")) for cl in c.get("claims", []))
    return "\n".join(parts)


def apply_checks(checks: dict[str, Any], res: ChatResult) -> list[str]:
    failures: list[str] = []
    text = norm(result_text(res))

    kind = checks.get("expect_kind")
    if kind is not None:
        allowed = [kind] if isinstance(kind, str) else list(kind)
        if res.kind not in allowed:
            failures.append(f"kind={res.kind} (expected {'/'.join(allowed)})")
    lang = checks.get("expect_lang")
    if lang and res.lang != lang:
        failures.append(f"lang={res.lang} (expected {lang})")
    for term in checks.get("must_contain_all", []):
        if norm(term) not in text:
            failures.append(f"missing '{term}'")
    any_terms = checks.get("must_contain_any", [])
    if any_terms and not any(norm(t) in text for t in any_terms):
        failures.append(f"missing any of {any_terms}")
    for term in checks.get("must_not_contain", []):
        if _contains_term(text, term):
            failures.append(f"contains forbidden '{term}'")
    if "expect_conflict" in checks:
        has = bool(res.conflicts)
        if has != bool(checks["expect_conflict"]):
            failures.append(f"conflict flagged={has} (expected {checks['expect_conflict']})")
    cited_docs = {c.doc_file for c in res.citations}
    for doc in checks.get("must_cite_docs", []):
        if doc not in cited_docs:
            failures.append(f"does not cite {doc}")
    min_cit = checks.get("min_citations")
    if min_cit is not None and len(res.citations) < int(min_cit):
        failures.append(f"{len(res.citations)} citations (< {min_cit})")
    return failures


# ------------------------------------------------------------------- judge


def judge(llm: OpenAIClient, history: list[dict[str, str]], user: str, res: ChatResult, expected: str) -> dict[str, Any]:
    convo = "\n".join(f"{t['role']}: {t['content']}" for t in history) or "(none)"
    reply = {
        "kind": res.kind,
        "language": res.lang,
        "answer": res.answer,
        "cited_sources": [citation_label(c) for c in res.citations],
        "flagged_conflicts": [
            {"topic": c["topic"], "values": [f"{cl['value']} ({cl['label']})" for cl in c["claims"]]}
            for c in res.conflicts
        ],
    }
    content = (
        f"Conversation so far:\n{convo}\n\n"
        f"Latest user message:\n{user}\n\n"
        f"Bot reply:\n{json.dumps(reply, ensure_ascii=False, indent=1)}\n\n"
        f"Expected behaviour:\n{expected}"
    )
    try:
        out = llm.chat_json(
            JUDGE_MODEL,
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": content}],
            "judge",
            JUDGE_SCHEMA,
        )
        return {
            "pass": bool(out.get("pass")),
            "reason": str(out.get("reason", "")),
            "analysis": str(out.get("analysis", "")),
        }
    except Exception as exc:  # judge failure counts as a failed turn
        return {"pass": False, "reason": f"judge error: {type(exc).__name__}"}


# ------------------------------------------------------------------ runner


def probe_dense_max(engine: ChatEngine, llm: OpenAIClient, res: ChatResult, user: str) -> float | None:
    """dense_max for turns the engine short-circuited before retrieval
    (greeting/injection), so threshold tuning can see off-topic scores."""
    if res.debug.get("dense_max") is not None:
        return res.debug["dense_max"]
    query = res.debug.get("standalone_query") or user
    try:
        vec = llm.embed([query])[0]
    except Exception:
        return None
    return round(retrieve(engine.index, query, vec, engine.conflicts).dense_max, 4)


def run_case(engine: ChatEngine, llm: OpenAIClient, case: dict[str, Any], use_judge: bool) -> dict[str, Any]:
    history: list[dict[str, str]] = []
    turns_out = []
    for turn in case["turns"]:
        user = turn["user"]
        started = time.time()
        res = engine.answer(user, history)
        elapsed = round(time.time() - started, 2)
        failures = apply_checks(turn.get("checks", {}), res)
        verdict = judge(llm, history, user, res, turn["expected_behavior"]) if use_judge else None
        passed = not failures and (verdict is None or verdict["pass"])
        turns_out.append(
            {
                "user": user,
                "expected_behavior": turn["expected_behavior"],
                "kind": res.kind,
                "lang": res.lang,
                "answer": res.answer,
                "citations": [{"chunk_id": c.chunk_id, "label": citation_label(c)} for c in res.citations],
                "conflicts": res.conflicts,
                "intent": res.debug.get("intent"),
                "standalone_query": res.debug.get("standalone_query"),
                "dense_max": res.debug.get("dense_max"),
                "probe_dense_max": probe_dense_max(engine, llm, res, user),
                "retrieved_ids": res.debug.get("retrieved_ids", []),
                "model_in_scope": res.debug.get("model_in_scope"),
                "check_failures": failures,
                "judge": verdict,
                "passed": passed,
                "seconds": elapsed,
            }
        )
        history.append({"role": "user", "content": user, "lang": res.lang})
        history.append({"role": "assistant", "content": res.answer, "lang": res.lang})
    return {
        "id": case["id"],
        "category": case.get("category", ""),
        "passed": all(t["passed"] for t in turns_out),
        "turns": turns_out,
    }


# ----------------------------------------------------------------- reports


def _cell(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def _notes(case: dict[str, Any]) -> str:
    notes = []
    multi = len(case["turns"]) > 1
    for i, t in enumerate(case["turns"], start=1):
        prefix = f"T{i}: " if multi else ""
        score = t["dense_max"] if t["dense_max"] is not None else t["probe_dense_max"]
        info = f"kind={t['kind']}, lang={t['lang']}, dense_max={score}"
        if t["conflicts"]:
            info += f", conflicts={len(t['conflicts'])}"
        problems = list(t["check_failures"])
        if t["judge"] is not None and not t["judge"]["pass"]:
            problems.append(f"judge: {t['judge']['reason']}")
        if problems:
            info += "; FAIL: " + "; ".join(problems)
        notes.append(prefix + info)
    return " / ".join(notes)


def score_ranges(cases: list[dict[str, Any]]) -> dict[str, list[float]]:
    """dense_max per reply kind: answered, out_of_scope (gate or generator
    declined) and injection/greeting (short-circuited; probe score)."""
    buckets: dict[str, list[float]] = {"answered": [], "out_of_scope": [], "injection_greeting": []}
    for case in cases:
        for t in case["turns"]:
            score = t["dense_max"] if t["dense_max"] is not None else t["probe_dense_max"]
            if score is None:
                continue
            if t["kind"] == "answer":
                buckets["answered"].append(score)
            elif t["kind"] == "out_of_scope":
                buckets["out_of_scope"].append(score)
            elif t["kind"] in ("injection", "greeting"):
                buckets["injection_greeting"].append(score)
    return {k: sorted(v) for k, v in buckets.items()}


def write_reports(cases: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    passed = sum(c["passed"] for c in cases)
    RESULTS_JSON.write_text(
        json.dumps({"meta": meta, "passed": passed, "total": len(cases), "cases": cases}, ensure_ascii=False, indent=1)
        + "\n",
        encoding="utf-8",
    )
    ranges = score_ranges(cases)

    def fmt(values: list[float]) -> str:
        return f"{values[0]:.3f}–{values[-1]:.3f} (n={len(values)})" if values else "n/a"

    lines = [
        "# Evaluation results",
        "",
        f"- Date: {meta['date']}",
        f"- Models: chat `{meta['chat_model']}`, condense `{meta['condense_model']}`, "
        f"embeddings `{meta['embed_model']}`, judge `{meta['judge_model'] or 'disabled'}`",
        f"- Relevance threshold (dense_max): {meta['threshold']}; top_k {meta['top_k']}; "
        f"max context chunks {meta['max_context_chunks']}",
        f"- dense_max by reply kind: answered {fmt(ranges['answered'])}; "
        f"out_of_scope {fmt(ranges['out_of_scope'])}; "
        f"injection/greeting (probe score, never reaches the gate) {fmt(ranges['injection_greeting'])}",
        "",
        f"**{passed}/{len(cases)} passed**",
        "",
        "Generated by `python eval/run_eval.py` (cases in `eval/questions.json`, full answers in `eval/results.json`). "
        "A turn passes when all deterministic checks pass and the judge agrees with the expected behaviour; "
        "a case passes when all its turns pass.",
        "",
        "| # | Question | Expected behavior | Result | Notes |",
        "|---|---|---|---|---|",
    ]
    for i, case in enumerate(cases, start=1):
        question = " → ".join(t["user"] for t in case["turns"])
        expected = " → ".join(t["expected_behavior"] for t in case["turns"])
        result = "PASS" if case["passed"] else "FAIL"
        lines.append(
            f"| {i} | {_cell(question)} | {_cell(expected)} | {result} | {_cell(case['id'] + ': ' + _notes(case))} |"
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge (deterministic checks only)")
    parser.add_argument("--only", help="comma-separated case id(s) to run; results files are not rewritten")
    args = parser.parse_args(argv)

    api_key = config.get_api_key()
    if not api_key:
        print("OPENAI_API_KEY is not configured (set it in .env or the environment).", file=sys.stderr)
        return 0

    cases = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in wanted]
        if not cases:
            print(f"No case matches --only {args.only}", file=sys.stderr)
            return 0

    index = load_index(config.INDEX_PATH, config.DATA_DIR)
    conflicts = load_conflicts(config.CONFLICTS_PATH, valid_ids=index.by_id.keys())
    llm = OpenAIClient(api_key=api_key, embed_model=config.EMBED_MODEL)
    engine = ChatEngine(index, conflicts, llm)

    results = []
    for case in cases:
        out = run_case(engine, llm, case, use_judge=not args.no_judge)
        results.append(out)
        print(f"[{'PASS' if out['passed'] else 'FAIL'}] {out['id']}: {_notes(out)}", flush=True)

    passed = sum(c["passed"] for c in results)
    print(f"\n{passed}/{len(results)} passed")
    if args.only:
        return 0

    meta = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "chat_model": engine.chat_model,
        "condense_model": engine.condense_model,
        "embed_model": config.EMBED_MODEL,
        "judge_model": None if args.no_judge else JUDGE_MODEL,
        "threshold": engine.threshold,
        "top_k": engine.top_k,
        "max_context_chunks": engine.max_context_chunks,
    }
    write_reports(results, meta)
    print(f"Wrote {RESULTS_MD.relative_to(REPO_ROOT)} and {RESULTS_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
