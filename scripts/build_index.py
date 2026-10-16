"""Build index/index.json (+ index/conflicts.json) from the PDFs in data/.

Usage: python scripts/build_index.py [--skip-conflicts]
Requires OPENAI_API_KEY in the environment or .env (never printed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag import config  # noqa: E402
from rag.conflicts import scan_conflicts  # noqa: E402
from rag.index import Index, build_index, write_json  # noqa: E402
from rag.llm import OpenAIClient  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-conflicts", action="store_true", help="only rebuild index.json")
    args = parser.parse_args(argv)

    api_key = config.get_api_key()
    if not api_key:
        print("OPENAI_API_KEY is not configured (set it in .env or the environment).", file=sys.stderr)
        return 1
    llm = OpenAIClient(api_key=api_key, embed_model=config.EMBED_MODEL)

    data = build_index(config.DATA_DIR, llm, config.EMBED_MODEL)
    write_json(config.INDEX_PATH, data)
    index = Index.from_data(data)
    print(f"Wrote {config.INDEX_PATH.relative_to(REPO_ROOT)}: {len(index.chunks)} chunks "
          f"({config.EMBED_MODEL}, dim {index.matrix.shape[1] if index.chunks else 0})")
    for c in index.chunks:
        print(f"  {c.chunk_id:<34} p.{c.page_start}-{c.page_end}  §{c.section_num} {c.section_title}")

    if args.skip_conflicts:
        return 0

    conflicts = scan_conflicts(index.chunks, llm, config.CHAT_MODEL)
    write_json(config.CONFLICTS_PATH, conflicts)
    print(f"Wrote {config.CONFLICTS_PATH.relative_to(REPO_ROOT)}: {len(conflicts['facts'])} facts, "
          f"{len(conflicts['conflicts'])} conflicts ({config.CHAT_MODEL})")
    for c in conflicts["conflicts"]:
        ids = ", ".join(cl["chunk_id"] for cl in c["claims"])
        print(f"  [{c['kind']}] {c['topic']}  <- {ids}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
