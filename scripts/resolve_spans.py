"""Check that every golden example's evidence text still exists in the corpus.

Run this after editing the corpus or the golden set. It fails loudly on drift,
which is the whole point: a golden set that silently loses examples produces
two runs that look comparable and are not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.spans import load_golden, resolve_spans  # noqa: E402
from src.config import load_config  # noqa: E402
from src.corpus import load_document_map  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local.yaml")
    args = parser.parse_args()

    app = load_config(args.config)
    documents = load_document_map(app.corpus_path)
    examples = load_golden(app.golden_path)
    resolved = resolve_spans(examples, documents)

    print(f"{len(resolved)}/{len(examples)} examples resolved cleanly\n")
    by_doc: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for ex in resolved:
        by_doc[ex.document_id] = by_doc.get(ex.document_id, 0) + 1
        by_type[ex.query_type] = by_type.get(ex.query_type, 0) + 1
        span = ex.spans[0]
        print(f"  {ex.id:<34} {ex.document_id:<22} chars {span.char_start}-{span.char_end}")

    print("\nby document:", by_doc)
    print("by query type:", by_type)


if __name__ == "__main__":
    main()
