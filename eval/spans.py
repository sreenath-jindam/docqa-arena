"""Golden set loading and span resolution.

The golden CSV stores an ``evidence_text`` snippet rather than raw character
offsets. Hand-labelling offsets is miserable and breaks the moment a document
is edited; a verbatim snippet is something a human can actually write and
verify, and it is resolved to offsets here by exact string search.

Resolved offsets then become the ground truth: a retrieved chunk is *relevant*
if its character range overlaps an evidence span. That definition is what makes
the metric independent of chunking strategy — the same label works whether the
chunker produced 200-character slivers or 1,200-character sections.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.types import Chunk, GoldenExample, RelevantSpan

REQUIRED_COLUMNS = {"id", "document_id", "query", "expected_answer", "evidence_text"}


def load_golden(csv_path: str | Path) -> list[GoldenExample]:
    path = Path(csv_path)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        examples = []
        for row in reader:
            if not row.get("id", "").strip():
                continue
            examples.append(
                GoldenExample(
                    id=row["id"].strip(),
                    document_id=row["document_id"].strip(),
                    query=row["query"].strip(),
                    query_type=(row.get("query_type") or "factual").strip(),
                    expected_answer=row["expected_answer"].strip(),
                    evidence_text=row["evidence_text"],
                )
            )
    return examples


def resolve_spans(examples: list[GoldenExample], documents: dict) -> list[GoldenExample]:
    """Turn ``evidence_text`` into character spans. Fails loudly, on purpose.

    A snippet that no longer appears in its document means the corpus changed
    under the golden set. Silently dropping that example would quietly shrink
    the benchmark and make two runs incomparable, so it raises instead.
    """
    resolved = []
    problems = []

    for example in examples:
        doc = documents.get(example.document_id)
        if doc is None:
            problems.append(f"{example.id}: no document '{example.document_id}'")
            continue

        spans = []
        # A multi-hop question needs evidence from two places. Snippets are
        # separated by '||' so one row can still describe one question.
        for needle in [n.strip() for n in example.evidence_text.split("||") if n.strip()]:
            start = doc.content.find(needle)
            if start == -1:
                normalized = " ".join(needle.split())
                start = _find_normalized(doc.content, normalized)
                if start == -1:
                    problems.append(f"{example.id}: evidence not found in {example.document_id}: {needle[:60]!r}")
                    continue
                end = start + len(normalized)
            else:
                end = start + len(needle)
            spans.append(RelevantSpan(example.document_id, start, end))

        if not spans:
            continue

        resolved.append(
            GoldenExample(
                id=example.id,
                document_id=example.document_id,
                query=example.query,
                query_type=example.query_type,
                expected_answer=example.expected_answer,
                evidence_text=example.evidence_text,
                spans=spans,
            )
        )

    if problems:
        raise ValueError("golden set could not be resolved:\n  " + "\n  ".join(problems))
    return resolved


def _find_normalized(haystack: str, needle: str) -> int:
    """Fallback for snippets whose whitespace was mangled by a spreadsheet."""
    flat = " ".join(haystack.split())
    idx = flat.find(needle)
    if idx == -1:
        return -1
    # Map the position in the flattened string back to the original.
    consumed, original_pos, in_space = 0, 0, False
    for pos, ch in enumerate(haystack):
        if ch.isspace():
            if not in_space:
                consumed += 1
                in_space = True
        else:
            in_space = False
            consumed += 1
        if consumed > idx:
            original_pos = pos
            break
    return original_pos


def ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and a_end > b_start


def relevant_chunk_ids(example: GoldenExample, chunks: list[Chunk]) -> list[str]:
    ids = []
    for span in example.spans:
        for chunk in chunks:
            if chunk.document_id != span.document_id:
                continue
            if ranges_overlap(span.char_start, span.char_end, chunk.char_start, chunk.char_end):
                ids.append(chunk.id)
    return sorted(set(ids))
