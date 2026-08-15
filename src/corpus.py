"""Corpus loading.

Documents are read verbatim — no normalisation, no whitespace stripping. The
golden set labels character offsets into these exact strings, so any cleanup
here would silently shift every span and quietly corrupt the retrieval metrics.
"""

from __future__ import annotations

from pathlib import Path

from .types import Document

SUPPORTED = {".md", ".txt", ".markdown"}


def load_corpus(corpus_dir: str | Path) -> list[Document]:
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"corpus directory not found: {root}")

    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            docs.append(
                Document(
                    id=path.name,
                    source=str(path.relative_to(root)),
                    content=path.read_text(encoding="utf-8"),
                )
            )
    if not docs:
        raise ValueError(f"no .md/.txt documents under {root}")
    return docs


def load_document_map(corpus_dir: str | Path) -> dict[str, Document]:
    return {doc.id: doc for doc in load_corpus(corpus_dir)}
