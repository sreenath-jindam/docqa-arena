"""Chunking strategies.

Every chunker must return chunks whose ``char_start``/``char_end`` index back
into the *original* document string. The evaluation depends on this: a golden
example labels a character span of evidence, and a retrieved chunk counts as
relevant if it overlaps that span. Lose the offsets and retrieval becomes
unscoreable.

Chunk ids are content-addressed (doc id + offsets + strategy), so re-ingesting
the same corpus produces identical ids and results stay comparable across runs.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

from .types import Chunk, Document

_WORD_RE = re.compile(r"\w+")


def count_tokens(text: str) -> int:
    """Approximate token count. Swap in tiktoken if exactness ever matters."""
    return len(_WORD_RE.findall(text))


def make_chunk_id(doc_id: str, start: int, end: int, strategy: str) -> str:
    payload = f"{doc_id}|{start}|{end}|{strategy}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


class Chunker(ABC):
    name: str
    strategy: str

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        ...

    def _build(self, doc: Document, start: int, end: int) -> Chunk:
        content = doc.content[start:end]
        return Chunk(
            id=make_chunk_id(doc.id, start, end, self.strategy),
            document_id=doc.id,
            content=content,
            strategy=self.strategy,
            char_start=start,
            char_end=end,
            token_count=count_tokens(content),
        )


class FixedSizeChunker(Chunker):
    """Slide a fixed character window with overlap. The dumb baseline."""

    strategy = "fixed"

    def __init__(self, chunk_size: int = 800, overlap: int = 120):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.name = f"fixed-{chunk_size}-{overlap}"

    def chunk(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        start = 0
        while start < len(doc.content):
            end = min(start + self.chunk_size, len(doc.content))
            if doc.content[start:end].strip():
                chunks.append(self._build(doc, start, end))
            if end == len(doc.content):
                break
            start += step
        return chunks


class RecursiveChunker(Chunker):
    """Split on the largest natural boundary that fits, then fall back.

    Headers -> blank lines -> sentences -> hard cut. Offsets survive because we
    only ever record positions, never rebuild strings.
    """

    strategy = "recursive"

    SEPARATORS = [
        re.compile(r"\n(?=#{1,6}\s)"),   # markdown headings
        re.compile(r"\n\s*\n"),          # paragraphs
        re.compile(r"(?<=[.!?])\s+"),    # sentences
    ]

    def __init__(self, chunk_size: int = 800, overlap: int = 120):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.name = f"recursive-{chunk_size}-{overlap}"

    def chunk(self, doc: Document) -> list[Chunk]:
        pieces = self._split(doc.content, 0, len(doc.content), 0)
        merged = self._merge(doc.content, pieces)
        return [self._build(doc, s, e) for s, e in merged if doc.content[s:e].strip()]

    def _split(self, text: str, start: int, end: int, depth: int) -> list[tuple[int, int]]:
        if end - start <= self.chunk_size:
            return [(start, end)]
        if depth >= len(self.SEPARATORS):
            # Hard cut, still overlapping so a fact on a boundary is not lost.
            out, cursor = [], start
            step = max(1, self.chunk_size - self.overlap)
            while cursor < end:
                out.append((cursor, min(cursor + self.chunk_size, end)))
                cursor += step
            return out

        segment = text[start:end]
        boundaries = [start] + [start + m.end() for m in self.SEPARATORS[depth].finditer(segment)] + [end]
        boundaries = sorted(set(boundaries))
        if len(boundaries) <= 2:
            return self._split(text, start, end, depth + 1)

        out: list[tuple[int, int]] = []
        for lo, hi in zip(boundaries, boundaries[1:]):
            if hi > lo:
                out.extend(self._split(text, lo, hi, depth + 1))
        return out

    def _merge(self, text: str, pieces: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Glue adjacent small pieces up to chunk_size so we do not emit slivers."""
        if not pieces:
            return []
        merged: list[tuple[int, int]] = []
        cur_start, cur_end = pieces[0]
        for start, end in pieces[1:]:
            if end - cur_start <= self.chunk_size:
                cur_end = end
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        merged.append((cur_start, cur_end))
        return merged


class SemanticChunker(Chunker):
    """Group consecutive sentences while they stay semantically similar.

    Sentences are embedded once; a new chunk starts when cosine similarity to
    the running window drops below ``threshold``. This is the strategy that
    actually needs a GPU on a real corpus, which is part of why the sweep runs
    on Kaggle.
    """

    strategy = "semantic"

    def __init__(self, embedder, threshold: float = 0.62, chunk_size: int = 1200, min_size: int = 200):
        self.embedder = embedder
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.min_size = min_size
        self.name = f"semantic-{threshold}"

    def chunk(self, doc: Document) -> list[Chunk]:
        sentences = _sentence_spans(doc.content)
        if len(sentences) <= 1:
            return [self._build(doc, 0, len(doc.content))] if doc.content.strip() else []

        texts = [doc.content[s:e] for s, e in sentences]
        vectors = self.embedder.embed(texts)

        groups: list[tuple[int, int]] = []
        start, end = sentences[0]
        window = vectors[0]
        count = 1

        for idx in range(1, len(sentences)):
            sim = _cosine(window, vectors[idx])
            s, e = sentences[idx]
            too_long = (e - start) > self.chunk_size
            if (sim < self.threshold and (end - start) >= self.min_size) or too_long:
                groups.append((start, end))
                start, end, window, count = s, e, vectors[idx], 1
            else:
                end = e
                # running mean keeps the window representative of the whole chunk
                window = [(w * count + v) / (count + 1) for w, v in zip(window, vectors[idx])]
                count += 1
        groups.append((start, end))
        return [self._build(doc, s, e) for s, e in groups if doc.content[s:e].strip()]


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, cursor = [], 0
    for match in _SENTENCE_END.finditer(text):
        end = match.start()
        if end > cursor:
            spans.append((cursor, end))
        cursor = match.end()
    if cursor < len(text):
        spans.append((cursor, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def build_chunker(name: str, embedder=None, chunk_size: int = 800, overlap: int = 120) -> Chunker:
    if name == "fixed":
        return FixedSizeChunker(chunk_size, overlap)
    if name == "recursive":
        return RecursiveChunker(chunk_size, overlap)
    if name == "semantic":
        if embedder is None:
            raise ValueError("semantic chunking needs an embedder")
        return SemanticChunker(embedder)
    raise ValueError(f"unknown chunker: {name}")
