"""Retrieval.

Three retrievers, one interface. BM25 is implemented here rather than pulled in
as a dependency because it is forty lines, it makes the scoring testable, and
"I know what BM25 does" is worth more in an interview than "I imported it".

Hybrid fuses the two ranked lists with Reciprocal Rank Fusion. RRF is chosen
over score-weighted fusion deliberately: cosine similarity and BM25 scores live
on incompatible scales, and normalising them introduces a knob that would need
its own sweep. RRF only uses rank, so there is nothing to tune per corpus.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter

from .embeddings import Embedder
from .store import ChromaStore
from .types import Chunk, RetrievalResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will", "with",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class Retriever(ABC):
    name: str

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        ...


class VectorRetriever(Retriever):
    name = "vector"

    def __init__(self, store: ChromaStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        vector = self.embedder.embed_query(query)
        return self.store.query(vector, k)


class BM25Retriever(Retriever):
    """Okapi BM25 over the same chunk set the vector index holds."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c.content) for c in chunks]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avg_len = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]

        df: Counter = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        n = len(chunks)
        # Robertson/Sparck-Jones idf with the +0.5 smoothing, floored at a small
        # positive value so a term appearing in every chunk cannot score negative.
        self.idf = {
            term: max(math.log((n - count + 0.5) / (count + 0.5) + 1.0), 1e-6)
            for term, count in df.items()
        }

    def score(self, query_tokens: list[str], idx: int) -> float:
        if self.avg_len == 0:
            return 0.0
        tf = self.term_freqs[idx]
        length = self.doc_lens[idx]
        total = 0.0
        for term in query_tokens:
            freq = tf.get(term)
            if not freq:
                continue
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * length / self.avg_len)
            total += self.idf.get(term, 0.0) * numerator / denominator
        return total

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        query_tokens = tokenize(query)
        scored = [(self.score(query_tokens, i), i) for i in range(len(self.chunks))]
        scored = [pair for pair in scored if pair[0] > 0]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))

        results = []
        for rank, (score, idx) in enumerate(scored[:k], start=1):
            chunk = self.chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    rank=rank,
                    score=float(score),
                    retriever_name="bm25",
                )
            )
        return results


class HybridRetriever(Retriever):
    """Reciprocal Rank Fusion over vector and BM25 result lists.

    RRF scores are computed from rank alone, so a rank-1 result always scores
    ~1/61 whether it is a perfect match or completely unrelated. That makes the
    fused score useless as a relevance threshold. To keep thresholding possible,
    the underlying cosine similarity is preserved on each result in
    `semantic_score`, and the relevance gate reads that instead.
    """

    name = "hybrid"

    def __init__(self, vector: VectorRetriever, bm25: BM25Retriever, rrf_k: int = 60, fetch_multiplier: int = 3):
        self.vector = vector
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.fetch_multiplier = fetch_multiplier

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        fetch = k * self.fetch_multiplier
        lists = [self.vector.retrieve(query, fetch), self.bm25.retrieve(query, fetch)]

        fused: dict[str, float] = {}
        best: dict[str, RetrievalResult] = {}
        semantic: dict[str, float] = {}
        for results in lists:
            for result in results:
                fused[result.chunk_id] = fused.get(result.chunk_id, 0.0) + 1.0 / (self.rrf_k + result.rank)
                best.setdefault(result.chunk_id, result)
                if result.retriever_name == "vector":
                    semantic[result.chunk_id] = result.score

        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        out = []
        for rank, (chunk_id, score) in enumerate(ordered[:k], start=1):
            base = best[chunk_id]
            out.append(
                RetrievalResult(
                    chunk_id=base.chunk_id,
                    document_id=base.document_id,
                    content=base.content,
                    char_start=base.char_start,
                    char_end=base.char_end,
                    rank=rank,
                    score=float(score),
                    retriever_name="hybrid",
                    semantic_score=semantic.get(chunk_id),
                )
            )
        return out


def build_retriever(name: str, store: ChromaStore, embedder: Embedder, chunks: list[Chunk] | None = None) -> Retriever:
    if name == "vector":
        return VectorRetriever(store, embedder)
    if name == "bm25":
        return BM25Retriever(chunks if chunks is not None else store.all_chunks())
    if name == "hybrid":
        return HybridRetriever(
            VectorRetriever(store, embedder),
            BM25Retriever(chunks if chunks is not None else store.all_chunks()),
        )
    raise ValueError(f"unknown retriever: {name}")
