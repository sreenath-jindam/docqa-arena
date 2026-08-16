"""Shared data types.

Every stage in the pipeline speaks in these objects, so a chunker, retriever or
reranker can be swapped without touching anything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    source: str
    content: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    content: str
    strategy: str
    char_start: int
    char_end: int
    token_count: int

    def to_metadata(self) -> dict[str, Any]:
        # Chroma metadata values must be str/int/float/bool.
        return {
            "document_id": self.document_id,
            "strategy": self.strategy,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    document_id: str
    content: str
    char_start: int
    char_end: int
    rank: int
    score: float
    retriever_name: str
    # Cosine similarity against the query, when a dense retriever produced or
    # scored this chunk. Unlike `score` (which may be an RRF or cross-encoder
    # value) this is comparable across queries, so it is what the relevance
    # gate thresholds on. None for BM25-only results.
    semantic_score: float | None = None

    def dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelevantSpan:
    """A character range in a source document that answers a golden question."""

    document_id: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class GoldenExample:
    id: str
    document_id: str
    query: str
    query_type: str
    expected_answer: str
    evidence_text: str
    spans: list[RelevantSpan] = field(default_factory=list)


@dataclass
class GenerationResult:
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float


@dataclass
class JudgeResult:
    correctness: str  # correct | partially-correct | incorrect
    faithfulness: str  # faithful | partially-faithful | unfaithful
    explanation: str


# Ordinal scales. Keeping the labels categorical and the numbers in one place
# means a change of scale is a one-line edit, not a search across the codebase.
CORRECTNESS_SCORE = {"correct": 1.0, "partially-correct": 0.5, "incorrect": 0.0}
FAITHFULNESS_SCORE = {"faithful": 1.0, "partially-faithful": 0.5, "unfaithful": 0.0}
