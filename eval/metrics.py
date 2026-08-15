"""Retrieval metrics.

These are scored against chunk ids, never against answer text. That separation
is the point of the whole harness: precision/recall/MRR/nDCG say whether the
right source text reached the generator, and correctness/faithfulness say what
the generator did with it. A single end-to-end accuracy number cannot tell you
which of the two failed.

Note on precision@k: with one evidence span and k=5, the ceiling is 0.2 or so.
Low precision here is not a bug, and MRR and nDCG are the numbers to compare.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for cid in top if cid in relevant_set) / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant_set) / len(relevant_set)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    for i, cid in enumerate(retrieved, start=1):
        if cid in relevant_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, cid in enumerate(retrieved[:k])
        if cid in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def hit_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Did any correct chunk make it into the top k? The blunt sanity check."""
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


@dataclass
class RetrievalEvaluation:
    example_id: str
    query: str
    k: int
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    hit_at_k: float
    retrieval_latency_ms: float
    rerank_latency_ms: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def evaluate_retrieval(
    example_id: str,
    query: str,
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
    retrieval_latency_ms: float,
    rerank_latency_ms: float = 0.0,
) -> RetrievalEvaluation:
    return RetrievalEvaluation(
        example_id=example_id,
        query=query,
        k=k,
        retrieved_chunk_ids=retrieved_chunk_ids,
        relevant_chunk_ids=relevant_chunk_ids,
        precision_at_k=precision_at_k(retrieved_chunk_ids, relevant_chunk_ids, k),
        recall_at_k=recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, k),
        reciprocal_rank=reciprocal_rank(retrieved_chunk_ids, relevant_chunk_ids),
        ndcg_at_k=ndcg_at_k(retrieved_chunk_ids, relevant_chunk_ids, k),
        hit_at_k=hit_at_k(retrieved_chunk_ids, relevant_chunk_ids, k),
        retrieval_latency_ms=retrieval_latency_ms,
        rerank_latency_ms=rerank_latency_ms,
    )


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. No numpy dependency, no interpolation surprises."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return float(ordered[idx])


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
