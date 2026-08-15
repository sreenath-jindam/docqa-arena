"""Reranking.

A separate stage rather than something folded into the retriever, so its cost
and its benefit each get their own column in the leaderboard. Default is off.

The cross-encoder scores every (query, chunk) pair jointly, which is why it is
better than bi-encoder cosine and also why it is roughly an order of magnitude
slower: N forward passes instead of one. On 30 questions x 18 configs x 20
candidates that is 10,800 forward passes, which is the practical reason the
sweep runs on a GPU.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .config import resolve_device
from .types import RetrievalResult


class Reranker(ABC):
    name: str

    @abstractmethod
    def rerank(self, query: str, results: list[RetrievalResult], k: int) -> list[RetrievalResult]:
        ...


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "auto", batch_size: int = 32):
        self.name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.name, device=self.device, max_length=512)
        return self._model

    def rerank(self, query: str, results: list[RetrievalResult], k: int) -> list[RetrievalResult]:
        if not results:
            return []
        pairs = [(query, r.content) for r in results]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        ranked = sorted(zip(results, scores), key=lambda pair: -float(pair[1]))
        out = []
        for rank, (result, score) in enumerate(ranked[:k], start=1):
            out.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    char_start=result.char_start,
                    char_end=result.char_end,
                    rank=rank,
                    score=float(score),
                    retriever_name=f"{result.retriever_name}+rerank",
                )
            )
        return out


class IdentityReranker(Reranker):
    """Truncate to k without reordering. Used in tests to keep the shape honest."""

    name = "identity"

    def rerank(self, query: str, results: list[RetrievalResult], k: int) -> list[RetrievalResult]:
        return list(results[:k])


def build_reranker(name: str | None, model_name: str = "BAAI/bge-reranker-base", device: str = "auto") -> Reranker | None:
    if name in (None, "none", "null", ""):
        return None
    if name == "identity":
        return IdentityReranker()
    if name in ("cross-encoder", "cross_encoder"):
        return CrossEncoderReranker(model_name, device=device)
    raise ValueError(f"unknown reranker: {name}")
