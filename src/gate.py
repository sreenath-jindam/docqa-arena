"""Relevance gating.

The problem this solves: a retriever always returns its top-k. Ask a document
Q&A service about Indian politics and it will hand the generator five passages
about refunds, ranked 1 through 5, with scores indistinguishable from a perfect
match. The generator then sometimes answers from them anyway — fluently, with a
citation, and wrongly.

**Why the fused score cannot be thresholded.** Reciprocal Rank Fusion computes
``1/(60 + rank)``, so a rank-1 result scores ~0.0164 whether it is the exact
answer or an unrelated paragraph. The number encodes position, not relevance.
Cross-encoder scores are calibrated but cost ~850 ms. Cosine similarity from the
dense retriever is the one score in this pipeline that is both cheap and
meaningful, so that is what the gate reads — see ``semantic_score`` on
``RetrievalResult``.

**On choosing a threshold.** Cosine similarity is not calibrated across models;
a value that means "unrelated" for bge-small means something else for bge-m3.
The default here was picked by measuring the golden set (see
``scripts/tune_threshold.py``), and it is deliberately loose: dropping a passage
that would have answered the question is a worse failure than passing through a
weak one, because the generator can decline but cannot retrieve.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import RetrievalResult


@dataclass
class GateDecision:
    """What the gate did, so the API can explain itself rather than just refusing."""

    passages: list[RetrievalResult]
    kept: int
    dropped: int
    best_score: float | None
    threshold: float
    passed: bool

    def to_dict(self) -> dict:
        return {
            "kept": self.kept,
            "dropped": self.dropped,
            "best_score": round(self.best_score, 4) if self.best_score is not None else None,
            "threshold": self.threshold,
            "passed": self.passed,
        }


class RelevanceGate:
    """Drops passages whose semantic similarity falls below a threshold.

    Two thresholds rather than one:

    - ``min_score`` filters individual passages.
    - ``min_top_score`` decides whether *anything* survives. If even the best
      passage is below this, the query is treated as out of scope and nothing is
      passed to the generator at all.

    The second is what makes the difference visible to a user: a query with no
    good match produces an explicit "not covered by these documents" rather than
    a plausible answer assembled from the least-bad chunk.
    """

    def __init__(
        self,
        min_score: float = 0.45,
        min_top_score: float = 0.50,
        enabled: bool = True,
        keep_at_least: int = 0,
    ):
        self.min_score = min_score
        self.min_top_score = min_top_score
        self.enabled = enabled
        self.keep_at_least = keep_at_least

    def apply(self, results: list[RetrievalResult]) -> GateDecision:
        if not self.enabled or not results:
            return GateDecision(
                passages=results,
                kept=len(results),
                dropped=0,
                best_score=_best(results),
                threshold=self.min_score,
                passed=bool(results),
            )

        scored = [r for r in results if r.semantic_score is not None]
        if not scored:
            # BM25-only retrieval has no comparable score, so the gate abstains
            # rather than guessing. Better to pass everything through than to
            # filter on a number that does not mean what the threshold assumes.
            return GateDecision(
                passages=results,
                kept=len(results),
                dropped=0,
                best_score=None,
                threshold=self.min_score,
                passed=True,
            )

        best = max(r.semantic_score for r in scored)

        if best < self.min_top_score:
            # Nothing in the corpus is close enough. Return no passages so the
            # generator has nothing to rationalise from.
            return GateDecision(
                passages=[],
                kept=0,
                dropped=len(results),
                best_score=best,
                threshold=self.min_top_score,
                passed=False,
            )

        kept = [r for r in results if (r.semantic_score or 0.0) >= self.min_score]
        if len(kept) < self.keep_at_least:
            ordered = sorted(results, key=lambda r: -(r.semantic_score or 0.0))
            kept = ordered[: self.keep_at_least]

        # Ranks must stay contiguous — citation numbers in the answer refer to
        # position in this list, so a gap would misattribute a claim.
        renumbered = [
            RetrievalResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                content=r.content,
                char_start=r.char_start,
                char_end=r.char_end,
                rank=i,
                score=r.score,
                retriever_name=r.retriever_name,
                semantic_score=r.semantic_score,
            )
            for i, r in enumerate(kept, start=1)
        ]

        return GateDecision(
            passages=renumbered,
            kept=len(renumbered),
            dropped=len(results) - len(renumbered),
            best_score=best,
            threshold=self.min_score,
            passed=bool(renumbered),
        )


def _best(results: list[RetrievalResult]) -> float | None:
    scores = [r.semantic_score for r in results if r.semantic_score is not None]
    return max(scores) if scores else None


OUT_OF_SCOPE_ANSWER = (
    "The provided documents do not cover this question. "
    "No sufficiently relevant passage was found."
)
