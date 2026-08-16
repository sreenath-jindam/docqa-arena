"""Pick a relevance threshold from data, not from intuition.

A threshold guessed from a blog post is a threshold tuned to somebody else's
corpus and somebody else's embedding model. This measures two distributions on
*your* setup:

- **in-scope**: the golden set's 31 questions, which the corpus does answer
- **out-of-scope**: questions the corpus demonstrably does not answer

The useful output is the gap between them. If the worst in-scope similarity sits
comfortably above the best out-of-scope one, a threshold in between will reject
irrelevant queries without ever dropping a real answer. If the two distributions
overlap, no threshold works and the honest conclusion is to say so rather than
to pick a number that trades one failure for another.

    python scripts/tune_threshold.py --config configs/local.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.spans import load_golden  # noqa: E402
from src.config import load_config  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

# Questions a payments-documentation corpus cannot answer. Deliberately varied:
# some share vocabulary with the corpus ("bank", "payment", "money"), because
# those are the cases a naive threshold gets wrong.
OUT_OF_SCOPE = [
    "Who is the Prime Minister of India?",
    "What is the capital of France?",
    "How do I train a neural network?",
    "Many people sent money to me at once and my bank account is down. What should I do?",
    "What is the best programming language for beginners?",
    "How do I reset my password?",
    "What are the payment terms for my mortgage?",
    "My bank account was frozen, who do I contact?",
    "What is the weather forecast tomorrow?",
    "How much does a payment processor charge for international transfers in Europe?",
    "Explain the difference between TCP and UDP.",
    "What is the refund policy for my gym membership?",
]


def best_similarity(pipeline: RAGPipeline, query: str) -> float:
    """Top cosine similarity for a query, bypassing the gate entirely."""
    vector = pipeline.embedder.embed_query(query)
    results = pipeline.store.query(vector, pipeline.cfg.top_k)
    return max((r.score for r in results), default=0.0)


def summarize(name: str, scores: list[float]) -> None:
    ordered = sorted(scores)
    n = len(ordered)
    print(
        f"{name:<14} n={n:<3} min={ordered[0]:.3f}  p10={ordered[max(0, n // 10 - 1)]:.3f}  "
        f"median={ordered[n // 2]:.3f}  p90={ordered[min(n - 1, n * 9 // 10)]:.3f}  max={ordered[-1]:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--show-all", action="store_true", help="print every query's score")
    args = parser.parse_args()

    app = load_config(args.config)
    pipeline = RAGPipeline(app)
    if pipeline.store.count() == 0:
        raise SystemExit("index is empty — run scripts/ingest.py first")

    print(f"embedding model: {pipeline.cfg.embed_model}")
    print(f"collection: {pipeline.cfg.collection_name} ({pipeline.store.count()} chunks)\n")

    golden = load_golden(app.golden_path)
    in_scope = [(e.query, best_similarity(pipeline, e.query)) for e in golden]
    out_scope = [(q, best_similarity(pipeline, q)) for q in OUT_OF_SCOPE]

    if args.show_all:
        print("--- in scope ---")
        for query, score in sorted(in_scope, key=lambda p: p[1]):
            print(f"  {score:.3f}  {query[:70]}")
        print("\n--- out of scope ---")
        for query, score in sorted(out_scope, key=lambda p: -p[1]):
            print(f"  {score:.3f}  {query[:70]}")
        print()

    in_scores = [s for _, s in in_scope]
    out_scores = [s for _, s in out_scope]
    summarize("in-scope", in_scores)
    summarize("out-of-scope", out_scores)

    worst_in = min(in_scores)
    best_out = max(out_scores)
    print(f"\nworst in-scope:  {worst_in:.3f}")
    print(f"best out-of-scope: {best_out:.3f}")

    if worst_in > best_out:
        # A clean gap: any threshold in between is correct on this sample.
        # Sitting nearer the lower edge is the safer choice, because dropping a
        # passage that would have answered the question is worse than passing a
        # weak one through — the generator can decline, but it cannot retrieve.
        suggested = best_out + (worst_in - best_out) * 0.35
        print(f"\nSeparable. Gap of {worst_in - best_out:.3f}.")
        print(f"Suggested gate_min_top_score: {suggested:.2f}")
        print(f"Suggested gate_min_score:     {suggested - 0.05:.2f}")
    else:
        overlap = [q for q, s in out_scope if s >= worst_in]
        print(f"\nOverlapping by {best_out - worst_in:.3f} — no threshold separates these cleanly.")
        print("Out-of-scope queries scoring above the weakest real question:")
        for query in overlap[:5]:
            print(f"  - {query[:70]}")
        print(
            "\nA threshold here trades false refusals for false answers. Either accept "
            "some of both, or use a cross-encoder as the gate — its scores are calibrated "
            "for relevance in a way cosine similarity is not."
        )


if __name__ == "__main__":
    main()
