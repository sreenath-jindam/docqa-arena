"""Turn eval/results/*/summary.json into a sorted markdown table.

Reads only what the sweep wrote to disk, so it works on results downloaded
from a Kaggle session that ended hours ago.

    python scripts/leaderboard.py --sort mrr --markdown > leaderboard.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402

COLUMNS = [
    ("config", "Config"),
    ("recall_at_k", "Recall@5"),
    ("mrr", "MRR"),
    ("ndcg_at_k", "nDCG@5"),
    ("correctness", "Correct"),
    ("faithfulness", "Faithful"),
    ("judge_coverage", "Judged"),
    ("retrieval_ms", "Retr ms"),
    ("rerank_ms", "Rerank ms"),
]

# Below this share of judged items, generation scores are reported as "--".
# A configuration scored on 3 of 31 answers is not comparable to one scored on
# all 31, and printing a number invites exactly that comparison.
MIN_COVERAGE = 0.9


def coverage_from_evaluations(config_dir: Path) -> float:
    """Recover coverage for artifacts written before it was tracked.

    Defaulting to 1.0 would be the dangerous choice: the artifacts most likely
    to lack the field are exactly the ones from a run where the judge failed,
    so an optimistic default prints unmeasured zeros as if they were scores.
    """
    path = config_dir / "evaluations.json"
    if not path.exists():
        return 0.0
    evaluations = json.loads(path.read_text())
    if not evaluations:
        return 0.0
    measured = 0
    for evaluation in evaluations:
        judge = evaluation.get("judge", {})
        if judge.get("measured") is True:
            measured += 1
            continue
        explanation = (judge.get("explanation") or "").lower()
        failed = "judge error" in explanation or "judge unavailable" in explanation
        if not failed and judge.get("correctness") is not None:
            measured += 1
    return measured / len(evaluations)


def collect(results_dir: Path) -> list[dict]:
    rows = []
    for summary_file in sorted(results_dir.glob("*/summary.json")):
        data = json.loads(summary_file.read_text())
        generation = data["generation"]
        coverage = generation.get("judge_coverage")
        if coverage is None:
            coverage = coverage_from_evaluations(summary_file.parent)
        rows.append(
            {
                "config": data["config"]["slug"],
                "recall_at_k": data["retrieval"]["recall_at_k"],
                "mrr": data["retrieval"]["mrr"],
                "ndcg_at_k": data["retrieval"]["ndcg_at_k"],
                "correctness": generation["correctness"] if coverage >= MIN_COVERAGE else None,
                "faithfulness": generation["faithfulness"] if coverage >= MIN_COVERAGE else None,
                "judge_coverage": coverage,
                "retrieval_ms": data["retrieval"]["retrieval_latency_ms_mean"],
                "rerank_ms": data["retrieval"]["rerank_latency_ms_mean"],
            }
        )
    return rows


def fmt(key: str, value) -> str:
    if key == "config":
        return str(value)
    if value is None:
        return "--"
    if key == "judge_coverage":
        return f"{value:.0%}"
    if key.endswith("_ms"):
        return f"{value:.0f}"
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--results", default=None, help="override the results directory")
    parser.add_argument("--sort", default="mrr", choices=[c[0] for c in COLUMNS if c[0] != "config"])
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results) if args.results else load_config(args.config).results_path
    rows = collect(results_dir)
    if not rows:
        print(f"no summary.json files under {results_dir}")
        return

    # None sorts last rather than crashing the comparison.
    rows.sort(key=lambda r: (r[args.sort] is None, -(r[args.sort] or 0)))
    headers = [label for _, label in COLUMNS]
    keys = [key for key, _ in COLUMNS]

    if args.markdown:
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join("---" for _ in headers) + "|")
        for row in rows:
            print("| " + " | ".join(fmt(k, row[k]) for k in keys) + " |")
    else:
        widths = [max(len(h), max(len(fmt(k, r[k])) for r in rows)) for h, k in zip(headers, keys)]
        print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
        print("  ".join("-" * w for w in widths))
        for row in rows:
            print("  ".join(fmt(k, row[k]).ljust(w) for k, w in zip(keys, widths)))

    print(f"\n{len(rows)} configurations · sorted by {args.sort}")
    unmeasured = [r["config"] for r in rows if r["faithfulness"] is None]
    if unmeasured:
        print(
            f"{len(unmeasured)} configuration(s) show -- for generation scores: fewer than "
            f"{MIN_COVERAGE:.0%} of answers were judged. Run scripts/rejudge.py to fill them in."
        )


if __name__ == "__main__":
    main()
