"""Re-score generation on an existing sweep, without re-running it.

Why this exists: retrieval and generation are expensive and were already done
correctly — every answer is sitting in ``evaluations.json``. Only the judge
failed, because a free-tier daily token quota ran out partway through. Nothing
about that requires re-embedding a corpus or re-calling the generator, so this
script re-judges from the saved artifacts and rewrites ``summary.json``.

Three things make it survive a quota that is smaller than the work:

1. **Deduplication.** Configurations that retrieve the same passages and
   generate the same answer produce the same verdict. Judging the unique
   (question, answer, context) triples instead of all 527 rows cuts the work by
   about a third on a corpus this size.
2. **A persistent cache.** Verdicts are written to disk as they arrive, so a run
   that dies at the quota resumes tomorrow instead of starting over.
3. **Null, not zero.** An item the judge could not score stays ``measured:
   false`` and is excluded from the averages, and ``judge_coverage`` records how
   much of each configuration was actually measured.

Usage:
    python scripts/rejudge.py --results eval/results --dry-run
    python scripts/rejudge.py --results eval/results --judge-model llama-3.1-8b-instant
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.judge import JudgeUnavailable, build_judge  # noqa: E402
from eval.metrics import mean, percentile  # noqa: E402
from src.types import CORRECTNESS_SCORE, FAITHFULNESS_SCORE, RetrievalResult  # noqa: E402


def triple_key(evaluation: dict) -> str:
    """A verdict depends on the question, the answer, and the context. Nothing else."""
    payload = "|".join(
        [
            evaluation["id"],
            evaluation["generated_answer"],
            ",".join(p["chunk_id"] for p in evaluation["passages"]),
        ]
    )
    return hashlib.sha1(payload.encode()).hexdigest()


class VerdictCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists():
            self.data = json.loads(path.read_text())

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, verdict: dict) -> None:
        self.data[key] = verdict
        # Written after every verdict — a quota can end the run at any moment.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))

    def __len__(self) -> int:
        return len(self.data)


def to_passages(raw: list[dict]) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=p["chunk_id"],
            document_id=p["document_id"],
            content=p["content"],
            char_start=p["char_start"],
            char_end=p["char_end"],
            rank=p["rank"],
            score=p["score"],
            retriever_name=p["retriever_name"],
        )
        for p in raw
    ]


def needs_judging(evaluation: dict) -> bool:
    judge = evaluation.get("judge", {})
    if judge.get("measured") is True:
        return False
    # Artifacts from before the fix recorded failures as a zero score with the
    # error text in `explanation`. Treat those as unmeasured too.
    explanation = (judge.get("explanation") or "").lower()
    return "judge error" in explanation or "judge unavailable" in explanation or judge.get("correctness") is None


def rebuild_summary(summary: dict, evaluations: list[dict]) -> dict:
    judges = [e["judge"] for e in evaluations]
    measured = [j for j in judges if j.get("measured")]
    generation = [e["generation"] for e in evaluations]

    summary["generation"] = {
        "judge_coverage": len(measured) / len(judges) if judges else 0.0,
        "judged_examples": len(measured),
        "correctness": mean([CORRECTNESS_SCORE[j["correctness"]] for j in measured]) if measured else None,
        "faithfulness": mean([FAITHFULNESS_SCORE[j["faithfulness"]] for j in measured]) if measured else None,
        "correct_count": sum(1 for j in measured if j["correctness"] == "correct"),
        "faithful_count": sum(1 for j in measured if j["faithfulness"] == "faithful"),
        "prompt_tokens_mean": mean([g["prompt_tokens"] for g in generation]),
        "completion_tokens_mean": mean([g["completion_tokens"] for g in generation]),
        "latency_ms_mean": mean([g["latency_ms"] for g in generation]),
        "latency_ms_p95": percentile([g["latency_ms"] for g in generation], 95),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="eval/results")
    parser.add_argument("--cache", default=None, help="verdict cache path (default: <results>/.verdict_cache.json)")
    parser.add_argument("--judge-model", default=None, help="override JUDGE_MODEL, e.g. llama-3.1-8b-instant")
    parser.add_argument("--backend", default="llm", choices=["llm", "ragas", "stub"])
    parser.add_argument("--limit", type=int, default=None, help="stop after N judge calls (to stay inside a quota)")
    parser.add_argument("--dry-run", action="store_true", help="report what needs judging and exit")
    args = parser.parse_args()

    if args.judge_model:
        os.environ["JUDGE_MODEL"] = args.judge_model

    results_dir = Path(args.results)
    cache_path = Path(args.cache) if args.cache else results_dir / ".verdict_cache.json"
    cache = VerdictCache(cache_path)

    config_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir() and (d / "evaluations.json").exists())
    if not config_dirs:
        raise SystemExit(f"no evaluations.json found under {results_dir}")

    # -- survey ---------------------------------------------------------
    pending: dict[str, dict] = {}
    per_config: dict[str, int] = {}
    for config_dir in config_dirs:
        evaluations = json.loads((config_dir / "evaluations.json").read_text())
        missing = [e for e in evaluations if needs_judging(e)]
        per_config[config_dir.name] = len(missing)
        for evaluation in missing:
            key = triple_key(evaluation)
            if key not in cache.data:
                pending.setdefault(key, evaluation)

    total_missing = sum(per_config.values())
    print(f"{len(config_dirs)} configurations, {total_missing} unmeasured items")
    print(f"{len(pending)} unique (question, answer, context) triples to judge")
    print(f"{len(cache)} verdicts already cached\n")
    for name, count in sorted(per_config.items(), key=lambda kv: -kv[1]):
        if count:
            print(f"  {name:<34} {count:>3} unmeasured")

    if args.dry_run:
        print("\ndry run: nothing called")
        return

    # -- judge ----------------------------------------------------------
    judge = build_judge(args.backend)
    calls = 0
    for key, evaluation in pending.items():
        if args.limit and calls >= args.limit:
            print(f"\nstopping at --limit {args.limit}")
            break
        try:
            verdict = judge.judge(
                evaluation["query"],
                evaluation["expected_answer"],
                evaluation["generated_answer"],
                to_passages(evaluation["passages"]),
            )
            cache.put(key, {**verdict.__dict__, "measured": True})
            calls += 1
            if calls % 10 == 0:
                print(f"  judged {calls}/{len(pending)}", flush=True)
        except JudgeUnavailable as exc:
            # Quota gone. Everything judged so far is already on disk; stop
            # cleanly and let the next run resume from the cache.
            print(f"\njudge unavailable after {calls} calls: {exc}")
            print("Cached verdicts are saved. Re-run this script when the quota resets.")
            break

    # -- write back -----------------------------------------------------
    print("\nrewriting summaries:")
    for config_dir in config_dirs:
        evaluations = json.loads((config_dir / "evaluations.json").read_text())
        changed = 0
        for evaluation in evaluations:
            if not needs_judging(evaluation):
                continue
            verdict = cache.get(triple_key(evaluation))
            if verdict:
                evaluation["judge"] = dict(verdict)
                changed += 1
            else:
                evaluation["judge"] = {
                    "correctness": None,
                    "faithfulness": None,
                    "measured": False,
                    "explanation": evaluation.get("judge", {}).get("explanation", "not judged"),
                }
        for evaluation in evaluations:
            evaluation["judge"].setdefault("measured", evaluation["judge"].get("correctness") is not None)

        (config_dir / "evaluations.json").write_text(json.dumps(evaluations, indent=2))
        summary = rebuild_summary(json.loads((config_dir / "summary.json").read_text()), evaluations)
        (config_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        coverage = summary["generation"]["judge_coverage"]
        faith = summary["generation"]["faithfulness"]
        faith_text = f"{faith:.3f}" if faith is not None else "unmeasured"
        print(f"  {config_dir.name:<34} +{changed:<3} coverage={coverage:>4.0%}  faithfulness={faith_text}")


if __name__ == "__main__":
    main()
