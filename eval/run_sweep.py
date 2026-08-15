"""The sweep.

Runs every configuration in the grid against a fixed golden set and writes
three files per config:

    results/<slug>/config.json       what was run
    results/<slug>/evaluations.json  per-question detail
    results/<slug>/summary.json      the aggregates

Written to disk, not printed. A printed number is gone the moment the Kaggle
session dies; a file can be downloaded, diffed against last week's run, and
pasted into RESULTS.md.

**Resumability.** A config whose ``summary.json`` already exists is skipped.
Kaggle sessions end at nine hours or when the tab closes, whichever comes
first, and a sweep that has to start over from config 1 every time will never
finish. Delete a config's folder to force a re-run.

Two entry points, one implementation: ``python eval/run_sweep.py --config
configs/kaggle.yaml`` from a shell, or ``from eval.run_sweep import run_sweep``
in a notebook. Nothing about the logic knows which one called it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make `python eval/run_sweep.py` work without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.judge import build_judge  # noqa: E402
from eval.metrics import evaluate_retrieval, mean, percentile  # noqa: E402
from eval.spans import load_golden, relevant_chunk_ids, resolve_spans  # noqa: E402
from src.config import AppConfig, PipelineConfig, expand_grid, load_config  # noqa: E402
from src.corpus import load_document_map  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.types import CORRECTNESS_SCORE, FAITHFULNESS_SCORE  # noqa: E402


def run_single_config(
    app: AppConfig,
    pipeline_config: PipelineConfig,
    examples,
    documents,
    judge,
    out_dir: Path,
    verbose: bool = True,
) -> dict:
    """Evaluate one configuration end to end and persist its artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    pipeline = RAGPipeline(app, pipeline_config)

    # Ingest only if this chunker's collection is empty. Three chunkers means
    # three indexes; the six retriever/reranker combinations reuse them.
    if pipeline.store.count() == 0:
        if verbose:
            print(f"  building index for {pipeline_config.collection_name} ...", flush=True)
        info = pipeline.ingest()
        if verbose:
            print(f"  indexed {info['chunks']} chunks in {info['elapsed_s']}s", flush=True)

    corpus_chunks = pipeline.chunks
    evaluations: list[dict[str, Any]] = []

    for i, example in enumerate(examples, start=1):
        relevant = relevant_chunk_ids(example, corpus_chunks)
        if not relevant:
            # The evidence span fell outside every chunk — impossible unless the
            # corpus and golden set have drifted apart. Recorded, not skipped.
            print(f"  ! {example.id}: no chunk overlaps the evidence span", flush=True)

        result = pipeline.answer(example.query)
        retrieved_ids = [p.chunk_id for p in result.passages]

        retrieval_eval = evaluate_retrieval(
            example_id=example.id,
            query=example.query,
            retrieved_chunk_ids=retrieved_ids,
            relevant_chunk_ids=relevant,
            k=pipeline_config.top_k,
            retrieval_latency_ms=result.timings_ms.get("retrieval", 0.0),
            rerank_latency_ms=result.timings_ms.get("rerank", 0.0),
        )

        try:
            verdict = judge.judge(example.query, example.expected_answer, result.answer, result.passages)
            judge_dict = verdict.__dict__.copy()
            judge_dict["measured"] = True
        except Exception as exc:  # a rate-limited judge must not kill the sweep
            # null, not zero. An unmeasured item is not an unfaithful one, and
            # averaging the two together turns an outage into a quality score.
            judge_dict = {
                "correctness": None,
                "faithfulness": None,
                "measured": False,
                "explanation": f"judge unavailable: {exc}",
            }

        evaluations.append(
            {
                "id": example.id,
                "query": example.query,
                "query_type": example.query_type,
                "expected_answer": example.expected_answer,
                "generated_answer": result.answer,
                "retrieval": retrieval_eval.to_dict(),
                "judge": judge_dict,
                "generation": {
                    "model": result.generation.model if result.generation else None,
                    "prompt_tokens": result.generation.prompt_tokens if result.generation else 0,
                    "completion_tokens": result.generation.completion_tokens if result.generation else 0,
                    "total_tokens": result.generation.total_tokens if result.generation else 0,
                    "latency_ms": result.timings_ms.get("generation", 0.0),
                },
                "passages": [p.dict() for p in result.passages],
            }
        )

        if verbose and i % 5 == 0:
            print(f"  {i}/{len(examples)} questions", flush=True)

    summary = summarize(pipeline_config, evaluations, started)

    (out_dir / "config.json").write_text(
        json.dumps({**pipeline_config.to_dict(), "slug": pipeline_config.slug, "started_at": started}, indent=2)
    )
    (out_dir / "evaluations.json").write_text(json.dumps(evaluations, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def summarize(cfg: PipelineConfig, evaluations: list[dict], started: str) -> dict:
    retrieval = [e["retrieval"] for e in evaluations]
    judges = [e["judge"] for e in evaluations]
    generation = [e["generation"] for e in evaluations]

    retrieval_latencies = [r["retrieval_latency_ms"] for r in retrieval]
    rerank_latencies = [r["rerank_latency_ms"] for r in retrieval]

    measured = [j for j in judges if j.get("measured")]
    coverage = len(measured) / len(judges) if judges else 0.0

    return {
        "config": {**cfg.to_dict(), "slug": cfg.slug},
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_examples": len(evaluations),
        "retrieval": {
            "precision_at_k": mean([r["precision_at_k"] for r in retrieval]),
            "recall_at_k": mean([r["recall_at_k"] for r in retrieval]),
            "mrr": mean([r["reciprocal_rank"] for r in retrieval]),
            "ndcg_at_k": mean([r["ndcg_at_k"] for r in retrieval]),
            "hit_at_k": mean([r["hit_at_k"] for r in retrieval]),
            "retrieval_latency_ms_mean": mean(retrieval_latencies),
            "retrieval_latency_ms_p95": percentile(retrieval_latencies, 95),
            "rerank_latency_ms_mean": mean(rerank_latencies),
            "rerank_latency_ms_p95": percentile(rerank_latencies, 95),
        },
        "generation": {
            # Averaged over measured items only; `judge_coverage` says how many
            # that was. A config scored on 3 of 31 answers is not comparable to
            # one scored on all 31, and the leaderboard refuses to rank it.
            "judge_coverage": coverage,
            "judged_examples": len(measured),
            "correctness": mean([CORRECTNESS_SCORE[j["correctness"]] for j in measured]) if measured else None,
            "faithfulness": mean([FAITHFULNESS_SCORE[j["faithfulness"]] for j in measured]) if measured else None,
            "correct_count": sum(1 for j in measured if j["correctness"] == "correct"),
            "faithful_count": sum(1 for j in measured if j["faithfulness"] == "faithful"),
            "prompt_tokens_mean": mean([g["prompt_tokens"] for g in generation]),
            "completion_tokens_mean": mean([g["completion_tokens"] for g in generation]),
            "latency_ms_mean": mean([g["latency_ms"] for g in generation]),
            "latency_ms_p95": percentile([g["latency_ms"] for g in generation], 95),
        },
    }


def run_sweep(
    config_path: str | Path | None = None,
    overrides: dict | None = None,
    only: list[str] | None = None,
    force: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """Run the whole grid. Safe to call twice — finished configs are skipped."""
    app = load_config(config_path, overrides)
    results_root = app.results_path
    results_root.mkdir(parents=True, exist_ok=True)

    documents = load_document_map(app.corpus_path)
    examples = resolve_spans(load_golden(app.golden_path), documents)
    if verbose:
        print(f"golden set: {len(examples)} questions over {len(documents)} documents")

    configs = expand_grid(app.sweep, app.pipeline)
    if only:
        configs = [c for c in configs if c.slug in only]

    summaries = []
    for i, cfg in enumerate(configs, start=1):
        out_dir = results_root / cfg.slug
        summary_file = out_dir / "summary.json"

        if summary_file.exists() and not force:
            if verbose:
                print(f"[{i}/{len(configs)}] {cfg.slug} — already done, skipping")
            summaries.append(json.loads(summary_file.read_text()))
            continue

        print(f"[{i}/{len(configs)}] {cfg.slug}", flush=True)
        start = time.perf_counter()
        judge = build_judge(cfg.judge_backend)
        try:
            summary = run_single_config(app, cfg, examples, documents, judge, out_dir, verbose)
            summaries.append(summary)
            gen = summary["generation"]
            faith = f"{gen['faithfulness']:.3f}" if gen["faithfulness"] is not None else "unmeasured"
            print(
                f"  done in {time.perf_counter() - start:.1f}s  "
                f"mrr={summary['retrieval']['mrr']:.3f}  "
                f"faithfulness={faith}  "
                f"judge_coverage={gen['judge_coverage']:.0%}",
                flush=True,
            )
        except Exception:
            # One bad config must not cost you the other seventeen.
            print(f"  FAILED: {cfg.slug}", flush=True)
            traceback.print_exc()
            (out_dir).mkdir(parents=True, exist_ok=True)
            (out_dir / "error.log").write_text(traceback.format_exc())

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval/generation sweep.")
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--only", nargs="*", help="slugs to run, e.g. recursive__hybrid__cross-encoder")
    parser.add_argument("--force", action="store_true", help="re-run configs that already have a summary.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    summaries = run_sweep(args.config, only=args.only, force=args.force, verbose=not args.quiet)
    print(f"\n{len(summaries)} configurations complete.")


if __name__ == "__main__":
    main()
