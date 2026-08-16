"""The sweep's two load-bearing properties: it writes the three artifacts, and
it does not redo work. Kaggle sessions die mid-run; resumability is not a nicety.
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("chromadb")

from src.config import PipelineConfig, expand_grid, load_config  # noqa: E402


def test_grid_produces_eighteen_configurations():
    sweep = {
        "chunkers": ["fixed", "recursive", "semantic"],
        "retrievers": ["vector", "bm25", "hybrid"],
        "rerankers": [None, "cross-encoder"],
    }
    configs = expand_grid(sweep, PipelineConfig())
    assert len(configs) == 18
    assert len({c.slug for c in configs}) == 18


def test_grid_order_is_stable():
    sweep = {"chunkers": ["fixed", "recursive"], "retrievers": ["vector"], "rerankers": [None]}
    first = [c.slug for c in expand_grid(sweep, PipelineConfig())]
    second = [c.slug for c in expand_grid(sweep, PipelineConfig())]
    assert first == second


def test_collection_is_shared_across_retrievers():
    """Three chunkers, not eighteen indexes — this is what keeps the sweep cheap."""
    base = PipelineConfig()
    configs = expand_grid({"chunkers": ["fixed"], "retrievers": ["vector", "bm25", "hybrid"], "rerankers": [None]}, base)
    assert len({c.collection_name for c in configs}) == 1


def test_slug_encodes_the_reranker():
    off = PipelineConfig(chunker="fixed", retriever="vector", reranker=None)
    on = PipelineConfig(chunker="fixed", retriever="vector", reranker="cross-encoder")
    assert off.slug == "fixed__vector__norerank"
    assert on.slug == "fixed__vector__cross-encoder"


def test_sweep_writes_artifacts_and_then_skips(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCQA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_BACKEND", "echo")
    monkeypatch.setenv("JUDGE_BACKEND", "stub")

    from eval.run_sweep import run_sweep

    overrides = {"sweep": {"chunkers": ["fixed"], "retrievers": ["bm25"], "rerankers": [None]}}
    summaries = run_sweep(ROOT / "configs" / "test.yaml", overrides=overrides, verbose=False)
    assert len(summaries) == 1

    out = tmp_path / ".pytest-results" / "fixed__bm25__norerank"
    for name in ("config.json", "summary.json", "evaluations.json"):
        assert (out / name).exists(), f"{name} was not written"

    summary = json.loads((out / "summary.json").read_text())
    assert summary["total_examples"] >= 30
    assert 0.0 <= summary["retrieval"]["mrr"] <= 1.0
    assert 0.0 <= summary["generation"]["faithfulness"] <= 1.0

    # Second run must not rewrite the artifacts.
    stamp = (out / "summary.json").stat().st_mtime_ns
    run_sweep(ROOT / "configs" / "test.yaml", overrides=overrides, verbose=False)
    assert (out / "summary.json").stat().st_mtime_ns == stamp


def test_unmeasured_judgements_are_null_not_zero(tmp_path):
    """A judge outage must not be recorded as a faithfulness score of zero.

    This is the bug that turned a rate-limited sweep into a leaderboard full of
    0.000 faithfulness. "Unfaithful" and "unmeasured" are different facts, and
    averaging them together lets an outage masquerade as a quality result.
    """
    import json
    from eval.run_sweep import summarize
    from src.config import PipelineConfig

    evaluations = [
        {
            "retrieval": {
                "precision_at_k": 0.2, "recall_at_k": 1.0, "reciprocal_rank": 1.0,
                "ndcg_at_k": 1.0, "hit_at_k": 1.0,
                "retrieval_latency_ms": 5.0, "rerank_latency_ms": 0.0,
            },
            "judge": {"correctness": "correct", "faithfulness": "faithful", "measured": True},
            "generation": {"prompt_tokens": 700, "completion_tokens": 30, "latency_ms": 500.0},
        },
        {
            "retrieval": {
                "precision_at_k": 0.2, "recall_at_k": 1.0, "reciprocal_rank": 0.5,
                "ndcg_at_k": 0.6, "hit_at_k": 1.0,
                "retrieval_latency_ms": 5.0, "rerank_latency_ms": 0.0,
            },
            "judge": {"correctness": None, "faithfulness": None, "measured": False,
                      "explanation": "judge unavailable: 429"},
            "generation": {"prompt_tokens": 700, "completion_tokens": 30, "latency_ms": 500.0},
        },
    ]

    summary = summarize(PipelineConfig(), evaluations, "2026-01-01T00:00:00Z")
    generation = summary["generation"]

    assert generation["judge_coverage"] == 0.5
    assert generation["judged_examples"] == 1
    # Averaged over the one measured item, not dragged to 0.5 by the failure.
    assert generation["faithfulness"] == 1.0
    assert generation["correctness"] == 1.0


def test_summary_reports_none_when_nothing_was_judged():
    from eval.run_sweep import summarize
    from src.config import PipelineConfig

    evaluations = [
        {
            "retrieval": {
                "precision_at_k": 0.0, "recall_at_k": 0.0, "reciprocal_rank": 0.0,
                "ndcg_at_k": 0.0, "hit_at_k": 0.0,
                "retrieval_latency_ms": 1.0, "rerank_latency_ms": 0.0,
            },
            "judge": {"correctness": None, "faithfulness": None, "measured": False},
            "generation": {"prompt_tokens": 1, "completion_tokens": 1, "latency_ms": 1.0},
        }
    ]
    generation = summarize(PipelineConfig(), evaluations, "2026-01-01T00:00:00Z")["generation"]
    assert generation["faithfulness"] is None
    assert generation["judge_coverage"] == 0.0


def test_judge_parser_repairs_trailing_commas():
    """Judges emit JavaScript-flavoured JSON; one stray comma must not cost a run.

    This exact payload — a trailing comma before the closing brace — stopped a
    186-call re-judging run after five calls.
    """
    from eval.judge import _parse_judge

    payload = '{"correctness": "correct", "faithfulness": "faithful", "explanation": "matches", }'
    verdict = _parse_judge(payload)
    assert verdict.correctness == "correct"
    assert verdict.faithfulness == "faithful"

    fenced = '```json\n{"correctness":"partially-correct","faithfulness":"faithful","explanation":"ok",}\n```'
    assert _parse_judge(fenced).correctness == "partially-correct"


def test_judge_parser_still_rejects_real_garbage():
    from eval.judge import JudgeUnavailable, _parse_judge

    with pytest.raises(JudgeUnavailable):
        _parse_judge("I'm sorry, I can't evaluate that.")