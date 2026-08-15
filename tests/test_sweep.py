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
