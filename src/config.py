"""Configuration.

Two rules that make the Kaggle story work:

1. No hardcoded paths. Every directory is resolved from a base that defaults to
   the repo root but is overridden by ``DOCQA_DATA_DIR`` (set to
   ``/kaggle/working`` inside a notebook).
2. Everything is a plain dict underneath, so ``run_sweep`` can be called from a
   notebook with a dict and from a shell with ``--config configs/local.yaml``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Root for everything written at runtime: indexes, caches, results."""
    return Path(os.environ.get("DOCQA_DATA_DIR", REPO_ROOT)).resolve()


@dataclass
class PipelineConfig:
    """One point in the sweep grid."""

    chunker: str = "recursive"          # fixed | recursive | semantic
    retriever: str = "hybrid"           # vector | bm25 | hybrid
    reranker: str | None = None         # None | cross-encoder  (off by default)
    top_k: int = 5
    rerank_candidates: int = 20         # retrieve this many before reranking
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    chunk_size: int = 800
    chunk_overlap: int = 120
    generator_model: str = "llama-3.1-8b-instant"
    judge_backend: str = "llm"          # llm | ragas
    device: str = "auto"                # auto | cpu | cuda

    @property
    def slug(self) -> str:
        """Stable directory name. Same config -> same folder -> resumable."""
        rr = self.reranker or "norerank"
        return f"{self.chunker}__{self.retriever}__{rr}"

    @property
    def collection_name(self) -> str:
        """Chunking + embedding model determine the index; retriever does not."""
        model_tag = self.embed_model.split("/")[-1].replace(".", "-")
        return f"{self.chunker}__{model_tag}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AppConfig:
    """Everything that is not swept."""

    corpus_dir: str = "data/corpus"
    golden_csv: str = "eval/golden.csv"
    results_dir: str = "eval/results"
    index_dir: str = "storage/chroma"
    cache_path: str = "storage/embedding_cache.sqlite"
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    sweep: dict[str, Any] = field(default_factory=dict)

    # --- resolved paths ------------------------------------------------
    def path(self, attr: str) -> Path:
        raw = Path(getattr(self, attr))
        return raw if raw.is_absolute() else data_dir() / raw

    @property
    def corpus_path(self) -> Path:
        # The corpus ships with the repo, so it is always relative to the repo,
        # never to the writable data dir.
        raw = Path(self.corpus_dir)
        return raw if raw.is_absolute() else REPO_ROOT / raw

    @property
    def golden_path(self) -> Path:
        raw = Path(self.golden_csv)
        return raw if raw.is_absolute() else REPO_ROOT / raw

    @property
    def results_path(self) -> Path:
        return self.path("results_dir")

    @property
    def index_path(self) -> Path:
        return self.path("index_dir")

    @property
    def cache_file(self) -> Path:
        return self.path("cache_path")


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load YAML (optional) then apply a dict of overrides (optional)."""
    raw: dict[str, Any] = {}
    if path:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)

    pipeline_raw = raw.pop("pipeline", {}) or {}
    cfg = AppConfig(**raw)
    cfg.pipeline = PipelineConfig(**pipeline_raw)
    return cfg


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def expand_grid(sweep: dict[str, Any], base: PipelineConfig) -> list[PipelineConfig]:
    """Cartesian product of chunkers x retrievers x rerankers.

    Order is fixed and deterministic so a resumed run walks the same sequence.
    """
    chunkers = sweep.get("chunkers", ["fixed", "recursive", "semantic"])
    retrievers = sweep.get("retrievers", ["vector", "bm25", "hybrid"])
    rerankers = sweep.get("rerankers", [None, "cross-encoder"])

    configs: list[PipelineConfig] = []
    for chunker in chunkers:
        for retriever in retrievers:
            for reranker in rerankers:
                cfg = PipelineConfig(**base.to_dict())
                cfg.chunker = chunker
                cfg.retriever = retriever
                cfg.reranker = None if reranker in (None, "none", "null") else reranker
                configs.append(cfg)
    return configs


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
