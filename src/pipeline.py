"""The pipeline.

One object that owns a config and exposes ``answer()``. Both the API and the
sweep call this, so there is exactly one code path that can be measured — a
benchmark that exercises different code from the service is a benchmark of
nothing.

Latency is reported per stage. An aggregate number tells you the request was
slow; the breakdown tells you which stage to fix.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import EmbeddingCache
from .chunkers import build_chunker
from .config import AppConfig, PipelineConfig
from .corpus import load_corpus
from .embeddings import build_embedder
from .gate import OUT_OF_SCOPE_ANSWER, RelevanceGate
from .generate import build_generator, build_messages
from .rerank import build_reranker
from .retrievers import build_retriever
from .store import ChromaStore
from .types import Chunk, GenerationResult, RetrievalResult


@dataclass
class AnswerResult:
    query: str
    answer: str
    passages: list[RetrievalResult]
    generation: GenerationResult | None
    timings_ms: dict[str, float] = field(default_factory=dict)
    cache_stats: dict[str, Any] = field(default_factory=dict)
    gate: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "passages": [p.dict() for p in self.passages],
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "cache_stats": self.cache_stats,
            "gate": self.gate,
            "model": self.generation.model if self.generation else None,
            "tokens": (
                {
                    "prompt": self.generation.prompt_tokens,
                    "completion": self.generation.completion_tokens,
                    "total": self.generation.total_tokens,
                }
                if self.generation
                else None
            ),
        }


class RAGPipeline:
    def __init__(self, app_config: AppConfig, pipeline_config: PipelineConfig | None = None, cache_enabled: bool = True):
        self.app = app_config
        self.cfg = pipeline_config or app_config.pipeline
        self.cache = EmbeddingCache(app_config.cache_file, enabled=cache_enabled)
        self.embedder = build_embedder(self.cfg.embed_model, device=self.cfg.device, cache=self.cache)
        self.store = ChromaStore(app_config.index_path, self.cfg.collection_name, self.embedder)
        self.gate = RelevanceGate(
            min_score=self.cfg.gate_min_score,
            min_top_score=self.cfg.gate_min_top_score,
            enabled=self.cfg.gate_enabled,
        )
        self._chunks: list[Chunk] | None = None
        self._retriever = None
        self._reranker = None
        self._generator = None

    # -- lazy stage construction ---------------------------------------
    @property
    def chunks(self) -> list[Chunk]:
        if self._chunks is None:
            self._chunks = self.store.all_chunks()
        return self._chunks

    @property
    def retriever(self):
        if self._retriever is None:
            needs_lexical = self.cfg.retriever in ("bm25", "hybrid")
            self._retriever = build_retriever(
                self.cfg.retriever,
                self.store,
                self.embedder,
                chunks=self.chunks if needs_lexical else None,
            )
        return self._retriever

    @property
    def reranker(self):
        if self._reranker is None and self.cfg.reranker:
            self._reranker = build_reranker(self.cfg.reranker, self.cfg.rerank_model, self.cfg.device)
        return self._reranker

    @property
    def generator(self):
        if self._generator is None:
            self._generator = build_generator(self.cfg.generator_model)
        return self._generator

    # -- ingestion ------------------------------------------------------
    def ingest(self, reset: bool = False) -> dict:
        """Chunk the corpus and write it to the index. Idempotent."""
        if reset:
            self.store.reset()

        documents = load_corpus(self.app.corpus_path)
        chunker = build_chunker(
            self.cfg.chunker,
            embedder=self.embedder,
            chunk_size=self.cfg.chunk_size,
            overlap=self.cfg.chunk_overlap,
        )
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(chunker.chunk(doc))

        start = time.perf_counter()
        added = self.store.add(all_chunks)
        self._chunks = None  # force reload
        return {
            "documents": len(documents),
            "chunks": added,
            "collection": self.cfg.collection_name,
            "chunker": chunker.name,
            "elapsed_s": round(time.perf_counter() - start, 2),
            "cache": self.cache.stats(),
        }

    # -- query ----------------------------------------------------------
    def retrieve(
        self, query: str, k: int | None = None
    ) -> tuple[list[RetrievalResult], dict[str, float], dict[str, Any] | None]:
        k = k or self.cfg.top_k
        timings: dict[str, float] = {}

        fetch_k = self.cfg.rerank_candidates if self.reranker else k
        t0 = time.perf_counter()
        results = self.retriever.retrieve(query, fetch_k)
        timings["retrieval"] = (time.perf_counter() - t0) * 1000

        if self.reranker:
            t1 = time.perf_counter()
            results = self.reranker.rerank(query, results, k)
            timings["rerank"] = (time.perf_counter() - t1) * 1000
        else:
            results = results[:k]

        decision = self.gate.apply(results)
        return decision.passages, timings, (decision.to_dict() if self.gate.enabled else None)

    def answer(self, query: str, k: int | None = None, generate: bool = True) -> AnswerResult:
        total_start = time.perf_counter()
        results, timings, gate_info = self.retrieve(query, k)

        generation = None
        answer_text = ""
        if generate:
            if not results and self.gate.enabled:
                # The gate found nothing relevant. Skipping the LLM call is not
                # just a saving: given only irrelevant context the generator
                # sometimes answers from it anyway, which is the exact failure
                # the gate exists to prevent.
                answer_text = OUT_OF_SCOPE_ANSWER
                timings["generation"] = 0.0
            else:
                t0 = time.perf_counter()
                generation = self.generator.generate(build_messages(query, results))
                timings["generation"] = (time.perf_counter() - t0) * 1000
                answer_text = generation.answer

        timings["total"] = (time.perf_counter() - total_start) * 1000
        return AnswerResult(
            query=query,
            answer=answer_text,
            passages=results,
            generation=generation,
            timings_ms=timings,
            cache_stats=self.cache.stats(),
            gate=gate_info,
        )

    def health(self) -> dict:
        return {
            "collection": self.cfg.collection_name,
            "indexed_chunks": self.store.count(),
            "config": self.cfg.to_dict(),
            "index_dir": str(self.app.index_path),
            "cache": self.cache.stats(),
        }
