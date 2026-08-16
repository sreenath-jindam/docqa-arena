"""HTTP service.

The pipeline is built once at startup and held for the process lifetime. That
matters more than it looks: the embedding model is roughly a second of load
time, and building it per request would put that second on every p95 number and
make the cache measurement meaningless.

The index this serves is normally built on a GPU and shipped in as a directory —
see ``scripts/export_index.py``. The container needs no GPU at runtime.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from src.config import load_config
from src.pipeline import RAGPipeline

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("docqa")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get("DOCQA_CONFIG", "configs/local.yaml")
    cache_enabled = os.environ.get("DOCQA_CACHE", "1") != "0"
    app_config = load_config(config_path)
    pipeline = RAGPipeline(app_config, cache_enabled=cache_enabled)

    logger.info("collection=%s chunks=%d cache=%s", pipeline.cfg.collection_name, pipeline.store.count(), cache_enabled)
    if pipeline.store.count() == 0 and os.environ.get("DOCQA_AUTO_INGEST", "1") == "1":
        logger.info("empty index — ingesting corpus")
        logger.info("ingested: %s", pipeline.ingest())

    state["pipeline"] = pipeline
    state["latencies"] = []
    yield
    pipeline.cache.close()


app = FastAPI(
    title="Document Q&A",
    version="1.0.0",
    description="Retrieval-augmented Q&A over a local corpus, with the retrieved passages exposed alongside every answer.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline() -> RAGPipeline:
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is still starting. Try again in a moment.")
    return pipeline


@app.middleware("http")
async def record_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    if request.url.path in ("/ask", "/search"):
        state.setdefault("latencies", []).append(elapsed_ms)
        del state["latencies"][:-1000]  # keep a rolling window
    return response


@app.get("/health", response_model=HealthResponse)
def health(pipeline: RAGPipeline = Depends(get_pipeline)) -> HealthResponse:
    info = pipeline.health()
    return HealthResponse(
        status="ok",
        collection=info["collection"],
        indexed_chunks=info["indexed_chunks"],
        config=info["config"],
        cache=info["cache"],
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> AskResponse:
    if pipeline.store.count() == 0:
        raise HTTPException(status_code=409, detail="Index is empty. POST /ingest first.")

    result = pipeline.answer(request.query, k=request.top_k, generate=request.generate)
    payload = result.to_dict()
    if not request.include_passages:
        payload["passages"] = []
    return AskResponse(
        query=payload["query"],
        answer=payload["answer"],
        passages=payload["passages"],
        timings_ms=payload["timings_ms"],
        cache=payload["cache_stats"],
        tokens=payload["tokens"],
        model=payload["model"],
        gate=payload["gate"],
    )


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> SearchResponse:
    """Retrieval without generation — the endpoint to hit when an answer looks
    wrong and you need to know whether the right passages were even fetched."""
    passages, timings, gate = pipeline.retrieve(request.query, k=request.top_k)
    return SearchResponse(
        query=request.query,
        passages=[p.dict() for p in passages],
        timings_ms=timings,
        gate=gate,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> IngestResponse:
    info = pipeline.ingest(reset=request.reset)
    return IngestResponse(**{k: v for k, v in info.items() if k != "cache"})


@app.get("/metrics")
def metrics(pipeline: RAGPipeline = Depends(get_pipeline)) -> dict:
    """Rolling service latency plus cache hit rate. Not the benchmark — that
    lives in eval/results/ and is a different measurement entirely."""
    from eval.metrics import mean, percentile

    latencies = state.get("latencies", [])
    return {
        "requests": len(latencies),
        "latency_ms": {
            "mean": round(mean(latencies), 2),
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
            "p99": round(percentile(latencies, 99), 2),
        },
        "cache": pipeline.cache.stats(),
        "config": pipeline.cfg.to_dict(),
    }


@app.get("/")
def root() -> dict:
    return {"service": "document-qa", "docs": "/docs", "health": "/health"}
