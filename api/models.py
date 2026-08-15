"""API schemas.

The response deliberately returns the retrieved passages alongside the answer.
The premise of the project is that a fluent answer tells you nothing about
whether retrieval worked, so the service exposes both and lets the caller
check.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, examples=["How long do refunds take to settle?"])
    top_k: int | None = Field(None, ge=1, le=20)
    include_passages: bool = True
    generate: bool = True


class Passage(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    char_start: int
    char_end: int
    rank: int
    score: float
    retriever_name: str


class Timings(BaseModel):
    retrieval: float | None = None
    rerank: float | None = None
    generation: float | None = None
    total: float | None = None


class AskResponse(BaseModel):
    query: str
    answer: str
    passages: list[Passage] = []
    timings_ms: Timings
    cache: dict[str, Any] = {}
    tokens: dict[str, int] | None = None
    model: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)


class SearchResponse(BaseModel):
    query: str
    passages: list[Passage]
    timings_ms: Timings


class HealthResponse(BaseModel):
    status: str
    collection: str
    indexed_chunks: int
    config: dict[str, Any]
    cache: dict[str, Any]


class IngestRequest(BaseModel):
    reset: bool = False


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    collection: str
    chunker: str
    elapsed_s: float
