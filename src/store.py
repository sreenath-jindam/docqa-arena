"""Vector store.

Chroma with a persistent client. Two things worth noting for the write-up:

1. Embeddings are passed in explicitly rather than letting Chroma call an
   embedding function. That keeps the cache in charge of every vector, which is
   the whole point of ``src/cache.py``.
2. A collection is keyed by ``chunker + embedding model``, not by retriever.
   Three chunkers therefore mean three indexes, and the six retriever/reranker
   combinations reuse them. That is what makes an 18-config sweep affordable.
"""

from __future__ import annotations

import os
from pathlib import Path

# Chroma's telemetry client is noisy and occasionally throws on version skew.
# Set before the first import so it never initialises.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")

from .embeddings import Embedder
from .types import Chunk, RetrievalResult


class ChromaStore:
    def __init__(self, index_dir: str | Path, collection_name: str, embedder: Embedder):
        import chromadb

        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(self.index_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # -- write ----------------------------------------------------------
    def count(self) -> int:
        return self.collection.count()

    def add(self, chunks: list[Chunk], batch_size: int = 128) -> int:
        if not chunks:
            return 0
        added = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embedder.embed([c.content for c in batch])
            self.collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.content for c in batch],
                embeddings=vectors,
                metadatas=[c.to_metadata() for c in batch],
            )
            added += len(batch)
        return added

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    # -- read -----------------------------------------------------------
    def all_chunks(self) -> list[Chunk]:
        """Full dump — BM25 needs the corpus in memory, vectors do not."""
        raw = self.collection.get(include=["documents", "metadatas"])
        chunks = []
        for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"]):
            chunks.append(
                Chunk(
                    id=cid,
                    document_id=meta["document_id"],
                    content=doc,
                    strategy=meta["strategy"],
                    char_start=int(meta["char_start"]),
                    char_end=int(meta["char_end"]),
                    token_count=int(meta.get("token_count", 0)),
                )
            )
        return chunks

    def query(self, query_vector: list[float], k: int) -> list[RetrievalResult]:
        raw = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for rank, (cid, doc, meta, dist) in enumerate(
            zip(raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]), start=1
        ):
            results.append(
                RetrievalResult(
                    chunk_id=cid,
                    document_id=meta["document_id"],
                    content=doc,
                    char_start=int(meta["char_start"]),
                    char_end=int(meta["char_end"]),
                    rank=rank,
                    score=1.0 - float(dist),  # cosine distance -> similarity
                    retriever_name="vector",
                )
            )
        return results
