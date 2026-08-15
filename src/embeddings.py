"""Embedding.

One interface, three reasons it exists:
- the cache lives here, so nothing downstream has to know about it;
- the model is loaded lazily, so importing the module in a test is free;
- ``device`` is resolved once, so the same code runs on a Kaggle T4 and a laptop.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from .cache import EmbeddingCache
from .config import resolve_device


class Embedder(ABC):
    name: str
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "auto",
        cache: EmbeddingCache | None = None,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self.name = model_name
        self.device = resolve_device(device)
        self.cache = cache
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None
        self._dim: int | None = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device=self.device)
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            _ = self.model
        return int(self._dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        pending_idx: list[int] = []
        pending_txt: list[str] = []

        if self.cache is not None:
            for i, text in enumerate(texts):
                hit = self.cache.get(self.name, text)
                if hit is None:
                    pending_idx.append(i)
                    pending_txt.append(text)
                else:
                    results[i] = hit
        else:
            pending_idx = list(range(len(texts)))
            pending_txt = list(texts)

        if pending_txt:
            vectors = self.model.encode(
                pending_txt,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            vectors = [[float(x) for x in row] for row in vectors]
            for i, vector in zip(pending_idx, vectors):
                results[i] = vector
            if self.cache is not None:
                self.cache.put_many(self.name, pending_txt, vectors)

        return [r for r in results if r is not None]


class HashEmbedder(Embedder):
    """Deterministic, dependency-free embedder used by tests and CI.

    Not semantically meaningful — it exists so the whole pipeline can be
    exercised without downloading a model or touching a GPU.
    """

    def __init__(self, dim: int = 64, cache: EmbeddingCache | None = None):
        self.name = f"hash-{dim}"
        self._dim = dim
        self.cache = cache

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vector = [0.0] * self._dim
            for token in text.lower().split():
                idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._dim
                vector[idx] += 1.0
            norm = sum(v * v for v in vector) ** 0.5 or 1.0
            out.append([v / norm for v in vector])
        return out


def build_embedder(
    model_name: str,
    device: str = "auto",
    cache: EmbeddingCache | None = None,
) -> Embedder:
    if model_name.startswith("hash"):
        dim = int(model_name.split("-")[-1]) if "-" in model_name else 64
        return HashEmbedder(dim=dim, cache=cache)
    return SentenceTransformerEmbedder(model_name, device=device, cache=cache)
