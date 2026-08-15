"""Embedding cache.

Query embedding is the first thing on the critical path of every request, and
on CPU it is also the most expensive thing. Caching it is what takes p95 from
"noticeably slow" to "feels instant" on repeat queries — the number worth
quoting is measured by ``scripts/bench_latency.py``, locally, never on Kaggle.

Design notes:
- Key is ``sha256(model_name | text)``. Including the model name means swapping
  embedding models cannot silently serve stale vectors.
- SQLite, not Redis. It survives container restarts via a bind mount, needs no
  second service, and the cache is small enough that the lookup is microseconds.
- Vectors are stored as raw float32 bytes, not JSON: ~4x smaller and no parse.
"""

from __future__ import annotations

import hashlib
import sqlite3
import struct
import threading
from collections import OrderedDict
from pathlib import Path


def cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


def _pack(vector) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class EmbeddingCache:
    """Two tiers: an in-process LRU in front of SQLite on disk."""

    def __init__(self, path: str | Path, memory_size: int = 2048, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self.memory_size = memory_size
        self._memory: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "  key TEXT PRIMARY KEY,"
                "  model TEXT NOT NULL,"
                "  dim INTEGER NOT NULL,"
                "  vector BLOB NOT NULL)"
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.commit()
        else:
            self._conn = None

    # -- reads ----------------------------------------------------------
    def get(self, model: str, text: str) -> list[float] | None:
        if not self.enabled:
            return None
        key = cache_key(model, text)
        with self._lock:
            if key in self._memory:
                self._memory.move_to_end(key)
                self.hits += 1
                return self._memory[key]

        row = self._conn.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
        if row is None:
            with self._lock:
                self.misses += 1
            return None

        vector = _unpack(row[0])
        self._remember(key, vector)
        with self._lock:
            self.hits += 1
        return vector

    def get_many(self, model: str, texts: list[str]) -> list[list[float] | None]:
        return [self.get(model, text) for text in texts]

    # -- writes ---------------------------------------------------------
    def put(self, model: str, text: str, vector: list[float]) -> None:
        if not self.enabled:
            return
        key = cache_key(model, text)
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (key, model, dim, vector) VALUES (?, ?, ?, ?)",
            (key, model, len(vector), _pack(vector)),
        )
        self._conn.commit()
        self._remember(key, vector)

    def put_many(self, model: str, texts: list[str], vectors: list[list[float]]) -> None:
        if not self.enabled or not texts:
            return
        rows = [
            (cache_key(model, t), model, len(v), _pack(v))
            for t, v in zip(texts, vectors)
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (key, model, dim, vector) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        for (key, _, _, _), vector in zip(rows, vectors):
            self._remember(key, vector)

    # -- housekeeping ---------------------------------------------------
    def _remember(self, key: str, vector: list[float]) -> None:
        with self._lock:
            self._memory[key] = vector
            self._memory.move_to_end(key)
            while len(self._memory) > self.memory_size:
                self._memory.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        rows = 0
        if self.enabled:
            rows = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "rows_on_disk": rows,
            "rows_in_memory": len(self._memory),
        }

    def clear(self) -> None:
        with self._lock:
            self._memory.clear()
        if self.enabled:
            self._conn.execute("DELETE FROM embeddings")
            self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
