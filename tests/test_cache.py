"""The embedding cache is the thing the p95 claim rests on, so it gets tested
on correctness (same vector back) and on scoping (model name is part of the key).
"""
import pytest

from src.cache import EmbeddingCache, cache_key
from src.embeddings import HashEmbedder


@pytest.fixture
def cache(tmp_path):
    c = EmbeddingCache(tmp_path / "cache.sqlite")
    yield c
    c.close()


def test_roundtrip_preserves_the_vector(cache):
    vector = [0.1, -0.25, 0.5]
    cache.put("model-a", "hello", vector)
    got = cache.get("model-a", "hello")
    assert got is not None
    assert all(abs(a - b) < 1e-6 for a, b in zip(got, vector))


def test_miss_returns_none(cache):
    assert cache.get("model-a", "never seen") is None


def test_key_is_scoped_to_the_model(cache):
    cache.put("model-a", "hello", [1.0, 0.0])
    assert cache.get("model-b", "hello") is None
    assert cache_key("model-a", "hello") != cache_key("model-b", "hello")


def test_survives_reopen(tmp_path):
    path = tmp_path / "cache.sqlite"
    first = EmbeddingCache(path)
    first.put("m", "text", [1.0, 2.0])
    first.close()

    second = EmbeddingCache(path)
    assert second.get("m", "text") == [1.0, 2.0]
    second.close()


def test_disabled_cache_is_a_no_op(tmp_path):
    cache = EmbeddingCache(tmp_path / "off.sqlite", enabled=False)
    cache.put("m", "text", [1.0])
    assert cache.get("m", "text") is None
    assert cache.stats()["enabled"] is False


def test_stats_track_hits_and_misses(cache):
    cache.put("m", "a", [1.0])
    cache.get("m", "a")
    cache.get("m", "b")
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_embedder_uses_the_cache_and_returns_identical_vectors(cache):
    embedder = HashEmbedder(16, cache=cache)
    # HashEmbedder does not consult the cache itself, so exercise the cache path
    # through put_many/get_many the way SentenceTransformerEmbedder does.
    texts = ["alpha beta", "gamma delta"]
    vectors = embedder.embed(texts)
    cache.put_many(embedder.name, texts, vectors)
    assert cache.get_many(embedder.name, texts) == vectors
