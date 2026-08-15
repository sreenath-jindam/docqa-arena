"""API contract tests, run entirely offline against the hash embedder and the
echo generator so CI needs no API key, no GPU, and no model download.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")
pytest.importorskip("chromadb")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["DOCQA_CONFIG"] = str(ROOT / "configs" / "test.yaml")
    os.environ["DOCQA_DATA_DIR"] = str(tmp_path_factory.mktemp("docqa"))
    os.environ["LLM_BACKEND"] = "echo"
    os.environ["JUDGE_BACKEND"] = "stub"

    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health_reports_an_indexed_collection(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] > 0


def test_search_returns_ranked_passages(client):
    body = client.post("/search", json={"query": "how long do card refunds take", "top_k": 3}).json()
    assert len(body["passages"]) <= 3
    assert [p["rank"] for p in body["passages"]] == list(range(1, len(body["passages"]) + 1))
    assert body["timings_ms"]["retrieval"] > 0


def test_ask_returns_an_answer_and_its_sources(client):
    body = client.post("/ask", json={"query": "what happens to the processing fee on a refund"}).json()
    assert body["answer"]
    assert body["passages"]
    assert body["timings_ms"]["total"] > 0


def test_ask_can_omit_passages(client):
    body = client.post("/ask", json={"query": "refunds", "include_passages": False}).json()
    assert body["passages"] == []


def test_ask_rejects_an_empty_query(client):
    assert client.post("/ask", json={"query": ""}).status_code == 422


def test_ask_rejects_an_out_of_range_top_k(client):
    assert client.post("/ask", json={"query": "refunds", "top_k": 500}).status_code == 422


def test_metrics_exposes_latency_percentiles_and_cache(client):
    client.post("/ask", json={"query": "refund timing"})
    body = client.get("/metrics").json()
    assert "p95" in body["latency_ms"]
    assert "hit_rate" in body["cache"]


def test_response_time_header_is_set(client):
    response = client.post("/search", json={"query": "disputes"})
    assert float(response.headers["X-Response-Time-Ms"]) > 0
