"""Measure service latency locally, with the cache on and off.

This is the number that belongs in RESULTS.md next to "p95". It is measured
here, on the machine the service actually runs on, against the running
container — never on Kaggle. A T4's p95 says nothing about a CPU-only service,
and quoting one would be the kind of claim that does not survive a follow-up.

    docker compose up -d
    python scripts/bench_latency.py --repeats 30 --compare-cache
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import percentile  # noqa: E402

DEFAULT_QUERIES = [
    "How long does a refund take to settle?",
    "What happens to the application fee when a charge is refunded?",
    "Which object represents a bill a customer needs to pay?",
    "Can a partial refund be issued more than once?",
    "What status does a disputed charge move to?",
]


def hit(api_url: str, query: str, generate: bool) -> float:
    start = time.perf_counter()
    response = requests.post(
        f"{api_url}/ask",
        json={"query": query, "generate": generate, "include_passages": False},
        timeout=120,
    )
    response.raise_for_status()
    return (time.perf_counter() - start) * 1000


def run(api_url: str, queries: list[str], repeats: int, generate: bool, warmup: int = 3) -> dict:
    for query in queries[:warmup]:
        hit(api_url, query, generate)

    samples = []
    for _ in range(repeats):
        for query in queries:
            samples.append(hit(api_url, query, generate))

    return {
        "n": len(samples),
        "mean": statistics.mean(samples),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
        "min": min(samples),
        "max": max(samples),
    }


def show(label: str, stats: dict) -> None:
    print(
        f"{label:<22} n={stats['n']:<5} "
        f"mean={stats['mean']:7.1f}ms  p50={stats['p50']:7.1f}ms  "
        f"p95={stats['p95']:7.1f}ms  p99={stats['p99']:7.1f}ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--no-generate", action="store_true", help="measure retrieval only, excluding the LLM call")
    parser.add_argument("--compare-cache", action="store_true", help="print the cache stats before and after")
    args = parser.parse_args()

    generate = not args.no_generate
    print(f"target: {args.api}   generation: {'on' if generate else 'off'}\n")

    if args.compare_cache:
        before = requests.get(f"{args.api}/metrics", timeout=30).json()["cache"]
        print(f"cache before: {before}")

    cold = run(args.api, DEFAULT_QUERIES, 1, generate, warmup=0)
    show("cold (first pass)", cold)

    warm = run(args.api, DEFAULT_QUERIES, args.repeats, generate)
    show("warm (cached)", warm)

    if args.compare_cache:
        after = requests.get(f"{args.api}/metrics", timeout=30).json()["cache"]
        print(f"cache after:  {after}")

    delta = cold["p95"] - warm["p95"]
    print(f"\np95 improvement: {delta:.0f} ms ({delta / cold['p95'] * 100:.0f}%)" if cold["p95"] else "")
    print("Paste these into RESULTS.md. Note the machine and whether generation was included.")


if __name__ == "__main__":
    main()
