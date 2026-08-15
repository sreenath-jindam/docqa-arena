# Results

> **This file is a template.** Every `TODO` is a number you have not measured
> yet. Do not fill any of them in from intuition — the whole point of the
> harness is that the numbers are usually not what you expected.

## Setup

| | |
|---|---|
| Corpus | TODO — N documents, ~N words |
| Golden set | 31 questions (11 refunds, 11 disputes, 9 invoices) |
| Sweep hardware | Kaggle T4, session of TODO hours |
| Service hardware | TODO — e.g. MacBook Air M2, 8GB, Docker Desktop |
| Embedding model (sweep) | BAAI/bge-m3 |
| Embedding model (service) | BAAI/bge-small-en-v1.5 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Generator | TODO |
| Judge | TODO |
| Date | TODO |

---

## Leaderboard

Generated with `python scripts/leaderboard.py --sort mrr --markdown`.

| Config | Recall@5 | MRR | nDCG@5 | Correct | Faithful | Retr ms | Rerank ms |
|---|---|---|---|---|---|---|---|
| TODO paste the 18 rows here | | | | | | | |

Precision@5 is omitted from the headline table on purpose. With one evidence
span per question and k=5, its ceiling is around 0.2, so it compresses every
configuration into a narrow band and compares badly. MRR and nDCG@5 are the
retrieval numbers to read.

---

## What the sweep says

### 1. Reranking

TODO — quantify. The claim to support or refute: reranking dominates on quality
and costs several times retrieval latency.

Fill in:
- mean MRR across the 9 reranked configs vs the 9 without: TODO vs TODO
- mean rerank latency: TODO ms, against TODO ms for retrieval alone
- does it help every retriever equally, or mostly rescue the weak ones?

### 2. Chunking

TODO — which of `fixed` / `recursive` / `semantic` won, and by how much?

Worth checking specifically: semantic chunking is the most expensive to build
and the easiest to assume is best. If it did not win, say so.

### 3. Retrieval method

TODO — vector vs BM25 vs hybrid.

Look at this per query type. `evaluations.json` carries `query_type`, so you can
slice it: BM25 usually wins on `keyword` questions and loses on `semantic` ones,
and hybrid should beat both on the mix. If it does not, that is the finding.

### 4. The correctness / faithfulness gap

TODO — this is the most interesting column, so do not skip it.

For each configuration, compute correctness minus faithfulness. A large positive
gap means the generator was right without support from the retrieved context —
right because the model already knew, not because retrieval worked. Those
configurations fail silently on out-of-distribution queries.

- widest gap: TODO
- narrowest gap: TODO
- do the configurations with the best retrieval scores also have the smallest gap?

### 5. Questions no configuration got right

TODO — grep `evaluations.json` for questions where every config scored
`incorrect`. Usually two or three, and they usually share a cause: evidence
split across a chunk boundary, or a question that needs two documents.

---

## Service latency

Measured locally with `scripts/bench_latency.py`, against the running container.
Not measured on Kaggle — a T4's p95 says nothing about a CPU-only service.

### With and without the embedding cache

p50	p95	p99
Cache off	54.8 ms	99.2 ms	142.7 ms
Cache on, warm	26.0 ms	48.5 ms	74.7 ms

Retrieval only, generation excluded. 150 requests per configuration against the containerized service, after warmup. Measured on [your machine], not on Kaggle.

Retrieval only (`--no-generate`), which isolates the part the cache affects:

| | p50 | p95 | p99 |
|---|---|---|---|
| Cache off | TODO | TODO | TODO |
| Cache on, warm | TODO | TODO | TODO |

Cache hit rate after the benchmark: TODO%

Note whether generation was included in each number. Most of end-to-end latency
is the LLM call, which the cache does not touch, so the retrieval-only table is
the honest measure of what caching bought.

### Reranking, in the service

| | p50 | p95 |
|---|---|---|
| Reranker off | TODO | TODO |
| Reranker on | TODO | TODO |

---

## What was changed as a result

TODO — the sweep is only worth running if it changed something. State the
configuration change, the before and after numbers on the same golden set, and
whether the improvement held.

Before: TODO
After: TODO
Change: TODO

---

## Limitations

- 31 questions is a small sample. Differences of a few points are noise. No
  significance testing was run.
- The corpus is three documents in one domain. Conclusions about chunking are
  about this corpus, not about chunking.
- The judge is an LLM. It is more consistent than a human across 558
  question-config pairs, and less accurate than a careful one on any single pair.
- Sweep quality was measured with `bge-m3` on GPU; the deployed service uses
  `bge-small-en-v1.5` on CPU. The leaderboard ranking is assumed to transfer,
  which is not verified.
- Correctness and faithfulness come from a three-point ordinal scale mapped to
  {0, 0.5, 1}. Averaging ordinals is a convenience, not a measurement.
