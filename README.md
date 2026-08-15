# Document Q&A with a benchmarked retrieval harness

A containerized RAG service over a document corpus, plus a benchmark that scores
**retrieval and generation separately** across 18 pipeline configurations.

The premise: two RAG configurations can produce equally fluent answers while
differing enormously in whether they retrieved the correct source text. A single
accuracy number cannot tell you which one you have. This repo measures the two
halves independently, so a bad score points at a stage rather than at a vibe.

```
                    ┌─────────────┐
   corpus ──chunk──▶│   Chroma    │──retrieve──▶ rerank? ──▶ generate ──▶ answer
                    └─────────────┘      │           │           │
                                         ▼           ▼           ▼
                              precision@k, recall@k,      correctness,
                              MRR, nDCG@k, latency        faithfulness
                              ── retrieval score ──       ── generation score ──
```

## Quick start

```bash
cp .env.example .env          # add a free-tier API key (Groq, OpenRouter, or a local Ollama)
docker compose up --build
```

- API: http://localhost:8000/docs
- UI: http://localhost:8501

The service ingests `data/corpus/` on first start. Ask a question and the UI
shows the answer next to the passages it was built from — if the passages are
irrelevant, the answer is a hallucination no matter how well it reads.

Without Docker:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/ingest.py --config configs/local.yaml
uvicorn api.main:app --reload
streamlit run app.py
```

## What gets measured

**Retrieval** — precision@k, recall@k, MRR, nDCG@k, hit@k. Scored against chunk
IDs, never against answer text. A golden question labels a character span of
evidence in the source document; a retrieved chunk counts as relevant if its
character range overlaps that span. That definition is independent of chunking
strategy, which is what makes an 18-way comparison meaningful.

**Generation** — correctness (does the answer match the expected answer?) and
faithfulness (is every claim supported by *the retrieved context alone*?).

The interesting cell is high correctness with low faithfulness: the model was
right because it already knew the answer, not because retrieval worked. That
configuration collapses on a corpus the model has never seen, and an end-to-end
accuracy score would never have warned you.

**Latency** sits alongside quality in every row. Reranking is off by default so
its marginal cost and benefit each get their own column rather than being
folded into the retriever.

## The sweep

3 chunkers × 3 retrievers × 2 reranker settings = 18 configurations, each run
against the same fixed golden set.

| | options |
|---|---|
| chunker | `fixed`, `recursive`, `semantic` |
| retriever | `vector`, `bm25`, `hybrid` (RRF) |
| reranker | off, `cross-encoder` |

```bash
python eval/run_sweep.py --config configs/local.yaml
python scripts/leaderboard.py --sort mrr
```

Each configuration writes three files to `eval/results/<slug>/`:

| file | contents |
|---|---|
| `config.json` | exactly what was run |
| `evaluations.json` | per-question detail: retrieved IDs, judge verdict, generated answer |
| `summary.json` | the aggregates the leaderboard reads |

Written to disk, not printed. A printed number is gone when the session ends.

**Resumable.** A configuration whose `summary.json` exists is skipped. Kaggle
sessions end at nine hours or when the tab closes; a sweep that restarts from
config 1 every time never finishes. Delete a folder to force a re-run, or pass
`--force`.

## Why Kaggle

Cross-encoder reranking over 30 questions × 18 configurations × 20 candidates is
about 10,800 forward passes. That is a batch job, not a CPU job. The sweep runs
on a free T4, the built Chroma index is exported, and the local CPU-only
container serves it.

Two separate measurements, deliberately:

- **Sweep quality** — on GPU, `notebooks/kaggle_sweep.ipynb`
- **Service latency** — locally, `scripts/bench_latency.py`

A Kaggle T4's p95 tells you nothing about a CPU container. Never quote one for
the other.

The notebook is about twenty lines and imports `src/`. Nothing runs there that
cannot run with `python eval/run_sweep.py`.

## Caching

Query embedding is first on the critical path of every request and the most
expensive part on CPU. `src/cache.py` is a two-tier cache — in-process LRU in
front of SQLite — keyed on `sha256(model_name | text)`. The model name is in the
key so swapping embedding models cannot serve stale vectors.

```bash
docker compose up -d
python scripts/bench_latency.py --repeats 30 --compare-cache
DOCQA_CACHE=0 docker compose up -d --force-recreate   # measure with it off
```

Numbers go in [RESULTS.md](RESULTS.md), measured locally, with the machine named.

## Golden set

`eval/golden.csv` — 31 questions across three documents, drafted against the
corpus and hand-corrected.

Each row stores a verbatim `evidence_text` snippet rather than raw character
offsets. Hand-labelling offsets is miserable and breaks the moment a document is
edited; a snippet is something a human can write and verify, and it is resolved
to offsets at load time by exact string search. Multi-hop questions separate two
snippets with `||`.

Resolution **fails loudly** on drift. A snippet that no longer appears in its
document means the corpus moved under the golden set, and silently dropping that
row would shrink the benchmark while making two runs look comparable.

```bash
python scripts/resolve_spans.py     # verify every row still resolves
```

To use your own corpus: replace `data/corpus/*.md`, write your own rows, then
run the check above before spending GPU hours.

## Layout

```
docker-compose.yml
api/       main.py  models.py            FastAPI service
src/       chunkers.py  retrievers.py  rerank.py
           pipeline.py  cache.py  store.py  embeddings.py
eval/      golden.csv  run_sweep.py  metrics.py  judge.py  spans.py
           results/<slug>/{config,summary,evaluations}.json
notebooks/ kaggle_sweep.ipynb           thin wrapper importing src/
scripts/   ingest.py  bench_latency.py  export_index.py  leaderboard.py
app.py     Streamlit client
tests/
RESULTS.md
```

## Tests

```bash
pytest
```

56 tests, all offline — a hash embedder and an echo generator stand in for the
real models, so CI needs no API key, no GPU, and no model download. The load
bearing ones: chunkers preserve exact character offsets, metrics match
hand-computed values, the golden set resolves under every chunking strategy, and
the sweep writes its artifacts and then skips finished work.

## Configuration

| profile | for |
|---|---|
| `configs/local.yaml` | CPU, `bge-small-en-v1.5`, the container |
| `configs/kaggle.yaml` | GPU, `bge-m3` + `bge-reranker-v2-m3`, writes to `/kaggle/working` |
| `configs/test.yaml` | offline, no downloads, used by pytest |

No paths are hardcoded. `DOCQA_DATA_DIR` controls where indexes, caches and
results are written; set it to `/kaggle/working` in a notebook and the same code
runs unchanged.

## Prior art

[Retrieval Arena](https://github.com/) (TypeScript), RAGEval, Google's
rag-playground. The evaluation approach here is a reimplementation of Retrieval
Arena's in Python. The contribution is the analysis on this corpus, not the
framework.

## Not done

- Contextualizers (full-document and window-summary chunk enrichment before
  embedding) — the reference project's biggest lever, dropped for time.
- Multiple embedding models as a sweep dimension.
- Statistical significance testing. With 31 questions, small leaderboard gaps
  are noise; treat differences under a few points as ties.
