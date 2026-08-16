# Results

## Setup

| | |
|---|---|
| Corpus | 3 markdown documents (~2,900 words), synthetic payments API documentation |
| Chunks | 14 (fixed) / 15 (recursive) / 16 (semantic) |
| Golden set | 31 questions — 11 refunds, 11 disputes, 9 invoices; 16 factual, 10 semantic, 3 keyword, 2 multi-hop |
| Sweep hardware | Kaggle T4 (single GPU), ~1.5 h wall clock for 17 configurations |
| Service hardware | Intel Core i7-1255U @ 1.70 GHz, 16 GB RAM, Windows 11, Docker Desktop, CPU only |
| Embedding model (sweep) | BAAI/bge-m3 |
| Embedding model (service) | BAAI/bge-small-en-v1.5 |
| Reranker | BAAI/bge-reranker-v2-m3 (cross-encoder) |
| Generator | llama-3.1-8b-instant via Groq |
| Judge | llama-3.3-70b-versatile and llama-3.1-8b-instant — see Limitations |
| Configurations | 17 of 18 completed (one lost to a generator rate limit mid-run) |

---

## Leaderboard

| Config | Recall@5 | MRR | nDCG@5 | Correct | Faithful | Retr ms | Rerank ms |
|---|---|---|---|---|---|---|---|
| recursive__bm25__cross-encoder | 1.000 | 0.984 | 0.988 | 0.887 | 0.984 | 0 | 589 |
| recursive__hybrid__cross-encoder | 1.000 | 0.984 | 0.988 | 0.871 | 0.984 | 7 | 815 |
| recursive__vector__cross-encoder | 1.000 | 0.984 | 0.988 | 0.855 | 0.984 | 7 | 809 |
| fixed__bm25__cross-encoder | 1.000 | 0.962 | 0.972 | 0.871 | 0.952 | 0 | 684 |
| fixed__hybrid__cross-encoder | 1.000 | 0.962 | 0.972 | 0.855 | 0.952 | 8 | 865 |
| fixed__vector__cross-encoder | 1.000 | 0.962 | 0.972 | 0.919 | 1.000 | 8 | 1151 |
| semantic__hybrid__cross-encoder | 1.000 | 0.952 | 0.964 | 0.855 | 1.000 | 7 | 1033 |
| semantic__vector__cross-encoder | 1.000 | 0.952 | 0.964 | 0.855 | 1.000 | 8 | 1044 |
| semantic__vector__norerank | 1.000 | 0.931 | 0.948 | 0.710 | 0.984 | 5 | 0 |
| semantic__bm25__cross-encoder | 0.968 | 0.919 | 0.932 | 0.855 | 1.000 | 0 | 737 |
| fixed__vector__norerank | 1.000 | 0.911 | 0.930 | 0.919 | 1.000 | 5 | 0 |
| recursive__vector__norerank | 1.000 | 0.909 | 0.928 | 0.903 | 0.968 | 5 | 0 |
| recursive__hybrid__norerank | 1.000 | 0.876 | 0.908 | 0.871 | 0.968 | 77 | 0 |
| semantic__hybrid__norerank | 1.000 | 0.869 | 0.902 | 0.806 | 0.984 | 5 | 0 |
| fixed__bm25__norerank | 0.903 | 0.855 | 0.868 | 0.855 | 0.887 | 0 | 0 |
| recursive__bm25__norerank | 0.935 | 0.828 | 0.856 | 0.823 | 0.935 | 0 | 0 |
| semantic__bm25__norerank | 0.935 | 0.793 | 0.829 | 0.806 | 0.968 | 0 | 0 |

Precision@5 is omitted on purpose. With one evidence span per question and k=5
its ceiling is about 0.2, so it compresses every configuration into a narrow
band. MRR and nDCG@5 are the retrieval numbers to read.

**Recall@5 saturates at 1.000 for 14 of 17 configurations.** With 14–16 chunks
and k=5, the top 5 nearly always contains the answer, so recall cannot
discriminate here. It is reported because the three exceptions are informative
(see §5), not because the column ranks anything.

---

## What the sweep says

### 1. Reranking improved retrieval in every pairing, and rescued weak retrievers most

Cross-encoder reranking won on MRR in **8 of 8** matched pairs, mean gain
**+0.091**. The size of the gain is inversely proportional to how good the base
retriever already was:

| pairing | MRR off | MRR on | gain |
|---|---|---|---|
| recursive + bm25 | 0.828 | 0.984 | **+0.156** |
| semantic + bm25 | 0.793 | 0.919 | +0.126 |
| recursive + hybrid | 0.876 | 0.984 | +0.108 |
| fixed + bm25 | 0.855 | 0.962 | +0.107 |
| semantic + hybrid | 0.869 | 0.952 | +0.083 |
| recursive + vector | 0.909 | 0.984 | +0.075 |
| fixed + vector | 0.911 | 0.962 | +0.051 |
| semantic + vector | 0.931 | 0.952 | +0.021 |

The three weakest pairings are all BM25; reranking recovers them to within
0.065 of the best configuration in the sweep. Reranking is a repair for poor
first-stage ranking more than an improvement on good ranking.

### 2. It cost about 67× retrieval latency to buy that

Mean rerank latency **859 ms** (589–1151) against mean retrieval latency **8 ms**
(0–77). Not the ~4× I assumed before measuring: reranking is two orders of
magnitude slower than the stage it corrects.

For an interactive service, +0.09 MRR does not obviously justify +859 ms, which
is why the reranker ships **off by default** and is exposed as its own
configuration flag rather than folded into the retriever.

### 3. The retrieval gain barely reached the answer

Mean correctness gain from reranking: **+0.034**, and it won on correctness in
only 5 of 8 pairings. One pairing got worse — `recursive + vector` fell from
0.903 to 0.855.

This is the finding the harness exists to produce. Reranking moves the right
chunks into better *positions*, but they were already inside the top 5, so the
generator sees nearly the same context and produces nearly the same answer.
Scoring retrieval and generation separately shows a large retrieval improvement
delivering a small end-to-end one; a single accuracy metric would have shown
+0.034 and hidden the +0.091 entirely.

### 4. Semantic chunking lost on both axes

| chunker | mean MRR | mean correctness | mean faithfulness |
|---|---|---|---|
| fixed | **0.930** | **0.884** | 0.958 |
| recursive | 0.927 | 0.868 | 0.971 |
| semantic | 0.903 | 0.815 | **0.989** |

The most expensive strategy to build — it embeds every sentence before grouping
— placed last on retrieval and last on correctness. Naive fixed-size windows
beat it.

The sharpest case is `semantic__vector__norerank`: MRR 0.931, one of the better
retrieval scores in the sweep, but correctness **0.710**, the worst of all 17.
Right chunks, wrong answers. Semantic boundaries cut where topics shift, which
on this corpus separated a definition from the qualifying sentence that followed
it — retrievable, but not sufficient to answer from.

### 5. BM25 is the only retriever that ever missed the evidence

| retriever | mean MRR | mean recall@5 |
|---|---|---|
| vector | **0.942** | 1.000 |
| hybrid | 0.929 | 1.000 |
| bm25 | 0.890 | 0.957 |

All four sub-1.0 recall scores in the sweep are BM25 configurations. And the
lowest-faithfulness configuration in the sweep, `fixed__bm25__norerank` at
0.887, is also the lowest-recall one at 0.903 — the two lowest numbers in the
table belong to the same row.

That is the causal chain the project set out to make visible: lexical retrieval
missed the evidence, the generator received context that did not contain the
answer, and its output drifted from the source. Adding a reranker to that same
pairing lifts faithfulness to 0.952. Without separate retrieval and generation
scoring, this would have looked like a generation problem.

Hybrid did not beat vector here, which the small corpus explains: RRF's value is
recovering documents one retriever missed, and with 14–16 chunks vector alone
already returns the evidence for every question.

### 6. No configuration was right without being grounded

Correctness minus faithfulness was **negative in 17 of 17** configurations, mean
gap **−0.120**. Not one configuration answered correctly from parametric
knowledge while ignoring its retrieved context.

This is the opposite of what I expected to find. The premise — that a
configuration can look accurate while the generator quietly answers from memory,
then collapse on unfamiliar data — did not appear here. Two plausible reasons,
and I cannot distinguish them with this data:

- the system prompt forbids answering outside the context and instructs the
  model to say so, and the generator complied;
- faithfulness ranges 0.887–1.000, near enough to saturation that the metric may
  simply lack resolution on a corpus this easy.

A corpus the model has genuinely never seen would settle it. The mechanism to
detect the failure is built and working; this corpus did not exhibit it.

### 7. Questions that no configuration answered fully

Four questions were never scored `correct` in any fully-judged configuration:

| question | type | why |
|---|---|---|
| `refund-settlement-time` | factual | answer needs card *and* bank-debit timings; the model returns one |
| `dispute-initial-status` | factual | status and the `due_by` timestamp are one sentence; the model returns the status only |
| `invoice-finalise-effect` | factual | three consequences of finalising; the model returns one or two |
| `dispute-efw-vs-rate` | multi-hop | evidence spans two sections; a single passage answers half |

The first three share a cause and it is not retrieval — every one of them
retrieved its evidence chunk at rank 1. The expected answers bundle two or three
facts, the generator returns the most prominent one, and the judge scores that
`partially-correct` (0.5). This is a generation-completeness failure that no
amount of reranking touches, and it puts a ceiling of roughly 0.90 on
correctness for this golden set.

The multi-hop question is the genuine retrieval failure of the four: its two
evidence spans sit in different sections, and a top-5 that surfaces one of them
looks like a partial hit to the metric and a complete miss to the generator.

---

## Service latency

Measured locally with `scripts/bench_latency.py` against the running container.
Not measured on Kaggle — a T4's p95 says nothing about a CPU-only service.

**Retrieval only, generation excluded.** 150 requests per configuration, after
warmup. This isolates the stage the cache affects; end-to-end latency is
dominated by the LLM call, which caching cannot touch.

| | p50 | p95 | p99 |
|---|---|---|---|
| Cache off (`DOCQA_CACHE=0`) | 54.8 ms | 99.2 ms | 142.7 ms |
| Cache on, warm | **26.0 ms** | **48.5 ms** | **74.7 ms** |

Caching query embeddings roughly **halves retrieval latency at every
percentile**. The consistency across p50/p95/p99 is what makes it credible — a
real effect shifts the whole distribution, not just the tail.

The absolute numbers are small because the corpus is small. The ratio is the
transferable part, and it grows with corpus size and query repetition.

Cold-start numbers are excluded deliberately. A first request after container
start pays a one-off embedding-model load (~11 s), which dominates any 5-sample
percentile and measures startup, not steady-state service latency.

### Reranking in the service

Not measured on CPU. The sweep's GPU figure of 859 ms mean is already
disqualifying for an interactive path, and a CPU cross-encoder would be
substantially slower. The reranker ships off by default.

---

## What changed as a result

**Default configuration: `recursive` chunking, `hybrid` retrieval, reranker
off.**

Not the sweep leader. `recursive__bm25__cross-encoder` tops the table at MRR
0.984, but it buys +0.108 MRR for +589 ms and +0.016 correctness over the
non-reranked hybrid. For an interactive service that is the wrong trade, and the
harness is what made the trade legible rather than a guess.

Recursive over fixed despite fixed's marginally higher mean MRR (0.930 vs
0.927): the difference is inside noise on 31 questions, and recursive scores
higher on faithfulness (0.971 vs 0.958) while producing chunks aligned to
document headings, which makes retrieved passages readable in the UI.

The reranker stays implemented, tested, and one config flag away — the sweep
established what it costs and what it buys, which is the point.

---

## Limitations

- **31 questions.** Differences of a few points are noise. No significance
  testing was run; treat gaps under ~0.05 as ties.
- **The corpus is 3 documents, ~2,900 words, one domain.** Recall@5 saturates at
  1.000 almost everywhere, so that column cannot rank anything. Conclusions
  about chunking are about this corpus, not about chunking.
- **Two judge models.** Groq's free-tier daily token quotas ran out mid-run.
  Roughly 200 items were judged by `llama-3.3-70b-versatile` and the remaining
  ~290 by `llama-3.1-8b-instant`, at temperature 0 with an identical prompt.
  The 8B judge is the same model that generated the answers, which is a weaker
  measurement — a judge should be stronger than what it grades. Verdicts are
  cached per (question, answer, context) triple, so a single-model re-judge is a
  matter of deleting the cache and spending three days of quota.
- **One configuration is missing.** `fixed__hybrid__norerank` hit a generator
  rate limit mid-run and is absent; the leaderboard has 17 rows, not 18.
- **Judge failures were originally recorded as scores of zero.** The first run
  produced a leaderboard where 14 of 17 configurations showed faithfulness
  0.000, which was a rate-limited judge, not a measurement. Unjudged items now
  record `null`, summaries carry `judge_coverage`, and the leaderboard prints
  `--` below 90% coverage rather than a number. All 17 configurations in this
  table are at 100% coverage.
- **Sweep quality used `bge-m3` on GPU; the service uses `bge-small-en-v1.5` on
  CPU.** The leaderboard ranking is assumed to transfer between them. Not
  verified.
- **Ordinal averaging.** Correctness and faithfulness are three-point scales
  mapped to {0, 0.5, 1}. Averaging ordinals is a convenience, not a measurement.
- **Latency was measured on one machine, single-threaded, no concurrency.**
  Nothing here says how the service behaves under load.