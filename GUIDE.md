# Step-by-step guide

Everything from unzipping this folder to a deployed service with a filled-in
RESULTS.md. Follow it in order — later steps assume earlier ones ran.

Times are rough. The 30-day plan at the end maps these steps onto weeks.

---

## Part 0 — Before you touch anything (20 min)

### 0.1 Get an LLM API key

You need one free-tier key for generation and judging. **Groq** is the
recommendation: it is free, fast, and OpenAI-compatible.

1. Open <https://console.groq.com> and sign up.
2. Left sidebar → **API Keys** → **Create API Key**.
3. Copy it. It starts with `gsk_`. You will not see it again.

Alternatives that work with zero code changes — set `LLM_BASE_URL` accordingly:

| provider | base URL |
|---|---|
| Groq | `https://api.groq.com/openai/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Local Ollama | `http://localhost:11434/v1` |
| OpenAI | `https://api.openai.com/v1` |

### 0.2 Install Docker Desktop

<https://docs.docker.com/desktop/> — install, launch it, wait for the whale
icon to stop animating. Verify:

```bash
docker --version
docker compose version
```

### 0.3 Create a Kaggle account

<https://www.kaggle.com> → sign up → **Settings** → **Phone Verification**.

You must verify your phone number. Without it, Kaggle will not give you GPU or
internet access in notebooks, and both are required in Part 3. Do this now, not
on the day you need the GPU.

---

## Part 1 — Get it running locally (45 min)

### 1.1 Unzip and look around

```bash
unzip docqa-arena.zip
cd docqa-arena
ls
```

You should see `docker-compose.yml`, `api/`, `src/`, `eval/`, `scripts/`,
`tests/`, `app.py`, `README.md`, `RESULTS.md`.

### 1.2 Set up a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This installs torch and takes a few minutes. Go make tea.

### 1.3 Run the tests

```bash
pytest
```

Expect **56 passed**. These run fully offline — no API key, no GPU, no model
download. If they pass, the core logic is intact and any later problem is
environmental.

### 1.4 Add your API key

```bash
cp .env.example .env
```

Open `.env` and replace both `gsk_replace_me` values with your real key.
`.env` is gitignored. Never commit it.

### 1.5 Check the golden set resolves

```bash
python scripts/resolve_spans.py
```

Expect `31/31 examples resolved cleanly` and a list of character offsets. If it
raises, the corpus and the golden set have drifted — fix that before anything
else, because every number downstream depends on it.

### 1.6 Build the index

```bash
python scripts/ingest.py --config configs/local.yaml
```

First run downloads `bge-small-en-v1.5` (~130 MB). It writes `storage/chroma/`.

### 1.7 Start the API

```bash
uvicorn api.main:app --reload
```

Open <http://localhost:8000/docs> — the interactive API docs. Try `POST /ask`
with:

```json
{"query": "How long does a card refund take to settle?"}
```

You should get an answer plus the passages it came from and per-stage timings.

### 1.8 Start the UI

In a **second terminal** (leave uvicorn running):

```bash
source .venv/bin/activate
streamlit run app.py
```

Opens <http://localhost:8501>. Ask a question. The answer appears on the left,
the retrieved passages on the right, the timings underneath.

**Checkpoint:** you have a working RAG service. Commit here.

---

## Part 2 — Containerize and measure latency (half a day)

### 2.1 Bring up the stack

Stop the manual uvicorn and streamlit processes first (Ctrl-C in both).

```bash
docker compose up --build
```

The first build takes 10–15 minutes — it installs torch and bakes the embedding
model into the image so no request ever pays a model download.

- API: <http://localhost:8000/docs>
- UI: <http://localhost:8501>

### 2.2 Measure latency with the cache on

```bash
docker compose up -d                  # detached
python scripts/bench_latency.py --repeats 30 --compare-cache
```

Then isolate retrieval, since the LLM call dominates end-to-end and the cache
does not touch it:

```bash
python scripts/bench_latency.py --repeats 30 --no-generate
```

Write every number down. These go in RESULTS.md.

### 2.3 Measure with the cache off

```bash
DOCQA_CACHE=0 docker compose up -d --force-recreate
python scripts/bench_latency.py --repeats 30 --no-generate
DOCQA_CACHE=1 docker compose up -d --force-recreate     # turn it back on
```

The gap between 2.2 and 2.3 is your caching result. **Whatever it is, that is
the number you quote.** Do not round it toward something more impressive — the
real figure survives a follow-up question and an invented one does not.

### 2.4 Fill in the latency section of RESULTS.md

Open `RESULTS.md` and replace the TODOs in **Service latency**. Name your
machine. Note whether generation was included in each row.

**Checkpoint:** commit. You now have the SDE half of the project.

---

## Part 3 — Run the sweep on Kaggle (one session, 3–6 hours)

### 3.1 Push the repo to GitHub

The notebook clones your repo, so it has to be somewhere Kaggle can reach.

```bash
git init
git add .
git commit -m "Document Q&A service with retrieval benchmark harness"
```

Then on GitHub: **+** (top right) → **New repository** → name it
`docqa-arena` → **Public** → **Create**. Do not add a README, you have one.
Follow the "push an existing repository" commands GitHub shows you.

Confirm `.env` is **not** in the repo. If you see it on GitHub, rotate your API
key immediately.

### 3.2 Create the Kaggle notebook

1. Go to <https://www.kaggle.com>.
2. Left sidebar → **Create** → **New Notebook**.
3. File → **Import Notebook** → **Upload** → choose
   `notebooks/kaggle_sweep.ipynb` from your unzipped folder.

### 3.3 Configure the session

Right sidebar (click **⋮** or the settings panel if it is collapsed):

- **Accelerator** → **GPU T4 x2**
- **Internet** → **On**

Both matter. No accelerator means no GPU. No internet means the notebook cannot
clone your repo, download models, or reach the LLM API.

Kaggle gives you roughly 30 GPU-hours per week, and a session ends after 9
hours or when you close the tab.

### 3.4 Add your API key as a secret

1. Top menu → **Add-ons** → **Secrets**.
2. **Add a new secret**. Label: `LLM_API_KEY`. Value: your `gsk_...` key.
3. Toggle it **attached** to this notebook.

Never paste a key into a cell. Kaggle notebooks are public by default and the
cell output is saved with the notebook.

### 3.5 Point the notebook at your repo

In **cell 1**, replace:

```python
REPO = "https://github.com/YOUR_USERNAME/docqa-arena.git"
```

with your actual URL.

### 3.6 Run the cells in order

Click each cell and press **Shift+Enter**, or **Run All** — but run them one at
a time the first time so you see where anything fails.

| cell | does | takes |
|---|---|---|
| 1 | clone the repo | seconds |
| 2 | install chromadb, openai, yaml | ~2 min |
| 3 | load the secret, set `DOCQA_DATA_DIR=/kaggle/working` | seconds |
| 4 | verify the golden set resolves | seconds |
| 5 | **smoke-test one config** | ~5 min |
| 6 | the full 18-config sweep | 2–5 hours |
| 7 | print the leaderboard | seconds |
| 8 | zip results + index for download | ~2 min |

**Do not skip cell 5.** It downloads `bge-m3` (2.2 GB) and runs one
configuration end to end. If your API key is wrong or a model name is bad, you
find out in five minutes instead of three hours.

### 3.7 While cell 6 runs

It prints progress per configuration. Leave the tab open — closing it kills the
session.

If it dies partway, that is fine and expected: restart the notebook, re-run
cells 1–3, then re-run cell 6. Configurations with a `summary.json` are skipped
and it picks up where it stopped. This is why the sweep writes after every
config instead of at the end.

### 3.8 Download before you close the tab

Run cell 8, then in the right sidebar open the **Output** panel, find
`sweep-artifacts.zip`, and click download.

**`/kaggle/working` is deleted when the session ends.** If you close the tab
without downloading, the entire sweep is gone and you run it again. Download
first, admire the leaderboard second.

### 3.9 Unpack locally

```bash
cd docqa-arena
unzip ~/Downloads/sweep-artifacts.zip -d kaggle-output
cp -r kaggle-output/results/* eval/results/
python scripts/leaderboard.py --sort mrr
```

**Checkpoint:** commit `eval/results/`. Those artifacts are the project.

---

## Part 4 — Analyze, fix, deploy (a week)

### 4.1 Read the leaderboard properly

```bash
python scripts/leaderboard.py --sort mrr --markdown > /tmp/board.md
python scripts/leaderboard.py --sort faithfulness
python scripts/leaderboard.py --sort ndcg_at_k
```

Sort by different columns. The configuration that wins on MRR is often not the
one that wins on faithfulness, and that disagreement is the interesting part.

### 4.2 Slice by query type

`eval/results/<slug>/evaluations.json` carries `query_type` on every row. Write
a few lines of pandas to group by it. Typical pattern worth confirming or
refuting on your data: BM25 wins on `keyword`, vector wins on `semantic`,
hybrid beats both on the mix.

### 4.3 Find the questions nothing got right

Look for question IDs that scored `incorrect` under every configuration. There
are usually two or three and they usually share one cause — evidence split
across a chunk boundary, or a question needing two documents. That cause is
your fix.

### 4.4 Compute the correctness minus faithfulness gap

For each config, subtract mean faithfulness from mean correctness. A large
positive gap means the model was right *without* support from the retrieved
context — right because it already knew, not because retrieval worked. Those
configurations look fine here and collapse on a corpus the model has not seen.

This is the single most valuable paragraph in your RESULTS.md.

### 4.5 Implement one fix and re-measure

Pick the one change the analysis points to. Change it. Re-run the affected
configurations with `--force`:

```bash
python eval/run_sweep.py --only recursive__hybrid__cross-encoder --force
```

Report before and after on the same golden set. One measured improvement beats
five speculative ones.

### 4.6 Serve the GPU-built index

```bash
python scripts/export_index.py --import kaggle-output/index.zip --replace
docker compose up -d --force-recreate
curl -s localhost:8000/health | python -m json.tool
```

Your CPU-only container is now serving an index built with `bge-m3` on a T4.
Note that `configs/local.yaml` must have `embed_model: BAAI/bge-m3` for query
embeddings to match the index — change it, and expect the service to download
2.2 GB on first start.

If that is too heavy for your deployment target, keep `bge-small` and rebuild
locally; just say so in RESULTS.md rather than implying the served index is the
benchmarked one.

### 4.7 Finish RESULTS.md

Replace every remaining TODO. If you cannot fill one in, delete the section
rather than inventing a number.

### 4.8 Deploy (optional)

Fly.io or Render both take a Dockerfile directly. Set `LLM_API_KEY` and
`LLM_BASE_URL` as environment secrets in their dashboard. Expect the image to
be large — the baked-in model is most of it.

---

## The 30-day plan

| week | do |
|---|---|
| 1 | Parts 0 and 1. Swap in your own corpus. Rewrite `eval/golden.csv` for it. |
| 2 | Part 2. Docker, caching, local p95 numbers written down. |
| 3 | Part 3. Golden set finalized, sweep run on Kaggle, results downloaded. |
| 4 | Part 4. Analyze, implement one fix, re-measure, export the index, RESULTS.md. |

---

## Using your own corpus

The bundled corpus is synthetic payments documentation — real enough to
exercise the pipeline, generic enough that nobody will be impressed by it. Swap
it for something you actually know.

1. Drop `.md` or `.txt` files into `data/corpus/`. Delete the samples.
2. Write 30 questions. For each: a question, the expected answer, and a
   **verbatim snippet** from the document that answers it. Multi-hop questions
   join two snippets with `||`.
3. Put them in `eval/golden.csv` with the same six columns.
4. `python scripts/resolve_spans.py` — it fails loudly if a snippet does not
   appear verbatim. Fix until it passes.
5. `python scripts/ingest.py --all-chunkers --reset`
6. `pytest tests/test_golden.py` — this checks every question still has a
   relevant chunk under *every* chunking strategy, which is what makes the
   18-way comparison fair.

A practical shortcut for step 2: ask an LLM to draft the 30 rows from your
documents, then hand-correct every one. Drafting is the boring part; checking
is the part that makes the golden set trustworthy, and it is not skippable.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`** — run commands from the repo
root, not from inside `eval/` or `scripts/`.

**Chroma telemetry errors in the console** — harmless. `ANONYMIZED_TELEMETRY`
is already set to False in `src/store.py`.

**`409 Index is empty`** — run `python scripts/ingest.py` or
`POST /ingest {"reset": true}`.

**Kaggle: "No GPU"** — accelerator not set, or your weekly quota is spent. The
quota resets Saturday 00:00 UTC.

**Kaggle: cannot clone the repo** — Internet is off in the session settings, or
the repo is private.

**Rate limits from the judge mid-sweep** — the sweep catches judge errors per
question and continues, recording the failure. If a whole config is affected,
delete its folder and re-run just that one.

**The sweep skips everything** — that means it already finished. Use `--force`
or delete `eval/results/`.

---

## Three things to be able to say out loud

**"Walk me through it."**
A document Q&A service — FastAPI, Chroma, a Streamlit client, all in
docker-compose. I could not tell whether it was any good, so I built a
benchmark: 31 questions each labelled with the source text that should be
retrieved, run across 18 configurations on a Kaggle T4. *[Your finding about
reranking.]* Caching embeddings took retrieval p95 from *[X]* to *[Y]* locally.

**"Why Kaggle?"**
Cross-encoder reranking over 558 question-config pairs is about ten thousand
forward passes — not practical on my CPU. I ran the sweep on a free T4,
exported the built index and the results, and serve them from a CPU-only
container. Sessions are ephemeral, so the sweep writes per config and resumes.

**"Doesn't this already exist?"**
Yes — Retrieval Arena, RAGEval, Google's rag-playground. I reimplemented the
evaluation approach in Python. The contribution is the analysis on my corpus,
not the framework.

One caution on the Kaggle framing: "I used a free GPU for a batch job" is the
honest version and it is enough. Do not let it grow into a distributed training
story, because that will not survive the follow-up question.
