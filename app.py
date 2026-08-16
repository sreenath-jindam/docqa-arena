"""Streamlit client.

The layout encodes the project's thesis: the answer and the evidence it was
built from sit side by side, and the per-stage timings sit under both. You
cannot look at this screen and mistake a confident answer for a well-retrieved
one — if the passages on the right are irrelevant, the answer on the left is a
hallucination no matter how well it reads.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Document Q&A", page_icon="◆", layout="wide")

st.markdown(
    """
    <style>
      .passage-card { border-left: 3px solid #4c6ef5; padding: 0.6rem 0.9rem;
                      margin-bottom: 0.7rem; background: rgba(76,110,245,0.06);
                      border-radius: 0 6px 6px 0; }
      .passage-meta { font-size: 0.75rem; opacity: 0.7; font-family: ui-monospace, monospace; }
      .stat { font-variant-numeric: tabular-nums; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str):
    return requests.get(f"{API_URL}{path}", timeout=30).json()


def api_post(path: str, payload: dict):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Service")
    try:
        health = api_get("/health")
        st.success(f"{health['indexed_chunks']} chunks indexed")
        st.caption(f"collection: `{health['collection']}`")
        cfg = health["config"]
        st.caption(
            f"chunker: **{cfg['chunker']}** · retriever: **{cfg['retriever']}** · "
            f"reranker: **{cfg['reranker'] or 'off'}**"
        )
        cache = health["cache"]
        st.metric("Embedding cache hit rate", f"{cache['hit_rate'] * 100:.0f}%")
        st.caption(f"{cache['rows_on_disk']} vectors cached on disk")
    except Exception as exc:
        st.error(f"Cannot reach the API at {API_URL}")
        st.caption(str(exc))

    st.divider()
    top_k = st.slider("Passages to retrieve", 1, 15, 5)
    show_passages = st.toggle("Show retrieved passages", value=True)

    if st.button("Re-index corpus"):
        with st.spinner("Ingesting…"):
            st.json(api_post("/ingest", {"reset": True}))


# ------------------------------------------------------------------- main
st.title("Document Q&A")
st.caption("Ask a question. The passages that produced the answer are shown next to it.")

query = st.text_input("Question", placeholder="How long does a refund take to settle?")
col_ask, col_search = st.columns([1, 1])
ask_clicked = col_ask.button("Ask", type="primary", use_container_width=True)
search_clicked = col_search.button("Retrieve only", use_container_width=True)

if (ask_clicked or search_clicked) and query.strip():
    endpoint = "/ask" if ask_clicked else "/search"
    payload = {"query": query, "top_k": top_k}
    if ask_clicked:
        payload["include_passages"] = show_passages

    try:
        with st.spinner("Working…"):
            data = api_post(endpoint, payload)
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        st.stop()

    left, right = st.columns([3, 2]) if show_passages else (st.container(), None)

    with left:
        if ask_clicked:
            st.subheader("Answer")
            st.write(data["answer"] or "_(no answer returned)_")
        else:
            st.subheader("Retrieval only")
            st.caption("No generation ran — this is what the retriever alone returned.")

    if show_passages and right is not None:
        with right:
            st.subheader("Retrieved passages")
            for i, passage in enumerate(data.get("passages", []), start=1):
                st.markdown(
                    f"<div class='passage-card'>"
                    f"<div class='passage-meta'>[{i}] {passage['document_id']} · "
                    f"chars {passage['char_start']}–{passage['char_end']} · "
                    f"score {passage['score']:.3f}"
                    f"{' · sim ' + format(passage['semantic_score'], '.3f') if passage.get('semantic_score') is not None else ''}"
                    f" · {passage['retriever_name']}</div>"
                    f"{passage['content'][:600].replace('<', '&lt;')}"
                    f"{'…' if len(passage['content']) > 600 else ''}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    gate = data.get("gate")
    if gate is not None:
        if not gate["passed"]:
            st.warning(
                f"Relevance gate: no passage cleared the threshold "
                f"(best similarity {gate['best_score']:.3f} < {gate['threshold']}). "
                f"Nothing was sent to the model."
            )
        elif gate["dropped"]:
            st.caption(
                f"Relevance gate kept {gate['kept']} of {gate['kept'] + gate['dropped']} passages "
                f"(best similarity {gate['best_score']:.3f})."
            )

    timings = data.get("timings_ms", {})
    st.divider()
    cols = st.columns(4)
    for col, key, label in zip(
        cols,
        ["retrieval", "rerank", "generation", "total"],
        ["Retrieval", "Rerank", "Generation", "Total"],
    ):
        value = timings.get(key)
        col.metric(label, f"{value:.0f} ms" if value else "—")

    if data.get("tokens"):
        st.caption(
            f"model `{data.get('model')}` · "
            f"{data['tokens']['prompt']} prompt + {data['tokens']['completion']} completion tokens"
        )
