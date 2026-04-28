"""
Streamlit UI for the Amazon ESCI Hybrid Search API.

Deployment:
  - Designed to run as its own Hugging Face Space (Streamlit SDK).
  - Calls the deployed API URL (Modal) — does NOT need any artifacts locally.
  - Just needs the API_URL env var or secret.

Local run:
  pip install streamlit requests
  ESCI_API_URL=http://localhost:8000 streamlit run app.py
"""
from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Amazon ESCI · Hybrid Search Demo",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_API_URL = os.getenv("ESCI_API_URL", "")
EXAMPLE_QUERIES = [
    "wireless gaming mouse with rgb",
    "noise cancelling over ear headphones",
    "iphone 13 case shockproof",
    "stainless steel water bottle 1 liter",
    "mens running shoes size 11",
    "mechanical keyboard brown switches",
]


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def render_header() -> None:
    st.title("🛒 Amazon ESCI · Hybrid Search")
    st.caption(
        "Two-stage retrieval pipeline: **Matryoshka 64-dim FAISS + SPLADE + BM25** → "
        "weighted RRF fusion → **cross-encoder reranking**. "
        "Powered by a FastAPI service deployed on Modal."
    )


def render_sidebar() -> dict[str, Any]:
    st.sidebar.header("⚙️ Configuration")
    api_url = st.sidebar.text_input(
        "API URL",
        value=DEFAULT_API_URL,
        help="Public URL of the deployed Modal endpoint (or http://localhost:8000 for local dev).",
        placeholder="https://your-workspace--esci-search-fastapi-app.modal.run",
    )

    top_k = st.sidebar.slider("Results to return", 1, 20, 5)
    show_signals = st.sidebar.toggle(
        "Show retrieval-signal breakdown",
        value=True,
        help="For each result, show which retrievers (BM25 / SPLADE / Dense) found it and at what rank.",
    )

    st.sidebar.divider()

    if api_url:
        if st.sidebar.button("🔌 Check API health", use_container_width=True):
            try:
                r = requests.get(f"{api_url.rstrip('/')}/info", timeout=20)
                if r.status_code == 200:
                    st.sidebar.success("API is healthy")
                    st.sidebar.json(r.json())
                else:
                    st.sidebar.error(f"API returned {r.status_code}")
            except requests.RequestException as e:
                st.sidebar.error(f"Cannot reach API: {e}")

    st.sidebar.divider()

    with st.sidebar.expander("ℹ️ About this demo"):
        st.markdown(
            """
            This is a search-engineering demo, not a product search engine.

            **What's interesting here:**
            - **Matryoshka embeddings** compress vectors from 768 → 64 dims
              with minimal recall loss (Recall@200 = 0.81 vs 0.43 for naive
              truncation of the base model).
            - **Hybrid retrieval** combines lexical (BM25) and neural sparse
              (SPLADE) and dense (FAISS) signals via RRF.
            - **Cross-encoder reranking** lifts nDCG@20 by +5 points at
              ~7.6× latency cost.

            Toggle the "Show retrieval-signal breakdown" to see which
            retriever found each result.
            """
        )

    return {"api_url": api_url, "top_k": top_k, "show_signals": show_signals}


def render_signals(signals: dict[str, Any]) -> str:
    """Format the per-retriever ranks compactly."""
    bits = []
    if signals.get("dense_rank") is not None:
        bits.append(f"🧠 Dense #{signals['dense_rank']}")
    if signals.get("splade_rank") is not None:
        bits.append(f"🕸️ SPLADE #{signals['splade_rank']}")
    if signals.get("bm25_rank") is not None:
        bits.append(f"🔤 BM25 #{signals['bm25_rank']}")
    if signals.get("fused_rrf_score") is not None:
        bits.append(f"RRF {signals['fused_rrf_score']:.4f}")
    return " · ".join(bits) if bits else "—"


def render_results(data: dict[str, Any]) -> None:
    timings = data.get("timings", {})
    results = data.get("results", [])

    cols = st.columns(4)
    cols[0].metric("Total", f"{timings.get('total_ms', 0):.0f} ms")
    cols[1].metric("Retrieval", f"{timings.get('retrieval_ms', 0):.0f} ms")
    cols[2].metric("Rerank", f"{timings.get('rerank_ms', 0):.0f} ms")
    cols[3].metric("Results", len(results))

    st.divider()

    for i, res in enumerate(results, start=1):
        with st.container(border=True):
            top_cols = st.columns([4, 1])
            with top_cols[0]:
                st.markdown(f"### #{i} · `{res['product_id']}`")
            with top_cols[1]:
                st.metric("Rerank score", f"{res['score']:.3f}", label_visibility="visible")

            if res.get("signals"):
                st.caption(render_signals(res["signals"]))

            st.write(res.get("text", ""))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    render_header()
    settings = render_sidebar()

    if not settings["api_url"]:
        st.info(
            "Set the **API URL** in the sidebar to point at your deployed Modal endpoint, "
            "or `http://localhost:8000` if running the API locally."
        )
        st.stop()

    st.subheader("Try a query")
    cols = st.columns(3)
    clicked: str | None = None
    for i, q in enumerate(EXAMPLE_QUERIES):
        with cols[i % 3]:
            if st.button(q, use_container_width=True, key=f"ex_{i}"):
                clicked = q

    user_q = st.text_input(
        "Or type your own query",
        placeholder="e.g., bluetooth speaker waterproof",
    )
    submit = st.button("🔍 Search", type="primary")

    query = clicked or (user_q if submit and user_q.strip() else None)
    if not query:
        return

    st.markdown(f"##### Query: _{query}_")

    with st.spinner("Searching..."):
        try:
            r = requests.post(
                f"{settings['api_url'].rstrip('/')}/api/v1/search",
                json={
                    "query": query,
                    "top_k": settings["top_k"],
                    "show_signals": settings["show_signals"],
                },
                timeout=120,
            )
        except requests.RequestException as e:
            st.error(f"Could not reach API: {e}")
            st.info(
                "If this is a Modal cold-start, the first request after idle "
                "can take 30–60s while the container loads models."
            )
            return

    if r.status_code != 200:
        st.error(f"API returned {r.status_code}")
        st.code(r.text)
        return

    render_results(r.json())


if __name__ == "__main__":
    main()