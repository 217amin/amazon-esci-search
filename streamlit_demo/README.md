---
title: Amazon ESCI Search Demo
emoji: 🛒
colorFrom: orange
colorTo: red
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
license: mit
---

# Amazon ESCI · Hybrid Search Demo

A thin Streamlit UI for the [Amazon ESCI Hybrid Search API](https://github.com/217amin/amazon-esci-search).

The UI calls a deployed FastAPI service (Modal) that runs:
- Matryoshka 64-dim FAISS dense retrieval
- SPLADE neural sparse retrieval
- BM25 lexical retrieval
- Weighted RRF fusion → cross-encoder reranking

