# Architecture

This project implements a **two‑stage retrieval system** commonly used in industry search stacks.

---

## 1) Data and representations

### Product text (dense)
A single dense text field is created per product (`product_text_dense`) from:
- `product_title`
- `product_brand`
- `bullet points`
- optional: color


Used by:
- Dense bi‑encoder encoding (System A)
- SPLADE indexing (System B)
- Cross‑encoder reranking text

### Query text
- Dense queries are encoded with a **BGE instruction prefix** (recommended for BGE).
- BM25 uses stemming/tokenization (lexical).
- SPLADE uses raw text (tokenizer‑driven); avoid aggressive normalization.

---

## 2) System A — Offline encoding and artifact creation

**Goal:** compute embeddings once and cache them for fast iteration.

Steps:
1) Filter to `split=test` (evaluation artifacts)
2) Create unique product list and unique query list
3) Encode:
   - products: no instruction
   - queries: BGE instruction prefix
4) Normalize embeddings
5) Save artifacts: `(prod_emb, qry_emb, prod_df, qry_df, pair_df)` in a consistent order

Why:
- Decouples expensive encoding from retrieval experiments
- Guarantees stable ID ordering (critical for FAISS mapping)
- Improves reproducibility

---

## 3) System B — Candidate generation + fusion + reranking

### 3.1 Dense retrieval (FAISS)
- Slice embeddings to `target_dim` (Matryoshka prefix)
- Renormalize sliced vectors
- Search using FAISS inner product (equivalent to cosine with normalization)

### 3.2 Sparse retrieval
#### BM25
- Robust lexical recall baseline
- Title‑focused corpus + stemming

#### SPLADE
- MLM‑based sparse expansion
- Builds a **CSR sparse matrix** for fast scoring
- Query scoring uses `doc_matrix @ q_vec` (GPU sparse matmul)

### 3.3 Fusion (Weighted RRF)
All retrievers produce ranked lists; fuse with Weighted RRF:

`score(pid) += weight / (k + rank + 1)`

Why:
- Strong and stable for hybrid search
- Avoids score calibration issues
- Simple to tune and explain

### 3.4 Reranking (Cross‑encoder)
Rerank top candidates with a cross‑encoder:
- Input: (query, product_text_dense)
- Output: final relevance score

Why:
- Candidate gen optimizes recall/speed
- Reranker optimizes precision at top ranks (nDCG@10/20)

---

## 4) Matryoshka: compression without collapsing retrieval

Matryoshka fine‑tuning trains the bi‑encoder to preserve retrieval quality in **embedding prefixes**:
- Full dim: 768
- Prefixes: 512, 256, 128, 64

Enables:
- smaller index footprint
- lower latency
- stronger retrieval at low dims than baseline prefixes
