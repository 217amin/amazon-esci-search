# Amazon ESCI Search — Hybrid Retrieval (Dense + SPLADE + BM25) + Matryoshka Embeddings 
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-IR%2FNLP-orange)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Project Summary

**Goal:** Retrieve the most relevant Amazon products for a shopping query (ESCI).  
**Approach:** Two-stage search (Candidate Gen + Rerank) with **Hybrid Retrieval** (Dense + Sparse) + **Matryoshka** (strong low-dim embeddings).  
**Deliverables:** Reproducible artifacts, evaluation suite (Recall / nDCG / QPS), and a production-oriented design (latency + scalability + monitoring).

---

## Project Overview

This project builds an **industry-style search/relevance pipeline** on the Amazon ESCI dataset:

- **Stage 1 (Fast):** Candidate generation using:
  - **Dense bi-encoder** (BGE embeddings + FAISS cosine/IP)
  - **Sparse retrievers**: BM25 + SPLADE
  - **Weighted RRF fusion** to merge ranked lists robustly
- **Stage 2 (Accurate):** Cross-Encoder reranking to optimize top-K precision.

The key engineering addition is **Matryoshka fine-tuning**, enabling strong retrieval at **compressed embedding sizes (64 dims)** without collapsing recall — a practical lever for **lower memory and higher QPS** in real systems.

---

## Objectives

- Implement a **hybrid retrieval** stack (dense + lexical + learned sparse)
- Fine-tune a **Matryoshka bi-encoder** for multi-dim serving (768→64)
- Evaluate ranking quality with **Recall@K** + **nDCG@K**
- Provide a **production-ready story**: artifacts, scalability choices, and monitoring plan

---

## Dataset & Problem Formulation (ESCI)

This project uses the Amazon Shopping Queries Dataset (ESCI), which provides difficult, real-world search queries alongside human-annotated relevance judgments.
Link to download the DataSet(Put both of the files under data/raw): 
- [shopping_queries_dataset_examples.parquet](https://github.com/amazon-science/esci-data/blob/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet)

- [shopping_queries_dataset_products.parquet](https://github.com/amazon-science/esci-data/blob/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet)

The relevance of a product to a query is graded on a 4-point categorical scale:

- **E (Exact)**: The product exactly matches the user's intent.
- **S (Substitute)**: A product that is similar and could fulfill the user's intent, though slightly different (e.g., Query: "iPhone 13 case", Product: Otterbox case for iPhone 13 Pro).
- **C (Complement)**: A product that is often bought with the requested item, but does not fulfill the primary intent.
- **I (Irrelevant)**: The product has no relevance to the query.

### Why use E and S as Positives in Retrieval?

During the Candidate Generation (Retrieval) phase, we treat both Exact (E) and Substitute (S) items as "positives" for bi-encoder training and Recall evaluation.

The primary job of Stage 1 retrieval is to **Maximize the Coverage (Recall)**. In e-commerce search, surfacing highly relevant substitutes is a valid and profitable strategy.

**Pipeline**: By training the Bi-encoder (Stage 1) to retrieve both E and S, we prevent the model from aggressively filtering out valid alternatives. We then rely on the Cross-Encoder Reranker (Stage 2) to better sort this broad pool.

### Feature Engineering for Retrieval

To optimize the strengths of different retrieval methods, we split the text representations:
* **`product_text_dense` (Dense and SPLADE):** Fed the comprehensive text (Title + Brand + Bullets). These models excel at semantic understanding, capturing synonyms and nuanced context from the entire product profile.
* **`product_title` (BM25):** Restricted strictly to the title. BM25 relies on exact keyword matching, so this maximizes precision and prevents irrelevant matches caused by "noisy" or repetitive vocabulary in long descriptions.

---

## Architecture

```text
Raw ESCI Dataset
    ↓
Text Building (product_text_dense, query handling)
    ↓
System A: Offline Encoding (BGE / Matryoshka)
    ↓
Artifacts (prod_emb, qry_emb, prod_df, qry_df, pairs)
    ↓
System B: Candidate Generation
    ├── Dense (FAISS, Matryoshka slice)
    ├── BM25 (lexical recall)
    ├── SPLADE (learned sparse expansion)
    └── Weighted RRF Fusion (top-K candidates)
    ↓
   Cross-Encoder Reranking
    ↓
Evaluation (Recall@200 | nDCG@20 | QPS)

```

---

## Repository Structure

```text
configs/
  └── esci.yaml

src/esci/
  ├── data.py               # preprocessing, text construction, label→grade mapping
  ├── artifacts.py          # save/load artifacts (embeddings + metadata)
  ├── matryoshka_train.py   # Matryoshka + MNRL training
  ├── system_a.py           # encoding pipeline (products + queries) → artifacts
  ├── sparse_retrievers.py  # BM25 + SPLADE
  ├── system_b.py           # candidate gen (dense+sparse+RRF) + rerank
  ├── faiss_utils.py        # FAISS index/search helpers
  └── metrics.py            # Recall@K and nDCG@K

notebooks/
  ├── 01_Preprocessing.ipynb
  ├── 02_Baseline.ipynb
  ├── 03_Matryoshka_Finetuning.ipynb
  ├── 04_Reranking.ipynb
  └── 05_Interactive_Query_Testing

```

---

## Modeling Choices

### ✅ Why Hybrid Retrieval (Dense + Sparse)?

* Dense models capture **semantic similarity** (synonyms, intent).
* BM25 provides **robust lexical recall** (exact tokens).
* SPLADE adds **learned sparse expansion**, improving recall/coverage without dense-only failure modes.
* RRF fusion is stable and avoids score calibration issues between retrievers.

### ✅ Why Matryoshka?

Matryoshka trains the bi-encoder so that **prefix dimensions** remain useful:

* enables **64-dim serving**
* reduces **index size**
* improves **latency / memory footprint**
* keeps 768-dim performance close to baseline (expected behavior)

### ✅ Why MNRL (MultipleNegativesRankingLoss)?

MNRL is a strong SOTA baseline for bi-encoder retrieval:

* by choosing this loss, Matryoshka will be used to finetune the bi-encoder to provide a higher recall.
* uses **in-batch negatives** (efficient contrastive learning). This forces the model to rapidly learn a highly discriminative vector space by penalizing it against dozens of distractors simultaneously.
* scales well
* forms a strong foundation before adding mined hard negatives (future improvement)

---

## Results (Best Recall@200 | nDCG@20)

### ✅ Baseline BGE embeddings (768 dim)

| Strategy | Dim | Recall@200 | nDCG@20 |
| --- | --- | --- | --- |
| Dense Only | 768 | 0.7398 | 0.4631 |
| Dense + BM25 | 768 | 0.7629 | 0.4811 |
| Dense + SPLADE | 768 | 0.7727 | 0.4943 |
| Dense + BM25  + SPLADE | 768 | **0.7805** | **0.5030** |

### ✅ Matryoshka Fine-tuned (768 dim)

| Strategy | Dim | Recall@200 | nDCG@20 |
| --- | --- | --- | --- |
| Dense Only | 768 | 0.7771 | 0.4512 |
| Dense + BM25 | 768 | 0.8054 | 0.4720 |
| Dense + SPLADE | 768 | 0.8199 | 0.4930 |
| Dense + BM25  + SPLADE | 768 | **0.8240** | **0.5034** |

### What Matryoshka is best at 64 dimensions 

At **64 dimensions** (the real Matryoshka target), dense-only and hybrid improve strongly vs baseline:

| Strategy @64 dim | Baseline Recall@200 | Matryoshka Recall@200 | Baseline nDCG@20 | Matryoshka nDCG@20 |
| --- | --- | --- | --- | --- |
| Dense Only | 0.4270 | **0.7394** | 0.2492 | **0.4512** |
| Dense + BM25 | 0.5574 | **0.7826** | 0.3153 | **0.4409** |
| Dense + SPLADE | 0.6600 | **0.8061** | 0.3503 | **0.4661** |
| Dense+BM25+SPLADE | 0.7005 | **0.8126** | 0.3895 | **0.4843** |

And after **7 Experiments** [MLFlow results](https://dagshub.com/aminlasri/Amazon-ESCI-MATRYOSHKA-Project.mlflow/#/experiments) with different weights for BM25 and SPLADE at 64-dim Matryoshka Embedding to choose the best retrieval model (at Recalll@200), these are the final results.

| Strategy @64 dim | Recall@200 | nDCG@20 | QPS |
| --- | --- | --- | --- |
| Retrieval | **0.8126** | 0.4843 | 71.91 |
| Reranker | --- | **0.5371** | 5.74 |

**Interpretation:** Matryoshka enables **compressed vectors** (64-dim) while boosting the retrieval which is a direct win for **performance/cost**.

### The Power of Matryoshka: Finetuned@64 vs. Baseline@64

The results show that the Finetuned Matryoshka model successfully outperformed the Baseline model at 64 dimensions, jumping from a Dense-Only Recall@200 of **0.42** to **0.74** (and up to **0.81** in the Hybrid setup).

* **The Baseline Failure:** Standard embedding models distribute meaning evenly across all 768 dimensions. Arbitrarily truncating them to 64 dimensions destroys the representation, leading to a catastrophic recall collapse (0.42).
* **The MNRL Key:** The success of this fine-tuning was driven by the combination of Matryoshka architecture and Multiple Negatives Ranking Loss (MNRL). Matryoshka explicitly forces the most critical semantic information into the earliest dimensions, while MNRL maximizes hardware utilization to quickly learn high-fidelity discriminative representations.

### Hardware Efficiency & The Path to SOTA

By engineering a robust pipeline and using highly efficient "base" models—`BAAI/bge-base-en-v1.5` for retrieval and `mixedbread-ai/mxbai-rerank-base-v1` for reranking—the system achieves excellent e-commerce metrics: **Recall@200 of 81.26%** and an **nDCG@20 of 53.71%** entirely on consumer-grade hardware (8GB RTX 4070).

With access to enterprise compute (e.g., A100s), these metrics can be aggressively pushed higher by:

1. **Model Parameter Scaling:** Upgrading to `bge-large-en-v1.5` and `bge-reranker-large` for deeper semantic knowledge.
2. **Exponentially Larger Batch Sizes:** MNRL's discriminative power scales directly with batch size. Enterprise VRAM allows for massive batches, exponentially increasing "in-batch negatives" for stricter vector training.
3. **Expanded Context Windows:** Increasing `max_seq_length` (beyond the current 256) to ingest complete technical specifications and customer reviews without OOM limits.

---

## Quickstart

1. **Install**

```bash
pip install -r requirements.txt

```

2. **Configure**
Edit:

* `configs/esci.yaml` (paths, models, dims, retrieval sources, RRF weights)

3. **Train Matryoshka**
Run `03_Matryoshka_Finetuning.ipynb`.
4. **Encode artifacts (System A)**
Encode products + queries and save embeddings:

* `encode_systemA(pair_df, cfg, model_override=...)`

5. **Retrieve + evaluate (System B)**

* Candidate gen: `build_candidates(cfg, override_dim=64)`
* Optional rerank: `rerank_candidates(...)`

---

## Future Improvements

* **Hard negative mining** (BM25/SPLADE/dense mined near-misses) for stronger discrimination
* ANN indexes (FAISS IVF/HNSW) for scalability beyond flat search
* Calibration/learning-to-rank for fusion (beyond RRF)
* Full MLOps: artifact versioning by config hash, drift monitoring, retraining triggers

---

## Skills

* Information Retrieval (bi-encoder, cross-encoder, hybrid retrieval)
* Contrastive learning (MNRL) + Matryoshka compression
* FAISS vector search + sparse indexing (BM25, SPLADE)
* Offline/online separation, artifact management, reproducible evaluation
* Production thinking: latency/QPS, memory footprint, monitoring plan