# Amazon ESCI Search — Two-Stage Hybrid Retrieval with Matryoshka Fine-Tuning

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-IR%2FNLP-orange)](https://sbert.net)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-brightgreen)](https://github.com/facebookresearch/faiss)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)](https://dagshub.com/aminlasri/Amazon-Search-Engine-Project.mlflow)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)](https://github.com/217amin/amazon-esci-search)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](https://github.com/217amin/amazon-esci-search)

> A production-oriented search engine built on the Amazon ESCI dataset. Combines Hybrid Retrieval (Matryoshka Dense + SPLADE + BM25) and Cross-Encoder Reranking to maximize relevance (nDCG) while compressing vector storage to 64 dimensions — without sacrificing recall.

---

## Results at a Glance

### Best Configuration: Matryoshka @ 64 dims + Full Hybrid + Reranker

| Stage | Recall@200 | nDCG@20 | QPS |
|---|---|---|---|
| Retrieval (hybrid) | **0.8148** | 0.4846 | 69.63 |
| After Reranking | — | **0.5378** | 5.72 |

### Why 64 Dimensions? — The Matryoshka Advantage

| Strategy | Baseline Recall@200 | Matryoshka Recall@200 |
|---|---|---|
| Dense Only | 0.43 | **0.74** |
| Dense + BM25 | 0.56 | **0.78** |
| Dense + SPLADE | 0.66 | **0.81** |
| Dense + BM25 + SPLADE | 0.70 | **0.81** |

Standard models collapse at 64 dims (0.43 recall). Matryoshka fine-tuning recovers that loss almost entirely — enabling smaller indexes, lower memory, and higher QPS at near-full quality.

**All experiments trained on a consumer GPU (8GB RTX 4070). [MLflow experiment log →](https://dagshub.com/aminlasri/Amazon-Search-Engine-Project.mlflow/#/experiments)**

---

## Problem

E-commerce search is hard. Queries like *"iPhone 13 case"* need to surface both exact matches and valid substitutes (e.g., OtterBox for iPhone 13 Pro) while filtering out complements and irrelevant results. The Amazon ESCI dataset provides human-annotated relevance judgments at four levels:

| Label | Meaning |
|---|---|
| **E** — Exact | Product directly fulfills the query |
| **S** — Substitute | Similar product, fulfills intent |
| **C** — Complement | Often bought together, doesn't fulfill intent |
| **I** — Irrelevant | No relation to the query |

Stage 1 retrieval treats E + S as positives (maximize recall). Stage 2 reranking sorts them (maximize precision).

---

## Architecture

```
Raw ESCI Dataset
    ↓
Text Building (product_text_dense / product_title split)
    ↓
System A: Offline Encoding (BGE / Matryoshka fine-tune)
    ↓
Artifacts (embeddings + metadata)
    ↓
System B: Candidate Generation
    ├── Dense (FAISS, Matryoshka 64-dim slice)
    ├── BM25  (lexical recall on title only)
    ├── SPLADE (learned sparse expansion)
    └── Weighted RRF Fusion → top-K candidates
    ↓
Cross-Encoder Reranking (mxbai-rerank-base-v1)
    ↓
Evaluation: Recall@K | nDCG@K | QPS
```

---

## Key Engineering Decisions

### Why Split Text Representations?

- **Dense + SPLADE** receive the full product profile (title + brand + bullets): these models excel at semantic understanding across rich context.
- **BM25** receives the title only: exact keyword matching on long, noisy descriptions degrades precision. Restricting to the title maximizes BM25's contribution.

### Why Hybrid Retrieval?

| Component | Strength |
|---|---|
| Dense bi-encoder | Semantic similarity, synonym handling |
| BM25 | Lexical recall, exact token matching |
| SPLADE | Learned sparse expansion, covers recall gaps |
| RRF Fusion | Stable merging without score calibration |

### Why Matryoshka + MNRL?

Matryoshka training forces the most critical semantic information into the first N dimensions. Combined with Multiple Negatives Ranking Loss (MNRL), which uses in-batch negatives for discriminative contrastive learning, this produces encoders that retain strong recall at 64 dims — making real-system deployment (lower memory, higher QPS) practical without a quality cliff.

---

## Full Results

### Baseline BGE (768 dims)

| Strategy | Recall@200 | nDCG@20 |
|---|---|---|
| Dense Only | 0.7398 | 0.4631 |
| Dense + BM25 | 0.7629 | 0.4811 |
| Dense + SPLADE | 0.7727 | 0.4943 |
| Dense + BM25 + SPLADE | **0.7805** | **0.5030** |

### Matryoshka Fine-tuned (768 dims)

| Strategy | Recall@200 | nDCG@20 |
|---|---|---|
| Dense Only | 0.7761 | 0.4497 |
| Dense + BM25 | 0.8054 | 0.4704 |
| Dense + SPLADE | 0.8210 | 0.4925 |
| Dense + BM25 + SPLADE | **0.8251** | **0.5029** |

### Matryoshka Fine-tuned (64 dims — the real target)

| Strategy | Baseline Recall@200 | Matryoshka Recall@200 | Δ |
|---|---|---|---|
| Dense Only | 0.4270 | **0.7404** | +73% |
| Dense + BM25 | 0.5574 | **0.7836** | +41% |
| Dense + SPLADE | 0.6600 | **0.8080** | +22% |
| Dense + BM25 + SPLADE | 0.7005 | **0.8148** | +16% |

---

## Repository Structure

```
configs/
  └── esci.yaml              # all hyperparameters

src/esci/
  ├── data.py                # preprocessing, text construction, label mapping
  ├── artifacts.py           # save/load embeddings + metadata
  ├── matryoshka_train.py    # Matryoshka + MNRL fine-tuning
  ├── system_a.py            # encoding pipeline → artifacts
  ├── sparse_retrievers.py   # BM25 + SPLADE
  ├── system_b.py            # candidate gen (dense+sparse+RRF) + rerank
  ├── faiss_utils.py         # FAISS index/search helpers
  └── metrics.py             # Recall@K and nDCG@K

notebooks/
  ├── 01_Preprocessing.ipynb
  ├── 02_Baseline.ipynb
  ├── 03_Matryoshka_Finetuning.ipynb
  ├── 04_Reranking.ipynb
  └── 05_Interactive_Query_Testing.ipynb

results/                     # saved experiment outputs
docs/                        # architecture diagrams
```

---

## Quickstart

### Option A — Notebook (recommended for exploration)

```bash
pip install -r requirements.txt
```

Download ESCI dataset files into `data/raw/`:
- [`shopping_queries_dataset_examples.parquet`](https://github.com/amazon-science/esci-data/blob/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet)
- [`shopping_queries_dataset_products.parquet`](https://github.com/amazon-science/esci-data/blob/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet)

Run notebooks in order: `01` → `02` → `03` → `04` → `05`

### Option B — Full Pipeline (production mode)

```bash
# Train Matryoshka encoder first
python -m esci.run_pipeline --mode train

# Run full inference pipeline
python -m esci.run_pipeline --mode inference
# Outputs: QPS / Recall@200 / nDCG@20
```

**Debug mode** (fast iteration on 2000 queries):
```yaml
# configs/esci.yaml
debug:
  use_sample: true
  sample_size: 2000
```

---

## Stack

- **Dense retrieval:** FAISS + BGE (`BAAI/bge-base-en-v1.5`)
- **Sparse retrieval:** BM25 + SPLADE
- **Fine-tuning:** Matryoshka + MNRL (sentence-transformers)
- **Reranking:** mxbai-rerank-base-v1
- **Fusion:** Weighted Reciprocal Rank Fusion (RRF)
- **Experiment tracking:** MLflow on DagsHub

---

## Future Improvements

- Hard negative mining (BM25/SPLADE/dense mined near-misses) for stronger discrimination
- ANN indexes (FAISS IVF/HNSW) for scalability beyond flat search
- Learning-to-rank for fusion weights (beyond static RRF)
- Full MLOps: artifact versioning by config hash, drift monitoring, retraining triggers
- Streamlit / Gradio live demo
