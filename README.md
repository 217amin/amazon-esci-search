# Amazon ESCI Search — Two-Stage Hybrid Retrieval with Matryoshka Fine-Tuning

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)](https://streamlit.io/)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-IR%2FNLP-orange)](https://sbert.net)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-brightgreen)](https://github.com/facebookresearch/faiss)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)](https://dagshub.com/aminlasri/Amazon-Search-Engine-Project.mlflow)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)](https://github.com/217amin/amazon-esci-search)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> A production-oriented e-commerce search engine built on the Amazon ESCI dataset.  
> Combines **Hybrid Retrieval** (Matryoshka Dense + SPLADE + BM25) and **Cross-Encoder Reranking** to maximize relevance (nDCG) while compressing vector storage to 64 dimensions — without sacrificing recall.  
> Served locally via a **FastAPI** backend and a **Streamlit** frontend.

---

## Results at a Glance

### Best Configuration: Matryoshka @ 64 dims + Full Hybrid + Reranker

| Stage | Recall@200 | nDCG@20 | QPS |
|---|---|---|---|
| Retrieval (hybrid) | **0.8148** | 0.4846 | 69.63 |
| After Reranking | — | **0.5378** | 5.72 |

### The Matryoshka Advantage at 64 Dimensions

Standard models collapse at 64 dims (0.43 recall). Matryoshka fine-tuning recovers that loss almost entirely — enabling smaller indexes, lower memory, and higher QPS at near-full quality.

| Strategy | Baseline Recall@200 | Matryoshka Recall@200 | Δ |
|---|---|---|---|
| Dense Only | 0.43 | **0.74** | +73% |
| Dense + BM25 | 0.56 | **0.78** | +41% |
| Dense + SPLADE | 0.66 | **0.81** | +22% |
| Dense + BM25 + SPLADE | 0.70 | **0.81** | +16% |

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
    ├── Dense  (FAISS, Matryoshka 64-dim slice)
    ├── BM25   (lexical recall on title only)
    ├── SPLADE (learned sparse expansion)
    └── Weighted RRF Fusion → top-K candidates
    ↓
Cross-Encoder Reranking (mxbai-rerank-base-v1)
    ↓
FastAPI — REST API (/api/v1/search)
    ↓
Streamlit UI — Interactive Search Frontend
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

### Why FastAPI + Streamlit?

A decoupled microservices architecture separates concerns cleanly:
- **FastAPI** loads all models once into GPU VRAM via a lifespan singleton (BGE bi-encoder, FAISS index, SPLADE, BM25, Cross-Encoder). Exposes a `/api/v1/search` POST endpoint with Pydantic validation and a `/health` check.
- **Streamlit** is a lightweight frontend that calls the local API and renders results with latency display. No model code in the UI layer.

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

### RRF Hyperparameter Search (MLflow — 7 experiments)

| Exp | rrf_k | w_bm25 | w_splade | Recall@200 | nDCG@20 | QPS |
|---|---|---|---|---|---|---|
| 1 | 60 | 0.3 | 0.5 | 0.815 | 0.485 | 69.09 |
| **2** | **60** | **0.3** | **0.7** | **0.813** | **0.500** | **72.49** |
| 3 | 60 | 0.2 | 0.5 | 0.814 | 0.479 | 73.48 |
| 4 | 60 | 0.3 | 0.3 | 0.811 | 0.470 | 71.10 |
| 5 | 80 | 0.3 | 0.5 | 0.815 | 0.484 | 72.09 |
| 6 | 40 | 0.3 | 0.5 | 0.814 | 0.486 | 71.42 |
| 7 | 20 | 0.3 | 0.5 | 0.813 | 0.489 | 72.52 |

---

## Repository Structure

```
configs/
  └── esci.yaml              # all hyperparameters (models, dims, RRF weights, paths)

src/
  ├── api/
  │   └── main.py            # FastAPI app — lifespan model loading + /search endpoint
  └── esci/
      ├── data.py            # preprocessing, text construction, label mapping
      ├── artifacts.py       # save/load embeddings + metadata
      ├── matryoshka_train.py # Matryoshka + MNRL fine-tuning
      ├── system_a.py        # encoding pipeline → artifacts
      ├── sparse_retrievers.py # BM25 + SPLADE
      ├── system_b.py        # candidate gen (dense+sparse+RRF) + rerank
      ├── faiss_utils.py     # FAISS index/search helpers
      ├── metrics.py         # Recall@K and nDCG@K
      ├── mlflow.py          # MLflow logging helpers
      └── run_pipeline.py    # CLI entrypoint (--mode train | inference)

app_ui.py                    # Streamlit frontend (calls FastAPI)
build_index.py               # Standalone FAISS index builder

notebooks/
  ├── 01_Data_Preprocessing_and_Featurization.ipynb
  ├── 02_Baseline_Evaluation_and_Indexing.ipynb
  ├── 03_Matryoshka_Finetuning.ipynb
  ├── 04_Hybrid_Retrieval_and_Reranking.ipynb
  └── 05_Interactive_Query_Testing.ipynb

results/                     # Saved experiment CSVs
docs/                        # Architecture, ModelCard, DataCard, Evaluation notes
```

---

## Quickstart

### Prerequisites

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` covers ML/retrieval dependencies. For the local app, also install:
> ```bash
> pip install fastapi uvicorn streamlit requests
> ```

Download ESCI dataset files into `data/raw/`:
- [`shopping_queries_dataset_examples.parquet`](https://github.com/amazon-science/esci-data/blob/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet)
- [`shopping_queries_dataset_products.parquet`](https://github.com/amazon-science/esci-data/blob/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet)

---

### Option A — Notebook (recommended for exploration)

Run notebooks in order: `01` → `02` → `03` → `04` → `05`

---

### Option B — Full Pipeline (production mode)

```bash
# Train Matryoshka encoder
python -m esci.run_pipeline --mode train

# Run full inference pipeline (encode → retrieve → rerank → evaluate)
python -m esci.run_pipeline --mode inference
```

**Debug mode** (fast iteration on 2000 queries):
```yaml
# configs/esci.yaml
debug:
  use_sample: true
  sample_size: 2000
```

---

### Option C — Local App (FastAPI + Streamlit)

This project implements a decoupled microservices architecture. Models are loaded **once** into GPU VRAM via a FastAPI lifespan singleton, ensuring minimal latency at query time.

**Step 1 — Build artifacts** (run once after training):
```bash
python build_index.py
```
This encodes all products, truncates to 64 dims, and saves `faiss_index.bin` + `faiss_mapping.pkl` to `artifacts/systemA/`.

**Step 2 — Start the API backend:**
```bash
python -m src.api.main
```
API starts on `http://localhost:8000`.  
Interactive docs available at [`http://localhost:8000/docs`](http://localhost:8000/docs).

**Step 3 — Start the frontend UI** (new terminal):
```bash
streamlit run app_ui.py
```
The Streamlit interface opens in your browser and connects to the local FastAPI backend.

**API Reference:**

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check + GPU status |
| `/api/v1/search` | POST | Two-stage hybrid search |

Example request:
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Sony wireless headphones over ear", "top_k": 5}'
```

---

## Stack

| Layer | Technology |
|---|---|
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Frontend UI** | Streamlit |
| **Dense retrieval** | FAISS + BGE (`BAAI/bge-base-en-v1.5`) |
| **Sparse retrieval** | BM25 (rank-bm25) + SPLADE (`naver/splade-cocondenser-ensembledistil`) |
| **Fine-tuning** | Matryoshka + MNRL (sentence-transformers) |
| **Reranking** | `mixedbread-ai/mxbai-rerank-base-v1` |
| **Fusion** | Weighted Reciprocal Rank Fusion (RRF) |
| **Experiment tracking** | MLflow on DagsHub |

---

## Future Improvements

- Hard negative mining (BM25/SPLADE/dense mined near-misses) for stronger discrimination
- ANN indexes (FAISS IVF/HNSW) for scalability beyond flat search
- Learning-to-rank for fusion weights (beyond static RRF)
- Full MLOps: artifact versioning by config hash, drift monitoring, retraining triggers
- Docker Compose to containerize FastAPI + Streamlit together

---

## Authors

- **Amin** — [@217amin](https://github.com/217amin)
- **Asmaa** — [@segmami](https://github.com/segmami)

---

## License

MIT
