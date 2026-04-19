import sys
import os
import time
import pickle
from pathlib import Path
from contextlib import asynccontextmanager
from collections import defaultdict

import yaml
import faiss
import torch
import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.esci.sparse_retrievers import SPLADEFast, BM25Fast

# --- Global State for Models & Data ---
class MLState:
    cfg = None
    dense_model = None
    faiss_index = None
    splade_model = None
    bm25_model = None
    cross_encoder = None
    
    # Data Mappings
    id_to_text = {}       # Maps product_id -> product_text_dense
    product_ids = []      # Maps FAISS integer index -> product_id

state = MLState()

# --- 1. Lifespan: Load Models at Startup (From Notebook 05) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up: Loading configurations, data, and models into VRAM...")
    
    # Load config
    with open(PROJECT_ROOT / "configs" / "esci.yaml", "r", encoding="utf-8") as f:
        state.cfg = yaml.safe_load(f)
    
    artifacts_dir = PROJECT_ROOT / state.cfg["paths"]["artifacts_dir"]
    processed_dir = PROJECT_ROOT / state.cfg["paths"]["processed_dir"]

    # 1. Load Data & Mappings
    df_path = processed_dir / "pair_df.parquet"
    if not df_path.exists():
        raise FileNotFoundError(f"Missing data file: {df_path}")
    pair_df = pd.read_parquet(df_path)
    
# Load exact unique IDs mapping to match the FAISS index mathematically
    state.product_ids = pd.read_pickle(artifacts_dir / "faiss_mapping.pkl").tolist()
    state.id_to_text = dict(zip(pair_df['product_id'], pair_df['product_text_dense']))
    print(f"📦 Loaded {len(state.product_ids)} unique products into memory.")

    # 2. Load Dense Model (Matryoshka BGE)
    print("🧠 Loading BGE-Base Model...")
    state.dense_model = SentenceTransformer(state.cfg["biencoder_model"])
    state.dense_model.eval()

    # 3. Load FAISS Index
    print("🔍 Loading FAISS Index...")
    state.faiss_index = faiss.read_index(str(artifacts_dir / "faiss_index.bin"))

    # 4. Load Sparse Models
    print("🕸️ Loading Sparse Retrievers (SPLADE & BM25)...")
    state.splade_model = SPLADEFast(state.cfg["sparse"]["splade_model"])
    with open(artifacts_dir / "bm25_retriever.pkl", "rb") as f:
        state.bm25_model = pickle.load(f)

    # 5. Load Cross-Encoder (Reranker)
    print("⚖️ Loading Cross-Encoder (Reranker)...")
    ce_max_len = state.cfg.get("reranker", {}).get("max_seq_length", 128)
    state.cross_encoder = CrossEncoder(
        state.cfg["cross_encoder_model"], 
        max_length=ce_max_len, 
        device="cuda"
    )
    state.cross_encoder.model.eval()

    print("✅ All systems ready!")
    yield
    print("🛑 Shutting down: Cleaning up GPU memory...")
    torch.cuda.empty_cache()

# --- 2. Initialize FastAPI ---
app = FastAPI(
    title="Amazon ESCI Internal Search API",
    description="Two-Stage Hybrid Retrieval Engine with Matryoshka Embeddings",
    version="1.0.0",
    lifespan=lifespan
)

# --- 3. Pydantic Validation Models ---
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=256, description="User search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return (max 50)")

class SearchResult(BaseModel):
    product_id: str
    score: float
    text: str

class SearchResponse(BaseModel):
    status: str
    latency_ms: float
    query: str
    results: list[SearchResult]

# --- 4. API Endpoints ---
@app.get("/health")
async def health_check():
    if state.cross_encoder is None:
        raise HTTPException(status_code=503, detail="Models are still loading.")
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}

@app.post("/api/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Real-time Inference Endpoint using Two-Stage Retrieval."""
    start_time = time.time()
    
    try:
        query_str = request.query
        retrieval_cfg = state.cfg["retrieval"]
        
        # ---------------------------------------------------------
        # STAGE 1: HYBRID RETRIEVAL (Dense + SPLADE + BM25)
        # ---------------------------------------------------------
        
        # A. Dense Search (FAISS)
        instruction = "Represent this sentence for searching relevant passages: "
        # Encode and normalize
        query_emb = state.dense_model.encode([instruction + query_str], normalize_embeddings=True)
        # Truncate to Matryoshka dimension (64)
        query_emb_64 = query_emb[:, :retrieval_cfg["matryoshka_dim"]]
        faiss.normalize_L2(query_emb_64) # FAISS inner product needs normalized vectors
        
        D, I = state.faiss_index.search(query_emb_64, k=retrieval_cfg["dense_top_k"])
        
        # Mapping FAISS indices back to Product IDs
        dense_candidates = [state.product_ids[idx] for idx in I[0] if idx != -1]
        
        # 1. SPLADE Search (uses .score_topk and returns internal matrix indices)
        splade_raw = state.splade_model.score_topk(query_str, top_k=retrieval_cfg["sparse_top_k"])
        splade_candidates = [state.splade_model.pid_list[idx] for idx, score in splade_raw]
        
        # 2. BM25 Search (uses .search and returns product_ids directly)
        bm25_raw = state.bm25_model.search(query_str, top_k=retrieval_cfg["sparse_top_k"])
        bm25_candidates = [pid for pid, score in bm25_raw]
        # C. RRF (Reciprocal Rank Fusion) Integration
        rrf_k = retrieval_cfg["rrf_k"]
        weights = retrieval_cfg["rrf_weights"]
        combined_scores = defaultdict(float)

        # Apply RRF Math
        for rank, pid in enumerate(dense_candidates):
            combined_scores[pid] += weights["dense"] / (rrf_k + rank + 1)
            
        for rank, pid in enumerate(splade_candidates):
            combined_scores[pid] += weights["splade"] / (rrf_k + rank + 1)
            
        for rank, pid in enumerate(bm25_candidates):
            combined_scores[pid] += weights["bm25"] / (rrf_k + rank + 1)

        # Sort and take top N candidates (200) for reranking
        sorted_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [pid for pid, score in sorted_candidates[:retrieval_cfg["candidate_top_k"]]]

        # ---------------------------------------------------------
        # STAGE 2: RERANKING (Cross-Encoder)
        # ---------------------------------------------------------
        
        # Extract texts for the candidates
        candidate_texts = [state.id_to_text.get(pid, "") for pid in top_candidates]
        
        # Prepare pairs for the Cross-Encoder: [[query, text1], [query, text2], ...]
        cross_inp = [[query_str, text] for text in candidate_texts]
        
        # Chrono Stage 1 (Retrieval)
        time_retrieval = time.time() - start_time
        start_rerank = time.time()

        # Predict exact relevance scores (Découpé en batchs pour sauver la VRAM)
        cross_scores = state.cross_encoder.predict(
            cross_inp,
            batch_size=16, 
            show_progress_bar=False
        )
        
        # Chrono Stage 2 (Reranking)
        time_rerank = time.time() - start_rerank
        print(f"⏱️ Debug [{query_str}] -> Retrieval: {time_retrieval:.2f}s | Reranking: {time_rerank:.2f}s")
        
        # Format and sort the final results
        final_results = [
            {"product_id": pid, "score": float(score), "text": text}
            for pid, score, text in zip(top_candidates, cross_scores, candidate_texts)
        ]
        
        # Sort descending by Cross-Encoder score and limit to requested top_k (max 50)
        final_results = sorted(final_results, key=lambda x: x["score"], reverse=True)[:request.top_k]
        
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        return SearchResponse(
            status="success",
            latency_ms=latency_ms,
            query=query_str,
            results=final_results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {str(e)}")

if __name__ == "__main__":
    # Tell Uvicorn exactly where the app is located
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)