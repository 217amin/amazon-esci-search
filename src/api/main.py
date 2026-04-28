"""
Amazon ESCI Hybrid Search API

Two-stage retrieval pipeline:
  Stage 1: Hybrid candidate generation (Dense FAISS @ 64-dim Matryoshka,
           SPLADE neural sparse, BM25 lexical) fused via weighted RRF.
  Stage 2: Cross-encoder reranking of the top candidates.

Production deployment notes:
  - The dense biencoder must be the FINE-TUNED matryoshka checkpoint that was
    used to encode the indexed products. Loading the base BGE model here
    silently mismatches the FAISS vector space and craters recall.
  - SPLADE state (doc_matrix + pid_list) is loaded from splade_data.pt.
  - Set CE_DEVICE=cpu in the environment to run cross-encoder on CPU
    (required for Modal CPU deployments). Defaults to cuda when available.
  - Set ARTIFACTS_DIR / PROCESSED_DIR / MATRYOSHKA_DIR env vars to override
    config paths — useful when artifacts are mounted from a Modal Volume at
    a different path than the repo's local layout.
"""
import sys
import os
import time
import pickle
from pathlib import Path
from contextlib import asynccontextmanager
from collections import defaultdict
from typing import Optional

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


# --- Helpers ---
def _resolve_path(env_var: str, cfg_path: str) -> Path:
    """Override a config path via env var, useful for Modal Volume mounts."""
    return Path(os.getenv(env_var, str(PROJECT_ROOT / cfg_path)))


def _resolve_device() -> str:
    """Resolve device for cross-encoder + SPLADE.
    Order: CE_DEVICE env var -> cuda if available -> cpu.
    """
    explicit = os.getenv("CE_DEVICE")
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


# --- Global State for Models & Data ---
class MLState:
    cfg = None
    device = "cpu"
    dense_model = None
    faiss_index = None
    splade_model = None
    bm25_model = None
    cross_encoder = None

    # Data Mappings
    id_to_text = {}       # Maps product_id -> product_text_dense
    product_ids = []      # Maps FAISS integer index -> product_id

    # Diagnostics
    dense_model_path = None  # for /info endpoint


state = MLState()


# --- Lifespan: Load Models at Startup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: loading configurations, data, and models...")

    # Load config
    with open(PROJECT_ROOT / "configs" / "esci.yaml", "r", encoding="utf-8") as f:
        state.cfg = yaml.safe_load(f)

    state.device = _resolve_device()
    print(f"Resolved device: {state.device}")

    artifacts_dir = _resolve_path("ARTIFACTS_DIR", state.cfg["paths"]["artifacts_dir"])
    processed_dir = _resolve_path("PROCESSED_DIR", state.cfg["paths"]["processed_dir"])
    matryoshka_dir = _resolve_path("MATRYOSHKA_DIR", state.cfg["paths"]["matryoshka_dir"])

    print(f"  artifacts_dir : {artifacts_dir}")
    print(f"  processed_dir : {processed_dir}")
    print(f"  matryoshka_dir: {matryoshka_dir}")

    # 1. Load Data & Mappings
    df_path = processed_dir / "pair_df.parquet"
    if not df_path.exists():
        raise FileNotFoundError(f"Missing data file: {df_path}")
    pair_df = pd.read_parquet(df_path)

    state.product_ids = pd.read_pickle(artifacts_dir / "faiss_mapping.pkl").tolist()
    state.id_to_text = dict(zip(pair_df['product_id'], pair_df['product_text_dense']))
    print(f"Loaded {len(state.product_ids)} unique products into memory.")

    # 2. Load Dense Model
    # CRITICAL: must be the matryoshka-fine-tuned checkpoint used to build
    # faiss_index.bin. The base BGE model would mismatch the vector space.
    matryoshka_subdir = os.getenv("MATRYOSHKA_SUBDIR", "us")
    matryoshka_path = matryoshka_dir / matryoshka_subdir
    if not matryoshka_path.exists():
        raise FileNotFoundError(
            f"Matryoshka checkpoint not found at {matryoshka_path}. "
            "Set MATRYOSHKA_SUBDIR env var or place the fine-tuned model there."
        )
    print(f"Loading Matryoshka biencoder from {matryoshka_path} ...")
    state.dense_model = SentenceTransformer(str(matryoshka_path), device=state.device)
    state.dense_model.eval()
    state.dense_model_path = str(matryoshka_path)

    # 3. Load FAISS Index
    print("Loading FAISS index...")
    state.faiss_index = faiss.read_index(str(artifacts_dir / "faiss_index.bin"))

    # 4. Load Sparse Models
    print("Loading SPLADE model + pre-encoded doc matrix...")
    state.splade_model = SPLADEFast(
        state.cfg["sparse"]["splade_model"],
        device=state.device,
    )
    splade_path = artifacts_dir / "splade_data.pt"
    if not splade_path.exists():
        raise FileNotFoundError(
            f"Missing SPLADE state at {splade_path}. "
            "Without it, SPLADE will return no candidates."
        )
    splade_data = torch.load(str(splade_path), map_location=state.device)
    state.splade_model.doc_matrix = splade_data["doc_matrix"].to(state.device)
    state.splade_model.pid_list = splade_data["pid_list"]
    print(f"  SPLADE doc_matrix shape: {tuple(state.splade_model.doc_matrix.shape)}")

    print("Loading BM25 retriever...")
    # The pickled BM25 may contain torch tensors that were on CUDA when
    # serialized. Plain pickle.load() can't remap devices on the fly.
    # torch.load() with map_location only handles the top level — nested
    # storages get re-deserialized via internal calls that don't see our
    # map_location. The robust fix: monkey-patch torch.load globally for
    # the duration of this load, so every recursive call defaults to CPU.
    bm25_path = artifacts_dir / "bm25_retriever.pkl"
    _orig_torch_load = torch.load

    def _cpu_torch_load(*args, **kwargs):
        kwargs.setdefault("map_location", state.device)
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _cpu_torch_load
    try:
        with open(bm25_path, "rb") as f:
            state.bm25_model = pickle.load(f)
    finally:
        torch.load = _orig_torch_load

    # BM25 may have been pickled with cuda tensors; coerce any remaining
    # device attributes after the fact.
    if hasattr(state.bm25_model, "doc_matrix") and state.bm25_model.doc_matrix is not None:
        try:
            state.bm25_model.doc_matrix = state.bm25_model.doc_matrix.to(state.device)
            state.bm25_model.device = state.device
        except Exception as e:
            print(f"  Warning: could not move BM25 doc_matrix to {state.device}: {e}")

    # 5. Load Cross-Encoder (Reranker)
    print("Loading cross-encoder reranker...")
    ce_max_len = state.cfg.get("reranker", {}).get("max_seq_length", 128)
    state.cross_encoder = CrossEncoder(
        state.cfg["cross_encoder_model"],
        max_length=ce_max_len,
        device=state.device,
    )
    state.cross_encoder.model.eval()

    print("All systems ready.")
    yield
    print("Shutting down.")
    if state.device == "cuda":
        torch.cuda.empty_cache()


# --- FastAPI App ---
app = FastAPI(
    title="Amazon ESCI Hybrid Search API",
    description=(
        "Two-stage hybrid retrieval (Matryoshka 64-dim + SPLADE + BM25) "
        "with cross-encoder reranking."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# --- Pydantic Models ---
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=256, description="User search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return (max 50)")
    show_signals: bool = Field(
        default=True,
        description="Include per-result retrieval-signal breakdown (which retrievers found it and at what rank)",
    )


class RetrievalSignals(BaseModel):
    """Which retrievers contributed to a result, and at what rank."""
    bm25_rank: Optional[int] = None
    splade_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    fused_rrf_score: Optional[float] = None


class SearchResult(BaseModel):
    product_id: str
    score: float                                  # Final cross-encoder score
    text: str
    signals: Optional[RetrievalSignals] = None


class StageTimings(BaseModel):
    retrieval_ms: float
    rerank_ms: float
    total_ms: float


class SearchResponse(BaseModel):
    status: str
    query: str
    timings: StageTimings
    results: list[SearchResult]


# --- Endpoints ---
@app.get("/")
async def root():
    """API metadata. Useful as a sanity check on the deployed URL."""
    return {
        "service": "Amazon ESCI Hybrid Search API",
        "version": app.version,
        "endpoints": {
            "POST /api/v1/search": "Hybrid retrieval + reranking",
            "GET /health": "Liveness probe",
            "GET /info": "Loaded model and config diagnostics",
            "GET /docs": "OpenAPI / Swagger UI",
        },
        "github": "https://github.com/217amin/amazon-esci-search",
    }


@app.get("/health")
async def health_check():
    if state.cross_encoder is None:
        raise HTTPException(status_code=503, detail="Models are still loading.")
    return {
        "status": "healthy",
        "device": state.device,
        "gpu_available": torch.cuda.is_available(),
    }


@app.get("/info")
async def info():
    """Diagnostics: which model produced query embeddings, retrieval config, etc."""
    if state.cfg is None:
        raise HTTPException(status_code=503, detail="Config not yet loaded.")
    return {
        "device": state.device,
        "dense_model_path": state.dense_model_path,
        "matryoshka_dim": state.cfg["retrieval"]["matryoshka_dim"],
        "splade_model": state.cfg["sparse"]["splade_model"],
        "cross_encoder_model": state.cfg["cross_encoder_model"],
        "rrf_weights": state.cfg["retrieval"]["rrf_weights"],
        "candidate_top_k": state.cfg["retrieval"]["candidate_top_k"],
        "num_products": len(state.product_ids),
    }


@app.post("/api/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Real-time inference: hybrid retrieval + cross-encoder reranking."""
    start_time = time.time()
    try:
        query_str = request.query
        retrieval_cfg = state.cfg["retrieval"]

        # =====================================================================
        # STAGE 1: HYBRID CANDIDATE GENERATION (Dense + SPLADE + BM25 -> RRF)
        # =====================================================================

        # A. Dense Search (FAISS) — encode query with the SAME matryoshka model
        # used to encode products, then truncate to 64 dims.
        instruction = "Represent this sentence for searching relevant passages: "
        query_emb = state.dense_model.encode(
            [instruction + query_str], normalize_embeddings=True
        )
        query_emb_64 = query_emb[:, : retrieval_cfg["matryoshka_dim"]]
        faiss.normalize_L2(query_emb_64)

        D, I = state.faiss_index.search(query_emb_64, k=retrieval_cfg["dense_top_k"])
        dense_candidates = [state.product_ids[idx] for idx in I[0] if idx != -1]

        # B. SPLADE Search
        splade_raw = state.splade_model.score_topk(
            query_str, top_k=retrieval_cfg["sparse_top_k"]
        )
        splade_candidates = [state.splade_model.pid_list[idx] for idx, _ in splade_raw]

        # C. BM25 Search
        bm25_raw = state.bm25_model.search(
            query_str, top_k=retrieval_cfg["sparse_top_k"]
        )
        bm25_candidates = [pid for pid, _ in bm25_raw]

        # D. Weighted Reciprocal Rank Fusion (RRF)
        rrf_k = retrieval_cfg["rrf_k"]
        weights = retrieval_cfg["rrf_weights"]
        combined_scores = defaultdict(float)

        # Track per-retriever ranks for signal breakdown response.
        dense_rank_of = {pid: i + 1 for i, pid in enumerate(dense_candidates)}
        splade_rank_of = {pid: i + 1 for i, pid in enumerate(splade_candidates)}
        bm25_rank_of = {pid: i + 1 for i, pid in enumerate(bm25_candidates)}

        for rank, pid in enumerate(dense_candidates):
            combined_scores[pid] += weights["dense"] / (rrf_k + rank + 1)
        for rank, pid in enumerate(splade_candidates):
            combined_scores[pid] += weights["splade"] / (rrf_k + rank + 1)
        for rank, pid in enumerate(bm25_candidates):
            combined_scores[pid] += weights["bm25"] / (rrf_k + rank + 1)

        sorted_candidates = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_candidates = [
            pid for pid, _ in sorted_candidates[: retrieval_cfg["candidate_top_k"]]
        ]
        fused_score_of = {pid: score for pid, score in sorted_candidates}

        time_retrieval = time.time() - start_time

        # =====================================================================
        # STAGE 2: CROSS-ENCODER RERANKING
        # =====================================================================
        start_rerank = time.time()

        candidate_texts = [state.id_to_text.get(pid, "") for pid in top_candidates]
        cross_inp = [[query_str, text] for text in candidate_texts]

        cross_scores = state.cross_encoder.predict(
            cross_inp,
            batch_size=16,
            show_progress_bar=False,
        )
        time_rerank = time.time() - start_rerank

        # Build response
        results = []
        for pid, score, text in zip(top_candidates, cross_scores, candidate_texts):
            signals = None
            if request.show_signals:
                signals = RetrievalSignals(
                    bm25_rank=bm25_rank_of.get(pid),
                    splade_rank=splade_rank_of.get(pid),
                    dense_rank=dense_rank_of.get(pid),
                    fused_rrf_score=round(fused_score_of.get(pid, 0.0), 6),
                )
            results.append(
                SearchResult(
                    product_id=pid,
                    score=float(score),
                    text=text,
                    signals=signals,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        results = results[: request.top_k]

        total_ms = (time.time() - start_time) * 1000.0

        return SearchResponse(
            status="success",
            query=query_str,
            timings=StageTimings(
                retrieval_ms=round(time_retrieval * 1000.0, 2),
                rerank_ms=round(time_rerank * 1000.0, 2),
                total_ms=round(total_ms, 2),
            ),
            results=results,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {e}")


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)