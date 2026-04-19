import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sentence_transformers import CrossEncoder
from .sparse_retrievers import SPLADEFast, BM25Fast
from .artifacts import load_artifacts
from .metrics import build_relevant_sets 
from .faiss_utils import build_faiss_index, faiss_search_topk
import torch

def _rrf_score(rank: int, k: int = 60, weight: float = 1.0) -> float:
    """
    Reciprocal Rank Fusion (RRF) Scorer.
    Engineering Choice: RRF is distribution-agnostic, making it safer than 
    Linear Combination (LC) when fusing dense (cosine) and sparse (BM25) scores.
    """
    return weight * (1.0 / (k + rank + 1))

def build_candidates(
    cfg: Dict[str, Any], 
    split="test", 
    override_dim: int = None,
    prebuilt_splade: Optional[SPLADEFast] = None,
    prebuilt_bm25: Optional[BM25Fast] = None
) -> Tuple[pd.DataFrame, float]:
    """
    Multi-Stage Retrieval Pipeline:
    1. Dense (Matryoshka) + Sparse (SPLADE/BM25) Retrieval.
    2. Reciprocal Rank Fusion (RRF) aggregation.
    3. Filtering of valid queries (The "Gatekeeper").
    """

    # --- 1. Load Artifacts & Initialize Gatekeeper ---
    prod_emb, qry_emb, prod_df, qry_df, pair_df = load_artifacts("us", split, cfg)
    
    # Gatekeeper: Filter out queries with 0 relevant items in the Ground Truth.
    q_to_rel = build_relevant_sets(pair_df, mode="broad")
    valid_qids = set(q_to_rel.keys())
    
    # --- 2. Initialize Retrievers ---
    sources = cfg["retrieval"].get("sources", ["dense"])
    
    # A. Dense Retrieval Strategy (with FAISS)
    index = None
    if "dense" in sources:

        target_dim = override_dim if override_dim else cfg["retrieval"]["matryoshka_dim"]
        
        # Slicing: Matryoshka efficient truncation
        P = prod_emb[:, :target_dim]
        Q = qry_emb[:, :target_dim]
        
        # Critical: L2 Normalization ensures Dot Product == Cosine Similarity
        P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-10)
        Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-10)
        
        print(f" Using FAISS (Dim={target_dim})")
        index = build_faiss_index(P)

    # B. Sparse Retrieval Strategy
    splade = None
    if "splade" in sources:
        # Re-use prebuilt object to save VRAM during iterative testing
        splade = prebuilt_splade if prebuilt_splade else SPLADEFast(
            cfg["sparse"]["splade_model"], 
            batch_size=cfg["sparse"]["batch_size"]
        )
        if not prebuilt_splade: 
            splade.build_index(prod_df["product_text_dense"].tolist(), prod_df["product_id"].tolist())

    bm25 = None
    if "bm25" in sources:
        # 1. Initialize empty (or with config params)
        bm25 = prebuilt_bm25 if prebuilt_bm25 else BM25Fast()
        
        if not prebuilt_bm25:
            print("⚡ Building BM25 Index on the fly (using Product Titles)...")
            
            #  Added .astype(str) to prevent crashes on numeric titles
            bm25.build_index(
                texts=prod_df["product_title"].fillna("").astype(str).tolist(), 
                pids=prod_df["product_id"].tolist()
            )

    # --- 3. Execution Loop ---
    rows = []
    
    # RRF Configuration
    rrf_k = cfg["retrieval"].get("rrf_k", 60) 
    candidate_top_k = cfg["retrieval"].get("candidate_top_k", 200)
    dense_k = cfg["retrieval"]["dense_top_k"]
    sparse_k = cfg["retrieval"]["sparse_top_k"]
    weights = cfg["retrieval"].get("rrf_weights", {"dense": 1.0, "splade": 0.5, "bm25": 0.2})
    
    # Pre-compute lookups for O(1) access inside loop
    q_raw = qry_df.set_index("query_id")["query"].to_dict()
    
    q_to_pos = {qid: i for i, qid in enumerate(qry_df["query_id"])}
    pid_list = prod_df["product_id"].tolist()
    pid_to_text = prod_df.set_index("product_id")["product_text_dense"].to_dict()
    grade_lookup = pair_df.set_index(["query_id", "product_id"])["grade"].to_dict()

    print(f"    -> Measuring QPS for {len(valid_qids)} queries...")
    start_time = time.time()

    for qid in valid_qids:
        if qid not in q_to_pos: continue
        rrf_scores = {}

        # 1. Dense Search
        if index:
            query_vec = Q[q_to_pos[qid]]
            w = weights["dense"]
            
            d_idxs, _ = faiss_search_topk(index, query_vec, dense_k)
            pids = [pid_list[i] for i in d_idxs]
            
            for rank, pid in enumerate(pids):
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + _rrf_score(rank, rrf_k, w)

        # 2. Sparse Search (SPLADE)
        if splade:
            w = weights["splade"]
            s_res = splade.score_topk(q_raw[qid], top_k=sparse_k)
            for rank, (db_idx, _) in enumerate(s_res):
                pid = splade.pid_list[db_idx]
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + _rrf_score(rank, rrf_k, w)
        
        # 3. Keyword Search (BM25)
        if bm25:
            w = weights["bm25"]
            
            # Explicitly use q_raw (raw query text)
            b_res = bm25.search(q_raw[qid], top_k=sparse_k)
            
            for rank, (pid, _) in enumerate(b_res):
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + _rrf_score(rank, rrf_k, w)

        # 4. Final Selection
        sorted_cands = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_top_k]
        
        for pid, score in sorted_cands:
            rows.append({
                "query_id": qid, "product_id": pid, 
                "grade": float(grade_lookup.get((qid, pid), 0.0)), 
                "rrf_score": score,
                "query": q_raw[qid],
                "product_text_dense": pid_to_text.get(pid, "")
            })
            
    end_time = time.time()
    qps = len(valid_qids) / (end_time - start_time) if (end_time - start_time) > 0 else 0.0
    
    return pd.DataFrame(rows), qps

def rerank_candidates(
    candidates_df: pd.DataFrame, 
    model_name: str, 
    batch_size: int = 2048, 
    top_k_to_rerank: int = 200,
    cfg: Dict[str, Any] = None
) -> Tuple[pd.DataFrame, float]:
    """
    Cross-Encoder Reranking.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_length = cfg["reranker"]["max_seq_length"]
    print(f" Loading Cross-Encoder: {model_name} on {device.upper()}... and Max length: {max_length}")
    
    ce = CrossEncoder(model_name, device=device, max_length=max_length)
    
    # Filtering: Only rerank the top K per query to manage latency
    subset = (candidates_df
          .sort_values(["query_id", "rrf_score"], ascending=[True, False])
          .groupby("query_id")
          .head(top_k_to_rerank)
          .copy())

    pairs = list(zip(subset["query"], subset["product_text_dense"]))
    
    print(f"    -> Reranking {len(pairs)} pairs...")
    start_time = time.time()
    
    # num_workers=0 prevents deadlocks on Windows/WSL
    scores = ce.predict(
        pairs, 
        batch_size=batch_size, 
        show_progress_bar=False
    )
    
    subset["ce_score"] = scores
    final_df = subset.sort_values(["query_id", "ce_score"], ascending=[True, False])
    
    total_time = time.time() - start_time
    qps = subset["query_id"].nunique() / total_time if total_time > 0 else 0.0
    
    print(f"    -> Rerank Speed: {qps:.2f} QPS")
    return final_df, qps