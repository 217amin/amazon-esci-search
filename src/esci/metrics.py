import numpy as np
import pandas as pd
from typing import Dict, List, Any

def build_relevant_sets(pair_df_loc: pd.DataFrame, mode: str):
    """
    The 'Gatekeeper' function.
    Role: Filters the Ground Truth to determine which queries are valid for evaluation.
    
    Modes:
    - 'strict': Only Exact (E) matches. Used for high-precision benchmarking.
    - 'broad': Exact (E) + Substitute (S). Used for Recall calculations.
    """
    if mode == "strict":
        rel_df = pair_df_loc[pair_df_loc["grade"] >= 0.99]
    elif mode == "broad":
        rel_df = pair_df_loc[pair_df_loc["grade"] >= 0.10]
    else:
        raise ValueError("mode must be 'broad' or 'strict'")
    return rel_df.groupby("query_id")["product_id"].apply(set).to_dict()

def dcg_at_k(gains: List[float], k: int) -> float:
    gains = np.asarray(gains, dtype=float)[:k]
    if gains.size == 0: return 0.0
    return np.sum((2**gains - 1) / np.log2(np.arange(len(gains)) + 2))

def ndcg_at_k(ranked_gains: List[float], k: int) -> float:
    actual = dcg_at_k(ranked_gains, k)
    ideal = dcg_at_k(sorted(ranked_gains, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0

def compute_recall_metrics(df: pd.DataFrame, q_rels: Dict[str, List[float]], ks=[50, 100, 200]) -> Dict[str, float]:
    """
    Computes Recall@K. 
    Engineering Note: Uses 'broad' relevance (E+S) by default, as Recall is about coverage.
    """
    recall_per_k = {k: [] for k in ks}

    for qid, group in df.groupby("query_id"):
        if qid not in q_rels: continue
        
        # Denominator: Universe of items with grade >= 0.5
        raw_grades = q_rels[qid]
        total_relevant = sum(1 for g in raw_grades if g >= 0.10)
        
        if total_relevant == 0: continue

        # Sort by system score
        score_col = "ce_score" if "ce_score" in group.columns else "rrf_score"
        group = group.sort_values(score_col, ascending=False)
        retrieved_grades = group["grade"].values.tolist()

        for k in ks:
            cutoff = min(k, len(group))
            hits = sum(1 for g in retrieved_grades[:cutoff] if g >= 0.10)
            recall_per_k[k].append(hits / total_relevant)

    metrics = {}
    for k in ks:
        metrics[f"Recall@{k}"] = np.mean(recall_per_k[k]) if recall_per_k[k] else 0.0
    return metrics

def compute_ndcg_metrics(df: pd.DataFrame, q_rels: Dict[str, List[float]], ks=[10, 20, 50]) -> Dict[str, float]:
    """
    Computes nDCG@K.
    Engineering Note: Uses 'graded' relevance (1.0, 0.1, 0.01) to reward exact ordering.
    """
    ndcg_per_k = {k: [] for k in ks}

    for qid, group in df.groupby("query_id"):
        if qid not in q_rels: continue
        
        # IDCG based on all positive grades > 0.0
        raw_grades = q_rels[qid]
        ideal_grades = sorted([g for g in raw_grades if g > 0.0], reverse=True)
        if not ideal_grades: continue

        score_col = "ce_score" if "ce_score" in group.columns else "rrf_score"
        group = group.sort_values(score_col, ascending=False)
        retrieved_grades = group["grade"].values.tolist()

        for k in ks:
            cutoff = min(k, len(group))
            current_k = retrieved_grades[:cutoff]
            idcg = dcg_at_k(ideal_grades, k)
            actual = dcg_at_k(current_k, k)
            ndcg_per_k[k].append(actual / idcg if idcg > 0 else 0.0)

    metrics = {}
    for k in ks:
        metrics[f"nDCG@{k}"] = np.mean(ndcg_per_k[k]) if ndcg_per_k[k] else 0.0
    return metrics