# src/esci/faiss_utils.py
from __future__ import annotations
import faiss
import numpy as np


def build_faiss_index(emb: np.ndarray):
    """
    Build an Inner-Product FAISS index on CPU, upgrade to GPU if available.
    emb: (N, D) float32 matrix, already normalized.
    """
    emb = emb.astype("float32")
    dim = emb.shape[1]

    index = faiss.IndexFlatIP(dim)

    # GPU if available
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(emb)
    return index


def faiss_search_topk(index, q_vec: np.ndarray, top_k: int):
    """
    q_vec: (D,) float32 vector
    Returns (idxs, scores) of the top_k items.
    """
    q = q_vec.astype("float32")[None, :]
    scores, idx = index.search(q, top_k)
    return idx[0], scores[0]


def faiss_search(index, q_vec: np.ndarray, top_k: int):
    """
    Backward-compatible alias (same return order as faiss_search_topk).
    """
    return faiss_search_topk(index, q_vec, top_k)
