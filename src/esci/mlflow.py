import os, re, mlflow
import pandas as pd

def _mlflow_safe_key(k: str) -> str:
    """
    MLflow metric/param name rules do NOT allow '@'.
    Allowed: alphanumerics, underscores, dashes, periods, spaces, slashes.
    We'll convert:
      Recall@200 -> Recall_200
      nDCG@20   -> nDCG_20
    and also remove any other unsafe characters defensively.
    """
    k = k.replace("@", "_")
    k = re.sub(r"[^0-9a-zA-Z_\-\. /]", "_", k)   # replace other invalid chars
    return k
    
def log_candidates_run(cfg, dim, sources, metrics_dict, qps, out_dir="../results"):
    os.makedirs(out_dir, exist_ok=True)

    # Save a small metrics CSV (artifact)
    run_csv = f"{out_dir}/candidates_dim{dim}_{'_'.join(sources)}.csv"
    pd.DataFrame([{
        "dim": dim,
        "sources": ",".join(sources),
        **{_mlflow_safe_key(k): float(v) for k, v in metrics_dict.items()},
        "QPS": qps
    }]).to_csv(run_csv, index=False)

    params = {
        "split": "test",
        "dim": dim,
        "sources": ",".join(sources),
        "finetuned_biencoder_model": cfg["biencoder_model"],
        "rrf_k": cfg["retrieval"]["rrf_k"],
        "candidate_top_k": cfg["retrieval"]["candidate_top_k"],
        "dense_top_k": cfg["retrieval"]["dense_top_k"],
        "sparse_top_k": cfg["retrieval"]["sparse_top_k"],
        "w_dense": cfg["retrieval"]["rrf_weights"].get("dense", 0.0),
        "w_bm25": cfg["retrieval"]["rrf_weights"].get("bm25", 0.0),
        "w_splade": cfg["retrieval"]["rrf_weights"].get("splade", 0.0),
    }

    with mlflow.start_run(run_name=f"candidates__dim{dim}__{'_'.join(sources)}"):
        # params (also sanitize keys just in case)
        mlflow.log_params({_mlflow_safe_key(k): v for k, v in params.items()})

        # metrics
        for k, v in metrics_dict.items():
            mlflow.log_metric(_mlflow_safe_key(k), float(v))
        mlflow.log_metric("QPS", float(qps))
        # artifacts (config + csv)
        mlflow.log_artifact("../configs/esci.yaml", artifact_path="config")
        mlflow.log_artifact(run_csv, artifact_path="results")
        
def log_rerank_run(cfg, dim, sources, metrics_dict, qps, out_dir="../results"):
    os.makedirs(out_dir, exist_ok=True)

    run_csv = f"{out_dir}/rerank_dim{dim}_{'_'.join(sources)}.csv"
    pd.DataFrame([{
        "dim": dim,
        "sources": ",".join(sources),
        **{_mlflow_safe_key(k): float(v) for k, v in metrics_dict.items()},
        "RERANK_QPS": qps
    }]).to_csv(run_csv, index=False)

    params = {
        "split": "test",
        "dim": dim,
        "sources": ",".join(sources),
        "cross_encoder_model": cfg["cross_encoder_model"],
        "top_k_to_rerank": cfg["retrieval"]["candidate_top_k"]
    }

    with mlflow.start_run(run_name=f"rerank__dim{dim}__{'_'.join(sources)}"):
        mlflow.log_params({_mlflow_safe_key(k): v for k, v in params.items()})
        for k, v in metrics_dict.items():
            mlflow.log_metric(_mlflow_safe_key(k), float(v))
        mlflow.log_metric("RERANK_QPS", float(qps))
        mlflow.log_artifact("../configs/esci.yaml", artifact_path="config")
        mlflow.log_artifact(run_csv, artifact_path="results")