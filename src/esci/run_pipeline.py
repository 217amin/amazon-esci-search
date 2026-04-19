import argparse
from pathlib import Path

import pandas as pd
import yaml

from .data import (
    sample_dataset,
    add_product_text,
    add_grades_and_pair_view,
    filter_queries_with_E,
    remove_train_test_overlap,
)
from .matryoshka_train import train_matryoshka
from .system_a import encode_systemA
from .system_b import build_candidates, rerank_candidates
from .metrics import compute_recall_metrics, compute_ndcg_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "esci.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    args = parser.parse_args()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("Loading data...")

    raw_examples_path = PROJECT_ROOT / cfg["paths"]["raw_examples"]
    raw_products_path = PROJECT_ROOT / cfg["paths"]["raw_products"]
    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    
    processed_dir.mkdir(parents=True, exist_ok=True)

    df_examples = pd.read_parquet(raw_examples_path)
    df_products = pd.read_parquet(raw_products_path)

    df = pd.merge(
        df_examples,
        df_products,
        how="left",
        on=["product_locale", "product_id"],
    )

    if "product_locale" in df.columns:
        df = df[df["product_locale"] == "us"].copy()

    if "small_version" in df.columns:
        df = df[df["small_version"] == 1].copy()

    if "query" in df.columns:
        df = df[df["query"].astype(str).map(str.isascii)].copy()

    df = sample_dataset(df, cfg)

    if "esci_label" in df.columns:
        df = filter_queries_with_E(df)

    if "split" in df.columns and "query" in df.columns:
        df = remove_train_test_overlap(df)

    print("Generating product text representations...")
    df = add_product_text(df)
    df = add_grades_and_pair_view(df)

    pair_df_path = processed_dir / "pair_df.parquet"
    df.to_parquet(pair_df_path, index=False)
    print(f"Saved pair dataframe to: {pair_df_path}")
    print(f"Final shape: {df.shape}")

    if args.mode == "train":
        train_matryoshka(df, cfg)

    elif args.mode == "inference":
        encode_systemA(df, cfg)
        cands, ret_qps = build_candidates(cfg)

        final, rerank_qps = rerank_candidates(
            cands,
            model_name=cfg["cross_encoder_model"],
            batch_size=cfg.get("reranker", {}).get("batch_size", 16),
            top_k_to_rerank=50,
            cfg=cfg,
        )

        # Ground-truth grades by query for evaluation
        test_df = df[df["split"] == "test"].copy()
        q_rels = test_df.groupby("query_id")["grade"].apply(list).to_dict()

        # Candidate-stage metrics
        cand_recall = compute_recall_metrics(cands, q_rels, ks=[200])
        cand_ndcg = compute_ndcg_metrics(cands, q_rels, ks=[20])

        # Reranked metrics
        rerank_ndcg = compute_ndcg_metrics(final, q_rels, ks=[20])

        print("\n=== Candidate Retrieval ===")
        print(f"QPS: {ret_qps:.2f}")
        print(f"Recall@200: {cand_recall.get('Recall@200', 0.0):.4f}")
        print(f"nDCG@20: {cand_ndcg.get('nDCG@20', 0.0):.4f}")

        print("\n=== Reranked Results ===")
        print(f"QPS: {rerank_qps:.2f}")
        print(f"nDCG@20: {rerank_ndcg.get('nDCG@20', 0.0):.4f}")


if __name__ == "__main__":
    main()