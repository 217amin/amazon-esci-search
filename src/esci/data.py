# src/esci/data.py
from __future__ import annotations
from typing import Any, Dict
import re, unicodedata
import pandas as pd
import numpy as np

def sample_dataset(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    DEBUG TOOL: Samples a subset of queries for rapid pipeline testing.
    This allows verifying logic without waiting for full training.
    """
    if not cfg.get("debug", {}).get("use_sample", False):
        return df

    n = cfg["debug"]["sample_size"]
    print(f"\n DEBUG MODE: Sampling {n} queries...")
    
    unique_qids = df["query_id"].unique()
    if len(unique_qids) > n:
        selected_qids = np.random.choice(unique_qids, size=n, replace=False)
        df = df[df["query_id"].isin(selected_qids)].reset_index(drop=True)
    
    print(f" DEBUG MODE: New shape: {df.shape}")
    return df

LABEL2GRADE = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}

def normalize_sparse_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^a-z0-9\-\.\#\+]+", " ", text)
    
    return re.sub(r"\s+", " ", text).strip()

def filter_queries_with_E(examples: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only queries that have at least one 'E' (Exact) labeled product.
    """
    print(f"Before filtering: {examples['query_id'].nunique():,} queries total")
    mask_has_E = (
        examples.groupby(["product_locale", "query_id"])["esci_label"]
        .transform(lambda labels: (labels == "E").any())
    )
    examples = examples[mask_has_E].copy()
    print(f"After filtering (require E): {examples['query_id'].nunique():,} queries total")
    print(examples.groupby("product_locale")["query_id"].nunique().rename("queries_with_E"))
    return examples

def remove_train_test_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any query that appears in both train and test splits for the same locale.
    """
    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]
    overlap_qids = (
        df_train[["query_id", "product_locale"]]
        .merge(df_test[["query_id", "product_locale"]], on=["query_id", "product_locale"], how="inner")["query_id"]
        .unique()
    )
    print(f" Overlapping queries between train and test: {len(overlap_qids):,}")
    df_clean = df[~((df["split"] == "train") & (df["query_id"].isin(overlap_qids)))]
    print(
        f"After removing overlap: train queries = {df_clean[df_clean['split']=='train']['query_id'].nunique():,}, "
        f"test queries = {df_clean[df_clean['split']=='test']['query_id'].nunique():,}"
    )
    return df_clean

def build_product_text_dense(row: Dict[str, Any]) -> str:
    parts = []
    
    # Helper to clean text and catch "None"/"nan"
    def clean_field(val):
        if val is None: return ""
        s = str(val).strip()
        if s.lower() in ["none", "nan", ""]: return ""
        return s

    title = clean_field(row.get("product_title"))
    if title: parts.append(title + ".")
        
    brand = clean_field(row.get("product_brand"))
    if brand: parts.append(brand)
        
    bullets = clean_field(row.get("product_bullet_point"))
    if bullets: parts.append(bullets)
        
    return " ".join(parts).strip()

def add_product_text(df: pd.DataFrame) -> pd.DataFrame:
    print("Generating product text representations...")
    df["product_text_dense"] = df.apply(build_product_text_dense, axis=1)
    return df

def add_grades_and_pair_view(df: pd.DataFrame) -> pd.DataFrame:
    df["grade"] = df["esci_label"].map(LABEL2GRADE)
    df = df.dropna(subset=["grade"]).reset_index(drop=True)
    base_cols = [
        "example_id", "query_id", "product_id", "product_locale", "query",
        "esci_label", "grade", "split",
        "product_title", "product_text_dense",
    ]
    df = df[base_cols].copy()
    print("Pair-level DataFrame shape:", df.shape)
    return df