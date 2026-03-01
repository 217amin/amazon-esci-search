# src/esci/artifacts.py
import numpy as np
import pandas as pd
from pathlib import Path

def save_artifacts(locale, split, cfg, prod_emb, qry_emb, df):
    base = Path(cfg["paths"]["artifacts_dir"])
    base.mkdir(parents=True, exist_ok=True)
    np.save(base / f"prod_emb_{split}.npy", prod_emb)
    np.save(base / f"qry_emb_{split}.npy", qry_emb)
    df.to_parquet(base / f"df_{split}.parquet")

def load_artifacts(locale, split, cfg):
    base = Path(cfg["paths"]["artifacts_dir"])
    prod_emb = np.load(base / f"prod_emb_{split}.npy")
    qry_emb = np.load(base / f"qry_emb_{split}.npy")
    df = pd.read_parquet(base / f"df_{split}.parquet")
    prod_df = df.drop_duplicates("product_id")
    qry_df = df.drop_duplicates("query_id")
    return prod_emb, qry_emb, prod_df, qry_df, df