import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .artifacts import save_artifacts

def encode_systemA(pair_df: pd.DataFrame, cfg: dict, model_override: str = None):
    split = "test"
    df = pair_df[pair_df["split"] == split].copy()
    
    # Logic: Override -> FineTuned (if exists) -> Config Base
    if model_override:
        model_path = model_override
    else:
        ft_path = Path(cfg["paths"]["matryoshka_dir"]) / "us"
        # If fine-tuned model exists, use it. Else use base model.
        model_path = str(ft_path) if ft_path.exists() else cfg["biencoder_model"]
    
    print(f"🚀 Encoding with System A model: {model_path}")
    model = SentenceTransformer(model_path, trust_remote_code=True)
    
    # --- Force Sequence Length ---
    target_seq_len = cfg.get("matryoshka", {}).get("max_seq_length", 128)
    print(f"📏 Max Seq Length: {target_seq_len}")
    model.max_seq_length = int(target_seq_len)
    
    # 1. ENCODE PRODUCTS (Standardize Order)
    print("    -> Preparing Product List...")
    prod_df_unique = df[["product_id", "product_text_dense"]].drop_duplicates("product_id")
    
    print(f"    -> Encoding {len(prod_df_unique)} Products...")
    # Products DO NOT use instructions
    prod_emb = model.encode(
        prod_df_unique["product_text_dense"].tolist(), 
        batch_size=32, 
        show_progress_bar=True,
        normalize_embeddings=True 
    )
    
    # 2. ENCODE QUERIES (Fix Alignment + Add Instruction)
    print("    -> Preparing Query List...")
    qry_df_unique = df[["query_id", "query"]].drop_duplicates("query_id")
    
    # BGE models ALWAYS need instructions for queries.
    instruction = "Represent this sentence for searching relevant passages: "
    
    print(f"    Applying BGE instruction prefix: '{instruction}'")
    queries = [instruction + q for q in qry_df_unique["query"].tolist()]
    
    print(f"    -> Encoding {len(queries)} Queries...")
    qry_emb = model.encode(
        queries, 
        batch_size=32, 
        show_progress_bar=True,
        normalize_embeddings=True 
    )
    
    # 3. SAVE ARTIFACTS
    # Manual safety normalization 
    prod_emb = prod_emb / np.maximum(np.linalg.norm(prod_emb, axis=1, keepdims=True), 1e-10)
    qry_emb = qry_emb / np.maximum(np.linalg.norm(qry_emb, axis=1, keepdims=True), 1e-10)
    
    save_artifacts("us", split, cfg, prod_emb, qry_emb, df)
    print(" System A Encoding Complete.")