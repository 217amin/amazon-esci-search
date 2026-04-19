import os
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading data...")
# Load the dataset
df = pd.read_parquet("data/processed/pair_df.parquet")

# 1. Get UNIQUE products exactly like your system_a.py does
prod_df_unique = df[["product_id", "product_text_dense"]].drop_duplicates("product_id")
print(f"Found {len(prod_df_unique)} unique products.")

# 2. Encode
print("Loading BGE-Base model...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

print("Encoding products (this will take a moment)...")
embeddings = model.encode(
    prod_df_unique["product_text_dense"].tolist(), 
    batch_size=32, 
    show_progress_bar=True, 
    normalize_embeddings=True
)

# 3. Matryoshka Truncation (to 64d) and L2 Normalization
print("Compressing to 64 dimensions...")
emb_64 = embeddings[:, :64].astype(np.float32)
faiss.normalize_L2(emb_64)

# 4. Build FAISS Index
print("Building FAISS index...")
index = faiss.IndexFlatIP(64)
index.add(emb_64)

# 5. Save the index AND the exact unique IDs mapping
os.makedirs("artifacts/systemA", exist_ok=True)
faiss.write_index(index, "artifacts/systemA/faiss_index.bin")
prod_df_unique["product_id"].to_pickle("artifacts/systemA/faiss_mapping.pkl")

print("✅ faiss_index.bin and mapping created successfully!")