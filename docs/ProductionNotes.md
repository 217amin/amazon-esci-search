# Production Notes

## Offline vs Online separation
### Offline
- Build product text fields
- Train Matryoshka model (optional)
- Encode products/queries (System A) and save artifacts
- Build and cache sparse indices (BM25/SPLADE)
- Build FAISS index for serving

### Online (serving)
- Receive query
- Encode query once (bi‑encoder)
- Retrieve candidates (dense + sparse)
- Fuse with weighted RRF
- Rerank top‑K with cross‑encoder
- Return ranked results

## Artifact versioning (strong recommendation)
Store artifacts per run with:
- config hash
- model version (baseline vs matryoshka)
- timestamp

Prevents accidental mixing of embeddings across runs.

## Monitoring (if deployed)
- Latency p50/p95 for each stage (encode / dense / sparse / rerank)
- Relevance drift: nDCG@20, Recall@200
- Data drift: query length distribution, category mix

## Scaling notes
- `IndexFlatIP` is correctness-first.
- For large corpora, consider ANN indexes:
  - HNSW / IVF / PQ (FAISS)

## Privacy
- Avoid storing raw queries unless required.
- If stored, add retention + anonymization policies.
