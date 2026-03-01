# ModelCard — Baseline and Matryoshka Bi‑Encoder

## Baseline bi‑encoder
- Backbone: `BAAI/bge-base-en-v1.5`
- Queries: instruction prefix  
  `Represent this sentence for searching relevant passages: {query}`
- Products: no instruction
- Embeddings: normalized

Why BGE:
- Strong general retrieval baseline
- Well supported in SentenceTransformers
- Instruction-tuned for retrieval queries

## Matryoshka fine‑tuning
### Objective
- Inner loss: **MultipleNegativesRankingLoss (MNRL)**
- Wrapper: **MatryoshkaLoss** with dims `[768, 512, 256, 128, 64]`
- Sampler: **NO_DUPLICATES** (stable in-batch negatives)
- (Optional) Extra weight on smallest dim to encourage compression quality

Why MNRL:
- SOTA bi‑encoder retrieval objective
- Efficient training via in-batch negatives
- Directly optimizes similarity separation used in retrieval

Why Matryoshka:
- Preserves full-dim retrieval while improving smaller prefixes
- Enables low-latency / low-memory serving (64/128 dims)
- Adds a practical “production” story: cheaper vectors without losing recall

### Training data choice (project decision)
- Option A: **E-only** positives (clean + strict)
- Option B: **E+S** positives (closer to real shopping relevance)

## Cross‑encoder reranker
- Model: `mixedbread-ai/mxbai-rerank-base-v1`
- Used for top‑K candidates per query
- Improves precision and nDCG at small K
