# Experiments (Ablation Plan)

These experiments demonstrate:
1) Matryoshka advantage at small dims
2) Hybrid retrieval benefits
3) Reranking impact
4) Latency/throughput trade-offs

---

## 1) Dimension sweep (dense-only)
Dims: 768, 512, 256, 128, 64  
Compare baseline vs Matryoshka embeddings.

Expected:
- Similar at 768
- Matryoshka significantly better at 64/128

---

## 2) Hybrid fusion sweep
For each dim:
- Dense-only
- Dense + BM25
- Dense + SPLADE
- Dense + BM25 + SPLADE

Tune:
- `rrf_k` (10, 60, 120)
- `candidate_top_k` (100, 200, 500)
- `rrf_weights` (BM25 0.2–0.5; SPLADE 0.5–1.0)

---

## 3) Reranking ablation
- Rerank top-100 vs top-200
- Compare nDCG@10/20 and QPS
- Report the latency budget impact

---

## 4) SPLADE configuration
- max_length: 128 vs 256 (make configurable)
- doc text: full `product_text_dense` vs (title)

Measure:
- Recall@K
- Index build time
- Query latency

---

## 5) MNRL (training)
We use in-batch negatives via MultipleNegativesRankingLoss (efficient contrastive learning).
Planned upgrade (hard negatives): 
Mine difficult negatives from BM25/SPLADE/dense candidates and train with positives 
+ hard negatives to improve discrimination on near-miss items.
