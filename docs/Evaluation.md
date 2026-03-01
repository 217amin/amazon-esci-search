# Evaluation

## Metrics
### nDCG@K (graded)
- Uses graded gains (E/S/C/I → 1.0/0.1/0.01/0.0)
- Evaluate at K ∈ {10, 20, 50}

Why:
- Measures top-ranking quality
- Supports graded relevance (not only binary)

### Recall@K
- Evaluate at K ∈ {50, 100, 200}
- Two modes:
  - **Strict:** Exact matches only (E)
  - **Broad:** includes additional relevant labels depending on threshold

Why:
- Candidate generation must retrieve relevant items before reranking can help
- Recall is the primary “coverage” metric for first-stage retrieval

## Experimental comparisons
Recommended reporting:
1) Dense-only: baseline vs matryoshka across dims
2) Hybrid: Dense+BM25, Dense+SPLADE, Dense+BM25+SPLADE across dims
3) Reranking: on/off for top-200 candidates
4) Speed: candidate generation QPS and reranking QPS

## Interpreting Matryoshka correctly
- At 768 dims, Matryoshka is expected to be close to baseline.
- Value appears at 64/128 dims where Matryoshka preserves recall/nDCG much better than baseline prefixes.
