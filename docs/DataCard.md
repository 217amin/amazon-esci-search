# DataCard — Amazon ESCI (Shopping Queries)

## Dataset
Amazon ESCI (Shopping Queries Dataset) provides query–product pairs labeled:
- **E**: Exact
- **S**: Substitute
- **C**: Complement
- **I**: Irrelevant

## Splits and leakage controls
- Uses provided `train` / `test` splits.
- Optional overlap removal utilities prevent train/test contamination when needed.

## Label → grade mapping (for graded metrics)
Used for nDCG (graded gains):
- E = 1.0
- S = 0.7
- C = 0.3
- I = 0.0

Rationale:
- Reflects decreasing relevance strength
- Enables meaningful graded ranking evaluation

## Text fields
### Dense field
`product_text_dense` = title + brand + bullet points + optional color 

Why:
- Captures identifiers + attributes
- Matches real catalog descriptions
- Improves semantic matching and reranker input quality

### Sparse fields
- **BM25:** stemming/tokenization for lexical robustness.
- **SPLADE:** raw text (tokenizer‑driven); avoid aggressive normalization.

## Sampling (debug)
Query-level sampling supports fast iteration without changing code paths.

## Known risks / biases
- Duplicates, near-duplicates, and missing fields are common in real catalogs.
- Labels can be noisy for ambiguous shopping intents.
- Locale filtering must be consistent across training and evaluation.
