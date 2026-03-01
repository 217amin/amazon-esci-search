# sparse_retrievers.py

import re, torch
import numpy as np
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from .data import normalize_sparse_text


class SPLADEFast:
    """
    GPU-Accelerated Sparse Neural Retriever (SPLADE).
    Stores an inverted index as a CSR sparse matrix and scores queries via torch.mv(doc_matrix, q_vec).
    """
    def __init__(self, model_name: str, batch_size: int = 128, device: Optional[str] = None, max_length: int = 128):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.batch_size = batch_size
        self.max_length = max_length

        self.doc_matrix = None
        self.pid_list: List[str] = []
        self.vocab_size = self.tokenizer.vocab_size

    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # SPLADE max(log(1+relu(logits))) over sequence dimension
        values, _ = torch.max(torch.log1p(torch.relu(logits)), dim=1)
        return values

    def build_index(self, texts: List[str], pids: List[str]):
        print(f"🏗️ Building Fast SPLADE index for {len(texts)} docs...")
        self.pid_list = pids

        all_rows, all_cols, all_vals = [], [], []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_vecs = self._encode_text(batch)

            # collect non-zeros
            nz = torch.nonzero(batch_vecs, as_tuple=False)
            if nz.numel() == 0:
                continue

            vals = batch_vecs[nz[:, 0], nz[:, 1]]
            all_rows.append(nz[:, 0] + i)
            all_cols.append(nz[:, 1])
            all_vals.append(vals)

        if not all_rows:
            self.doc_matrix = None
            print("⚠️ SPLADE index: no non-zero entries found.")
            return

        rows = torch.cat(all_rows)
        cols = torch.cat(all_cols)
        vals = torch.cat(all_vals)

        self.doc_matrix = torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            vals,
            (len(texts), self.vocab_size)
        ).to_sparse_csr().to(self.device)

        print("✅ SPLADE index built. Matrix shape:", tuple(self.doc_matrix.shape))

    def score_topk(self, query: str, top_k: int = 500) -> List[Tuple[int, float]]:
        if self.doc_matrix is None:
            return []

        q_vec = self._encode_text([str(query)])[0]
        scores = torch.mv(self.doc_matrix, q_vec)

        k = min(top_k, scores.size(0))
        top_scores, top_indices = torch.topk(scores, k)
        return list(zip(top_indices.tolist(), top_scores.tolist()))


class BM25Fast:
    """
    GPU-Accelerated BM25 implemented as CSR(doc_term_bm25_weights) @ query_tf_vector.

    Design intent:
    - "Dumb" lexical recall baseline (stems, classic IR behavior)
    - Should improve recall coverage, not necessarily precision
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75, device: Optional[str] = None, min_df: int = 1):
        self.k1 = k1
        self.b = b
        self.min_df = min_df
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.doc_matrix = None
        self.vectorizer: Optional[CountVectorizer] = None
        self.pids: List[str] = []

        self.stemmer = PorterStemmer()

    def _stem_tokenize(self, text: str) -> List[str]:
        # Normalize first (same function you use for sparse), then tokenize + stem
        text = normalize_sparse_text(str(text))
        words = re.findall(r"(?u)\b\w+\b", text)
        return [self.stemmer.stem(w) for w in words]

    def build_index(self, texts: List[str], pids: List[str]):
        print(f"🏗️ Building Fast GPU BM25 Index (stemming + normalize) for {len(texts)} docs...")
        self.pids = pids

        # Custom tokenizer => token_pattern must be None
        self.vectorizer = CountVectorizer(
            tokenizer=self._stem_tokenize,
            token_pattern=None,
            lowercase=False,   # we already normalize+lowercase ourselves
            min_df=self.min_df
        )

        X = self.vectorizer.fit_transform(texts)  # CSR matrix on CPU

        n_docs = X.shape[0]
        doc_lengths = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        avgdl = float(doc_lengths.mean()) if n_docs > 0 else 0.0

        # document frequency per term
        df = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float32)

        # BM25 idf
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

        X_coo = X.tocoo()
        tf = X_coo.data.astype(np.float32)
        rows = X_coo.row
        cols = X_coo.col

        dl = doc_lengths[rows]
        numerator = tf * (self.k1 + 1.0)
        denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(avgdl, 1e-9)))
        weights = (numerator / denom) * idf[cols]

        indices = torch.from_numpy(np.vstack([rows, cols])).long()
        values = torch.from_numpy(weights).float()

        self.doc_matrix = torch.sparse_coo_tensor(
            indices,
            values,
            (n_docs, len(self.vectorizer.vocabulary_))
        ).to_sparse_csr().to(self.device)

        print("✅ BM25 index built. Matrix shape:", tuple(self.doc_matrix.shape))
        print("✅ BM25 vocab size:", len(self.vectorizer.vocabulary_))

    def search(self, query: str, top_k: int = 500) -> List[Tuple[str, float]]:
        if self.doc_matrix is None or self.vectorizer is None:
            return []

        # IMPORTANT: CountVectorizer will call our tokenizer (normalize+stem)
        q_sparse = self.vectorizer.transform([str(query)])

        # If query contains no in-vocab tokens, BM25 contributes nothing.
        if q_sparse.nnz == 0:
            return []

        q_indices = torch.from_numpy(q_sparse.indices).long().to(self.device)
        q_values = torch.from_numpy(q_sparse.data.astype(np.float32)).float().to(self.device)

        # Build query vector in vocab space (dense is fine; vocab dim is manageable)
        q_vec = torch.zeros(self.doc_matrix.shape[1], device=self.device, dtype=torch.float32)
        q_vec.index_add_(0, q_indices, q_values)

        scores = torch.mv(self.doc_matrix, q_vec)

        k = min(top_k, scores.size(0))
        top_scores, top_indices = torch.topk(scores, k)

        return [(self.pids[i], float(s)) for i, s in zip(top_indices.tolist(), top_scores.tolist())]
