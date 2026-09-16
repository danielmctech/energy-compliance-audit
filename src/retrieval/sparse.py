"""Lexical BM25 retrieval (rank_bm25) over the 03b chunk corpus."""
from __future__ import annotations

from typing import Iterable, List, Optional

from ._corpus import tokenize


class SparseIndex:
    def __init__(self, texts: List[str]):
        doc_tokens = [tokenize(t) for t in texts]
        self._has_tokens = [bool(t) for t in doc_tokens]
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi(doc_tokens) if any(self._has_tokens) else None

    def search(self, query: str, k: int = 10) -> List[tuple]:
        """Return [(doc_idx, score)] sorted desc, scores >= 0 kept."""
        q = tokenize(query)
        if self.bm25 is None or not q:
            return []
        scores = self.bm25.get_scores(q)
        idx = [i for i in range(len(scores)) if scores[i] > 0]
        idx.sort(key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in idx[:k]]
