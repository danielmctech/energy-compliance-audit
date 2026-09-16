"""Shared cached semantic-similarity embedder (bge-m3).

Documented per "Answer-Level Metrics": model=bge-m3, similarity=cosine,
normalization=L2.  One SentenceTransformer instance loaded once and cached so
the answer-similarity loops are fast and deterministic.
"""
from __future__ import annotations

import threading
from typing import Optional, Tuple

from sentence_transformers import SentenceTransformer

_EMB: Optional[Tuple[str, SentenceTransformer]] = None
_LOCK = threading.Lock()
DEFAULT_MODEL = "BAAI/bge-m3"


def embedder(model: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Return a cached SentenceTransformer (reloaded only if model differs)."""
    global _EMB
    with _LOCK:
        if _EMB is None or _EMB[0] != model:
            import warnings
            warnings.filterwarnings("ignore")
            _EMB = (model, SentenceTransformer(model))
    return _EMB[1]


def cosine(a: str, b: str, model: str = DEFAULT_MODEL) -> Optional[float]:
    """L2-normalised cosine similarity; ``None`` if either text is empty."""
    if not (a and b):
        return None
    try:
        e = embedder(model)
        va = e.encode([a], normalize_embeddings=True)[0]
        vb = e.encode([b], normalize_embeddings=True)[0]
        import numpy as np
        return float(np.dot(va, vb))
    except Exception:
        return None


__all__ = ["DEFAULT_MODEL", "cosine", "embedder"]
