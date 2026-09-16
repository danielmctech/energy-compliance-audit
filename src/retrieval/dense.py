"""Dense retrieval: sentence-transformers embeddings + FAISS cosine index.

Model loading is lazy and cached, so the index builds instantly even when
only sparse/graph modes are used. Finetuned variants plug in by passing a
different ``model_name`` (local path or HF id).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional

_MODEL_CACHE: dict = {}
_LOCK = threading.Lock()


def load_model(model_name: str):
    with _LOCK:
        if model_name not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer

            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        return _MODEL_CACHE[model_name]


def encode(model_name: str, texts: List[str], batch_size: int = 128):
    import numpy as np
    import torch

    model = load_model(model_name)
    with torch.no_grad():
        vecs = model.encode(texts, batch_size=batch_size,
                            show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")


def _index(vecs):
    import faiss
    import numpy as np

    iv = faiss.IndexFlatIP(vecs.shape[1])
    iv.add(np.asarray(vecs, dtype="float32"))
    return iv


class DenseIndex:
    def __init__(self, texts: List[str], model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.vecs = encode(model_name, texts)
        self.index = _index(self.vecs)
        self.dim = int(self.vecs.shape[1])

    def search(self, query: str, k: int = 10) -> List[tuple]:
        import numpy as np

        qv = encode(self.model_name, [query])
        scores, idx = self.index.search(qv, min(k, len(self.vecs)))
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0])]
