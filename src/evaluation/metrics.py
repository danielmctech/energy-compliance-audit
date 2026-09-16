"""Information-retrieval + answer metrics with documented edge cases.

The Retrieval Metrics family (Recall@K, Precision@K, Hit Rate / Success@K, MRR, nDCG) and Answer-Level Metrics (EM / F1).

Conventions (per "Retrieval Metrics": "Be careful with cases where a query
has zero relevant items"):

* ``relevant_ids`` may be empty.
* ``precision_at_k`` divides by ``min(k, len(retrieved_ids))`` and returns
  ``None`` (not 0.0) when fewer than k items were retrieved, so callers can
  report "insufficient retrievals" rather than silently deflating the score
  (the "Precision@K" requirement).
* ``recall_at_k`` returns ``None`` when a query has zero relevant items, so
  an "unanswerable from the benchmark" query does not count as 0.0.
* ``hit_rate_at_k`` returns 0.0 when there are zero relevant items (there is
  nothing to hit), which is the natural reading of the piecewise definition
  in the "Hit Rate / Success@K" definition.
* ``reciprocal_rank`` is 0.0 when the first rank exceeds 0 with no relevant
  result (per "Mean Reciprocal Rank").
* ``ndcg_at_k`` uses binary relevance (relevance in {0,1}) because the
  benchmark carries no graded judgments (per "nDCG@K": "do not invent graded
  relevance scores").  With binary relevance the ideal DCG places all
  relevant items at the top.
* Duplicates in ``retrieved_ids`` are de-duplicated on first occurrence
  (per "Unit Tests": duplicate retrieved IDs); the first position wins.
"""
from __future__ import annotations

import math
import re
from typing import Iterable, List, Optional, Sequence, Tuple


def _dedupe(ids: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for i in ids:
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def recall_at_k(retrieved_ids: Sequence[str],
                relevant_ids: Iterable[str], k: int) -> Optional[float]:
    """Recall@K = |relevant ∩ topK| / |relevant|.

    Returns ``None`` when ``relevant`` is empty (undefined, not zero) so the
    caller can exclude it from the mean rather than biasing it down.
    """
    rel = set(relevant_ids)
    if not rel:
        return None
    topk = _dedupe(retrieved_ids)[:k]
    hits = len(rel & set(topk))
    return hits / len(rel)


def precision_at_k(retrieved_ids: Sequence[str],
                   relevant_ids: Iterable[str], k: int) -> Optional[float]:
    """Precision@K = |relevant ∩ topK| / min(K, retrieved).

    Returns ``None`` when fewer than K results were retrieved (per "Precision@K": do
    not blindly divide by K) and ``None`` when there are zero relevant
    items.  Otherwise divides by ``min(k, len(topk))`` so that a 5-result
    list evaluated at K=10 is scored on the 5 it actually returned.
    """
    rel = set(relevant_ids)
    if not rel:
        return None
    topk = _dedupe(retrieved_ids)[:k]
    if not topk:
        return None
    denom = min(k, len(topk))
    hits = len(rel & set(topk))
    return hits / denom


def hit_rate_at_k(retrieved_ids: Sequence[str],
                  relevant_ids: Iterable[str], k: int) -> float:
    """Hit@K: 1.0 if at least one relevant item is in topK, else 0.0.

    Returns 0.0 when there are no relevant items (per "Hit Rate / Success@K"; nothing to hit).
    """
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    topk = _dedupe(retrieved_ids)[:k]
    return 1.0 if rel & set(topk) else 0.0


def reciprocal_rank(retrieved_ids: Sequence[str],
                    relevant_ids: Iterable[str]) -> float:
    """RR = 1 / rank of the first relevant item; 0.0 when none in the list."""
    rel = set(relevant_ids)
    for rank, lid in enumerate(_dedupe(retrieved_ids), start=1):
        if lid in rel:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(retrieved_ids: Sequence[str],
                         relevant_ids: Iterable[str]) -> float:
    """Per-query MRR.  0.0 when no relevant result is retrieved (per "Mean Reciprocal Rank")."""
    return reciprocal_rank(retrieved_ids, relevant_ids)


def dcg(relevances: Sequence[float], k: Optional[int] = None) -> float:
    """Discounted cumulative gain at rank (1-indexed).

    DCG@K = sum_{i=1..K} rel_i / log2(i + 1).
    """
    if k is not None:
        relevances = relevances[:k]
    total = 0.0
    for i, rel in enumerate(relevances, start=1):
        if rel > 0:
            total += rel / math.log2(i + 1)
    return total


def ndcg_at_k(retrieved_ids: Sequence[str],
              relevant_ids: Iterable[str],
              k: int) -> float:
    """nDCG@K under binary relevance (see module docstring for why binary).

    ``None``-safe: returns 0.0 when there are no relevant items (IDCG=0),
    because with nothing to rank against, the retrieved list is neither
    better nor worse -- 0.0 is the neutral, documented value and it matches
    the per-query mean behaviour (a query with no relevant items contributes
    0, same as a recall=None query contributing nothing).
    """
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    topk = _dedupe(retrieved_ids)[:k]
    relevances = [1.0 if lid in rel else 0.0 for lid in topk]
    # ideal case: |rel| relevant items all packed at the top, capped by how
    # many of the top-k window we actually retrieved (or len(rel) if the list
    # is shorter).
    ideal_n = min(len(rel), max(len(topk), 1))
    idcg = dcg([1.0] * ideal_n, k=k)
    if idcg == 0.0:
        return 0.0
    return dcg(relevances, k=k) / idcg


def ndcg_per_query(retrieved_by_query: Sequence[Sequence[str]],
                   relevant_by_query: Sequence[Iterable[str]],
                   k: int) -> Tuple[Optional[float], int]:
    """Mean nDCG@K across queries.  Returns ``(mean, n_counted)``.

    Queries with zero relevant items are excluded and counted separately so
    the caller can report coverage (the "Retrieval Metrics" requirement).
    """
    vals = []
    skipped = 0
    for retrieved, rel in zip(retrieved_by_query, relevant_by_query):
        rel_set = set(rel)
        if not rel_set:
            skipped += 1
            continue
        vals.append(ndcg_at_k(retrieved, rel_set, k))
    if not vals:
        return None, skipped
    return sum(vals) / len(vals), skipped


def recall_curve(retrieved_ids: Sequence[str],
                 relevant_ids: Iterable[str],
                 k_values: Sequence[int]
                 ) -> List[Tuple[int, Optional[float], Optional[float], float]]:
    """Return ``[(k, recall, precision, hit), ...]`` for each k.

    ``recall``/``precision`` may be ``None`` (see module edge-case notes).
    """
    rel = set(relevant_ids)
    out = []
    uniq = _dedupe(retrieved_ids)
    for k in k_values:
        topk = uniq[:k]
        r = (len(rel & set(topk)) / len(rel)) if rel else None
        p = (len(rel & set(topk)) / min(k, len(topk))) if (rel and topk) else None
        out.append((k, r, p, hit_rate_at_k(uniq, rel, k)))
    return out


def exact_match(prediction: str, reference: str) -> float:
    """Case-insensitive, whitespace/punctuation-normalised exact match."""
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
    return 1.0 if norm(prediction) == norm(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1.  0.0 when either side has no tokens."""
    def toks(s: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", (s or "").lower())
    p, r = toks(prediction), toks(reference)
    if not p or not r:
        return 0.0
    common = set(p) & set(r)
    if not common:
        return 0.0
    prec = len(common) / len(set(p))
    rec = len(common) / len(set(r))
    return 2 * prec * rec / (prec + rec)
