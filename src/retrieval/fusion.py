"""Reciprocal-Rank Fusion over method results + graph-boosted hybrid mode."""
from __future__ import annotations

from typing import List

RRF_K = 60
GRAPH_BOOST = 0.05  # score added to graph-reachable neighbors in hybrid


def rrf_fuse(rank_lists: List[List[tuple]], k: int = RRF_K) -> dict:
    """Fuse ranked [(lineage_id, raw_score)] lists via reciprocal rank.

    Returns {lineage_id: {"score": float, "methods": [name, ...],
    "ranks": {name: rank}}} -- methods are the list labels given by the
    caller via rank_lists (list of (label, ranked) pairs).
    """
    fused: dict = {}
    for label, ranked in rank_lists:
        for rank0, item in enumerate(ranked):
            lid = item[0]
            score = 1.0 / (k + rank0 + 1)
            rec = fused.setdefault(lid, {"score": 0.0, "methods": [],
                                        "ranks": {}})
            rec["score"] += score
            rec["methods"].append(label)
            rec["ranks"][label] = rank0 + 1
    return fused
