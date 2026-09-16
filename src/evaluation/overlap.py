"""Retrieval overlap + complementarity analysis (per "Retrieval Overlap Analysis").

Complementarity evidence for the thesis: beyond "final metric", show *which
evidence each method uniquely surfaces*.

* Jaccard(A, B) = |A ∩ B| / |A ∪ B| over the top-K lineage_id sets per query,
  averaged across queries.
* Unique-relevant counts: for each method, how many of the *relevant*
  evidence items are retrieved by that method alone (not by any other).
* "Found only after fusion": targets in the hybrid top-K that appear in none
  of the single-method top-K sets.

Uses the SAME benchmark and SAME K as the aggregate metrics so tables are
internally consistent (the "Methodological Neutrality" requirement)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Set

from . import config as E
from .benchmark import BenchmarkItem


def jaccard(a: Set[str], b: Set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _topset(rows_by_q: Dict[str, list], query_id: str, k: int) -> Set[str]:
    rows = rows_by_q.get(query_id, [])
    return set(r["lineage_id"] for r in rows[:k])


def overlap_matrix(per_query_by_system: Dict[str, Dict[str, list]],
                   systems: Sequence[str], k: int) -> Dict[str, float]:
    """Mean Jaccard between each pair of systems at top-K, across queries.

    ``per_query_by_system[system]`` maps ``query_id -> [row, ...]`` where each
    row carries at least ``lineage_id``; rows are in rank order.
    """
    out: Dict[str, float] = {}
    for i, a in enumerate(systems):
        if a not in per_query_by_system:
            continue
        for b in systems[i + 1:]:
            if b not in per_query_by_system:
                continue
            per_q = []
            for qid in per_query_by_system[a]:
                if qid not in per_query_by_system[b]:
                    continue
                per_q.append(jaccard(
                    _topset(per_query_by_system[a], qid, k),
                    _topset(per_query_by_system[b], qid, k)))
            if per_q:
                out[f"{a}~{b}"] = round(sum(per_q) / len(per_q), 4)
    return out


def complementarity(per_query_by_system: Dict[str, Dict[str, list]],
                    items: Sequence[BenchmarkItem],
                    systems: Sequence[str], k: int) -> dict:
    """Unique relevant evidence per method + post-fusion additions (per "Retrieval Overlap Analysis")."""
    core = [s for s in ("dense", "sparse", "graph") if s in per_query_by_system]
    only_by: Dict[str, int] = {s: 0 for s in core}
    no_method_hit = 0
    found_after_fusion = 0
    fusion = "hybrid" if "hybrid" in per_query_by_system else None
    for item in items:
        target = item.target_lineage_id
        sets = {s: _topset(per_query_by_system[s], item.query_id, k)
                for s in core}
        hit_by = [s for s, st in sets.items() if target in st]
        if not hit_by:
            no_method_hit += 1
        elif len(hit_by) == 1:
            only_by[hit_by[0]] += 1
        # relevant evidence found only after hybridization ("Retrieval Overlap Analysis"):
        # in the fused system's top-k but in NO single-method top-k
        if (fusion is not None
                and target in _topset(per_query_by_system[fusion],
                                      item.query_id, k)
                and not any(target in st for st in sets.values())):
            found_after_fusion += 1
    return {
        "k": k,
        "unique_relevant_only_by": only_by,
        "queries_with_no_method_hitting_target": no_method_hit,
        "targets_found_only_after_fusion": found_after_fusion,
        "core_systems": core,
    }


def save(overlap: Dict[str, float], comp: dict,
         out_dir: Path = E.OUT_RETRIEVAL) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "overlap.json"
    p.write_text(json.dumps(
        {"jaccard": overlap, "complementarity": comp}, indent=2))
    return p


def load(path: Path = E.OUT_RETRIEVAL / "overlap.json") -> dict:
    return json.loads(path.read_text())
