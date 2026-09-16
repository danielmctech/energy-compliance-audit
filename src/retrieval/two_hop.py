"""Deterministic within-2-hop graph expansion for GCG-2hop-50 (spec §4/§6).

Pure function -- no corpus, no LLM, no I/O.  Operates in *node* space (the
production regulatory graph's article nodes).  This keeps it unit-testable
against a synthetic adjacency exactly like ``tests/test_graph_rerank.py``.

The driver maps seeds' article nodes -> real corpus chunks and calls
:func:`expand_within_2hop`.  Expansion order and candidate order are fully
deterministic:

  1. all 1-hop real-chunk neighbours of the seeds (seeds sorted, adjacency in
     stored edge order);
  2. then all 2-hop real-chunk neighbours (same seed/edge ordering, second hop
     in stored edge order).

Every candidate carries construction-time provenance (spec §6):

    node, source, graph_distance, graph_relation(s), graph_direction,
    graph_seed, intermediate (None for 1-hop), graph_path

A 2-hop candidate through ``A --K1--> B --K2--> C`` records:

    graph_distance = 2
    graph_path     = [A, K1, B, K2, C]
    relation       = (K1, K2)
    intermediate   = B

Only candidates whose node maps to a REAL corpus chunk are returned
(``real_nodes``); arbitrary graph nodes with no retrievable evidence are
excluded (spec §4).  Seeds and already-emitted nodes are never re-emitted.
"""
from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

from retrieval.graph import GRAPH_EDGE_KINDS

Node = Hashable
EdgeKind = str


def _out(node: Node, adj: Dict[Node, List[Tuple[Node, EdgeKind]]],
         allowed: Tuple[EdgeKind, ...]):
    """Stored-order allowed-kind outgoing edges of ``node`` (deterministic)."""
    return [(dst, k) for dst, k in adj.get(node, ()) if k in allowed]


def expand_within_2hop(
    adj: Dict[Node, List[Tuple[Node, EdgeKind]]],
    seed_nodes: Sequence[Node],
    real_nodes: Iterable[Node],
    allowed: Tuple[EdgeKind, ...] = GRAPH_EDGE_KINDS,
    keep: Optional[int] = None,
) -> List[Dict]:
    """Deterministically expand ``seed_nodes`` within 2 hops.

    Parameters
    ----------
    adj : adjacency ``node -> [(dst, edge_kind)]``.
    seed_nodes : the A0 hybrid pool's article nodes (deduped internally).
    real_nodes : nodes that map to REAL corpus chunks (the only candidates kept).
    allowed : edge kinds treated as regulatory links (default CROSS_REFERENCES/AMENDS).
    keep : if set, truncate the candidate list to this many (spec §5 budget).

    Returns
    -------
    list of provenance dicts, 1-hop first then 2-hop, deterministic order.
    """
    seedset = set(seed_nodes)
    real = set(real_nodes)
    allowed = tuple(allowed)
    out: List[Dict] = []
    seen = set()

    def emit(dst: Node, dist: int, seed: Node, k_rel: str,
             k2: Optional[str], inter: Optional[Node]) -> None:
        out.append({
            "node": dst,
            "source": "graph_2hop" if dist == 2 else "graph_1hop",
            "graph_distance": dist,
            "graph_relation": [k_rel, k2] if dist == 2 else [k_rel],
            "graph_direction": "outgoing",
            "graph_seed": seed,
            "intermediate": inter,
            "graph_path": ([seed, k_rel, inter, k2, dst] if dist == 2
                          else [seed, k_rel, dst]),
        })

    # 1-hop first (seeds sorted; adjacency in stored order) -- deterministic.
    for seed in sorted(seedset):
        for dst, kind in _out(seed, adj, allowed):
            if dst in seedset or dst not in real or dst in seen:
                continue
            seen.add(dst)
            emit(dst, 1, seed, kind, None, None)

    # then 2-hop (same seed order; second hop in stored order).
    for seed in sorted(seedset):
        for mid, k1 in _out(seed, adj, allowed):
            for dst, k2 in _out(mid, adj, allowed):
                if dst in seedset or dst not in real or dst in seen:
                    continue
                seen.add(dst)
                emit(dst, 2, seed, k1, k2, mid)

    if keep is not None and len(out) > keep:
        out = out[:keep]
    return out


def apply_admission_policy(
    recs: Sequence[Dict],
    policy: str = "first",
    keep: Optional[int] = None,
) -> List[Dict]:
    """Pure, deterministic application of a 20-slot admission ``policy`` (spec §5).

    ``recs`` are the provenance dicts of :func:`expand_within_2hop` (1-hop first,
    then 2-hop).  Supported ``policy`` values:

    * ``"first"`` (default, = the as-specified §5 order): the combined d1-then-d2
      list truncated to the first ``keep``.
    * ``"2hop_first"`` (distance-priority): same candidate set, ``keep`` taken
      preferring the distance-2 candidates (d2 then d1).
    * ``"split_N_M"`` (distance-quota): ``N`` slots to distance-1, ``M`` to
      distance-2 (with ``N + M == keep``).  ``d1[:N] + d2[:M]`` in stable
      stored order.  This is the only policy that preserves BOTH one-hop and
      two-hop golds simultaneously on this benchmark: a *total* re-order would
      trade one class off the other (1-hop golds sit at d1-rank 3–12, 2-hop
      golds at d2-rank 1–9; the minimal feasible pair is 12/8 or 11/9).

    Only the *order* differs between policies -- the candidate set, the
    real-chunk filter and every provenance field are untouched.  Pure; unit-
    testable without a corpus or an LLM.
    """
    if policy == "first":
        ordered = list(recs)
    elif policy == "2hop_first":
        ordered = sorted(recs, key=lambda x: -int(x.get("graph_distance", 1)))
    elif isinstance(policy, str) and policy.startswith("split_"):
        try:
            _, a, b = policy.split("_")
            n1, n2 = int(a), int(b)
        except (ValueError, AttributeError):
            raise ValueError(
                f"invalid split policy {policy!r} -- expected 'split_N_M'")
        d1 = [r for r in recs if int(r.get("graph_distance", 1)) == 1]
        d2 = [r for r in recs if int(r.get("graph_distance", 1)) == 2]
        ordered = d1[:n1] + d2[:n2]
        if keep is not None:
            ordered = ordered[:keep]
        return ordered
    else:
        raise ValueError(f"unknown admission policy {policy!r}")
    if keep is not None and len(ordered) > keep:
        return ordered[:keep]
    return ordered


def two_hop_only(
    adj: Dict[Node, List[Tuple[Node, EdgeKind]]],
    seed_nodes: Sequence[Node],
    target_node: Node,
    allowed: Tuple[EdgeKind, ...] = GRAPH_EDGE_KINDS,
) -> bool:
    """True iff ``target_node`` is reachable from any seed in 2 hops but NOT in
    1 hop (spec §7 "2-hop-only" verification against the actual production
    graph).  Pure; used by tests and the reachability pre-check."""
    allow = tuple(allowed)
    seedset = set(seed_nodes)
    h1 = set()
    for s in seedset:
        for dst, k in _out(s, adj, allow):
            h1.add(dst)
    if target_node in h1 or target_node in seedset:
        return False
    for s in seedset:
        for mid, k1 in _out(s, adj, allow):
            for dst, k2 in _out(mid, adj, allow):
                if dst == target_node:
                    return True
    return False
