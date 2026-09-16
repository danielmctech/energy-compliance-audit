"""Graph retrieval: 1-2 hop expansion over CROSS_REFERENCES / AMENDS edges."""
from __future__ import annotations

from typing import List

# only these edge kinds carry regulatory linkage for retrieval
GRAPH_EDGE_KINDS = ("CROSS_REFERENCES", "AMENDS")
HOP_WEIGHT = {1: 1.0, 2: 0.5}


class GraphRetriever:
    def __init__(self, graph):
        self.nodes = graph.nodes
        self.adj = graph.adj

    def _outgoing(self, src: str) -> List[tuple]:
        return [(d, k) for d, k in self.adj.get(src, [])
                if k in GRAPH_EDGE_KINDS]

    def expand(self, seed_ids: List[str], max_hops: int = 2) -> dict:
        """Return {target_lineage_id: (best_hop, [edge_kinds])} for nodes
        reachable from any seed within max_hops on graph-kind edges.
        Seeds are excluded from the result."""
        seeds = {s for s in dict.fromkeys(seed_ids) if s in self.nodes}
        seen: dict = {}
        frontier = sorted(seeds)
        for hop in range(1, max_hops + 1):
            nxt = {}
            for src in frontier:
                for dst, kind in self._outgoing(src):
                    if dst in seeds or dst in seen:
                        continue
                    rec = nxt.setdefault(dst, (hop, []))
                    rec[1].append(kind)
            for dst, (h, kinds) in nxt.items():
                seen[dst] = (h, sorted(set(kinds)))
            if not nxt:
                break
            frontier = list(nxt)
        return seen

    def search(self, seed_ids: List[str], max_hops: int = 2) -> List[tuple]:
        """Return [(target_lineage_id, score, hop, edge_kinds)] sorted."""
        out = [
            (dst, HOP_WEIGHT.get(hop, 0.25), hop, kinds)
            for dst, (hop, kinds) in self.expand(seed_ids, max_hops).items()
        ]
        out.sort(key=lambda r: (-r[1], r[0]))
        return out
