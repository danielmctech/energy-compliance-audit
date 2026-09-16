#!/usr/bin/env python
"""GCG (Graph-assisted Candidate Generation) eligibility analysis (spec §20).

Primary pre-posed question: of the 60 benchmark queries, how many have the
gold target ABSENT from the A0-30 30-pool (the plain hybrid RRF pool that the
frozen A0-30 reference re-ranks) yet PRESENT in the <=1-hop regulatory
neighbourhood (CROSS_REFERENCES / AMENDS) of that pool's article nodes?  Those
are the queries a 1-hop graph candidate generator could rescue.

Offline: no LLM, no rerank.  Read-only over the frozen corpus + graph.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluation.config as E          # noqa: E402
import evaluation.benchmark as BM     # noqa: E402
from retrieval import Retriever                          # noqa: E402
from retrieval._corpus import article_node_for           # noqa: E402
from retrieval.graph import GRAPH_EDGE_KINDS             # noqa: E402


def article_node_of_chunk(r, lid):
    idx = r._by_lineage.get(lid)
    if idx is None:
        return None, None
    return (article_node_for(r.chunks[idx], r.corpus["article_index"],
                             r.corpus["graph"]), idx)


def main() -> int:
    t0 = time.time()
    DIAG = E.EVAL_RESULTS / "diagnostic"
    items = BM.build_benchmark()
    r = Retriever(dense_models=["base"],
                  dense_specs={"base": "all-MiniLM-L6-v2"})
    graph = r.corpus["graph"]            # GraphLike: .nodes .adj .adj_in .edges
    adj = graph.adj
    A0_POOL_N = 30

    rows = []
    for it in items:
        qid, q, tgt = it.query_id, it.question, it.target_lineage_id
        # A0-30 pre-rerank pool (the plain hybrid RRF, cap 30)
        pool_lids, _, _ = r._hybrid_pool(q, A0_POOL_N)
        tgt_in_pool = tgt in pool_lids

        tgt_node, tgt_idx = article_node_of_chunk(r, tgt)
        tgt_is_chunk = tgt in r._by_lineage

        # 1-hop regulatory neighbourhood of the pool seeds' article nodes
        pool_nodes = set()
        for lid in pool_lids:
            node, _i = article_node_of_chunk(r, lid)
            if node:
                pool_nodes.add(node)

        one_hop: dict = {}               # dst -> [kinds]
        for node in pool_nodes:
            for dst, kind in adj.get(node, []):
                if kind in GRAPH_EDGE_KINDS and dst != node:
                    one_hop.setdefault(dst, [])
                    if kind not in one_hop[dst]:
                        one_hop[dst].append(kind)

        # is the gold target inside that 1-hop set (as a node or a chunk)?
        rescued = False
        evidence = None
        if tgt_node is not None and tgt_node in one_hop:
            rescued, evidence = True, {"via": "node",
                                       "kinds": one_hop[tgt_node]}
        if not rescued and tgt_idx is not None and tgt in one_hop:
            rescued, evidence = True, {"via": "chunk_lid",
                                       "kinds": one_hop[tgt]}

        # 2-hop (informational only; not used by GCG-1hop)
        two_hop_only = False
        if not rescued:
            for node in pool_nodes:
                for mid, k1 in adj.get(node, []):
                    if mid in pool_nodes or k1 not in GRAPH_EDGE_KINDS:
                        continue
                    for dst, k2 in adj.get(mid, []):
                        if (k2 in GRAPH_EDGE_KINDS
                                and (dst == tgt_node or dst == tgt)):
                            two_hop_only = True
                            break
                    if two_hop_only:
                        break
                if two_hop_only:
                    break

        rows.append({
            "query_id": qid,
            "category": it.category,
            "target": tgt,
            "target_is_chunk": tgt_is_chunk,
            "target_node": tgt_node,
            "target_in_a0_30_30pool": tgt_in_pool,
            "pool_size": len(pool_lids),
            "n_pool_nodes": len(pool_nodes),
            "rescuable_by_1hop": rescued,
            "rescue_evidence": evidence,
            "two_hop_only": two_hop_only,
        })

    n = len(rows)
    absent = [x for x in rows if not x["target_in_a0_30_30pool"]]
    rescuable_absent = [x for x in absent if x["rescuable_by_1hop"]]
    rescuable_any = [x for x in rows if x["rescuable_by_1hop"]]
    out = {
        "run": "GCG-1hop-50 eligibility (spec §20 primary question)",
        "n_queries": n,
        "a0_30_pool_n": A0_POOL_N,
        "target_in_a0_30_30pool": n - len(absent),
        "target_absent_from_30pool": len(absent),
        "absent_but_1hop_rescuable_by_graph": len(rescuable_absent),
        "two_hop_only": sum(1 for x in absent if x["two_hop_only"]),
        "rescuable_incl_already_in_pool": len(rescuable_any),
        "rescuable_ids": [x["query_id"] for x in rescuable_absent],
        "rescuable_ids_any": [x["query_id"] for x in rescuable_any],
        "rows": rows,
        "graph_edge_kinds": list(GRAPH_EDGE_KINDS),
        "timing_s": round(time.time() - t0, 2),
    }
    DIAG.mkdir(parents=True, exist_ok=True)
    (DIAG / "gcg_eligibility.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"n_queries={n}")
    print(f"A0-30 30-pool: target present in {n - len(absent)}, "
          f"ABSENT in {len(absent)}")
    print(f"ABSENT from 30-pool but 1-hop rescuable: {len(rescuable_absent)}")
    print(f"2-hop-only (not 1-hop): {sum(1 for x in absent if x['two_hop_only'])}")
    print(f"1-hop rescuable (incl. already in pool): {len(rescuable_any)}")
    print(f"rescuable-absent ids: {out['rescuable_ids']}")
    fams = sorted({x['category'] for x in rescuable_absent})
    print("by family:",
          {fam: sum(1 for x in rescuable_absent if x["category"] == fam)
           for fam in fams})
    print(f"wrote {DIAG / 'gcg_eligibility.json'}  ({out['timing_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
