"""A0-30-full: A0 reranker with a 30-candidate model input.

Why this script exists (and is distinct from run_a0_30.py):

  * The graph-rerank modes (hybrid_graph_rerank / hybrid_gr_1hop /
    hybrid_gr_relations) build a 30-candidate hybrid pool and call
    ``rr.rerank(query, pool, max_candidates=len(pool))`` -- the LLM ranks all
    30.
  * A0 (`hybrid_rerank`) builds a max(k,5)=20 pool and calls
    ``rr.rerank(query, pool)`` -- the LLM ranks all 20 (cap not the binding
    factor; the pool itself is 20).
  * run_a0_30.py (the earlier `A0-30` attempt) builds a 30-pool but still
    calls ``rr.rerank`` without ``max_candidates`` -- the reranker's
    MAX_CANDIDATES=20 cap takes over, so the model only ranks 20 of the 30
    (candidates 21-30 stay in RRF order at the tail).  That is *not* "A0 with
    the same model input as the graph modes", so "same pool, only graph
    evidence differs" still isn't a clean comparison.

This script does it for real:
  1. Same pool construction as A0 (sparse + dense RRF of top-30)
  2. Same candidate dict assembly (_gr_candidates, graph_cfg=None)
  3. ``rr.rerank(query, pool, max_candidates=len(pool))`` -- the LLM ranks
     ALL 30 candidates
  4. Result assembled via the same helpers the public A0 path uses
     (_from_top, graph-boost, ...), so the only delta vs A0-20 is the
     reranker's model input size (30 vs 20) and the (distinct, see
     src/reranking/__init__.py:119-126) cache key for the new call.

Cache-key note: the reranker's bare-branch hash was (model, temp, query,
ids, n) -- two calls with different explicit ``max_candidates`` but the same
ids/n would collide.  ``src/reranking/__init__.py:119-126`` now includes
``(mc, max_candidates)`` when one is given, so this call has its own cache
entry and does NOT inherit the 20-cap result cached by run_a0_30.py.

Usage:
    PYENV_VERSION=energy-audit ENERGY_AUDIT_ROOT=$(pwd) \
        python scripts/run_a0_30_clean.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(os.getenv("ENERGY_AUDIT_ROOT") or ".").resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("ENERGY_AUDIT_ROOT", str(ROOT))

from evaluation import config as E       # noqa: E402
from evaluation import benchmark as BM   # noqa: E402
from evaluation import metrics           # noqa: E402
from retrieval.fusion import GRAPH_BOOST  # noqa: E402


POOL_N = 30


def main() -> None:
    t0 = time.time()
    items = BM.build_benchmark()
    print(f"[a0-30-full] n_items={len(items)}")

    from retrieval import Retriever

    r = Retriever(dense_models=["base"],
                  dense_specs={"base": "all-MiniLM-L6-v2"})
    rr = r._reranker.get()

    k_set = E.EvalConfig().k_values
    k_max = max(k_set)

    rows = []
    for qno, item in enumerate(items, start=1):
        t_q = time.time()
        # 1-3: identical pool + candidate assembly to A0, EXCEPT no
        # graph_cfg (baseline) and pool_n=30.
        top_lids, fused, rank_lists = r._hybrid_pool(item.question, POOL_N)
        neigh = [g for g in r.search_graph(item.question, 30)
                 if g[1] >= 1 and g[0] in set(top_lids)]
        for g in neigh:
            fused[g[0]]["score"] += GRAPH_BOOST * (1.0 if g[1] == 1 else 0.5)
        candidates, _ = r._gr_candidates(
            item.question, top_lids, graph_cfg=None, neigh=neigh, fused=fused)
        pool = list(candidates)
        if not pool:
            out_list = r._from_top(top_lids, fused, rank_lists, k_max,
                                   graph_edges=neigh)
        else:
            # The clean step: send ALL 30 to the model.
            result = rr.rerank(item.question, pool,
                               max_candidates=len(pool))
            if {x for x in result["order"]} != set(top_lids):
                new_order = list(top_lids)
                result = {**result, "status": "fallback",
                          "note": "permutation-mismatch"}
            else:
                new_order = list(result["order"])
            out_list = r._from_top(new_order, fused, rank_lists, k_max,
                                   graph_edges=neigh, rerank_meta=result)

        lids = [x.lineage_id for x in out_list]
        tgt = item.target_lineage_id
        m = {}
        for k in k_set:
            rl = lids[:k]
            m[str(k)] = {
                "system": "hybrid_rerank_30_full", "query_id": item.query_id,
                "k": k,
                "recall": metrics.recall_at_k(rl, [tgt], k),
                "precision": metrics.precision_at_k(rl, [tgt], k),
                "hit": metrics.hit_rate_at_k(rl, [tgt], k),
                "mrr": metrics.reciprocal_rank(rl, [tgt]),
                "ndcg": metrics.ndcg_at_k(rl, [tgt], k),
            }
        first = lids.index(tgt) + 1 if tgt in lids else None
        rows.append({
            "query_id": item.query_id,
            "question": item.question,
            "system": "hybrid_rerank_30_full",
            "category": item.category,
            "target": tgt,
            "retrieved_top30": lids,
            "retrieved_top20": lids[:20],
            "scores": [round(x.score, 6) for x in out_list[:30]],
            "methods": [list(x.source_methods) for x in out_list[:30]],
            "target_rank": first,
            "metrics": m,
            "retrieval_ms": round((time.time() - t_q) * 1000, 1),
        })
        if qno % 10 == 0:
            print(f"[a0-30-full]   {qno}/{len(items)} done "
                  f"({time.time()-t_q:5.0f}s)")

    out_dir = E.OUT_PER_QUERY
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "retrieval_hybrid_rerank_30_full.jsonl"
    with open(p, "w") as f:
        for r_ in rows:
            f.write(json.dumps(r_, ensure_ascii=False) + "\n")
    print(f"[a0-30-full] wrote {p}  rows={len(rows)}  "
          f"elapsed={time.time()-t0:.1f}s")

    in_top5 = sum(1 for r_ in rows if r_["target_rank"] is not None
                  and r_["target_rank"] <= 5)
    in_top20 = sum(1 for r_ in rows if r_["target_rank"] is not None
                   and r_["target_rank"] <= 20)
    print(f"[a0-30-full]   target in top-5: {in_top5}/{len(rows)}   "
          f"in top-20: {in_top20}/{len(rows)}")


if __name__ == "__main__":
    main()
