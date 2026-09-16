"""Clean causal experiment: A0-30 vs A0-20 on the same 60 queries.

Why: A0 (`hybrid_rerank`) currently caps candidates at 20, and the public
`retrieve(mode="hybrid_rerank", pool_n=...)` API ignores pool_n
(retrieval/__init__.py:182-186). The graph-rerank modes (hybrid_graph_rerank,
hybrid_gr_1hop, hybrid_gr_relations) *do* pass pool_n=30 (retrieval_run.py:98)
and override the reranker's 20-cap with `max_candidates=len(pool)`
(retrieval/__init__.py:367-370). So "same pool, only graph changes" was never
actually true in the original benchmark; A0 was structurally constrained.

This script runs A0 with the same 30-candidate pool the graph modes use,
so that A0-30 vs Graph-30/1hop-30/Relations-30 is a clean causal comparison:
same query, same sparse+dense hybrid retrieval, same 30-candidate pool,
same answer generator (5-chunk window), same judge, only graph context in
the reranker prompt differs.

We call `_hybrid_rerank(query, k, pool_n=30)` directly (bypassing the public
`retrieve` A0 branch) — this is exactly what the graph-rerank mode does,
minus the `graph_cfg` evidence. The reranker's own 20-cap still applies to
the LLM call, but the top-20 pool drawn is RRF's best 20 of 30 (rather than
20 of 20), and a rank-21+ chunk could still surface if the RRF scores put
it there. That is the honest "A0-30" — same mechanism, just a deeper pool.

Usage:
    PYENV_VERSION=energy-audit ENERGY_AUDIT_ROOT=$(pwd) \
        python scripts/run_a0_30.py
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

from evaluation import config as E     # noqa: E402
from evaluation import benchmark as BM  # noqa: E402
from evaluation import metrics         # noqa: E402


def main() -> None:
    t0 = time.time()
    items = BM.build_benchmark()
    print(f"[a0-30] n_items={len(items)}")

    # Build one Retriever (dense model loads once).
    from retrieval import Retriever
    r = Retriever(dense_models=["base"],
                  dense_specs={"base": "all-MiniLM-L6-v2"})

    k_set = E.EvalConfig().k_values  # (1, 3, 5, 10, 20)
    k_max = max(k_set)

    rows = []
    for qno, item in enumerate(items, start=1):
        t_q = time.time()
        # Call the private rerank method directly, bypassing the A0 branch
        # of the public retrieve() which ignores pool_n.
        out = r._hybrid_rerank(item.question, k=k_max, pool_n=30)
        lids = [x.lineage_id for x in out]
        tgt = item.target_lineage_id
        m = {}
        for k in k_set:
            rl = lids[:k]
            m[str(k)] = {
                "system": "hybrid_rerank_30", "query_id": item.query_id,
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
            "system": "hybrid_rerank_30",
            "category": item.category,
            "target": tgt,
            "retrieved_top20": lids[:20],
            "scores": [round(x.score, 6) for x in out[:20]],
            "methods": [list(x.source_methods) for x in out[:20]],
            "target_rank": first,
            "metrics": m,
            "retrieval_ms": round((time.time() - t_q) * 1000, 1),
        })
        if qno % 10 == 0:
            print(f"[a0-30]   {qno}/{len(items)} done ({time.time()-t_q:5.0f}s)")

    # save
    out_dir = E.OUT_PER_QUERY
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "retrieval_hybrid_rerank_30.jsonl"
    with open(p, "w") as f:
        for r_ in rows:
            f.write(json.dumps(r_, ensure_ascii=False) + "\n")
    print(f"[a0-30] wrote {p}  rows={len(rows)}  elapsed={time.time()-t0:.1f}s")

    # quick sanity: target rank distribution
    in_top5 = sum(1 for r_ in rows if r_["target_rank"] is not None and r_["target_rank"] <= 5)
    in_top20 = sum(1 for r_ in rows if r_["target_rank"] is not None)
    print(f"[a0-30]   target in top-5: {in_top5}/{len(rows)}  in top-20: {in_top20}/{len(rows)}")


if __name__ == "__main__":
    main()
