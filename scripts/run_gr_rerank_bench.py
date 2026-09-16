"""Scoped benchmark driver: run the graph-aware re-ranker v2 systems + the
A0 baseline (``hybrid_rerank``) over the shared 60-query benchmark, persist
per-query logs, and write a system-vs-system comparison report.

Usage:
    PYENV_VERSION=energy-audit ENERGY_AUDIT_ROOT=$(pwd) \
        python scripts/run_gr_rerank_bench.py

The run is scoped to the new systems so that:
  * the dense model loads once,
  * the LLM cache serves the A0 baseline and any non-context re-ranks
    (prompt-hash keyed),
  * and the report is limited to the ablation matrix (A0 vs A2/A4/A5+A6/A7)
    instead of the full ``EXPERIMENTS`` set (which includes finetuned
    variants, Neo4j, etc. -- out of scope for this comparison).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(os.getenv("ENERGY_AUDIT_ROOT") or ".").resolve()
sys.path.insert(0, str(ROOT / "src"))
os.environ["ENERGY_AUDIT_ROOT"] = str(ROOT)

from evaluation import config as E       # noqa: E402
from evaluation import benchmark as BM   # noqa: E402
from evaluation import retrieval_run as RR  # noqa: E402
from evaluation import compare as C      # noqa: E402


SYSTEMS = [
    "hybrid_rerank",          # A0 baseline
    "hybrid_graph_rerank",    # A3 + A5 + A6 (main system)
    "hybrid_gr_1hop",         # A2
    "hybrid_gr_relations",    # A4
    "hybrid_gr_expand",       # A7
]


def main() -> None:
    print(f"[bench] ROOT={ROOT}")
    # Load the already-built benchmark (avoids re-deriving / the missing
    # pairs_stage1.jsonl; the historical 60-query set is on disk).
    bench_path = E.OUT_PER_QUERY / "benchmark.jsonl"
    print(f"[bench] loading benchmark from {bench_path}")
    items = BM.load(bench_path)
    if not items:
        raise SystemExit(f"[bench] no items loaded from {bench_path}")
    print(f"[bench] n_items={len(items)}")

    cfg = E.EvalConfig()
    print(f"[bench] gr_enable={cfg.gr_enable} gr_2hop={cfg.gr_2hop} "
          f"gr_snippets={cfg.gr_snippets} gr_query_aware={cfg.gr_query_aware} "
          f"gr_max_edges={cfg.gr_max_edges} gr_max_paths={cfg.gr_max_paths} "
          f"gr_token_limit={cfg.gr_token_limit} "
          f"gr_candidate_k={cfg.gr_candidate_k}")

    t0 = time.time()
    results = RR.run_retrieval(items, systems=SYSTEMS)
    dt = time.time() - t0
    for s, rows in results.items():
        failed = "SKIPPED" if not rows else (f"{len(rows)} rows")
        print(f"[bench]   {s}: {failed}")
    print(f"[bench] aggregate done in {dt:.1f}s")

    # ---- comparison report -----------------------------------------------
    t0 = time.time()
    comparisons = []
    per_query_cls = []
    for b in SYSTEMS[1:]:
        comparisons.extend(C.compare("hybrid_rerank", b))
        # spec §14: per-query IMPROVED / UNCHANGED / DEGRADED rows
        per_query_cls.extend(C.classify_per_query(
            "hybrid_rerank", b, k=10))
    out_md = C.save_report(comparisons, per_query_cls=per_query_cls)
    print(f"[bench] report -> {out_md}")
    print(f"[bench] comparison done in {time.time() - t0:.1f}s")
    print("\n".join(["=" * 60, open(out_md).read(), "=" * 60]))


if __name__ == "__main__":
    main()
