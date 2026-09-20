"""Re-run retrieval over the fresh 8-family benchmark for ALL EXPERIMENTS,
then re-render every aggregate-dependent table (A, comparison) so
real values -- never n/a -- are available. Table B is re-rendered from the
already-fresh answer_aggregate.csv for internal consistency.

Why:
  * The on-disk retrieval_*.jsonl + retrieval_aggregate.{csv,json} +
    table_*.csv + tables.md were produced before the 8-family benchmark
    landed, so Table A / Comparison show n/a (dense/sparse/hybrid rows
    missing from the aggregate).
  * The answer/judge files were already refreshed (Sep 5) on the 8-family
    set and MUST NOT be re-judged -- Table B's source stays unchanged.

Plan:
  1. build_benchmark() -> 60 fresh items (deterministic, verified on disk).
  2. save benchmark.jsonl (the stale one was backed up by the caller).
  3. run_retrieval(items) ONCE over every EXPERIMENTS system -- this is the
     correct call shape for the aggregate (one write), with graceful per-
     system skip (recorded in skipped_systems.json, never abort, never
     fabricate). The dense model loads once; the LLM re-rank cache serves
     repeated (query, pool) prompts.
  4. re-render Table A + Comparison from the fresh aggregate.
  5. re-render Table B from answer_aggregate.csv (unchanged source).
  6. write table_{A,B,comparison}.csv + tables.md (overwrite stale).

Usage:
    PYENV_VERSION=energy-audit \
    ENERGY_AUDIT_ROOT=$(pwd) \
    python scripts/rerun_retrieval_and_tables.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.getenv("ENERGY_AUDIT_ROOT") or ".").resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("ENERGY_AUDIT_ROOT", str(ROOT))

from evaluation import config as E        # noqa: E402
from evaluation import benchmark as BM    # noqa: E402
from evaluation import retrieval_run as RR  # noqa: E402
from evaluation import tables as TB       # noqa: E402


def line(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    t0 = time.time()
    line(f"[setup] ROOT={ROOT}")
    line(f"[setup] EVAL_RESULTS={E.EVAL_RESULTS}")

    # ---- 1. build the fresh 8-family benchmark ------------------------------
    items = BM.build_benchmark()
    s = BM.summary(items)
    line(f"[bench] n_items={len(items)} categories={s['category_counts']}")
    line(f"[bench] hop_counts={s['hop_count_counts']}")

    # ---- 2. persist the benchmark (backup already made by caller) -----------
    bm_path = E.OUT_PER_QUERY / "benchmark.jsonl"
    BM.save(items, bm_path)
    line(f"[bench] saved -> {bm_path}")

    # ---- 3. retrieval over ALL EXPERIMENTS (single correct call) ------------
    t1 = time.time()
    line(f"[retrieval] running {len(E.EXPERIMENTS)} systems over {len(items)} queries ...")
    RR.run_retrieval(items)
    line(f"[retrieval] done in {time.time()-t1:.1f}s")

    skip_path = E.OUT_PER_QUERY / "skipped_systems.json"
    if skip_path.exists():
        line(f"[retrieval] SKIPPED: {json.loads(skip_path.read_text())}")

    # per-system row counts
    for sname in E.EXPERIMENTS:
        f = E.OUT_PER_QUERY / f"retrieval_{sname}.jsonl"
        if f.exists():
            n = sum(1 for _ in open(f))
            line(f"[retrieval]   {sname:<20} {n} rows")
        else:
            line(f"[retrieval]   {sname:<20} (not run)")

    # ---- 4. fresh aggregate -------------------------------------------------
    agg = TB.load_aggregate()
    line(f"[agg] rows={len(agg)}  systems={sorted(set(r['system'] for r in agg))}")

    # ---- 5. render tables ---------------------------------------------------
    tA = TB.table_a(agg)
    tComp = TB.comparison(agg, k=10)
    ans_agg = TB.load_answer_aggregate()
    tB = TB.table_b(ans_agg)
    line(f"[table] A={len(tA)}  B={len(tB)}  comparison={len(tComp)}")

    # ---- 6. write CSVs + tables.md ------------------------------------------
    written = TB.write({"A": tA, "B": tB, "comparison": tComp})
    for name, p in written.items():
        line(f"[write] table {name} -> {p}")
    line(f"[write] markdown -> {E.OUT_AGGREGATE / 'tables.md'}")

    line(f"[done] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
