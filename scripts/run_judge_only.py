"""Run Track A: doc's 5 primary systems on the existing 60 benchmark queries.

Per "Run answer generation on systems.md" the primary answer-scoring set is:

    #3 hybrid                 (retrieval baseline)
    #7 hybrid_rerank (A0)     (re-ranker without graph context)
    #8 hybrid_graph_rerank    (main graph-conditioned hypothesis)
    #9 hybrid_gr_1hop         (ablation: 1-hop context only)
    #10 hybrid_gr_relations   (ablation: typed/directional relations)

All 5 systems' per-query rows are already on disk (notebooks/data/
evaluation/per_query/retrieval_{system}.jsonl, 60 rows each).

Generation: only `#3 hybrid` and its 4 ablations are new (the qwen cache is
cold for these); `hybrid` is already cached.

Judge: gpt-oss:latest primary + nemotron-3-nano:30b escalation. Each row
takes ~30-90 s depending on whether the escalator fires.  ~2h-4h total,
expected to write:

  - notebooks/data/evaluation/generation/answers_{system}.jsonl
  - notebooks/data/evaluation/aggregate/answer_aggregate.csv
  - notebooks/data/evaluation/aggregate/tables.md (Table B rendered)

The legacy sparse/dense/hybrid_graph/neo4j answer rows (pre-refactor)
remain untouched on disk for the 04_Evaluation_and_Results.md appendix.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import evaluation.config as E
import evaluation.benchmark as BM
import evaluation.generation as GEN
import evaluation.answer_metrics as AM
import evaluation.tables as TB


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    cfg = E.EvalConfig()
    log(f"repo={ROOT}")
    log(f"generation={list(E.GENERATION_SYSTEMS)}")
    log(f"judge primary={cfg.judge_model}  "
        f"escalator={cfg.judge_escalator}  "
        f"threshold={cfg.judge_confidence_threshold}  "
        f"escalate_scores={list(cfg.judge_escalate_scores)}")

    log("building benchmark (60 items) ...")
    items = BM.build_benchmark()
    log(f"n_items={len(items)}")

    gen_systems = list(E.GENERATION_SYSTEMS)

    retrieval: dict = {}
    for s in gen_systems:
        f = E.OUT_PER_QUERY / f"retrieval_{s}.jsonl"
        if not f.exists():
            log(f"!! missing per-query retrieval for {s}: {f}")
            return 1
        rows = [json.loads(line) for line in f.read_text().splitlines()
                if line.strip()]
        if len(rows) != len(items):
            log(f"!! {s}: expected {len(items)} rows, got {len(rows)}")
        retrieval[s] = rows
        log(f"loaded {s}: {len(rows)} rows")

    log("GEN.run_generation ...")
    t1 = time.time()
    gen = GEN.run_generation(items, retrieval,
                             systems=gen_systems, out_dir=E.OUT_GENERATION)
    log(f"generation done in {time.time() - t1:.1f}s  "
        + ", ".join(f"{s}={len(gen.get(s, []))}" for s in gen_systems))

    log("AM.run (judge) ...")
    t2 = time.time()
    all_scored: dict = {}
    for s in gen_systems:
        t_s = time.time()
        rows = gen[s]
        log(f"  judge for {s} ({len(rows)} rows) ...")
        out = AM.run(items, {s: rows}, run_judge=True,
                     judge_model=cfg.judge_model,
                     out_dir=E.OUT_GENERATION)
        log(f"  {s} done in {time.time() - t_s:.1f}s")
        all_scored[s] = out[s]
        n = len(all_scored[s])
        esc = sum(1 for r in all_scored[s]
                  if (r.get("judge") or {}).get("escalated"))
        log(f"    {n} rows, {esc} escalated ({(esc/max(n,1)):.0%})")

    log("AM.aggregate ...")
    ans_agg = AM.aggregate(all_scored)
    import csv
    E.OUT_AGGREGATE.mkdir(parents=True, exist_ok=True)
    with open(E.OUT_AGGREGATE / "answer_aggregate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "metric", "value", "n"])
        w.writeheader()
        for r in ans_agg:
            w.writerow(r)
    log(f"wrote answer_aggregate.csv ({len(ans_agg)} rows)")

    tB = TB.table_b(ans_agg)
    for r in tB:
        log(f"   Table B {json.dumps(r)}")

    dt = time.time() - t0
    log(f"[done] {dt:.1f}s ({dt / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
