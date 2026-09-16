"""Answer-generation + judge pass for A0-30-FULL (`hybrid_rerank_30_full`).

Why this script exists separately from `scripts/run_judge_only.py`:

* `run_generation` (generation.py:150) re-calls the live Retriever for chunks,
  which requires the system to be registered in `E.EXPERIMENTS` (config.py:191)
  so the mode string is known.  A0-30-FULL is a *retrieval-only* diagnostic
  mode (its full 30-rank permutation is already on disk in
  `per_query/retrieval_hybrid_rerank_30_full.jsonl`).  Re-retrieving A0-30
  live would hit the public `retrieve(mode="hybrid_rerank")` path which
  ignores the pool_n / depth override -- the whole point of the clean run
  was to *avoid* a 20-cap re-rank.

* So: this script reads the on-disk 30-permutation, slices the top-5 by the
  same ids, pulls the chunk text from the corpus, and calls `GEN.generate`
  with that explicit chunk set.  The `user_prompt` cache key is
  (llm_model, temperature, max_tokens, SYSTEM_PROMPT, user_prompt) and the
  user_prompt is the standard template rendering those 5 chunks -- so if the
  same 5 chunks ever appear in another system's top-5, the LLM call is a
  free cache hit.  (For A0-30-FULL the top-5 was shown to differ from all
  four other systems on ~38 of the 60 graphs, so most rows are fresh.)

* Judge: gpt-oss:latest primary, nemotron-3-nano:30b escalator (per
  EvalConfig).  Each row ~30-90 s.  ~2-4 h for 60 rows.

* Output:
    notebooks/data/evaluation/generation/answers_hybrid_rerank_30_full.jsonl
  (the standard `answers_{system}.jsonl` naming convention the report uses.)

Usage:
    PYENV_VERSION=energy-audit ENERGY_AUDIT_ROOT=$(pwd) \
        python scripts/run_judge_a0_30_full.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import evaluation.config as E             # noqa: E402
import evaluation.benchmark as BM         # noqa: E402
import evaluation.generation as GEN       # noqa: E402
import evaluation.answer_metrics as AM    # noqa: E402

SYSTEM = "hybrid_rerank_30_full"
PER_QUERY_FILE = E.OUT_PER_QUERY / f"retrieval_{SYSTEM}.jsonl"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    t0 = time.time()
    log(f"system={SYSTEM}")
    log(f"per_query file: {PER_QUERY_FILE}")
    if not PER_QUERY_FILE.exists():
        log("!! missing per_query retrieval -- run scripts/run_a0_30_clean.py first")
        return 1

    cfg = E.EvalConfig()
    log(f"judge primary={cfg.judge_model}   escalator={cfg.judge_escalator}")

    log("building benchmark (60 items) ...")
    items = BM.build_benchmark()
    if not items:
        log("!! benchmark empty"); return 1
    log(f"n_items={len(items)}")

    log(f"loading {PER_QUERY_FILE} ...")
    per_q = {json.loads(l)["query_id"]: json.loads(l)
             for l in PER_QUERY_FILE.read_text().splitlines() if l.strip()}
    if len(per_q) != len(items):
        log(f"!! expected {len(items)} rows, got {len(per_q)}")
        return 1
    log(f"loaded {len(per_q)} retrieval rows")

    log("loading corpus (for top-5 chunk text) ...")
    t1 = time.time()
    from retrieval import Retriever
    r = Retriever(dense_models=["base"],
                  dense_specs={"base": "all-MiniLM-L6-v2"})
    by_lid = dict(r._by_lineage)
    log(f"corpus loaded {len(by_lid)} chunks in {time.time()-t1:.1f}s")

    log(f"GEN.generate (system={SYSTEM}) ...")
    t2 = time.time()
    rows: List[dict] = []
    for item in items:
        qid = item.query_id
        rq = per_q.get(qid)
        if rq is None:
            log(f"  !! missing per_q row for {qid}")
            continue
        ids = rq.get("retrieved_top30") or rq.get("retrieved_top20") or []
        top5 = ids[:GEN.CONTEXT_WINDOW]
        chunks = []
        for lid in top5:
            idx = by_lid.get(lid)
            if idx is None:
                continue
            c = r.chunks[idx]
            chunks.append({"lineage_id": c.lineage_id,
                           "doc_id": c.doc_id,
                           "text": c.text})
        gen = GEN.generate(item.question, chunks, cfg)
        gen.update({
            "query_id": item.query_id,
            "system": SYSTEM,
            "category": item.category,
            "target": item.target_lineage_id,
            "reference_answer": item.reference_answer,
            "reference_basis": "target_chunk_text",
            "retrieval_target_rank": rq.get("target_rank"),
            "target_in_context": any(
                c["lineage_id"] == item.target_lineage_id
                for c in chunks),
            "graph_evidence": GEN.format_graph_evidence(item.gold_edges),
            "intended_relation": item.intended_relation,
            "intended_direction": item.intended_direction,
        })
        rows.append(gen)

    log(f"generation done in {time.time()-t2:.1f}s   n_rows={len(rows)}")
    GEN._save(SYSTEM, rows, E.OUT_GENERATION)
    tgt_in = sum(1 for r_ in rows if r_.get("target_in_context"))
    log(f"  target_in_context: {tgt_in}/{len(rows)}")

    log("AM.run (judge) ...")
    t3 = time.time()
    out = AM.run(items, {SYSTEM: rows}, run_judge=True,
                 judge_model=cfg.judge_model,
                 out_dir=E.OUT_GENERATION)
    srows = out[SYSTEM]
    log(f"judge done in {time.time()-t3:.1f}s  n={len(srows)}")
    n = len(srows)
    ok = sum(1 for r_ in srows
             if bool((r_.get("judge") or {}).get("correct")))
    esc = sum(1 for r_ in srows
              if (r_.get("judge") or {}).get("escalated"))
    correct_rate = ok / max(n, 1)
    esc_rate = esc / max(n, 1)
    log(f"judge correct = {ok}/{n} ({correct_rate:.3f})   "
        f"escalated = {esc}/{n} ({esc_rate:.0%})")

    # Write the aggregate for this system only so the CSV does not clobber
    # the existing multi-system table; if the user wants a merged table they
    # can re-run run_judge_only.py afterwards.
    ans_agg = AM.aggregate({SYSTEM: srows})
    E.OUT_AGGREGATE.mkdir(parents=True, exist_ok=True)
    with open(E.OUT_AGGREGATE / f"answer_aggregate_{SYSTEM}.csv",
              "w", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=["system", "metric", "value", "n"])
        w.writeheader()
        for r_ in ans_agg:
            w.writerow(r_)
    log(f"wrote answer_aggregate_{SYSTEM}.csv  ({len(ans_agg)} rows)")

    dt = time.time() - t0
    log(f"[done] {dt:.1f}s ({dt/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
