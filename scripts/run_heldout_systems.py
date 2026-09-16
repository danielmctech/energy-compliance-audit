"""Spec 08 §2/§9 — run the frozen systems on the held-out benchmark.

Runs the four conditions specified by spec 08 on the frozen held-out
benchmark (the on-disk ``heldout_benchmark.jsonl`` artifact, restricted to
the 160 audit-valid items — 174 built, 14 excluded by doc-09 Stage 1 audit):

  B0        = A0-30 frozen baseline (spec 08 §2: ``hybrid_rerank_30_full``,
              scripts/run_a0_30_clean.py — 30 hybrid seeds, A0 bare prompt,
              listwise LLM rerank of all 30)
  G1        = GCG-1hop-50 frozen (spec 08 §2: scripts/run_gcg_1hop_50.py —
              30 hybrid seeds + up to 20 1-hop graph-new, budget 50,
              bare rerank of all)
  G2        = GCG-2hop Lever-2 frozen, the "12/14" configuration
              (notebooks/data/evaluation/diagnostic/gcg_2hop_two_levers_report.md
              §C: split_12_9 + budget 21 + graph-ctx; scripts/run_gcg_2hop_50.py
              with --policy split_12_9 --pool-cap 51 --graph-ctx)
  G2_nogc   = spec 08 §15 Q2 control: identical G2 with graph_ctx=False
              ("G2 without graph context")

Nothing here is tuned (spec 08 §1): candidate generation, expansion,
budget, reranker prompt/model, answer generator and judge settings are
all the exact frozen values above.  This script is a *runner*, not a
config.

Phases (all resumable — re-running skips systems/queries already on disk):
  1. retrieval  -> evaluation/heldout_{b0,g1,g2,g2_nogc}_retrieval.jsonl
                    (per-query trajectory, spec 08 §13 + candidate
                     provenance, spec 08 §14, + admission pool)
  2. generation -> evaluation/heldout_{...}_answers.jsonl
                   (frozen generator over top-5; AM.run attaches the
                    frozen judge: gpt-oss:latest primary,
                    nemotron-3-nano:30b escalator)

Usage:
    PYENV_VERSION=energy-audit ENERGY_AUDIT_ROOT=$(pwd) \
        python scripts/run_heldout_systems.py --phase retrieval --limit 4
    PYENV_VERSION=energy-audit ENERGY_AUDIT_ROOT=$(pwd) \
        python scripts/run_heldout_systems.py --phase all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(os.getenv("ENERGY_AUDIT_ROOT") or ".").resolve()
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (str(SRC), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("ENERGY_AUDIT_ROOT", str(ROOT))

import evaluation.config as E                    # noqa: E402
import evaluation.benchmark as BM                # noqa: E402
import evaluation.generation as GEN              # noqa: E402
import evaluation.answer_metrics as AM           # noqa: E402
import evaluation.metrics as metrics             # noqa: E402
from retrieval.fusion import GRAPH_BOOST         # noqa: E402
from reranking.graph_context import GraphContextConfig  # noqa: E402

# Reuse the EXACT frozen helpers (spec 08 §1 + §22 'do not overwrite'):
import run_gcg_1hop_50 as G1                    # noqa: E402
import run_gcg_2hop_50 as G2                    # noqa: E402

# ---- frozen configs (pinned from the completed GCG series) ------------------
#: Spec 08 §2: "Use the exact configuration from the completed GCG-1hop
#: experiment."  run_gcg_1hop_50.py: POOL_N=30, CAND_CAP=50 (30+20).
G1_POOL_N = G1.POOL_N          # 30
G1_CAND_CAP = G1.CAND_CAP      # 50

#: Spec 08 §2: "Use the configuration established in the completed
#: two-lever experiment."  That is Lever 2 = split_12_9 + budget 21
#: (pool-51) + graph-ctx — the 12/14 best condition on answers.
G2_POLICY = "split_12_9"
G2_POOL_CAP = 51               # 30 seeds + 21 graph-new slots
G2_GRAPH_CTX = True            # Lever 2: GraphContextConfig(include_paths=False)
GC_CONFIG = GraphContextConfig(include_paths=False)  # as in run_gcg_2hop_50.py:222

SYSTEMS = ["b0", "g1", "g2", "g2_nogc"]
OUT = E.EVAL_RESULTS


def log(tag: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Per-system retrieval (frozen pipelines, reused by import)
# ---------------------------------------------------------------------------

def _common_pool(r, q: str, pool_n: int = G1_POOL_N):
    """Step 1, identical for B0/G1/G2 (spec 08 §9)."""
    top_lids, fused, rank_lists = r._hybrid_pool(q, pool_n)
    neigh = [g for g in r.search_graph(q, 30)
             if g[1] >= 1 and g[0] in set(top_lids)]
    for g in neigh:
        fused[g[0]]["score"] += GRAPH_BOOST * (1.0 if g[1] == 1 else 0.5)
    return top_lids, fused, rank_lists, neigh


def _rerank_all(rr, r, q, pool_lids, top_lids, fused, rank_lists,
                neigh, k_max, graph_cfg):
    """Frozen A0 listwise rerank of the WHOLE pool (run_*_50.py pattern)."""
    candidates, _ = r._gr_candidates(q, pool_lids, graph_cfg=graph_cfg,
                                     neigh=neigh, fused=fused)
    pool = list(candidates)
    if not pool:
        out_list = r._from_top(top_lids, fused, rank_lists, k_max,
                               graph_edges=neigh)
        meta = {"status": "fallback", "model": None, "pool": 0}
    else:
        result = rr.rerank(q, pool, max_candidates=len(pool))
        if {x for x in result["order"]} != set(pool_lids):
            result = {**result, "status": "fallback",
                      "note": "permutation-mismatch"}
            out_list = r._from_top(list(pool_lids),
                                   fused | {l: {"score": 0.0,
                                               "methods": ["graph:new"]}
                                           for l in pool_lids[len(top_lids):]},
                                   rank_lists, k_max, graph_edges=neigh,
                                   rerank_meta=result)
        else:
            out_list = r._from_top(list(result["order"]),
                                   fused | {l: {"score": 0.0,
                                               "methods": ["graph:new"]}
                                           for l in pool_lids[len(top_lids):]},
                                   rank_lists, k_max, graph_edges=neigh,
                                   rerank_meta=result)
        meta = result
    return out_list, meta


def _candidate_provenance(r, q: str, pool_lids: List[str],
                          out_list, prov_by_lid: Dict[str, dict],
                          fused: Optional[dict], top_lids: List[str]):
    """Spec 08 §14: graph provenance for every candidate, pre/post rerank."""
    lids = [x.lineage_id for x in out_list]
    final_score = {x.lineage_id: x.score for x in out_list}
    sparse_rank, dense_rank = G2._sparse_dense_rank(r, q)
    out = []
    for pre, lid in enumerate(pool_lids, start=1):
        fr = (fused or {}).get(lid, {}) if fused else {}
        seed_rank = (top_lids.index(lid) + 1) if lid in top_lids else None
        g = prov_by_lid.get(lid)
        out.append({
            "chunk_id": lid,
            "document_id": (r.chunks[r._by_lineage[lid]].doc_id
                            if lid in r._by_lineage else None),
            "source": (g and g.get("source")) or "hybrid",
            "sparse_rank": sparse_rank.get(lid),
            "dense_rank": dense_rank.get(lid),
            "hybrid_rank": seed_rank,
            "rrf_score": (round(fr.get("score", 0.0), 8) if fr else None),
            "graph_distance": g.get("graph_distance") if g else None,
            "graph_seed": g.get("graph_seed") if g else None,
            "graph_relation": g.get("graph_relation") if g else None,
            "graph_direction": g.get("graph_direction") if g else None,
            "graph_path": g.get("graph_path") if g else None,
            "graph_intermediate": g.get("intermediate") if g else None,
            "candidate_rank_before_rerank": pre,
            "final_rerank_rank": (lids.index(lid) + 1) if lid in lids else None,
            "final_rerank_score": (round(final_score.get(lid, 0.0), 8)
                                   if lid in final_score else None),
        })
    return out


def _run_query(system: str, r, rr, cfg, item,
               k_set, k_max, a0_by_qid: Dict[str, dict]):
    """Run one frozen system for one held-out item (spec 08 §9: identical
    corpus/graph/indexes/scoring across systems; only the graph mechanism
    differs)."""
    q, tgt = item.question, item.target_lineage_id
    q_start = time.time()

    top_lids, fused, rank_lists, neigh = _common_pool(r, q)

    if system == "b0":
        # ---- B0: A0-30 (no graph-new candidates)
        out_list, meta = _rerank_all(rr, r, q, list(top_lids), top_lids,
                                     fused, rank_lists, neigh, k_max,
                                     graph_cfg=None)
        pool_lids = list(top_lids)
        new_prov = []
    elif system == "g1":
        # ---- G1: 1-hop graph-new, budget 50 (run_gcg_1hop_50.py:166-172)
        full_new = G1._one_hop_new_prov(r, list(top_lids))
        keep = max(G1_CAND_CAP - len(top_lids), 0)
        new_prov = full_new[:keep]
        new_cands = [rec["chunk_id"] for rec in new_prov]
        pool_lids = list(top_lids)
        pool_lids.extend(n for n in new_cands if n not in pool_lids)
        out_list, meta = _rerank_all(rr, r, q, pool_lids, top_lids,
                                     fused, rank_lists, neigh, k_max,
                                     graph_cfg=None)
    else:
        # ---- G2 / G2_nogc: 2-hop, split_12_9, budget 21, graph_ctx for G2
        new_prov = G2._two_hop_new_prov(r, list(top_lids), untruncated=False,
                                        policy=G2_POLICY, pool_cap=G2_POOL_CAP)
        graph_cfg = GC_CONFIG if (system == "g2") else None
        new_cands = [rec["chunk_id"] for rec in new_prov]
        pool_lids = list(top_lids)
        pool_lids.extend(n for n in new_cands if n not in pool_lids)
        out_list, meta = _rerank_all(rr, r, q, pool_lids, top_lids,
                                     fused, rank_lists, neigh, k_max,
                                     graph_cfg=graph_cfg)

    lids = [x.lineage_id for x in out_list]
    first = lids.index(tgt) + 1 if tgt in lids else None
    tgt_final_score = (round(next(x.score for x in out_list
                                  if x.lineage_id == tgt), 8)
                       if tgt in lids else None)
    tgt_in_pool = tgt in pool_lids

    # spec §14 candidates provenance
    by_lid = {c["chunk_id"]: c for c in new_prov}
    cand_prov = _candidate_provenance(r, q, pool_lids, out_list, by_lid,
                                      fused, top_lids)
    tgt_prov = next((c for c in cand_prov if c["chunk_id"] == tgt), None)

    m = {}
    for k in k_set:
        rl = lids[:k]
        m[str(k)] = {
            "system": system, "query_id": item.query_id, "k": k,
            "recall": metrics.recall_at_k(rl, [tgt], k),
            "precision": metrics.precision_at_k(rl, [tgt], k),
            "hit": metrics.hit_rate_at_k(rl, [tgt], k),
            "mrr": metrics.reciprocal_rank(rl, [tgt]),
            "ndcg": metrics.ndcg_at_k(rl, [tgt], k),
        }

    # spec §13 per-query trajectory
    row = {
        "query_id": item.query_id,
        "question": q,
        "system": system,
        "query_type": item.category,
        "target": tgt,
        "gold_chunk_ids": item.gold_chunks,
        "gold_document_ids": item.gold_documents,
        "gold_entities": item.gold_entities,
        "gold_edges": item.gold_edges,
        "gold_path": item.gold_path,
        "candidate_pool_size": len(pool_lids),
        "target_in_pool": tgt_in_pool,
        "n_seeds": len(top_lids),
        "n_graph_new": len(new_prov),
        "target_hybrid_rank": None,
        "target_graph_distance": (tgt_prov["graph_distance"]
                                  if tgt_prov else None),
        "target_graph_relation": (tgt_prov["graph_relation"]
                                  if tgt_prov else None),
        "target_graph_direction": (tgt_prov["graph_direction"]
                                   if tgt_prov else None),
        "target_graph_path": (tgt_prov["graph_path"] if tgt_prov else None),
        "target_rank_before_rerank": (tgt_prov["candidate_rank_before_rerank"]
                                      if tgt_prov else None),
        "target_final_rank": first,
        "target_final_score": tgt_final_score,
        "retrieved_top_1": lids[:1],
        "retrieved_top_5": lids[:5],
        "retrieved_top_10": lids[:10],
        "retrieved_top30": lids[:30],
        "rerank_status": (meta.get("status") if isinstance(meta, dict)
                          else None),
        "rerank_pool": (meta.get("pool") if isinstance(meta, dict) else None),
        "candidates": cand_prov,
        "retrieval_ms": round((time.time() - q_start) * 1000.0, 1),
        # spec §15 Q1 baseline axis: frozen A0-30 rank for the same item
        "b0_target_rank": a0_by_qid.get(item.query_id, {}).get(
            "target_final_rank"),
    }
    return row


def _heldout_items(limit: Optional[int] = None) -> List:
    """The FROZEN held-out benchmark (the on-disk ``heldout_benchmark.jsonl``
    artifact, restricted to the audit-valid subset), as the runnable set.

    Spec 08 §4/§16/§20 fixes the held-out set at build time (deterministic
    seed 13), and doc 09 Stage 1 (``scripts/audit_benchmark.py``) audits the
    same on-disk artifact and marks 160/174 items valid (152 valid_exact +
    8 valid_subarticle); the 14 invalid (4 malformed / "Article None",
    4 mica:Article 7-10 disagreement, 6 answer-unsupported term-lookups) are
    excluded from the runnable set with reasons (spec 08 §4: quality).

    We deliberately do NOT call ``BM.build_benchmark_heldout()`` here: that
    regeneration path is non-reproducible once the corpus changes (adding
    the 12 recovered article chunks shifts its per-family allocations and
    re-shuffles the ``hNNN`` ids, which no longer match the frozen artifact
    nor the audit). The on-disk artifact is the authoritative frozen object
    (``audit_benchmark.py`` and ``run_gr_rerank_bench.py`` both read it).
    """
    audit_path = E.EVAL_RESULTS / "benchmark_audit.jsonl"
    bench_path = E.EVAL_RESULTS / "heldout_benchmark.jsonl"
    valid_qids = BM.audit_valid_qids(audit_path)
    items = BM.load_jsonl(bench_path, valid_qids)
    if limit:
        items = items[:limit]
    return items


def phase_retrieval(limit: Optional[int]) -> int:
    items = _heldout_items(limit)
    cfg = E.EvalConfig()
    k_set = cfg.k_values
    k_max = max(k_set)
    n = len(items)
    log("retrieval", f"n_items={n}  systems={SYSTEMS}  k={k_set}")

    from retrieval import Retriever
    r = Retriever(dense_models=["base"],
                  dense_specs={"base": "all-MiniLM-L6-v2"})
    rr = r._reranker.get()

    # B0 runs first; other systems read its on-disk rows for the
    # cross-system `b0_target_rank` field (spec §15 Q1).
    order = ["b0", "g1", "g2", "g2_nogc"]
    for system in order:
        p = OUT / f"heldout_{system}_retrieval.jsonl"
        done = set()
        if p.exists():
            done = {json.loads(l)["query_id"]
                    for l in p.read_text().splitlines() if l.strip()}
        todo = [it for it in items if it.query_id not in done]
        log(system, f"todo={len(todo)} (already on disk: {len(done)})")

        a0_by_qid = {}
        if system != "b0":
            p0 = OUT / "heldout_b0_retrieval.jsonl"
            if p0.exists():
                a0_by_qid = {json.loads(l)["query_id"]: json.loads(l)
                             for l in p0.read_text().splitlines()
                             if l.strip()}

        t0 = time.time()
        rows = [json.loads(l) for l in p.read_text().splitlines()
                if l.strip()] if p.exists() else []
        p.parent.mkdir(parents=True, exist_ok=True)

        def _flush():
            # Persist after every item so a mid-run crash loses at most the
            # in-flight item (idempotent on resume: `done` is keyed by the
            # query_id already on disk). No config/system change.
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(
                json.dumps(x, ensure_ascii=False) for x in rows) + "\n")

        for i, item in enumerate(todo, start=1):
            row = _run_query(system, r, rr, cfg, item, k_set, k_max,
                             a0_by_qid)
            rows.append(row)
            _flush()
            if i % 5 == 0 or i == len(todo):
                el = time.time() - t0
                log(system, f"  {i}/{len(todo)}  "
                            f"tgt_in_pool={row['target_in_pool']} "
                            f"final={row['target_final_rank']}  "
                            f"({el:.0f}s elapsed, "
                            f"{el/i:.0f}s/query)")
        _flush()
        log(system, f"wrote {p}  ({len(rows)} rows)")
    log("retrieval", "DONE")
    return 0


# ---------------------------------------------------------------------------
# Phase 2: frozen generation (top-5) + frozen judge via AM.run
# ---------------------------------------------------------------------------

def phase_answers(limit: Optional[int]) -> int:
    items = _heldout_items(limit)
    cfg = E.EvalConfig()
    log("answers", f"n_items={len(items)}  generator={cfg.llm_model} "
                   f"judge={cfg.judge_model} esc={cfg.judge_escalator}")

    from retrieval import Retriever
    r = Retriever(dense_models=["base"],
                  dense_specs={"base": "all-MiniLM-L6-v2"})

    for system in SYSTEMS:
        p = OUT / f"heldout_{system}_retrieval.jsonl"
        rows = {json.loads(l)["query_id"]: json.loads(l)
                for l in p.read_text().splitlines() if l.strip()}
        pgen = OUT / f"heldout_{system}_answers.jsonl"
        if pgen.exists():
            have = {json.loads(l)["query_id"]
                    for l in pgen.read_text().splitlines() if l.strip()}
            missing = [it for it in items if it.query_id not in have]
        else:
            missing = list(items)
        log(system, f"answers todo={len(missing)} (have={len(items)-len(missing)})")

        gen_rows = (json.loads(l)
                    for l in pgen.read_text().splitlines() if l.strip()) \
            if pgen.exists() else ()
        gen_list = list(gen_rows)
        for it in missing:
            rq = rows[it.query_id]
            top5 = (rq.get("retrieved_top_10") or rq.get("retrieved_top30")
                    or [])[:GEN.CONTEXT_WINDOW]
            chunks = []
            for lid in top5:
                idx = r._by_lineage.get(lid)
                if idx is None:
                    continue
                c = r.chunks[idx]
                chunks.append({"lineage_id": c.lineage_id,
                               "doc_id": c.doc_id, "text": c.text})
            gen = GEN.generate(it.question, chunks, cfg)
            gen.update({
                "query_id": it.query_id,
                "question": it.question,
                "system": system,
                "category": it.category,
                "target": it.target_lineage_id,
                "reference_answer": it.reference_answer,
                "reference_basis": "target_chunk_text",
                "retrieval_target_rank": rq.get("target_final_rank"),
                "target_in_context": any(
                    c["lineage_id"] == it.target_lineage_id for c in chunks),
                "graph_evidence": GEN.format_graph_evidence(it.gold_edges),
                "intended_relation": it.intended_relation,
                "intended_direction": it.intended_direction,
            })
            gen_list.append(gen)

        log(system, f"scoring {len(gen_list)} rows with frozen judge ...")
        out = AM.run(items, {system: gen_list}, run_judge=True,
                     judge_model=cfg.judge_model, out_dir=OUT)
        srows = out[system]
        ok = sum(1 for x in srows
                 if bool((x.get("judge") or {}).get("correct")))
        log(system, f"judge correct={ok}/{len(srows)}")
    log("answers", "DONE")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["retrieval", "answers", "all"],
                    default="all")
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N held-out items (validation)")
    a = ap.parse_args()
    limit = a.limit or None

    if a.phase in ("retrieval", "all"):
        if phase_retrieval(limit):
            return 1
    if a.phase in ("answers", "all"):
        if phase_answers(limit):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
