#!/usr/bin/env python
"""GCG-2hop-50 (spec 06 §3/§19): graph-assisted CANDIDATE GENERATION extended
to 2 hops, over the 8 queries whose gold target is ABSENT from the A0-30 pool
but reachable within exactly 2 hops of that pool's article nodes.

Design (graph = candidate generator, NOT a graph-aware reranker; spec §13-14):
  1. pool   = A0 30 hybrid pool (sparse+dense RRF), exactly as A0-30-FULL.
  2. expand = within-2-hop regulatory neighbours (CROSS_REFERENCES / AMENDS) of
             the pool's article nodes that are REAL corpus chunks, appended
             BEHIND the 30 seeds.  Deterministic order (1-hop first, then
             2-hop); bounded to a total pool of 50 (max 20 new).  Only ONE new
             variable vs A0: 2-hop reachability (spec §5).
  3. rerank = the SAME listwise LLM reranker, A0 bare text prompt
             (graph_cfg=None -- NO evidence lines, spec §14), max_candidates=len.
             Cache key stays distinct from A0-30 (['mc',50] + distinct set).
  4. answer = neutral GEN.generate over top-5 + AM.run judge (gpt-oss primary,
             nemotron escalator) -- identical to A0-30 / GCG-1hop-50 (§13).

Primary set   = the 8 "2-hop-only" queries (from gcg_eligibility.json, NOT
                hand-built): two_hop_only AND absent from the A0-30 30-pool.
Controls      = the 6 known 1-hop-rescuable queries (regression; §8).

The question answered: does 2-hop graph-assisted candidate generation put the
missing evidence into the candidate pool, and can the UNCHANGED text-only
reranker exploit it -- while A0-30 and GCG-1hop-50 remain byte-identical?

Usage:
    PYENV_VERSION=energy-audit \
    ENERGY_AUDIT_ROOT=$(pwd) \
    python scripts/run_gcg_2hop_50.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import evaluation.config as E             # noqa: E402
import evaluation.benchmark as BM         # noqa: E402
import evaluation.generation as GEN       # noqa: E402
import evaluation.answer_metrics as AM    # noqa: E402
import evaluation.metrics as metrics      # noqa: E402
from retrieval.fusion import GRAPH_BOOST  # noqa: E402
from retrieval._corpus import article_node_for  # noqa: E402
from retrieval.graph import GRAPH_EDGE_KINDS    # noqa: E402
from retrieval.two_hop import (  # noqa: E402
    apply_admission_policy, expand_within_2hop, two_hop_only)
from reranking.graph_context import GraphContextConfig  # noqa: E402


POOL_N = 30                # A0 hybrid seeds
CAND_CAP = 50              # 30 seeds + up to 20 graph-new candidates


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_sets() -> List[str]:
    """Primary 8 (2-hop-only, absent from A0-30) + 6 (1-hop controls), read from
    the eligibility artifact (spec §1: do NOT manually reconstruct)."""
    p = E.EVAL_RESULTS / "diagnostic" / "gcg_eligibility.json"
    d = json.loads(p.read_text())
    primary = [r["query_id"] for r in d["rows"]
               if r.get("two_hop_only") and not r.get("target_in_a0_30_30pool")]
    controls = list(d.get("rescuable_ids", []))
    return primary, controls


def _node_map(r, lids: List[str]) -> List[str]:
    """Map lineage ids to their article nodes (deduped, sorted)."""
    ns = set()
    for l in lids:
        if l in r._by_lineage:
            n = article_node_for(r.chunks[r._by_lineage[l]],
                                 r.corpus["article_index"], r.corpus["graph"])
            if n:
                ns.add(n)
    return sorted(ns)


def _two_hop_new_prov(r, seeds: List[str],
                      untruncated: bool = False,
                      policy: str = "first",
                      pool_cap: int = CAND_CAP) -> List[dict]:
    """Within-2-hop expansion of the seeds' article nodes to REAL corpus
    chunks (deterministic), with construction-time provenance (§6).

    ``untruncated`` (spec §7/§12 diagnostic): when True, admit ALL 1+2-hop
    real-chunk neighbours (no 20-slot budget).  When False (the as-specified
    primary), admit at most CAND_CAP - len(seeds) = 20 new (spec §5).

    ``policy``: the admission *order* within the 20-slot budget.
      * ``"first"``      (default; = the as-specified §5 order): d1-then-d2,
        truncate to first 20 of the combined list.  All 8 two-hop golds land
        BEYOND position 20, hence never admitted -- that is the H1 failure.
      * ``"2hop_first"`` (distance-priority): d2-then-d1, truncate to first 20.
        A pure re-order of the SAME candidate set (same budget, same traversal,
        same provenance, same real-chunk filter); the only delta is which 20
        survive.  This is the §20 rule-4 "budget" lever tested in isolation.

    Each entry:
        {chunk_id, document_id, source, graph_distance, graph_relation tuple,
         graph_direction, graph_seed, intermediate, graph_path}
    """
    adj = r.corpus["graph"].adj
    real = {n for n in r.corpus["graph"].nodes if n in r._by_lineage}
    seed_nodes = _node_map(r, seeds)
    keep = None if untruncated else max(pool_cap - len(seeds), 0)
    recs = expand_within_2hop(adj, seed_nodes, real,
                              allowed=GRAPH_EDGE_KINDS, keep=None)
    recs = apply_admission_policy(recs, policy=policy, keep=keep)
    out: List[dict] = []
    seen = set()
    for rec in recs:
        cid = rec["node"]
        if cid in seen:
            continue
        seen.add(cid)
        out.append({
            "chunk_id": cid,
            "document_id": (r.chunks[r._by_lineage[cid]].doc_id
                            if cid in r._by_lineage else None),
            "source": rec["source"],                 # graph_1hop / graph_2hop
            "graph_distance": rec["graph_distance"],  # 1 or 2
            "graph_relation": list(rec["graph_relation"]),
            "graph_direction": rec["graph_direction"],
            "graph_seed": rec["graph_seed"],
            "intermediate": rec.get("intermediate"),
            "graph_path": rec["graph_path"],
        })
    return out


def _sparse_dense_rank(r, query: str, n: int = 50) -> tuple:
    sparse_rank = {lid: i + 1 for i, (lid, _s)
                   in enumerate(r.search_sparse(query, n))}
    dense_rank: Dict[str, int] = {}
    for _name, ranked in r.search_dense(query, n).items():
        for i, (lid, _s) in enumerate(ranked):
            dense_rank.setdefault(lid, i + 1)
    return sparse_rank, dense_rank


def run(tag: str = "gcg_2hop_50", untruncated: bool = False,
        only: List[str] = None, policy: str = "first",
        all60: bool = False,
        pool_cap: int = CAND_CAP,
        graph_ctx: bool = False) -> int:
    t0 = time.time()
    primary, controls = _load_sets()
    tag = tag or "gcg_2hop_50"
    log(f"pol={policy}  primary8={primary}  controls6={controls}")
    items = {it.query_id: it for it in BM.build_benchmark()}
    if all60:
        qids = list(dict.fromkeys(items.keys()))
        log(f"[§18-D all-60] scope={len(qids)} queries")
    elif untruncated:
        qids = list(dict.fromkeys(primary)) if only is None \
            else list(dict.fromkeys(only or primary))
        log(f"[UNTRUNCATED diagnostic] scope={qids}")
    else:
        qids = list(dict.fromkeys(primary + controls))
        log(f"primary 2-hop set ({len(primary)}): {primary}")
        log(f"control 1-hop set ({len(controls)}): {controls}")
    log(f"untruncated={untruncated}")
    for qid in qids:
        if qid not in items:
            log(f"!! {qid} not in benchmark"); return 1

    from retrieval import Retriever
    r = Retriever(dense_models=["base"],
                  dense_specs={"base": "all-MiniLM-L6-v2"})
    rr = r._reranker.get()
    cfg = E.EvalConfig()
    k_set = cfg.k_values
    k_max = max(k_set)
    log(f"judge={cfg.judge_model} esc={cfg.judge_escalator} "
        f"reasoner={cfg.llm_model} k={k_set}")

    a0p = E.OUT_PER_QUERY / "retrieval_hybrid_rerank_30_full.jsonl"
    a0 = {json.loads(l)["query_id"]: json.loads(l)
          for l in a0p.read_text().splitlines() if l.strip()} \
        if a0p.exists() else {}

    rows: List[dict] = []
    prov_rows: List[dict] = []
    for qid in qids:
        item = items[qid]
        q, tgt = item.question, item.target_lineage_id
        # 1. A0 30 pool
        top_lids, fused, rank_lists = r._hybrid_pool(q, POOL_N)
        neigh = [g for g in r.search_graph(q, 30)
                 if g[1] >= 1 and g[0] in set(top_lids)]
        for g in neigh:
            fused[g[0]]["score"] += GRAPH_BOOST * (1.0 if g[1] == 1 else 0.5)
        # 2. within-2-hop graph-new candidates + construction-time provenance
        new_prov = _two_hop_new_prov(r, list(top_lids), untruncated=untruncated,
                                     policy=policy, pool_cap=pool_cap)
        # uncapped reachability (spec §12 causality proof): is the target in
        # the FULL 1+2-hop real-node set before the 20-slot budget is applied?
        _adj = r.corpus["graph"].adj
        _real = {n for n in r.corpus["graph"].nodes if n in r._by_lineage}
        _seeds_n = _node_map(r, list(top_lids))
        _full = expand_within_2hop(_adj, _seeds_n, _real,
                                   allowed=GRAPH_EDGE_KINDS, keep=None)
        _tgt_in_full = any(x["node"] == tgt for x in _full)
        _full_n = len(_full)
        new_cands = [rec["chunk_id"] for rec in new_prov]
        pool_lids = list(top_lids)
        pool_lids.extend([n for n in new_cands if n not in pool_lids])
        prov_by_lid = {rec["chunk_id"]: rec for rec in new_prov}
        # 3. assemble candidates (A0 bare, NO graph evidence) + rerank ALL
        #    Lever 2: graph_ctx=True enables per-candidate graph-context evidence
        _gc = GraphContextConfig(include_paths=False) if graph_ctx else None
        candidates, _ = r._gr_candidates(q, pool_lids, graph_cfg=_gc,
                                         neigh=neigh, fused=fused)
        pool = list(candidates)
        tgt_in_pool = tgt in [c["lineage_id"] for c in pool]
        if not pool:
            out_list = r._from_top(top_lids, fused, rank_lists, k_max,
                                   graph_edges=neigh)
            meta = {"status": "fallback", "model": None, "pool": 0}
        else:
            result = rr.rerank(q, pool, max_candidates=len(pool))
            if {x for x in result["order"]} != set(pool_lids):
                result = {**result, "status": "fallback",
                          "note": "permutation-mismatch"}
                new_order = list(pool_lids)
            else:
                new_order = list(result["order"])
            out_list = r._from_top(new_order,
                                   fused | {l: {"score": 0.0,
                                               "methods": ["graph:new"]}
                                           for l in pool_lids[len(top_lids):]},
                                   rank_lists, k_max, graph_edges=neigh,
                                   rerank_meta=result)
            meta = result
        lids = [x.lineage_id for x in out_list]
        final_score_by_lid = {x.lineage_id: x.score for x in out_list}
        first = lids.index(tgt) + 1 if tgt in lids else None

        # ---- spec §6 provenance (captured at construction) ----------------
        sparse_rank, dense_rank = _sparse_dense_rank(r, q)
        cand_prov: List[dict] = []
        for pre, lid in enumerate(pool_lids, start=1):
            fr = fused.get(lid, {})
            seed_rank = (top_lids.index(lid) + 1) if lid in top_lids else None
            g = prov_by_lid.get(lid)
            cand_prov.append({
                "chunk_id": lid,
                "document_id": (r.chunks[r._by_lineage[lid]].doc_id
                                if lid in r._by_lineage else None),
                "source": g["source"] if g else "hybrid",
                "sparse_rank": sparse_rank.get(lid),
                "dense_rank": dense_rank.get(lid),
                "hybrid_rank": seed_rank,
                "rrf_score": (round(fr.get("score", 0.0), 8) if fr else None),
                "graph_distance": g["graph_distance"] if g else None,
                "graph_seed": g["graph_seed"] if g else None,
                "graph_relation": g["graph_relation"] if g else None,
                "graph_direction": g["graph_direction"] if g else None,
                "graph_path": g["graph_path"] if g else None,
                "graph_intermediate": g["intermediate"] if g else None,
                "candidate_rank_before_rerank": pre,
                "final_rerank_rank": (lids.index(lid) + 1) if lid in lids else None,
                "final_rerank_score": (round(final_score_by_lid.get(lid, 0.0), 8)
                                       if lid in final_score_by_lid else None),
            })
        tgt_prov = next((c for c in cand_prov if c["chunk_id"] == tgt), None)
        # spec §9 verification: 2-hop-only against the ACTUAL graph + chunk map
        adj = r.corpus["graph"].adj
        tgt_node = (article_node_for(r.chunks[r._by_lineage[tgt]],
                                     r.corpus["article_index"], r.corpus["graph"])
                    if tgt in r._by_lineage else tgt)
        ver_two_hop_only = two_hop_only(
            adj, _node_map(r, list(top_lids)), tgt_node,
            allowed=GRAPH_EDGE_KINDS) if tgt_node else False

        prov_rows.append({
            "query_id": qid,
            "gold_chunk": tgt,
            "family": item.category,
            "primary_set": (
                "2hop-primary" if qid in primary
                else "1hop-control" if qid in controls
                else "other"),
            "a0_rank": a0.get(qid, {}).get("target_rank"),
            "gcg_generated": tgt in set(pool_lids),
            "generated_by_graph": tgt in {rec["chunk_id"] for rec in new_prov},
            "target_distance": (tgt_prov["graph_distance"] if tgt_prov else None),
            "target_reachable_within_2hop": bool(_tgt_in_full),
            "n_uncapped_1hop_plus_2hop": int(_full_n),
            "target_in_pool_after_cap": tgt_in_pool,
            # §12/§7 causal: generated (reachable in-graph) but excluded by the
            # 20-slot / 50-cap budget (NOT a traversal or reranker failure).
            "lost_by_cap": bool(_tgt_in_full) and (not tgt_in_pool),
            "excluded_by_budget": bool(_tgt_in_full) and (not tgt_in_pool),
            "target_pre_rerank_rank": (tgt_prov["candidate_rank_before_rerank"]
                                       if tgt_prov else None),
            "target_final_rank": first,
            "target_relation": (tgt_prov["graph_relation"] if tgt_prov else None),
            "target_direction": (tgt_prov["graph_direction"] if tgt_prov else None),
            "target_seed": (tgt_prov["graph_seed"] if tgt_prov else None),
            "target_intermediate": (tgt_prov["graph_intermediate"] if tgt_prov else None),
            "target_path": (tgt_prov["graph_path"] if tgt_prov else None),
            "verified_two_hop_only": ver_two_hop_only,
            "n_candidates": len(pool_lids),
            "n_seeds": len(top_lids),
            "n_new": len(new_prov),
            "n_new_dist2": sum(1 for x in new_prov if x["graph_distance"] == 2),
            "answer_correct": None,
            "answer_overall": None,
            "candidates": cand_prov,
        })

        m = {}
        for k in k_set:
            rl = lids[:k]
            m[str(k)] = {
                "system": tag, "query_id": qid, "k": k,
                "recall": metrics.recall_at_k(rl, [tgt], k),
                "precision": metrics.precision_at_k(rl, [tgt], k),
                "hit": metrics.hit_rate_at_k(rl, [tgt], k),
                "mrr": metrics.reciprocal_rank(rl, [tgt]),
                "ndcg": metrics.ndcg_at_k(rl, [tgt], k),
            }
        a0r = a0.get(qid, {}).get("target_rank")
        rows.append({
            "query_id": qid, "question": q, "system": tag,
            "category": item.category, "target": tgt,
            "primary_set": (
                "2hop-primary" if qid in primary
                else "1hop-control" if qid in controls
                else "other"),
            "a0_30_target_rank": a0r,
            "a0_30_target_in_top5": (a0r is not None and a0r <= 5),
            "pool_seeds": len(top_lids), "graph_new_candidates": len(new_cands),
            "graph_new_dist2": sum(1 for x in new_prov if x["graph_distance"] == 2),
            "target_reachable_within_2hop": bool(_tgt_in_full),
            "excluded_by_budget": bool(_tgt_in_full) and (not tgt_in_pool),
            "n_uncapped_1hop_plus_2hop": int(_full_n),
            "pool_total": len(pool), "target_in_pool": tgt_in_pool,
            "retrieved_top30": lids[:30], "retrieved_top20": lids[:20],
            "target_rank": first, "metrics": m,
            "rerank_status": meta.get("status"), "rerank_pool": meta.get("pool"),
        })
        _ps = ("2hop" if qid in primary
               else "1hop" if qid in controls else "other")
        log(f"  {qid} [{_ps}]: "
            f"new={len(new_cands)} (d2={sum(1 for x in new_prov if x['graph_distance']==2)}) "
            f"pool={len(pool)} tgt_in_pool={tgt_in_pool} "
            f"pre={prov_rows[-1]['target_pre_rerank_rank']} final={first} "
            f"(A0-30={a0r})")

    # ---- persist retrieval rows ------------------------------------------
    E.OUT_PER_QUERY.mkdir(parents=True, exist_ok=True)
    p = E.OUT_PER_QUERY / f"retrieval_{tag}.jsonl"
    with open(p, "w") as f:
        for _r in rows:
            f.write(json.dumps(_r, ensure_ascii=False) + "\n")
    log(f"wrote {p}  ({len(rows)} rows)")

    # ---- persist provenance (per-query; answers backfilled after judge) ----
    DIAG = E.EVAL_RESULTS / "diagnostic"
    DIAG.mkdir(parents=True, exist_ok=True)
    _prov_path = DIAG / f"{tag}_provenance.jsonl"
    with open(_prov_path, "w") as f:
        for pr in prov_rows:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")
    log(f"wrote {_prov_path}  ({len(prov_rows)} queries, "
        f"{sum(len(pr['candidates']) for pr in prov_rows)} candidates)")

    # ---- answer generation (neutral) over top-5 ---------------------------
    for _r in rows:
        _r["top5"] = (_r["retrieved_top30"])[:GEN.CONTEXT_WINDOW]
    gen_rows = []
    for _r in rows:
        item = items[_r["query_id"]]
        chunks = []
        for lid in _r["top5"]:
            idx = r._by_lineage.get(lid)
            if idx is None:
                continue
            c = r.chunks[idx]
            chunks.append({"lineage_id": c.lineage_id, "doc_id": c.doc_id,
                           "text": c.text})
        gen = GEN.generate(_r["question"], chunks, cfg)
        gen.update({
            "query_id": item.query_id, "system": tag,
            "category": item.category, "target": item.target_lineage_id,
            "reference_answer": item.reference_answer,
            "reference_basis": "target_chunk_text",
            "retrieval_target_rank": _r["target_rank"],
            "target_in_context": any(
                c["lineage_id"] == item.target_lineage_id for c in chunks),
            "a0_30_target_rank": _r["a0_30_target_rank"],
            "graph_evidence": GEN.format_graph_evidence(item.gold_edges),
            "intended_relation": item.intended_relation,
            "intended_direction": item.intended_direction,
        })
        gen_rows.append(gen)
    E.OUT_GENERATION.mkdir(parents=True, exist_ok=True)
    with open(E.OUT_GENERATION / f"generation_{tag}.jsonl", "w") as f:
        for g in gen_rows:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    log(f"generation done; wrote generation_{tag}.jsonl")

    # ---- judge (gpt-oss + escalator) -------------------------------------
    out = AM.run([items[qid] for qid in qids], {tag: gen_rows},
                 run_judge=True, judge_model=cfg.judge_model,
                 out_dir=E.OUT_GENERATION)
    srows = out[tag]
    ok = sum(1 for g in srows if bool((g.get("judge") or {}).get("correct")))
    log(f"judge: correct = {ok}/{len(srows)}")
    judge_by_qid = {g["query_id"]: (g.get("judge") or {}) for g in srows}
    for pr in prov_rows:
        j = judge_by_qid.get(pr["query_id"], {})
        pr["answer_correct"] = bool(j.get("correct"))
        pr["answer_overall"] = j.get("overall_score")
    with open(_prov_path, "w") as f:
        for pr in prov_rows:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")
    E.OUT_AGGREGATE.mkdir(parents=True, exist_ok=True)
    import csv
    ans_agg = AM.aggregate({tag: srows})
    with open(E.OUT_AGGREGATE / f"answer_aggregate_{tag}.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "metric", "value", "n"])
        w.writeheader()
        for a in ans_agg:
            w.writerow(a)
    log(f"[done] {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--untruncated", action="store_true",
                    help="§7/§12 diagnostic: no 20-slot cap (2-hop targets enter the pool)")
    ap.add_argument("--scope", default="primary",
                    choices=["primary", "all"],
                    help="untruncated scope: primary 8 (default) or all 14")
    ap.add_argument("--policy", default=None,
                    choices=[None, "first", "2hop_first",
                             "split_10_10", "split_11_9",
                             "split_12_9", "split_12_8", "split_8_12"],
                    help="admission order within the 20-slot budget (spec §5/§20 rule-4 "
                         "'budget' lever). 'first' = as-specified (d1-then-d2). "
                         "'2hop_first' = d2-then-d1. 'split_N_M' = N slots to "
                         "distance-1 + M to distance-2 (the only class that keeps "
                         "BOTH the 1-hop and 2-hop golds).")
    ap.add_argument("--all", dest="all60", action="store_true", default=False,
                    help="run ALL 60 benchmark queries (spec §18-D 'Overall 60-query "
                         "effect'). Budget stays 20, reranker unchanged.")
    ap.add_argument("--pool-cap", type=int, default=CAND_CAP,
                    help="Lever 1 (spec §5 override): total candidate pool cap. "
                         "51 (default 50) => 21 graph-new slots, the minimal budget "
                         "that also secures control q011 (d1-rank 12) and primary "
                         "q051 (d2-rank 9).")
    ap.add_argument("--graph-ctx", dest="graph_ctx", action="store_true",
                    default=False,
                    help="Lever 2 (spec §20 'investigate graph-aware reranking'): "
                         "give the unchanged listwise reranker per-candidate 1-hop "
                         "graph-relations as evidence. Distinct cache key.")
    args = ap.parse_args()
    policy = args.policy or "first"
    if args.untruncated:
        scope = list(dict.fromkeys(_load_sets()[0]))
        sys.exit(run(tag="gcg_2hop_50_untruncated", untruncated=True,
                        only=scope, policy="first"))
    # tag reflects: policy + budget bucket (pool-30 = graph-new count) + graph-ctx
    budget_bucket = args.pool_cap - POOL_N          # 20 -> "20", 21 -> "21"
    suffix = policy.replace("_", "") + "_" + str(budget_bucket) + \
        ("_gc" if args.graph_ctx else "")
    tag = "gcg_2hop_" + suffix
    if args.all60:
        tag += "_60"
    sys.exit(run(tag=tag, policy=policy, all60=args.all60,
                 pool_cap=args.pool_cap, graph_ctx=args.graph_ctx))
