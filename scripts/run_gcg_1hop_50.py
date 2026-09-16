#!/usr/bin/env python
"""GCG-1hop-50 (spec §20): graph-assisted CANDIDATE GENERATION over the 6
queries whose gold target is ABSENT from the A0-30 30-pool but reached within
1-hop (CROSS_REFERENCES / AMENDS) of that pool's article nodes.

Design (graph = candidate generator, NOT the v2 graph-context reranker):
  1. pool   = A0 30 hybrid pool (sparse+dense RRF), exactly as A0-30-FULL.
  2. expand = 1-hop regulatory neighbours of the pool's article nodes that are
             REAL corpus chunks, appended BEHIND the 30 seeds (new-node
             admission).  Deterministic order; bounded to a total pool of 50.
  3. rerank = the SAME listwise LLM reranker, A0 bare prompt
     (graph_cfg=None -- no evidence lines), max_candidates=len(pool).  The
     distinct candidate set + n keeps the cache key distinct from A0-30 so the
     frozen baseline is untouched.  Graph is here ONLY to build the pool.
  4. answer = neutral GEN.generate over top-5 chunks + AM.run judge
     (gpt-oss primary, nemotron escalator) -- identical to A0-30's judge.

The question answered: for these 6 queries, does admitting 1-hop graph
candidates + the same reranker recover the gold target into the top-k and/or
improve the judged answer, while the A0-30 numbers are unchanged?

Usage:
    PYENV_VERSION=energy-audit \
    ENERGY_AUDIT_ROOT=$(pwd) \
    python scripts/run_gcg_1hop_50.py
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


SYSTEM = "gcg_1hop_50"
POOL_N = 30          # A0 hybrid seeds
CAND_CAP = 50        # GCG-1hop-50: 30 seeds + up to 20 graph-new candidates


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_eligible() -> List[str]:
    p = E.EVAL_RESULTS / "diagnostic" / "gcg_eligibility.json"
    d = json.loads(p.read_text())
    return d["rescuable_ids"]


def _one_hop_new_prov(r, seeds: List[str]
                      ) -> List[dict]:
    """1-hop regulatory neighbours of the seeds' article nodes that are REAL
    corpus chunks and not already in the seeds -- in deterministic order.

    Each entry carries the construction-time provenance the spec §1 requires
    (captured here, at candidate generation -- NOT inferred after the run):

        {chunk_id, source, graph_seed, graph_relation, graph_direction,
         graph_path, graph_distance}

    Direction is defined against the stored edge direction (the graph stores
    ``src --kind--> dst``; a neighbour is a *dst* of an outgoing edge from a
    pool-seed node, so it is an OUTGOING neighbour of that seed).
    """
    adj = r.corpus["graph"].adj
    seedset = set(seeds)
    _nodes = {
        article_node_for(r.chunks[r._by_lineage[l]], r.corpus["article_index"],
                         r.corpus["graph"])
        for l in seeds if l in r._by_lineage
    }
    pool_nodes = sorted(p for p in _nodes if p)
    out: List[dict] = []
    seen = set()
    for node in pool_nodes:
        for dst, kind in adj.get(node, []):
            if kind not in GRAPH_EDGE_KINDS:
                continue
            if dst in seedset or dst in seen or dst not in r._by_lineage:
                continue
            seen.add(dst)
            out.append({
                "chunk_id": dst,
                "source": "graph_1hop",
                "graph_distance": 1,
                "graph_relation": kind,          # stored edge kind (CROSS_REFERENCES/AMENDS)
                "graph_direction": "outgoing",   # dst of an outgoing edge from the seed node
                "graph_seed": node,
                "graph_path": [node, kind, dst],
            })
    return out


def _one_hop_new(r, seeds: List[str]) -> List[str]:
    """Compatibility wrapper: just the chunk ids (deterministic order)."""
    return [rec["chunk_id"] for rec in _one_hop_new_prov(r, seeds)]


def _sparse_dense_rank(r, query: str, n: int = 50) -> tuple:
    """Deterministic sparse/dense rank maps for provenance (spec §1).

    Mirrors scripts/build_baseline_diagnostic.py: rank is the position (1-based)
    of the chunk in the channel's own ranking; ``None`` when the channel did
    not rank it.  Read-only; does not alter the A0 pools.
    """
    sparse_rank = {lid: i + 1 for i, (lid, _s)
                   in enumerate(r.search_sparse(query, n))}
    dense_rank: Dict[str, int] = {}
    for _name, ranked in r.search_dense(query, n).items():
        for i, (lid, _s) in enumerate(ranked):
            dense_rank.setdefault(lid, i + 1)
    return sparse_rank, dense_rank


def main() -> int:
    t0 = time.time()
    eligible = _load_eligible()
    log(f"eligible GCG-1hop set ({len(eligible)}): {eligible}")
    items = {it.query_id: it for it in BM.build_benchmark()}
    for qid in eligible:
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

    # baseline A0-30 (frozen, from disk) for the same 6, to show no drift
    a0p = E.OUT_PER_QUERY / "retrieval_hybrid_rerank_30_full.jsonl"
    a0 = {json.loads(l)["query_id"]: json.loads(l)
          for l in a0p.read_text().splitlines() if l.strip()} \
        if a0p.exists() else {}

    rows: List[dict] = []
    prov_rows: List[dict] = []
    for qid in eligible:
        item = items[qid]
        q, tgt = item.question, item.target_lineage_id
        # 1. A0 30 pool
        top_lids, fused, rank_lists = r._hybrid_pool(q, POOL_N)
        neigh = [g for g in r.search_graph(q, 30)
                 if g[1] >= 1 and g[0] in set(top_lids)]
        for g in neigh:
            fused[g[0]]["score"] += GRAPH_BOOST * (1.0 if g[1] == 1 else 0.5)
        # 2. 1-hop graph-new candidates + construction-time provenance (§1)
        full_new_prov = _one_hop_new_prov(r, list(top_lids))
        keep = max(CAND_CAP - len(top_lids), 0)
        new_prov = full_new_prov[:keep]
        new_cands = [rec["chunk_id"] for rec in new_prov]
        pool_lids = list(top_lids)
        pool_lids.extend([n for n in new_cands if n not in pool_lids])
        prov_by_lid = {rec["chunk_id"]: rec for rec in new_prov}
        # 3. assemble candidates (A0 bare, no graph evidence) + rerank ALL
        candidates, _ = r._gr_candidates(q, pool_lids, graph_cfg=None,
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
                new_order = list(pool_lids)
                result = {**result, "status": "fallback",
                          "note": "permutation-mismatch"}
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

        # ---- spec §1/§6/§9 provenance (captured at construction) ----------
        sparse_rank, dense_rank = _sparse_dense_rank(r, q)
        cand_prov: List[dict] = []
        for pre_rerank_rank, lid in enumerate(pool_lids, start=1):
            fr = fused.get(lid, {})
            seed_rank = (top_lids.index(lid) + 1) if lid in top_lids else None
            g = prov_by_lid.get(lid)
            cand_prov.append({
                "chunk_id": lid,
                "document_id": (r.chunks[r._by_lineage[lid]].doc_id
                                if lid in r._by_lineage else None),
                "source": "graph_1hop" if g else "hybrid",
                "sparse_rank": sparse_rank.get(lid),
                "dense_rank": dense_rank.get(lid),
                "hybrid_rank": seed_rank,
                "rrf_score": (round(fr.get("score", 0.0), 8) if fr else None),
                "graph_distance": g["graph_distance"] if g else None,
                "graph_seed": g["graph_seed"] if g else None,
                "graph_relation": g["graph_relation"] if g else None,
                "graph_direction": g["graph_direction"] if g else None,
                "graph_path": g["graph_path"] if g else None,
                "candidate_rank_before_rerank": pre_rerank_rank,
                "final_rerank_rank": (lids.index(lid) + 1)
                                     if lid in lids else None,
                "final_rerank_score": (round(final_score_by_lid.get(lid, 0.0), 8)
                                       if lid in final_score_by_lid else None),
            })
        tgt_prov = next((c for c in cand_prov if c["chunk_id"] == tgt), None)
        full_new_ids = {rec["chunk_id"] for rec in full_new_prov}
        prov_rows.append({
            "query_id": qid,
            "gold_chunk": tgt,
            "family": item.category,
            "a0_rank": a0.get(qid, {}).get("target_rank"),
            "gcg_generated": tgt in set(pool_lids),
            "generated_by_graph": tgt in full_new_ids,
            "target_in_pool_after_cap": tgt in pool_lids,
            "lost_by_cap": (tgt in full_new_ids) and (tgt not in pool_lids),
            "target_pre_rerank_rank": (tgt_prov["candidate_rank_before_rerank"]
                                       if tgt_prov else None),
            "target_final_rank": first,
            "target_relation": (tgt_prov["graph_relation"] if tgt_prov else None),
            "target_direction": (tgt_prov["graph_direction"] if tgt_prov else None),
            "target_seed": (tgt_prov["graph_seed"] if tgt_prov else None),
            "n_candidates": len(pool_lids),
            "n_seeds": len(top_lids),
            "n_new": len(new_prov),
            "answer_correct": None,        # filled after judge
            "answer_overall": None,
            "candidates": cand_prov,
        })

        m = {}
        for k in k_set:
            rl = lids[:k]
            m[str(k)] = {
                "system": SYSTEM, "query_id": qid, "k": k,
                "recall": metrics.recall_at_k(rl, [tgt], k),
                "precision": metrics.precision_at_k(rl, [tgt], k),
                "hit": metrics.hit_rate_at_k(rl, [tgt], k),
                "mrr": metrics.reciprocal_rank(rl, [tgt]),
                "ndcg": metrics.ndcg_at_k(rl, [tgt], k),
            }
        a0r = a0.get(qid, {}).get("target_rank")
        rows.append({
            "query_id": qid,
            "question": q,
            "system": SYSTEM,
            "category": item.category,
            "target": tgt,
            "a0_30_target_rank": a0r,
            "a0_30_target_in_top5": (a0r is not None and a0r <= 5),
            "pool_seeds": len(top_lids),
            "graph_new_candidates": len(new_cands),
            "pool_total": len(pool),
            "target_in_pool": tgt_in_pool,
            "retrieved_top30": lids[:30],
            "retrieved_top20": lids[:20],
            "target_rank": first,
            "metrics": m,
            "rerank_status": meta.get("status"),
            "rerank_pool": meta.get("pool"),
        })
        log(f"  {qid}: new={len(new_cands)} pool={len(pool)} "
            f"tgt_in_pool={tgt_in_pool} pre={prov_rows[-1]['target_pre_rerank_rank']} "
            f"final={first} (A0-30 rank={a0r})")

    # ---- persist retrieval rows ------------------------------------------
    E.OUT_PER_QUERY.mkdir(parents=True, exist_ok=True)
    p = E.OUT_PER_QUERY / f"retrieval_{SYSTEM}.jsonl"
    with open(p, "w") as f:
        for r_ in rows:
            f.write(json.dumps(r_, ensure_ascii=False) + "\n")
    log(f"wrote {p}  ({len(rows)} rows)")

    # ---- persist spec §1/§6/§9 provenance (per-query, answer filled later) ---
    DIAG = E.EVAL_RESULTS / "diagnostic"
    DIAG.mkdir(parents=True, exist_ok=True)
    _prov_path = DIAG / "gcg_provenance.jsonl"
    with open(_prov_path, "w") as f:
        for pr in prov_rows:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")
    log(f"wrote {_prov_path}  ({len(prov_rows)} queries, "
        f"{sum(len(pr['candidates']) for pr in prov_rows)} candidates)")

    # ---- answer generation (neutral) over top-5 ---------------------------
    top5_by_id = {}
    for r_ in rows:
        r_["top5"] = (r_["retrieved_top30"])[:GEN.CONTEXT_WINDOW]
    gen_rows = []
    for r_ in rows:
        item = items[r_["query_id"]]
        chunks = []
        for lid in r_["top5"]:
            idx = r._by_lineage.get(lid)
            if idx is None:
                continue
            c = r.chunks[idx]
            chunks.append({"lineage_id": c.lineage_id, "doc_id": c.doc_id,
                           "text": c.text})
        gen = GEN.generate(r_["question"], chunks, cfg)
        gen.update({
            "query_id": item.query_id,
            "system": SYSTEM,
            "category": item.category,
            "target": item.target_lineage_id,
            "reference_answer": item.reference_answer,
            "reference_basis": "target_chunk_text",
            "retrieval_target_rank": r_["target_rank"],
            "target_in_context": any(
                c["lineage_id"] == item.target_lineage_id for c in chunks),
            "a0_30_target_rank": r_["a0_30_target_rank"],
            "graph_evidence": GEN.format_graph_evidence(item.gold_edges),
            "intended_relation": item.intended_relation,
            "intended_direction": item.intended_direction,
        })
        gen_rows.append(gen)
    E.OUT_GENERATION.mkdir(parents=True, exist_ok=True)
    with open(E.OUT_GENERATION / f"generation_{SYSTEM}.jsonl", "w") as f:
        for g in gen_rows:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    log(f"generation done; wrote generation_{SYSTEM}.jsonl")

    # ---- judge (gpt-oss + escalator) -------------------------------------
    out = AM.run([items[qid] for qid in eligible], {SYSTEM: gen_rows},
                 run_judge=True, judge_model=cfg.judge_model,
                 out_dir=E.OUT_GENERATION)
    srows = out[SYSTEM]
    n = len(srows)
    ok = sum(1 for g in srows if bool((g.get("judge") or {}).get("correct")))
    log(f"judge: correct = {ok}/{n}")
    # backfill answer correctness into the provenance rows (spec §6)
    judge_by_qid = {g["query_id"]: (g.get("judge") or {}) for g in srows}
    for pr in prov_rows:
        j = judge_by_qid.get(pr["query_id"], {})
        pr["answer_correct"] = bool(j.get("correct"))
        pr["answer_overall"] = j.get("overall_score")
    with open(_prov_path, "w") as f:
        for pr in prov_rows:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")
    E.OUT_AGGREGATE.mkdir(parents=True, exist_ok=True)
    ans_agg = AM.aggregate({SYSTEM: srows})
    import csv
    with open(E.OUT_AGGREGATE / f"answer_aggregate_{SYSTEM}.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "metric", "value", "n"])
        w.writeheader()
        for a in ans_agg:
            w.writerow(a)
    log(f"[done] {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
