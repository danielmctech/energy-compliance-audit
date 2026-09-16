"""Build the pre-experiment baseline diagnostic (spec sections 2, 3, 4, 16,
18 of "Pre-Experiment Fixes Before Graph-Assisted Candidate Generation").

Scope (exactly the two frozen reference systems):

    Hybrid   : sparse + dense + RRF (top-20 pool, graph-boost annotation,
               no LLM re-ranking)
    A0-30    : sparse + dense + RRF (top-30 pool) + listwise LLM reranker
               over ALL 30 (the ``hybrid_rerank_30_full`` clean baseline,
               whose post-permutation ranks are already on disk)

What it produces (under notebooks/data/evaluation/diagnostic/):

    run_config.json           run id + full frozen configuration (spec 16)
    candidate_pools.jsonl     per (system, query): the candidate pool
                              BEFORE reranking, with provenance per
                              candidate (sparse_rank, dense_rank,
                              rrf_score, hybrid_rank, graph_distance,
                              graph_relations) + final rank after
                              reranking (spec 2, 4)
    failure_analysis.json     per (system, query): failure class
                              A/B/C/D/S (spec 3)
    metrics.json              aggregate + per-family tables (spec 18)
    diagnostic_report.md      the human-readable report (spec 18)

No LLM calls are made: the A0-30 post-rerank order is loaded from
per_query/retrieval_hybrid_rerank_30_full.jsonl (already produced), the
final ``hybrid`` order from per_query/retrieval_hybrid.jsonl, and judge
correctness from generation/answers_{system}.jsonl.  The pre-rerank pools
are recomputed from the (deterministic) sparse + dense + RRF pipeline.

Usage:
    PYENV_VERSION=energy-audit ENERGY_AUDIT_ROOT=$(pwd) \
        python scripts/build_baseline_diagnostic.py
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os_env_done = False  # noqa: F841

import os

os.environ.setdefault("ENERGY_AUDIT_ROOT", str(ROOT))

import evaluation.config as E          # noqa: E402
import evaluation.benchmark as BM     # noqa: E402

DIAG = E.EVAL_RESULTS / "diagnostic"

#: the two frozen reference systems (spec 1 / 18)
SYSTEMS = {
    "hybrid":   {"per_query": "retrieval_hybrid.jsonl",
                 "answers": "answers_hybrid.jsonl",
                 "pool_n": 20, "label": "Hybrid"},
    "a0_30":    {"per_query": "retrieval_hybrid_rerank_30_full.jsonl",
                 "answers": "answers_hybrid_rerank_30_full.jsonl",
                 "pool_n": 30, "label": "A0-30"},
}
CAND_KS = (10, 20, 30, 50)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_jsonl(p: Path) -> List[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def run_id() -> str:
    return time.strftime("baseline_diag_%Y%m%d_%H%M%S")


def build_run_config(rid: str, n_items: int) -> dict:
    cfg = E.EvalConfig()
    import platform
    bench_p = E.OUT_PER_QUERY / "benchmark.jsonl"
    return {
        "run_id": rid,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "frozen-baseline diagnostic preceding GCG (spec 18)",
        "benchmark": {
            "path": str(bench_p),
            "n_queries": n_items,
            "sha256": _sha256(bench_p),
        },
        "retrieval": {
            "sparse": "regex tokenizer over corpus articles (src/retrieval/_corpus.py)",
            "dense": cfg and "all-MiniLM-L6-v2",
            "fusion": "RRF k=60 (src/retrieval/fusion.py)",
            "graph_boost": "0.05 annotation on 1-2 hop neighbours (hybrid only, reranking-free)",
            "pool_size": {"hybrid": 20, "a0_30": 30},
        },
        "reranker": {
            "model": "qwen3.8:27b (listwise prompt v2, temp 0.0)",
            "hybrid": "none",
            "a0_30": "max_candidates = 30 (whole pool; cache key distinct per src/reranking/__init__.py:119-126)",
        },
        "answer_generator": {
            "model": cfg.llm_model, "temperature": cfg.llm_temperature,
            "max_tokens": cfg.llm_max_tokens, "context_window": 5,
        },
        "answer_judge": {
            "primary": cfg.judge_model,
            "escalator": cfg.judge_escalator,
            "confidence_threshold": cfg.judge_confidence_threshold,
            "escalate_scores": list(cfg.judge_escalate_scores),
        },
        "python": platform.python_version(),
        "frozen": [
            "embedding model", "sparse config", "dense config", "RRF",
            "reranker model+prompt", "answer generator", "answer judge",
            "benchmark queries", "gold labels",
        ],
    }


def _sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main() -> int:
    t0 = time.time()
    DIAG.mkdir(parents=True, exist_ok=True)
    items = BM.build_benchmark()
    log(f"n_items={len(items)}  diag_dir={DIAG}")
    by_qid = {it.query_id: it for it in items}

    # ---- retrieval objects -------------------------------------------------
    from retrieval import Retriever
    r = Retriever(dense_models=["base"],
                  dense_specs={"base": "all-MiniLM-L6-v2"})

    # on-disk final orders + judge flags
    finals: Dict[str, Dict[str, List[str]]] = {}
    correct_flag: Dict[str, Dict[str, bool]] = {}
    for sys_name, spec in SYSTEMS.items():
        pq = {row["query_id"]: row
              for row in load_jsonl(E.OUT_PER_QUERY / spec["per_query"])}
        ans = {row["query_id"]: row
               for row in load_jsonl(E.OUT_GENERATION / spec["answers"])}
        finals[sys_name] = {}
        correct_flag[sys_name] = {}
        for qid in by_qid:
            row = pq.get(qid)
            if row is None:
                log(f"!! {sys_name}: missing per_query row for {qid}")
                continue
            lids = (row.get("retrieved_top30") or
                    row.get("retrieved_top20") or [])
            finals[sys_name][qid] = lids
            a = ans.get(qid)
            correct_flag[sys_name][qid] = bool(
                (a or {}).get("judge", {}).get("correct")) if a else False
    log("loaded final orders + judge flags for "
        f"{list(finals.keys())}")

    # ---- pools, provenance, failure classes --------------------------------
    pool_rows: List[dict] = []
    failure: List[dict] = []
    cand_hit: Dict[str, int] = defaultdict(int)
    fam_count: Dict[str, int] = defaultdict(int)
    for item in items:
        qid, q = item.query_id, item.question
        tgt = item.target_lineage_id
        # deterministic, cache-free sparse + dense + RRF pools per system
        # (hybrid: cap 20; a0_30: cap 30) -- exactly the pools each pipeline
        # feeds its next stage before reranking.
        pools = {}
        for sys_name, spec in SYSTEMS.items():
            n = spec["pool_n"]
            top_lids, fused, _ = r._hybrid_pool(q, n)
            neigh = [g for g in r.search_graph(q, 30)
                     if g[1] >= 1 and g[0] in set(top_lids)]
            edge_by = {g[0]: (g[1], list(g[2])) for g in neigh}
            # full sparse / dense rank lookups for provenance
            sparse_rank = {lid: i + 1
                           for i, (lid, _s) in enumerate(r.search_sparse(q, 50))}
            dense_rank = {}
            for name, ranked in r.search_dense(q, 50).items():
                for i, (lid, _s) in enumerate(ranked):
                    dense_rank.setdefault(lid, i + 1)
            cands = []
            for hybrid_rank, lid in enumerate(top_lids, start=1):
                fr = fused.get(lid, {})
                c = {
                    "chunk_id": lid,
                    "sparse_rank": sparse_rank.get(lid),
                    "dense_rank": dense_rank.get(lid),
                    "rrf_score": round(fr.get("score", 0.0), 8),
                    "hybrid_rank": hybrid_rank,
                    "graph_distance": edge_by[lid][0] if lid in edge_by else None,
                    "graph_relations": edge_by[lid][1] if lid in edge_by else [],
                }
                cands.append(c)
            pools[sys_name] = {
                "pool_lids": top_lids,
                "target_rank_before_rerank": (top_lids.index(tgt) + 1)
                                              if tgt in top_lids else None,
                "candidates": cands,
            }
        # final ranks + classes
        for sys_name, spec in SYSTEMS.items():
            final = finals[sys_name].get(qid, [])
            top5 = final[:5]
            in_pool = pools[sys_name]["target_rank_before_rerank"] is not None
            final_rank = final.index(tgt) + 1 if tgt in final else None
            corr = correct_flag[sys_name].get(qid, False)
            if corr and (tgt in top5):
                cls, note = "S", "answer correct; target in final top-5"
            elif corr and tgt not in top5:
                cls, note = "D", "answer correct despite target not in top-5"
            elif not in_pool:
                cls, note = "A", "target absent from candidate pool " \
                             "(candidate-generation failure)"
            elif final_rank is None or final_rank > 5:
                cls, note = "B", "target in pool but not in final top-5 " \
                               "(reranking failure)"
            else:
                cls, note = "C", "target in final top-5 but answer incorrect " \
                               "(answer-generation/judging failure)"
            # target source attribution for the GCG question (spec 20)
            if in_pool:
                c0 = pools[sys_name]["candidates"][
                    pools[sys_name]["pool_lids"].index(tgt)]
                src = []
                if c0["sparse_rank"]:
                    src.append("sparse")
                if c0["dense_rank"]:
                    src.append("dense")
                found_by = "+".join(src) if src else "unknown"
            else:
                found_by = None
            pool_rows.append({
                "run_id": None,  # filled after run_config
                "system": sys_name,
                "query_id": qid,
                "category": item.category,
                "target": tgt,
                "candidate_pool_size": len(pools[sys_name]["pool_lids"]),
                "target_rank_before_rerank":
                    pools[sys_name]["target_rank_before_rerank"],
                "target_final_rank": final_rank,
                "target_found_by": found_by,
                "candidates": pools[sys_name]["candidates"],
            })
            failure.append({
                "system": sys_name,
                "query_id": qid,
                "category": item.category,
                "target": tgt,
                "failure_class": cls,
                "target_in_candidate_pool": in_pool,
                "target_rank_before_rerank":
                    pools[sys_name]["target_rank_before_rerank"],
                "target_final_rank": final_rank,
                "answer_correct": corr,
                "note": note,
            })
            # candidate recall tallies (pre-rerank: target rank within pool)
            rank_before = pools[sys_name]["target_rank_before_rerank"]
            fam = item.category
            fam_count[f"{sys_name}|{fam}"] += 1
            pool_size = len(pools[sys_name]["pool_lids"])
            for k in CAND_KS:
                cap = min(k, pool_size)
                if rank_before is not None and rank_before <= cap:
                    cand_hit[f"{sys_name}|{fam}|@{k}"] += 1
                    cand_hit[f"{sys_name}|ALL|@{k}"] += 1
    log(f"pools + provenance + failure classes computed over "
        f"{len(items)} queries x {len(SYSTEMS)} systems")

    # ---- metrics (final-rank) per family + overall ------------------------
    def _final_stats(sys_name: str) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = defaultdict(dict)
        pq_rows = load_jsonl(E.OUT_PER_QUERY / SYSTEMS[sys_name]["per_query"])
        for row in pq_rows:
            fam = row["category"]
            for k in (1, 5, 10):
                m = row["metrics"].get(str(k)) or {}
                if m.get("recall") is not None:
                    out[fam].setdefault(f"recall@{k}", []).append(m["recall"])
            m10 = row["metrics"].get("10") or {}
            for mm in ("mrr", "ndcg"):
                if m10.get(mm) is not None:
                    out[fam].setdefault(f"{mm}@10", []).append(m10[mm])
        return out

    fams = sorted({it.category for it in items})
    metrics_doc = {"overall": {}, "per_family": {}, "failure_attribution": {}}
    for sys_name in SYSTEMS:
        fs = _final_stats(sys_name)
        agg = {
            "recall@1": _mean(fs, None, "recall@1"),
            "recall@5": _mean(fs, None, "recall@5"),
            "recall@10": _mean(fs, None, "recall@10"),
            "mrr@10": _mean(fs, None, "mrr@10"),
            "ndcg@10": _mean(fs, None, "ndcg@10"),
            "candidate_recall@10": _cand("ALL", "10", sys_name, len(items), cand_hit),
            "candidate_recall@20": _cand("ALL", "20", sys_name, len(items), cand_hit),
            "candidate_recall@30": _cand("ALL", "30", sys_name, len(items), cand_hit),
            "candidate_recall@50": _cand("ALL", "50", sys_name, len(items), cand_hit),
            "answer_correct": sum(
                1 for r_ in failure
                if r_["system"] == sys_name and r_["answer_correct"]) /
                max(len(items), 1),
        }
        metrics_doc["overall"][SYSTEMS[sys_name]["label"]] = agg
        pf = {}
        for fam in fams:
            n = fam_count.get(f"{sys_name}|{fam}", 0)
            if n == 0:
                continue
            pf[fam] = {
                "n": n,
                "recall@1": _mean(fs, fam, "recall@1"),
                "recall@5": _mean(fs, fam, "recall@5"),
                "recall@10": _mean(fs, fam, "recall@10"),
                "mrr@10": _mean(fs, fam, "mrr@10"),
                "ndcg@10": _mean(fs, fam, "ndcg@10"),
                "candidate_recall@10": _cand(fam, "10", sys_name, n, cand_hit),
                "candidate_recall@20": _cand(fam, "20", sys_name, n, cand_hit),
                "candidate_recall@30": _cand(fam, "30", sys_name, n, cand_hit),
                "candidate_recall@50": _cand(fam, "50", sys_name, n, cand_hit),
                "answer_correct": sum(
                    1 for r_ in failure
                    if r_["system"] == sys_name and r_["category"] == fam
                    and r_["answer_correct"]) / max(n, 1),
            }
        metrics_doc["per_family"][SYSTEMS[sys_name]["label"]] = pf
        fa = defaultdict(lambda: defaultdict(int))
        for r_ in failure:
            if r_["system"] == sys_name:
                fa[r_["category"]][r_["failure_class"]] += 1
                fa["ALL"][r_["failure_class"]] += 1
        metrics_doc["failure_attribution"][SYSTEMS[sys_name]["label"]] = {
            k: dict(v) for k, v in fa.items()}
    # GCG primary question (spec 20): queries whose target is absent from the
    # 30-pool of the plain hybrid pool but present after the same 30 RRF
    # (trivially 0 here, since the 30 RRF pool IS the GCG-eligible candidate
    # space).  More useful: how many are absent from the hybrid (no-rerank)
    # system's top-20 vs present in the A0-30 pool -- i.e. pool-depth effect.
    metrics_doc["pool_depth_effect"] = {
        "hybrid_pool_n": 20, "a0_30_pool_n": 30,
        "target_in_hybrid_20pool": sum(
            1 for r_ in failure if r_["system"] == "hybrid"
            and r_["target_in_candidate_pool"]),
        "target_in_a0_30_30pool": sum(
            1 for r_ in failure if r_["system"] == "a0_30"
            and r_["target_in_candidate_pool"]),
    }

    # ---- persist ------------------------------------------------------------
    rid = run_id()
    run_cfg = build_run_config(rid, len(items))
    for r_ in pool_rows:
        r_["run_id"] = rid
    (DIAG / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2, ensure_ascii=False))
    with open(DIAG / "candidate_pools.jsonl", "w") as f:
        for r_ in pool_rows:
            f.write(json.dumps(r_, ensure_ascii=False) + "\n")
    (DIAG / "failure_analysis.json").write_text(
        json.dumps({"run_id": rid,
                    "definitions": {
                        "A": "target absent from candidate pool "
                            "(candidate-generation failure)",
                        "B": "target in pool but not in final top-5 "
                            "(reranking failure)",
                        "C": "target reachable but answer incorrect "
                            "(answer-generation/judging failure)",
                        "D": "answer correct despite target not in final "
                            "top-5 (useful diagnostic, not automatic "
                            "retrieval success)",
                        "S": "answer correct AND target in final top-5 "
                            "(success; not a failure class)",
                    },
                    "rows": failure},
                   indent=2, ensure_ascii=False, default=str))
    (DIAG / "metrics.json").write_text(
        json.dumps(metrics_doc, indent=2, ensure_ascii=False, default=str))
    _write_report(DIAG, rid, run_cfg, items, metrics_doc, failure,
                  by_qid, cand_hit, fam_count)
    log(f"wrote run_config.json, candidate_pools.jsonl "
        f"({len(pool_rows)} rows), failure_analysis.json ({len(failure)} "
        f"rows), metrics.json, diagnostic_report.md")
    log(f"[done] {time.time() - t0:.1f}s")
    return 0


def _mean(fs, fam, key):
    if fam is None:
        vals = [x for k in fs.values() for x in k.get(key, [])]
    else:
        vals = fs.get(fam, {}).get(key, [])
    return round(sum(vals) / len(vals), 4) if vals else None


def _cand(fam, k, sys_name, n, cand_hit):
    key = f"{sys_name}|{fam}|@{k}"
    return round(cand_hit.get(key, 0) / max(n, 1), 4)


def _write_report(d: Path, rid: str, run_cfg: dict, items: list,
                  metrics_doc: dict, failure: List[dict],
                  by_qid: dict, cand_hit, fam_count) -> None:
    fams = sorted({it.category for it in items})
    lines = []
    lines.append("# Baseline diagnostic (pre-GCG)\n")
    lines.append(f"Run ID: `{rid}`  \nGenerated "
                 f"{run_cfg['timestamp_utc']}  \n"
                 f"Benchmark: {run_cfg['benchmark']['n_queries']} queries, "
                 f"sha256 "
                 f"`{run_cfg['benchmark']['sha256'][:16]}…`\n")
    lines.append("Frozen per spec sections 1 & 14/15: sparse/dense/RRF, "
                 "reranker model+prompt, answer generator "
                 f"({run_cfg['answer_generator']['model']}), judge "
                 f"({run_cfg['answer_judge']['primary']} + "
                 f"{run_cfg['answer_judge']['escalator']}).\n")

    def _fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

    def _tbl(row_of):
        out = ["| Metric | Hybrid | A0-30 |", "|---|---:|---:|"]
        for label in ("recall@1 | Recall@1", "recall@5 | Recall@5",
                      "recall@10 | Recall@10", "mrr@10 | MRR@10",
                      "ndcg@10 | nDCG@10",
                      "candidate_recall@10 | Candidate Recall@10",
                      "candidate_recall@20 | Candidate Recall@20",
                      "candidate_recall@30 | Candidate Recall@30",
                      "candidate_recall@50 | Candidate Recall@50",
                      "answer_correct | Answer Correct"):
            key, name = label.split(" | ")
            row = row_of(key)
            out.append(f"| {name} | {_fmt(row['Hybrid'])} | "
                       f"{_fmt(row['A0-30'])} |")
        return "\n".join(out)

    lines.append("## Overall (all 60 queries)\n")
    ov = metrics_doc["overall"]
    lines.append(_tbl(lambda k: {
        "Hybrid": ov["Hybrid"].get(k), "A0-30": ov["A0-30"].get(k)}))
    lines.append("")
    lines.append("## Per-family\n")
    for fam in fams:
        h = metrics_doc["per_family"]["Hybrid"].get(fam)
        a = metrics_doc["per_family"]["A0-30"].get(fam)
        lines.append(f"### {fam}\n")
        rows = []
        for label, key in [("Recall@1", "recall@1"), ("Recall@5", "recall@5"),
                           ("Recall@10", "recall@10"), ("MRR@10", "mrr@10"),
                           ("nDCG@10", "ndcg@10"),
                           ("Cand Recall@10", "candidate_recall@10"),
                           ("Cand Recall@20", "candidate_recall@20"),
                           ("Cand Recall@30", "candidate_recall@30"),
                           ("Cand Recall@50", "candidate_recall@50"),
                           ("Answer Correct", "answer_correct")]:
            rows.append(f"| {label} | {_fmt(h and h.get(key))} | "
                        f"{_fmt(a and a.get(key))} |")
        lines.append("| Metric | Hybrid | A0-30 |\n|---|---:|---:|"
                     + "\n".join(rows) + "\n")
    lines.append("## Failure attribution (spec section 3)\n")
    lines.append("Classes: **A** absent from pool · **B** in pool, "
                 "not top-5 · **C** reachable, wrong answer · "
                 "**D** correct despite miss · **S** success (correct + "
                 "top-5)\n")
    for sys_label in ("Hybrid", "A0-30"):
        fa = metrics_doc["failure_attribution"][sys_label]
        lines.append(f"### {sys_label}\n")
        hdr = ("| Family | A: absent | B: rerank | C: answer | "
               "D: correct-on-miss | S: success |")
        sep = "|---|---:|---:|---:|---:|---:|"
        lines.append(hdr)
        lines.append(sep)
        for fam in fams + ["ALL"]:
            v = fa.get(fam) or {}
            lines.append(f"| {fam} | {v.get('A', 0)} | {v.get('B', 0)} | "
                         f"{v.get('C', 0)} | {v.get('D', 0)} | "
                         f"{v.get('S', 0)} |")
        lines.append("")
    lines.append("## Notes\n")
    lines.append("- **Candidate Recall@K is computed on the pre-rerank pool "
                 "(spec section 2); final Recall/MRR/nDCG are on the "
                 "system's final ordering.**")
    lines.append("- **Pool depths differ by design**: Hybrid's pool is 20 "
                 "(`max(k,5)`, k=20); A0-30's pool is 30 (the re-ranker pool "
                 "depth). This asymmetry is part of what the candidate "
                 "column captures and is preserved for the next GCG "
                 "experiment.")
    lines.append("- **Graph relations are preserved as typed/directional "
                 "metadata** (per spec section 7); they do *not* feed a "
                 "scoring function in either baseline reranker here.")
    lines.append("- **Temporal-family queries are flagged diagnostic-only** "
                 "for spec section 8: the graph schema lacks "
                 "SUPERSEDES/effective-date/in-force semantics, so "
                 "performance on `temporal_version` is *not* evidence "
                 "about graph-assisted retrieval.")
    lines.append("- **`non_relational_semantic` family** is audited in a "
                 "separate pass (spec section 9); until then its "
                 "Recall@K numbers are reported but not used to draw "
                 "retrieval conclusions.")
    (d / "diagnostic_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
