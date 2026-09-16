"""Spec 08 report analysis — reads the on-disk evaluation artifacts
(retrieval_*.jsonl, answers_*.jsonl, heldout_benchmark.jsonl,
benchmark_audit.jsonl) and produces:

  evaluation/heldout_per_query_comparison.json
  evaluation/heldout_failure_attribution.json
  evaluation/heldout_statistics.json

Then a thin printout of key numbers for the report.

No LLM, no tuning, purely deterministic over existing artifacts.
"""
from __future__ import annotations
import json, math, os, random, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "scripts"):
    if str(ROOT/p) not in sys.path: sys.path.insert(0, str(ROOT/p))
os.environ.setdefault("ENERGY_AUDIT_ROOT", str(ROOT))

import evaluation.benchmark as BM   # noqa: E402
import evaluation.config as E       # noqa: E402
import evaluation.metrics as M      # noqa: E402

OUT = E.EVAL_RESULTS
SYSTEMS = ["b0", "g1", "g2", "g2_nogc"]
K_SET = (1, 3, 5, 10, 20)
FAMS = ["single_document","non_relational_semantic","one_hop_relational",
        "two_hop_relational","relation_direction","temporal_version",
        "graph_distractor","multi_document_synthesis"]


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def mean(xs): return (sum(xs)/len(xs)) if xs else None


def _rank_of(row: dict) -> Optional[int]:
    """1-based rank of the single gold target in the row's top-30 (or None)."""
    lids = row.get("retrieved_top30") or []
    tgt = row.get("target")
    if tgt in lids:
        return lids.index(tgt) + 1
    return row.get("target_final_rank")


def _recall_at(row: dict, k: int) -> float:
    lids = (row.get(f"retrieved_top_{k}")
            or (row.get("retrieved_top30") or [])[:k])
    return 1.0 if row.get("target") in lids else 0.0


def _mrr(row: dict, k: int) -> float:
    rank = _rank_of(row)
    return (1.0 / rank) if (rank is not None and rank <= k) else 0.0


def _ndcg_at(row: dict, k: int) -> float:
    rank = _rank_of(row)
    return (1.0 / math.log2(rank + 1)) if (rank is not None and rank <= k) else 0.0


def retrieval_metrics(rows: list[dict], k=10) -> dict:
    """Aggregate retrieval metrics for a set of per-query rows.

    Computed directly from each row's ``retrieved_top_{k}`` list and its
    single gold ``target`` (spec §10 candidate/final retrieval metrics).
    """
    in_pool = [int(bool(r.get("target_in_pool"))) for r in rows]
    return {
        "n": len(rows),
        "target_in_pool_rate": mean(in_pool),
        "recall_at_1": mean([_recall_at(r, 1) for r in rows]),
        "recall_at_5": mean([_recall_at(r, 5) for r in rows]),
        "recall_at_10": mean([_recall_at(r, 10) for r in rows]),
        "mrr_at_10": mean([_mrr(r, 10) for r in rows]),
        "ndcg_at_10": mean([_ndcg_at(r, 10) for r in rows]),
    }


def answer_metrics(rows: list[dict]) -> dict:
    correct = [int(bool(r.get("judge", {}).get("correct"))) for r in rows]
    faithful = [int(bool(r.get("judge", {}).get("faithful"))) for r in rows]
    complete = [int(bool(r.get("judge", {}).get("complete"))) for r in rows]
    evsup = [int(bool(r.get("judge", {}).get("evidence_supported"))) for r in rows]
    unsup = [int(bool(r.get("judge", {}).get("unsupported_claims"))) for r in rows]
    grc = [r["judge"].get("graph_reasoning_correct") for r in rows if isinstance(r.get("judge", {}).get("graph_reasoning_correct"), bool)]
    oscore = [r["judge"].get("overall_score") for r in rows if isinstance(r.get("judge", {}).get("overall_score"), (int, float))]
    conf = [r["judge"].get("confidence") for r in rows if isinstance(r.get("judge", {}).get("confidence"), (int, float))]
    esc = [int(bool(r.get("judge", {}).get("escalated"))) for r in rows]
    n = len(rows)
    return {
        "n": n,
        "correct_rate": mean(correct),
        "faithful_rate": mean(faithful),
        "complete_rate": mean(complete),
        "evidence_supported_rate": mean(evsup),
        "unsupported_claims_rate": mean(unsup),
        "graph_reasoning_correct_rate": mean(grc) if grc else None,
        "overall_score_mean": mean(oscore),
        "confidence_mean": mean(conf),
        "escalation_rate": mean(esc),
        "empty_answer_rate": (sum(1 for r in rows if not (r.get("answer") or "").strip()) / n) if n else 0.0,
    }


def bootstrap_ci(rows, fn, n_boot=2000, alpha=0.05, seed=13):
    """Nonparametric bootstrap 95% CI of a per-row-mean metric.
    ``fn(rows)`` should return the mean of a numeric list built from rows.
    """
    if not rows: return None
    rng = random.Random(seed)
    n = len(rows)
    vals = []
    base = fn(rows)
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        vals.append(fn(sample))
    vals.sort()
    lo = vals[int((alpha/2)*n_boot)]
    hi = vals[int((1-alpha/2)*n_boot)]
    return {"mean": base, "ci95_low": lo, "ci95_high": hi, "n": n}


def mcnemar(lst_a: list[bool], lst_b: list[bool]):
    """Paired McNemar's test on per-query correct booleans.

    Returns a dict with the discordant-pair counts (a_correct/b_wrong and
    vice versa), the chi-square statistic (asymp), the exact binomial p
    when discordants < 25, and the test name used.
    """
    b_wins_ab = sum(1 for x, y in zip(lst_a, lst_b) if x and not y)
    a_wins_ba = sum(1 for x, y in zip(lst_a, lst_b) if not x and y)
    s = b_wins_ab + a_wins_ba
    if s == 0:
        return {"a_wins": b_wins_ab, "b_wins": a_wins_ba,
                "chi_sq": 0.0, "p_value": 1.0, "n_discordant": 0,
                "note": "no discordant pairs; systems agree on every query"}
    if s < 25:
        # exact binomial  p = 2 * P(X <= min(b,c) | X~Bin(s, 0.5))
        from math import comb
        k = min(b_wins_ab, a_wins_ba)
        p = 0.0
        for i in range(k+1):
            p += comb(s, i) * (0.5 ** s)
        p = min(1.0, 2.0 * p)
        chi2 = None
    else:
        chi2 = (b_wins_ab - a_wins_ba) ** 2 / float(s)
        # chi-square df=1 two-sided tail
        p = 0.5 * _chi2_sf(chi2, df=1)
    return {
        "b0_wins": b_wins_ab, f"other_wins": a_wins_ba,
        "n_discordant": s,
        "chi_sq": chi2,
        "p_value": float(p),
        "test": "mcnemar_exact" if s < 25 else "mcnemar_asymp",
    }


def _chi2_sf(x, df=1):
    """Approximate chi-square upper-tail (df=1) via erfc(sqrt(x/2))."""
    from math import sqrt, erfc
    return erfc(sqrt(x / 2.0))


def failure_class(row: dict, sysname: str) -> str:
    """Assign spec §12 primary failure class to one (system, query).

    A — Admission failure    (target not in pool).
    B — Ranking failure      (in pool, final rank > 5 — not in top-5 context).
    C — Answer failure       (target in top-5, judge correct=False).
    D — Correct              (judge correct=True; retrieval may have been imperfect).
    For graph systems additionally split:
        G1 — graph-expansion failure: target absent from pool, but present in
             the corpus for the doc (so expansion missed it).
        G2 — graph candidate prioritization failure: target in pool, final_rank > 5.
        G3 — graph-context demotion / graph-candidate promotion: target in
             top-5 but judge correct=False (answer failure attributed to context
             interaction).
        G4 — successful graph recovery: target in pool, final_rank <= 5, but the
             seed-rank (b0_target_rank) was absent/None, i.e. only graph made it
             reachable.
    We emit the primary class (A/B/C/D) plus the graph refinement when present.
    """
    judge_c = bool((row.get("judge") or {}).get("correct"))
    tgt_r = row.get("target_final_rank")  # 1-based; None if not in top30
    in_pool = bool(row.get("target_in_pool"))
    b0_r = row.get("b0_target_rank")
    graph_system = sysname in ("g1", "g2", "g2_nogc")

    if not in_pool:
        base = "A_admission"
    elif tgt_r is None or tgt_r > 5:
        base = "B_ranking"
    elif not judge_c:
        base = "C_answer"
    else:
        base = "D_correct"

    refinement = None
    if graph_system:
        if base == "A_admission":
            refinement = "G1_graph_expansion_failure"
        elif base == "B_ranking":
            refinement = "G2_graph_prioritization_failure"
        elif base == "C_answer":
            refinement = "G3_graph_context_failure"
        else:  # D
            if b0_r is None and tgt_r is not None:
                refinement = "G4_successful_graph_recovery"
            elif b0_r is None:
                refinement = "G4_successful_graph_recovery"
    return base, refinement


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load frozen 160-item set from on-disk artifact
    audit = load_jsonl(OUT / "benchmark_audit.jsonl")
    vh = {r["query_id"] for r in audit
          if r.get("set") == "heldout_174"
          and r["status"] in ("valid_exact", "valid_subarticle")}
    print(f"valid qids: {len(vh)}")
    items = BM.load_jsonl(OUT / "heldout_benchmark.jsonl", vh)
    byq = {it.query_id: it for it in items}

    # 2. Load retrieval + answers by (system, query)
    ret: dict[str, dict[str, dict]] = {s: {} for s in SYSTEMS}
    ans: dict[str, dict[str, dict]] = {s: {} for s in SYSTEMS}
    for s in SYSTEMS:
        for rec in load_jsonl(OUT / f"heldout_{s}_retrieval.jsonl"):
            ret[s][rec["query_id"]] = rec
        for rec in load_jsonl(OUT / f"answers_{s}.jsonl"):
            ans[s][rec["query_id"]] = rec

    # 3. Per-query combined rows (spec §13)
    per_query: dict[str, dict] = {}
    for q in vh:
        it = byq[q]
        row = {
            "query_id": q,
            "query_type": it.category,
            "difficulty": it.difficulty,
            "doc_id": it.doc_id,
            "question": it.question,
            "gold_chunks": it.gold_chunks,
            "gold_documents": it.gold_documents or [it.doc_id],
            "gold_entities": it.gold_entities,
            "gold_edges": it.gold_edges,
            "gold_path": it.gold_path,
            "hop_count": it.hop_count,
            "intended_relation": it.intended_relation,
            "intended_direction": it.intended_direction,
            "target_lineage_id": it.target_lineage_id,
            "systems": {},
        }
        for s in SYSTEMS:
            r = ret[s].get(q)
            a = ans[s].get(q)
            base, ref = (None, None)
            if r and a:
                base, ref = failure_class({**r, **a}, s)
            row["systems"][s] = {
                "retrieval": r,
                "answer": a,
                "failure_class": base,
                "graph_refinement": ref,
            }
        per_query[q] = row

    # 4. Aggregate per-system retrieval + answer metrics
    retrieval_agg = {}
    answer_agg = {}
    for s in SYSTEMS:
        rows = [ret[s][q] for q in vh if q in ret[s]]
        retrieval_agg[s] = retrieval_metrics(rows)
        arows = [ans[s][q] for q in vh if q in ans[s]]
        answer_agg[s] = answer_metrics(arows)

    # 5. Family breakdown
    FAM_AGG = {}
    for fam in FAMS:
        fam_qids = [q for q in vh if byq[q].category == fam]
        fam_row = {"n": len(fam_qids), "qids": fam_qids, "systems_retrieval": {}, "systems_answer": {}}
        for s in SYSTEMS:
            rows = [ret[s][q] for q in fam_qids if q in ret[s]]
            if rows:
                fam_row["systems_retrieval"][s] = retrieval_metrics(rows)
            arows = [ans[s][q] for q in fam_qids if q in ans[s]]
            if arows:
                fam_row["systems_answer"][s] = {
                    "n": len(arows),
                    "correct_rate": answer_metrics(arows)["correct_rate"],
                }
        FAM_AGG[fam] = fam_row

    # 6. Statistics: bootstrap CI on recall@10, MRR@10, answer-correct per system
    STAT = {}
    seed = 13
    for s in SYSTEMS:
        rrows = [ret[s][q] for q in vh if q in ret[s]]
        arows = [ans[s][q] for q in vh if q in ans[s]]
        STAT[s] = {
            "retrieval": {
                "recall_at_10": bootstrap_ci(rrows,
                    lambda rs: mean([_recall_at(r, 10) for r in rs]),
                    seed=seed),
                "mrr_at_10": bootstrap_ci(rrows,
                    lambda rs: mean([_mrr(r, 10) for r in rs]),
                    seed=seed),
                "target_in_pool_rate": bootstrap_ci(rrows,
                    lambda rs: mean([int(bool(r.get("target_in_pool"))) for r in rs]),
                    seed=seed),
            },
            "answer": {
                "correct_rate": bootstrap_ci(arows,
                    lambda rs: mean([int(bool(r.get("judge", {}).get("correct"))) for r in rs]),
                    seed=seed),
                "overall_score": bootstrap_ci(arows,
                    lambda rs: mean([r["judge"]["overall_score"]
                                    for r in rs
                                    if isinstance((r.get("judge") or {}).get("overall_score"), (int, float))]) or None,
                    seed=seed),
            },
        }

    # 7. Pairwise (B0 vs each graph system) per-query correct + McNemar
    PAIRS = {}
    for s in ["g1", "g2", "g2_nogc"]:
        a = [bool((ans["b0"].get(q, {}).get("judge") or {}).get("correct")) for q in vh]
        b = [bool((ans[s].get(q, {}).get("judge") or {}).get("correct")) for q in vh]
        mc = mcnemar(a, b)

        # retrieval pairwise: recall@10 / mrr@10 / target_in_pool
        def _r10(rows_): return [_recall_at(r, 10) for r in rows_]
        def _mrr10(rows_): return [_mrr(r, 10) for r in rows_]
        def _inpool(rows_): return [int(bool(r.get("target_in_pool", False))) for r in rows_]
        r_a = _r10([ret["b0"][q] for q in vh])
        r_b = _r10([ret[s][q] for q in vh])
        m_a = _mrr10([ret["b0"][q] for q in vh])
        m_b = _mrr10([ret[s][q] for q in vh])
        w = sum(1 for x, y in zip(r_a, r_b) if x < y)
        l = sum(1 for x, y in zip(r_a, r_b) if x > y)
        t = sum(1 for x, y in zip(r_a, r_b) if x == y)
        PAIRS[f"b0_vs_{s}"] = {
            "answer_correct": {
                "b0_correct": sum(a), f"{s}_correct": sum(b),
                "paired_mcnemar": mc,
            },
            "retrieval_recall_at_10": {
                "b0_mean": mean(r_a), f"{s}_mean": mean(r_b),
                "win_tie_loss": {f"{s}_wins(w)": w, "ties": t, "b0_wins(l)": l},
                "delta_mean": (mean(r_b) - mean(r_a)),
            },
            "retrieval_mrr_at_10": {
                "b0_mean": mean(m_a), f"{s}_mean": mean(m_b),
                "delta_mean": (mean(m_b) - mean(m_a)),
            },
            "retrieval_target_in_pool": {
                "b0_rate": mean(_inpool([ret["b0"][q] for q in vh])),
                f"{s}_rate": mean(_inpool([ret[s][q] for q in vh])),
            },
        }

    # 8. Graph-specific metrics: recovery, promotion/demotion vs b0, induced false positives
    GRAPH = {}
    for s in ["g1", "g2", "g2_nogc"]:
        n = len(vh)
        b0_hit = {}
        for q in vh:
            r = ret[s].get(q) or {}
            b0r = r.get("b0_target_rank")
            b0_hit[q] = (b0r is not None)
            _ = r
        # G4: system final hit but b0 not hit
        g4 = sum(1 for q in vh
                  if (ret[s][q].get("target_final_rank") is not None
                      and (ret[s][q].get("b0_target_rank") is None)))
        n_sys_hit = sum(1 for q in vh if ret[s][q].get("target_final_rank") is not None)
        n_b0_hit = sum(1 for q in vh if b0_hit[q])
        # G1: expansion failure (target not in pool by graph system but in b0)
        g1f = sum(1 for q in vh
                  if (not ret[s][q].get("target_in_pool")
                      and (ret[s][q].get("b0_target_rank") is not None)))
        # G2: prioritization failure (in pool, not in top-5) for graph systems
        g2f = sum(1 for q in vh
                  if (ret[s][q].get("target_in_pool")
                      and (ret[s][q].get("target_final_rank") is None
                           or ret[s][q].get("target_final_rank") > 5)
                      and (ret[s][q].get("b0_target_rank") is not None
                           and ret[s][q]["b0_target_rank"] <= 5)))
        # G3: graph-induced demotion vs b0 (top-5 in b0 but out of top-5 in graph)
        demote = sum(1 for q in vh
                     if (ret[s][q].get("b0_target_rank") is not None and ret[s][q]["b0_target_rank"] <= 5
                         and (ret[s][q].get("target_final_rank") is not None
                              and ret[s][q]["target_final_rank"] > 5)))
        promote = sum(1 for q in vh
                      if (ret[s][q].get("b0_target_rank") is None
                          or ret[s][q].get("b0_target_rank", 999) > 5)
                      and ret[s][q].get("target_final_rank") is not None
                      and ret[s][q]["target_final_rank"] <= 5)
        GRAPH[s] = {
            "n": n,
            "g4_successful_graph_recovery_count": g4,
            "g4_rate": g4 / n if n else 0.0,
            "graph_system_top5_hit_count": n_sys_hit,
            "b0_top5_hit_count": n_b0_hit,
            "delta_top5": n_sys_hit - n_b0_hit,
            "g1_expansion_failure_count": g1f,
            "g1_rate": g1f / n if n else 0.0,
            "g2_prioritization_failure_count": g2f,
            "g3_demotions": demote,
            "g3_promotions": promote,
            "net_top5_delta_vs_b0": n_sys_hit - n_b0_hit,
        }

    # 9. Failure-class attribution (A/B/C/D + graph refinement)
    FAILCOUNT = {s: Counter() for s in SYSTEMS}
    for q in vh:
        row = per_query[q]
        for s in SYSTEMS:
            base = row["systems"][s]["failure_class"]
            ref = row["systems"][s]["graph_refinement"]
            if base: FAILCOUNT[s][base] += 1
            if ref: FAILCOUNT[s][ref] += 1
    FAILATTR = {s: {"counts": dict(FAILCOUNT[s]),
                    "n": len(vh),
                    "rates": {k: v/len(vh) for k, v in FAILCOUNT[s].items()}}
                for s in SYSTEMS}

    # 10. Leakage check (already on disk) — read + verify
    leak = json.loads((OUT / "heldout_leakage_audit.json").read_text())
    checks = leak.get("checks", {})
    failing = [k for k, v in checks.items()
               if isinstance(v, dict) and v.get("pass") is not True]
    leak_check = {
        "file": "heldout_leakage_audit.json",
        "leak_audit_present": True,
        "all_checks_pass": bool(leak.get("all_checks_pass")),
        "n_audit_checks": len(checks),
        "failing_checks": failing,
        "n_heldout": len(vh),
        "diagnostic_14_query_ids": leak.get("diagnostic_set", {}).get("query_ids", []),
    }

    # 11. Anomaly: empty answers
    empty = []
    for s in SYSTEMS:
        for q in vh:
            r = ans[s].get(q, {})
            if not (r.get("answer") or "").strip():
                empty.append({"system": s, "query_id": q})
    ANOMALIES = {
        "empty_answers": empty,
        "note": "Rows where the generator returned empty answer. "
                "Likely cause: ollama_chat error path in generate() "
                "(returns '' on any exception). Check generation_errors.jsonl.",
    }

    # 12. Corpus / graph / models / config provenance
    PROVENANCE = {
        "corpus_total_chunks": 3652,
        "corpus_baseline": 3640,
        "corpus_recovered_chunks_in_stage1_repair": 12,
        "recovered_chunk_ids": [
            "data_act:article:2", "dora:article:3", "dora:article:35",
            "eidas_2:article:1", "eidas_2:article:16",
            "eidas_2:article:5a", "eidas_2:article:46e",
            "elec_dir:article:2", "elec_dir:article:40",
            "emd_reform:article:2", "nis2:article:46", "remit_ii:article:1",
        ],
        "generator_model": "qwen3.8:27b",
        "generator_temperature": 0.0,
        "judge_model": "gpt-oss:latest",
        "judge_escalator": "nemotron-3-nano:30b",
        "judge_confidence_threshold": 0.85,
        "frozen_systems": {
            "B0": "hybrid_rerank_30_full (A0-30)",
            "G1": "gcg_1hop_50 (30 seeds + 20 graph-new, budget 50)",
            "G2": "gcg_2hop_lever2 pool-51 split_12_9 + graph-ctx",
            "G2_noctx": "G2 minus graph context (spec §15 Q2 control)",
        },
        "generator_context_window": 5,
        "k_values": list(K_SET),
        "seed_heldout_assignment": 13,
        "seed_bootstrap": 13,
    }

    # --------------------------------------------------------------
    # Dump
    # --------------------------------------------------------------
    def clean(x):
        def _f(o):
            if o is None or isinstance(o, (int, float, bool, str)):
                return o
            if isinstance(o, dict):
                return {str(k): _f(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_f(v) for v in o]
            return str(o)
        return json.loads(json.dumps(x, default=_f))

    (OUT / "heldout_per_query_comparison.json").write_text(
        json.dumps(clean(per_query), ensure_ascii=False, indent=2))

    (OUT / "heldout_failure_attribution.json").write_text(
        json.dumps(clean(FAILATTR), ensure_ascii=False, indent=2))

    STAT["retrieval_agg"] = clean(retrieval_agg)
    STAT["answer_agg"] = clean(answer_agg)
    STAT["family_breakdown"] = clean(FAM_AGG)
    STAT["pairs"] = clean(PAIRS)
    STAT["graph_specific"] = clean(GRAPH)
    STAT["leakage_check"] = clean(leak_check)
    STAT["anomalies"] = clean(ANOMALIES)
    STAT["provenance"] = clean(PROVENANCE)
    (OUT / "heldout_statistics.json").write_text(
        json.dumps(STAT, ensure_ascii=False, indent=2))

    # --------------------------------------------------------------
    # Print compact summary
    # --------------------------------------------------------------
    print("\n=== per-system retrieval ===")
    for s in SYSTEMS:
        r = retrieval_agg[s]
        print(f"  {s}: in_pool={r['target_in_pool_rate']:.3f} "
              f"r@1={r['recall_at_1']:.3f} r@5={r['recall_at_5']:.3f} "
              f"r@10={r['recall_at_10']:.3f} mrr@10={r['mrr_at_10']:.3f} "
              f"ndcg@10={r['ndcg_at_10']:.3f}")

    print("\n=== per-system answer ===")
    for s in SYSTEMS:
        a = answer_agg[s]
        print(f"  {s}: correct={a['correct_rate']:.3f} faithful={a['faithful_rate']:.3f} "
              f"overall={a['overall_score_mean']:.2f} esc={a['escalation_rate']:.2f} "
              f"empty={a['empty_answer_rate']*100:.1f}%")

    print("\n=== family correct_rate ===")
    for fam in FAMS:
        row = FAM_AGG[fam]
        cs = [f"{s}:{(row['systems_answer'].get(s, {}).get('correct_rate') or 0):.2f}(n={len(row['qids'])})"
              for s in SYSTEMS if fam in row and row['systems_answer'].get(s, {}).get('correct_rate') is not None]
        print(f"  {fam}: " + "  ".join(cs))

    print("\n=== pairwise b0 vs graph ===")
    for name, pair in PAIRS.items():
        ac = pair["answer_correct"]
        mc = ac["paired_mcnemar"]
        r10 = pair["retrieval_recall_at_10"]; mrr10 = pair["retrieval_mrr_at_10"]
        print(f"  {name}: correct b0={ac['b0_correct']} vs {ac.get('b0_correct', '?') and [v for k,v in ac.items() if 'correct' in k and k!='b0_correct']} "
              f"| r@10 b0={r10['b0_mean']:.3f} other={r10.get('g1_mean', r10.get('g2_mean', r10.get('g2_nogc_mean'))):.3f} "
              f"| mrr@10 b0={mrr10['b0_mean']:.3f} other={mrr10.get('g1_mean', mrr10.get('g2_mean', mrr10.get('g2_nogc_mean'))):.3f}")
        print(f"    mcnemar: b0_wins={mc.get('b0_wins')} other_wins={mc.get('other_wins')} "
              f"n_d={mc.get('n_discordant')} chi2={mc.get('chi_sq')} p={mc.get('p_value'):.4f} [{mc.get('test')}]")

    print("\n=== failure class counts ===")
    for s in SYSTEMS:
        c = FAILATTR[s]["counts"]
        print(f"  {s}: {c}")

    print("\n=== graph-specific ===")
    for s in ["g1", "g2", "g2_nogc"]:
        g = GRAPH[s]
        print(f"  {s}: g4={g['g4_successful_graph_recovery_count']} "
              f"g1f={g['g1_expansion_failure_count']} g2f={g['g2_prioritization_failure_count']} "
              f"demote={g['g3_demotions']} promote={g['g3_promotions']} "
              f"net_top5={g['net_top5_delta_vs_b0']:+d}")

    print("\n=== anomalies ===")
    for e in ANOMALIES.get("empty_answers", []):
        print(f"  {e['system']} {e['query_id']}")

    print("\nArtifacts written to:")
    for f in ("heldout_per_query_comparison.json", "heldout_failure_attribution.json",
              "heldout_statistics.json"):
        p = OUT / f
        print(f"  {p}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
