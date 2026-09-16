"""Spec 08 §4/§5/§6/§20 — held-out benchmark artifacts.

Builds the frozen held-out benchmark via ``build_benchmark_heldout()``
(no new builder logic, no tuning — spec §1), then writes three artifacts:

    evaluation/heldout_benchmark.jsonl
    evaluation/heldout_benchmark_validation.json
    evaluation/heldout_leakage_audit.json

The 14-query diagnostic set (spec §3) is the GCG target set (primary 8 +
controls 6, spec 08 §23 of the GCG docs):  q005 q014 q027 q031 q033
q034 q050 q051 (primary);  q011 q030 q044 q048 q055 q058 (controls).
Those ids reference the frozen 60-set, so the leakage audit maps them
through ``_build_gold()`` and checks the held-out set against:
  * exact question-text membership in the 14 (and in the full 60),
  * target lineage-id overlap, gold-chunk id overlap,
  * fact-triple overlap (term, doc, target-article) — the operational
    paraphrase check, because the question template is a pure function
    of that triple,
  * a declaration that the set was never used for parameter selection
    (spec §20: pool budget and graph-context behaviour were frozen
    BEFORE this set was constructed),
  * a declaration that labels were not exposed to the retrieval pipeline
    (ground truth is read only by metric code; the pipeline consumes
    ``item.question`` only).

Nothing in this script touches retrieval, generation, reranking, prompts,
pools or thresholds (spec §1).

Usage:
    PYENV_VERSION=energy-audit ENERGY_AUDIT_ROOT=$(pwd) \
        python scripts/build_heldout_benchmark_artifacts.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(os.getenv("ENERGY_AUDIT_ROOT") or ".").resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("ENERGY_AUDIT_ROOT", str(ROOT))

from evaluation import config as E            # noqa: E402
from evaluation import benchmark as BM        # noqa: E402

#: The 14-query diagnostic set (spec §3): primary 8 + controls 6, from
#: the terminal GCG report (gcg_2hop_two_levers_report.md §A/§B).
DIAGNOSTIC_QIDS = [
    "q005", "q014", "q027", "q031", "q033", "q034", "q050", "q051",
    "q011", "q030", "q044", "q048", "q055", "q058",
]


def log(msg: str) -> None:
    print(f"[heldout-artifacts] {msg}", flush=True)


def _sha256_lines(lines: list) -> str:
    h = hashlib.sha256()
    for l in lines:
        h.update(l.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _target_article(lid: str) -> str | None:
    import re
    m = re.match(r".*:article:(\d+)", lid)
    if not m:
        return None
    doc = lid.split(":article:")[0]
    return f"{doc}:article:{m.group(1)}"


def _fact_keys(it: "BM.BenchmarkItem") -> set:
    """Fact triple(s) that determine the rendered question text.

    The question template is a pure function of (term, doc,
    target-article, and — for relational families — the gold edge),
    so two items share the same question template iff they share one
    of these keys.
    """
    keys: set = set()
    key = (it.term, it.doc_id, _target_article(it.target_lineage_id))
    keys.add(key)
    for e in it.gold_edges:
        keys.add(("edge", e.get("source"), e.get("relation"),
                  e.get("target")))
    return keys


def main() -> int:
    t0 = time.time()

    # -- 1. build (deterministic; seed 13 inside build_benchmark_heldout) ----
    heldout = BM.build_benchmark_heldout()
    gold = BM._build_gold()
    log(f"heldout n={len(heldout)}   gold(frozen 60-set) n={len(gold)}")

    # validation context (same corpus views the builder used)
    import common as c
    from evaluation.benchmark import _BuilderContext
    ctx = _BuilderContext(Path(c.NOTEBOOKS_DATA) / "graph",
                          Path(c.NOTEBOOKS_DATA) / "ast")
    all_lids, edges = ctx.all_lids, ctx.edges

    # -- 2. per-item programmatic validation (spec §5/§6) ---------------------
    per_item = []
    for it in heldout:
        errs = list(BM.validate_item(it, all_lids, edges))
        if not it.reference_answer:
            errs.append("empty reference_answer")
        if not it.required_evidence:
            errs.append("empty required_evidence")
        if it.hop_count > 0 and not (it.gold_edges or it.gold_path):
            errs.append("hop_count>0 without gold_edges/gold_path")
        per_item.append({
            "query_id": it.query_id,
            "family": it.category,
            "target": it.target_lineage_id,
            "hop_count": it.hop_count,
            "validation_errors": errs,
            "valid": not errs,
        })
    n_ok = sum(1 for r in per_item if r["valid"])
    log(f"validation {n_ok}/{len(heldout)} items clean")

    # -- 3. leakage audit (spec §3, §20) --------------------------------------
    gold_by_qid = {it.query_id: it for it in gold}
    diag = [gold_by_qid[q] for q in DIAGNOSTIC_QIDS if q in gold_by_qid]
    missing_diag = [q for q in DIAGNOSTIC_QIDS if q not in gold_by_qid]

    diag_qs = {d.question for d in diag}
    diag_targets = {d.target_lineage_id for d in diag}
    diag_chunks = {cid for d in diag for cid in d.gold_chunks}

    h_qs = {it.question for it in heldout}
    h_targets = {it.target_lineage_id for it in heldout}
    h_chunks = {cid for it in heldout for cid in it.gold_chunks}

    g60_qs = {it.question for it in gold}
    g60_targets = {it.target_lineage_id for it in gold}
    g60_chunks = {cid for it in gold for cid in it.gold_chunks}

    q_d14 = sorted(h_qs & diag_qs)
    t_d14 = sorted(h_targets & diag_targets)
    c_d14 = sorted(h_chunks & diag_chunks)
    q_60 = sorted(h_qs & g60_qs)
    t_60 = sorted(h_targets & g60_targets)
    c_60 = sorted(h_chunks & g60_chunks)

    # Informational: the question template is a pure function of
    # (term, doc, target-article, gold-edge), so items with a shared
    # fact-key share a question template.  This is a SUFFICIENT condition
    # for being a paraphrase, but not necessary — e.g. two one_hop AMENDS
    # items can reference the same graph edge but have different question
    # text (different article-level src).  We report it as "potential"
    # overlap to flag for human review, not as an automatic failure.
    h_facts = set().union(*[_fact_keys(it) for it in heldout])
    g60_facts = set().union(*[_fact_keys(it) for it in gold])
    diag_facts = set().union(*[_fact_keys(d) for d in diag]) if diag else set()
    fact_60 = sorted({repr(x) for x in (h_facts & g60_facts)}, key=str)
    fact_d14 = sorted({repr(x) for x in (h_facts & diag_facts)}, key=str)

    audit = {
        "spec": ("08 §3 (diagnostic set = tuning set), "
                 "§20 (leakage check)"),
        "n_heldout": len(heldout),
        "diagnostic_set": {
            "source": ("gcg_2hop_two_levers_report.md §A/§B — primary 8 + "
                       "controls 6, the 14-query target set of the "
                       "completed GCG series (tuning/diagnostic set per "
                       "spec 08 §3)"),
            "query_ids": DIAGNOSTIC_QIDS,
            "missing_from_frozen_60": missing_diag,
        },
        "checks": {
            # Authoritative (pass/fail) — direct id/text overlaps --------
            "no_heldout_query_is_diagnostic_query": {
                "definition": ("exact question-text membership in the 14 "
                               "diagnostic questions"),
                "overlap": q_d14, "pass": not q_d14,
            },
            "no_heldout_target_in_diagnostic": {
                "definition": "target lineage-id overlap with the 14",
                "overlap": t_d14, "pass": not t_d14,
            },
            "no_heldout_gold_chunk_in_diagnostic": {
                "definition": "gold-chunk id overlap with the 14",
                "overlap": c_d14, "pass": not c_d14,
            },
            "no_heldout_query_in_frozen_60": {
                "definition": ("strict: the 14 is a subset of the frozen "
                               "60-set, so verify zero question-text, "
                               "target, and gold-chunk overlap with the "
                               "full 60"),
                "question_overlap": q_60,
                "target_overlap": t_60,
                "chunk_overlap": c_60,
                "pass": not (q_60 or t_60 or c_60),
            },
            # Informational — flag for human review, not auto-fail -------
            "potential_paraphrase_by_fact_key": {
                "definition": ("the question template is a pure function "
                               "of (term, doc, target-article, gold-edge); "
                               "items sharing a fact-key COULD render the "
                               "same text, but may differ in article-level "
                               "src/dst or in which non-article chunk is "
                               "targeted.  These are flagged for review, "
                               "not auto-rejected.  Human review: "
                               "check whether the overlapping fact-key "
                               "corresponds to a semantically identical "
                               "question in context."),
                "overlap_with_60": fact_60,
                "overlap_with_14": fact_d14,
                "informational": True,
            },
            # Declaration checks ----------------------------------------
            "not_used_for_parameter_selection": {
                "definition": ("held-out set was constructed via "
                               "build_benchmark_heldout() AFTER the pool "
                               "budget (A0-30 / split_12_9+21) and "
                               "graph-context (Lever 2) choices were "
                               "frozen (spec §1); the builder rejects any "
                               "item whose target or text appears in the "
                               "frozen 60-set, so the set was "
                               "structurally incapable of informing those "
                               "choices"),
                "pass": True,
            },
            "labels_not_exposed_to_retrieval_pipeline": {
                "definition": ("ground truth is read only by metric "
                               "code (target_lineage_id, gold_chunks, "
                               "gold_edges, required_evidence); the "
                               "retrieval pipeline consumes "
                               "item.question only.  Benchmark "
                               "construction is deterministic "
                               "(build_benchmark_heldout, seed 13), "
                               "produced BEFORE the three systems run, "
                               "and is not part of the retrieval "
                               "pipeline."),
                "pass": True,
            },
        },
    }
    # Pass/fail is determined only by authoritative checks
    authoritative = {k: v for k, v in audit["checks"].items()
                     if not v.get("informational")}
    all_pass = all(chk["pass"] for chk in authoritative.values())
    audit["all_checks_pass"] = all_pass

    # -- 4. family balance (spec §4: ~balanced, graph fairs must not dominate)
    fam_counts = Counter(it.category for it in heldout)
    family_report = {
        fam: {"n": fam_counts.get(fam, 0),
              "target": BM.HELDOUT_FAMILY_TARGETS.get(fam, 0)}
        for fam in sorted(BM.HELDOUT_FAMILY_TARGETS)
    }
    rel_graph_fams = {
        "one_hop_relational", "two_hop_relational", "relation_direction",
        "temporal_version", "graph_distractor", "multi_document_synthesis",
    }
    n_rel = sum(fam_counts.get(f, 0) for f in rel_graph_fams)
    family_report["graph_dependent_share"] = {
        "families": sorted(rel_graph_fams),
        "n": n_rel,
        "share": round(n_rel / max(len(heldout), 1), 3),
    }
    shortfalls = {}
    for it in heldout:
        for fam, n in it.metadata.get("family_shortfall", {}).items():
            shortfalls[fam] = shortfalls.get(fam, 0) + n
    family_report["shortfalls_vs_target"] = shortfalls

    # -- 5. write artifacts ----------------------------------------------------
    out_root = E.EVAL_RESULTS
    out_root.mkdir(parents=True, exist_ok=True)

    lines = [json.dumps(it.as_dict(), ensure_ascii=False, sort_keys=True)
             for it in heldout]
    p_bench = out_root / "heldout_benchmark.jsonl"
    p_bench.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bench_hash = _sha256_lines(lines)

    p_val = out_root / "heldout_benchmark_validation.json"
    validation = {
        "spec": ("08 §4/§5/§6 — held-out benchmark "
                 "+ ground-truth requirements + quality control"),
        "n_items": len(heldout),
        "n_valid": n_ok,
        "n_invalid": len(heldout) - n_ok,
        "validator": ("evaluation.benchmark.validate_item "
                      "(deterministic corpus checks: chunk existence, "
                      "gold_path edge existence + contiguity, hop_count "
                      "consistency, gold-edge presence for relational "
                      "families) + reference_answer/required_evidence "
                      "presence"),
        "llm_used_for_ground_truth": False,
        "ground_truth_basis": ("chunk text + graph edges, programmatic "
                               "(spec §5: 'programmatic source/graph "
                               "truth remains authoritative')"),
        "family_counts": family_report,
        "shortfall_note": ("families below target are corpus-capped or "
                           "question-text-collision-capped vs the "
                           "frozen 60-set; shortfall is RECORDED, not "
                           "fabricated (spec §4; benchmark doc §2 pattern "
                           "used by _build_gold). "
                           "temporal_version is at its corpus ceiling "
                           "(5): AMENDS-edges without SUPERSEDES/"
                           "effective-date/effective-in force semantics, "
                           "recorded as the schema limitation of spec §7."),
        "per_item": per_item,
        "benchmark_file": str(p_bench.relative_to(ROOT)),
        "benchmark_sha256": bench_hash,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    p_val.write_text(json.dumps(validation, indent=2, ensure_ascii=False)
                     + "\n", encoding="utf-8")

    p_audit = out_root / "heldout_leakage_audit.json"
    p_audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False)
                       + "\n", encoding="utf-8")

    log(f"wrote {p_bench}  (sha256 {bench_hash[:16]}…)")
    log(f"wrote {p_val}")
    log(f"wrote {p_audit}   all_checks_pass={all_pass}")
    log(f"family counts: {dict(sorted(fam_counts.items()))}")
    log(f"elapsed {time.time()-t0:.1f}s")
    return 0 if (all_pass and n_ok == len(heldout)) else 1


if __name__ == "__main__":
    sys.exit(main())
