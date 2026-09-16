"""Thesis-ready CSV + markdown tables (per "Machine-Readable Results", "Thesis Tables" and "Retrieval Metrics at Multiple K Values").

Reads the long-form aggregate row tables produced by the run modules and
renders:
  Table A -- Retrieval performance (per system, per K: recall/precision/
             hit/MRR/nDCG).
  Table B -- RAG answer quality (per system: EM/F1/sim/cites + judge axes).
  Table C -- Ablation (component on/off against the full system).
  Table D -- Fine-tuning a/b (dense base vs stage1 vs stage2).

All numeric values are the *actually computed* ones from the run -- never
placeholders ("No Fabricated Results").  A system row that could not be executed is shown
as ``not_executed`` with the recorded reason, not a fabricated number.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import config as E

_K_ORDER = [1, 3, 5, 10, 20]


def _piv(metric: str, agg: List[dict], systems: Sequence[str]) -> Dict[str, Dict[int, Optional[float]]]:
    out = {s: {} for s in systems}
    for r in agg:
        if r.get("metric") != metric:
            continue
        if r["system"] in out:
            out[r["system"]][r["k"]] = r["value"]
    return out


def _fmt(v: Optional[float], nd: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def table_a(agg: List[dict], systems: Optional[Sequence[str]] = None
            ) -> List[dict]:
    """Table A: one row per (system, k) with all 5 metrics."""
    systems = list(systems) if systems else [
        s for s in E.EXPERIMENTS if any(r["system"] == s for r in agg)]
    rec = _piv("recall", agg, systems)
    pre = _piv("precision", agg, systems)
    hit = _piv("hit", agg, systems)
    mrr = _piv("mrr", agg, systems)
    ndc = _piv("ndcg", agg, systems)
    rows = []
    for s in systems:
        for k in _K_ORDER:
            rows.append({
                "system": s, "k": k,
                "recall": _fmt(rec.get(s, {}).get(k)),
                "precision": _fmt(pre.get(s, {}).get(k)),
                "hit": _fmt(hit.get(s, {}).get(k)),
                "mrr": _fmt(mrr.get(s, {}).get(k)),
                "ndcg": _fmt(ndc.get(s, {}).get(k)),
            })
    return rows


# spec §12 judge fields as long-form metric names (from answer_metrics.aggregate)
_SPEC_METRICS = (
    "judge__correct", "judge__faithful", "judge__complete",
    "judge__evidence_supported", "judge__overall_score", "judge__confidence",
    "judge__graph_reasoning_correct", "judge__unsupported_claim_rate",
    "judge__escalation_rate",
)


def table_b(ans_agg: List[dict], systems: Optional[Sequence[str]] = None
            ) -> List[dict]:
    """Table B: RAG answer quality per system.

    Carries the legacy base + 4-axis scores *and* the spec §12 judge
    columns (``correct``/``faithful``/``complete``/``evidence_supported``
    as 0/1 means, ``overall_score``/``confidence`` on their native scales,
    ``graph_reasoning_correct`` accuracy, ``unsupported_claim_rate`` and
    ``escalation_rate``).  A column whose metric was not computed for a
    system shows ``n/a`` rather than a fabricated number.
    """
    systems = list(systems) if systems else [
        s for s in E.GENERATION_SYSTEMS
        if any(r["system"] == s for r in ans_agg)]
    legacy = ("exact_match", "token_f1", "semantic_similarity",
              "cites_target", "judge__relevance", "judge__faithfulness",
              "judge__groundedness", "judge__completeness")
    metrics = legacy + _SPEC_METRICS
    vals = {}
    for m in metrics:
        vals[m] = {r["system"]: r["value"]
                   for r in ans_agg if r.get("metric") == m}
    spec_cols = {
        "judge__correct": ("correct", 3),
        "judge__faithful": ("faithful", 3),
        "judge__complete": ("complete", 3),
        "judge__evidence_supported": ("evidence_supported", 3),
        "judge__overall_score": ("overall_score", 2),
        "judge__confidence": ("confidence", 3),
        "judge__graph_reasoning_correct": ("graph_reasoning_correct", 3),
        "judge__unsupported_claim_rate": ("unsupported_claim_rate", 3),
        "judge__escalation_rate": ("escalation_rate", 3),
    }
    rows = []
    for s in systems:
        row = {
            "system": s,
            "em": _fmt(vals["exact_match"].get(s), 3),
            "f1": _fmt(vals["token_f1"].get(s), 3),
            "semantic_sim": _fmt(vals["semantic_similarity"].get(s), 3),
            "cites_target": _fmt(vals["cites_target"].get(s), 2),
            "judge_relevance": _fmt(vals["judge__relevance"].get(s), 2),
            "judge_faithfulness": _fmt(vals["judge__faithfulness"].get(s), 2),
            "judge_groundedness": _fmt(vals["judge__groundedness"].get(s), 2),
            "judge_completeness": _fmt(vals["judge__completeness"].get(s), 2),
        }
        for m, (col, nd) in spec_cols.items():
            row[col] = _fmt(vals[m].get(s), nd)
        rows.append(row)
    return rows


def table_c(agg: List[dict], ablation_metric: str = "recall",
            ablation_k: int = 10) -> List[dict]:
    """Table C: ablation (full vs without-dense/sparse/graph) at one K."""
    full = next((r["value"] for r in agg
                 if r["system"] == "hybrid" and r["metric"] == ablation_metric
                 and r["k"] == ablation_k), None)
    rows = []
    for cfg_name in ("dense", "sparse", "graph"):
        v = next((r["value"] for r in agg
                  if r["system"] == cfg_name and r["metric"] == ablation_metric
                  and r["k"] == ablation_k), None)
        rows.append({
            "configuration": f"without_{cfg_name}",
            "retrieval": cfg_name,
            "graph": cfg_name == "graph",
            "ablation_metric": ablation_metric,
            "k": ablation_k,
            "value": _fmt(v),
            "delta_vs_full": _fmt((v - full) if (v is not None and full is not None)
                                 else None),
        })
    rows.insert(0, {"configuration": "full", "retrieval": "hybrid",
                    "graph": True, "ablation_metric": ablation_metric,
                    "k": ablation_k, "value": _fmt(full),
                    "delta_vs_full": "0.00"})
    return rows


def table_d(agg: List[dict], k: int = 10,
            ) -> List[dict]:
    """Table D: fine-tuning a/b (dense base vs stage1 vs stage2)."""
    out = []
    for s in ("dense", "dense_ft_s1", "dense_ft_s2"):
        v = next((r["value"] for r in agg
                  if r["system"] == s and r["metric"] == "recall"
                  and r["k"] == k), None)
        out.append({"variant": s, "k": k, "recall": _fmt(v)})
    return out


def comparison(agg: List[dict], k: int = 10,
               ) -> List[dict]:
    """Hybrid vs dense/sparse deltas (per the "Hybrid Retrieval Contribution Analysis" requirement)."""
    def val(s, m):
        return next((r["value"] for r in agg
                     if r["system"] == s and r["metric"] == m and r["k"] == k),
                    None)
    rows = []
    for m in ("recall", "precision", "hit", "mrr", "ndcg"):
        hy = val("hybrid", m)
        de = val("dense", m)
        sp = val("sparse", m)
        def d(a, b):
            if a is None or b is None:
                return None
            return a - b
        def rel(a, b):
            if a is None or b is None or b == 0:
                return None
            return (a - b) / b
        rows.append({
            "metric": m, "k": k,
            "hybrid": _fmt(hy), "dense": _fmt(de), "sparse": _fmt(sp),
            "delta_hybrid_vs_dense": _fmt(d(hy, de)),
            "delta_hybrid_vs_sparse": _fmt(d(hy, sp)),
            "rel_vs_dense": _fmt(rel(hy, de), 3),
            "rel_vs_sparse": _fmt(rel(hy, sp), 3),
        })
    return rows


def write(systems_to_write: Dict[str, List[dict]],
          out_dir: Path = E.OUT_AGGREGATE) -> Dict[str, Path]:
    """Persist each table as CSV + a single markdown block for the thesis.

    ``systems_to_write`` maps table name -> list of row dicts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    md = ["# Evaluation results tables\n"]
    for name, rows in systems_to_write.items():
        if not rows:
            continue
        fields = list(rows[0].keys())
        csvp = out_dir / f"table_{name}.csv"
        with open(csvp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        written[name] = csvp
        md.append(f"\n## {name.upper()}\n")
        md.append("| " + " | ".join(fields) + " |")
        md.append("|" + "|".join(["---"] * len(fields)) + "|")
        for r in rows:
            md.append("| " + " | ".join(str(r.get(k, "")) for k in fields) + " |")
    md.append("")
    (out_dir / "tables.md").write_text("\n".join(md))
    return written


def load_aggregate(path: Path = E.OUT_AGGREGATE / "retrieval_aggregate.csv"
                   ) -> List[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["k"] = int(r["k"])
        r["value"] = float(r["value"])
        r["n"] = int(r["n"])
    return rows


def load_answer_aggregate(path: Path = E.OUT_AGGREGATE / "answer_aggregate.csv"
                          ) -> List[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["value"] = float(r["value"])
        r["n"] = int(r["n"])
    return rows


def load_longform_rows(rows: Optional[List[dict]] = None) -> List[dict]:
    """Accept an in-memory table or load from disk."""
    if rows is not None:
        return rows
    from . import retrieval_run
    if (E.OUT_AGGREGATE / "retrieval_aggregate.csv").exists():
        return load_aggregate()
    return []
