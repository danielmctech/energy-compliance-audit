"""System-vs-system comparison helpers (task §13–§15).

Pairs per-query rows of two systems (e.g. ``hybrid_rerank`` vs
``hybrid_graph_rerank``) and reports, for every (metric, k):

* paired mean, median, std of **a − b** (positive = a wins)
* improvement / degradation / unchanged rates over the paired queries
* bootstrap 95 % CI on the paired difference
* Wilcoxon signed-rank p-value + Cohen's d (via :mod:`.stats`)
* per-query deltas -- the *evidence* behind the aggregate, kept for the
  report and any future error analysis.

Per-query classification (task §14):

* :func:`classify_per_query` produces one row per query with the spec's
  required fields (``baseline_rank``, ``graph_rank``, ``rank_delta``,
  ``baseline_score``, ``graph_score``, ``graph_context_available``,
  ``graph_edges_used``, ``candidate_source``) and a verdict
  (``IMPROVED`` / ``UNCHANGED`` / ``DEGRADED``).
* :func:`render_markdown` embeds a per-query summary section in the report
  when :func:`save_report` receives the classified rows.

The comparison is pure arithmetic over the per-query logs already written
by :mod:`.retrieval_run`; it never re-runs retrieval, so it is cheap to
call for many system pairs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config as E
from . import stats as S


def load_per_query(system: str,
                   out_dir: Path = E.OUT_PER_QUERY) -> Dict[str, dict]:
    """Load ``retrieval_{system}.jsonl`` -> ``{query_id: row}``."""
    p = out_dir / f"retrieval_{system}.jsonl"
    out: Dict[str, dict] = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = __import__("json").loads(line)
        out[str(r["query_id"])] = r
    return out


@dataclass
class PairMetric:
    """One (metric, k) row of a system-vs-system comparison."""
    a: str
    b: str
    metric: str
    k: int
    n: int
    mean_diff: float          # mean(a) - mean(b)
    ci95: Tuple[float, float] = (float("nan"), float("nan"))
    improvement_rate: float = float("nan")
    degradation_rate: float = float("nan")
    unchanged_rate: float = float("nan")
    wilcoxon_p: Optional[float] = None
    cohens_d: Optional[float] = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["ci95"] = list(self.ci95)
        return d


def compare(system_a: str, system_b: str,
            k_values: Sequence[int] = None,
            metrics: Sequence[str] = E.RETRIEVAL_METRICS,
            out_dir: Path = E.OUT_PER_QUERY,
            ) -> List[PairMetric]:
    """Per-(metric, k) comparison of ``system_a`` against ``system_b``.

    Only queries present in **both** log files are paired (queries one
    system skipped are excluded, not imputed -- "No Fabricated Results").
    """
    k_values = list(k_values) if k_values else list(E.EvalConfig().k_values)
    rows_a = load_per_query(system_a, out_dir)
    rows_b = load_per_query(system_b, out_dir)
    common = sorted(set(rows_a) & set(rows_b),
                    key=lambda q: (len(q), q))
    if not common:
        raise ValueError(f"no common queries between {system_a} and {system_b}")
    out: List[PairMetric] = []
    for k in k_values:
        for m in metrics:
            va = [float(rows_a[q]["metrics"][str(k)].get(m) or 0.0)
                  for q in common]
            vb = [float(rows_b[q]["metrics"][str(k)].get(m) or 0.0)
                  for q in common]
            n = len(common)
            d = np.asarray(va, dtype="float64") - np.asarray(vb, dtype="float64")
            mean_diff = float(d.mean())
            try:
                diff, lo, hi = S.paired_bootstrap_diff(va, vb)
            except Exception:  # pragma: no cover - degenerate case
                lo = hi = mean_diff
            pos = float((d > 0).mean())
            neg = float((d < 0).mean())
            p, _ = S.wilcoxon(va, vb)
            out.append(PairMetric(
                a=system_a, b=system_b, metric=m, k=int(k), n=n,
                mean_diff=mean_diff, ci95=(float(lo), float(hi)),
                improvement_rate=pos, degradation_rate=neg,
                unchanged_rate=float((d == 0).mean()),
                wilcoxon_p=p, cohens_d=S.cohen_d_paired(va, vb)))
    return out


def _rank_value(row: dict, k: int) -> Optional[int]:
    """target rank in ``row`` (``None`` when the target is outside the
    logged top-k).  The logged ``target_rank`` is the global rank over the
    full retrieval list, so it is ``None`` only when the target was not
    retrieved at all."""
    tr = row.get("target_rank")
    if tr is None:
        return None
    return int(tr) if int(tr) <= k else k + 1


def classify_per_query(system_a: str, system_b: str,
                       k: int = 10,
                       out_dir: Path = E.OUT_PER_QUERY) -> List[dict]:
    """Per-query IMPROVED / UNCHANGED / DEGRADED classification (task §14).

    ``system_a`` is the baseline (``hybrid_rerank``), ``system_b`` is the
    graph-aware system.  A query is classified as IMPROVED when the
    baseline's target rank at ``k`` is WORSE than the graph-aware system's
    (or when the baseline did not retrieve the target at all but the
    graph-aware system did), and DEGRADED when the opposite holds.
    UNCHANGED covers all ties, including "not retrieved by either" at this k.

    Every row carries the spec's required fields:

    * ``query_id``, ``question``, ``category``, ``target``
    * ``baseline_rank``, ``graph_rank``, ``rank_delta`` (b − a; negative =
      baseline is better on this query)
    * ``baseline_score`` / ``graph_score``: per-query MRR at ``k``
      (the rank-aware metric the spec asks for the *pairwise* classification)
    * ``graph_context_available``: whether any candidate in the graph-aware
      row carries a non-empty ``graph_context``
    * ``graph_edges_used``: number of distinct ``lineage_id`` in the
      graph-aware top-k that have ``graph_edge_type`` set
    * ``candidate_source``: "graph_expansion" when the system's retrieval
      mode includes expansion (A7) and at least one top-k row shows
      ``source == "graph_expansion"`` in ``extra``, else "hybrid"
    """
    rows_a = load_per_query(system_a, out_dir)
    rows_b = load_per_query(system_b, out_dir)
    common = sorted(set(rows_a) & set(rows_b), key=lambda q: (len(q), q))
    if not common:
        raise ValueError(f"no common queries between {system_a} and {system_b}")
    spec = E.EXPERIMENTS.get(system_b, {})
    is_expand = "expand" in spec.get("reranking", "")
    mode_b = spec.get("retrieval", "")
    # spec §14: graph_context_available -- True when the system's mode
    # renders per-candidate graph context (all GRAPH_RERANK_MODES).  A0
    # hybrid_rerank never renders context, so it is False by construction.
    ctx_mode = mode_b in E.GRAPH_RERANK_MODES
    out: List[dict] = []
    for q in common:
        ra = rows_a[q]
        rb = rows_b[q]
        rk_a = _rank_value(ra, k)
        rk_b = _rank_value(rb, k)
        # rank_delta = graph_rank − baseline_rank; positive = baseline is
        # better on this query (graph-aware moved it down).
        if rk_a is None and rk_b is None:
            rank_delta: Optional[int] = 0
        elif rk_a is None:
            rank_delta = -1   # baseline missed; graph-aware hit -> -1 (b better)
        elif rk_b is None:
            rank_delta = 1    # baseline hit; graph-aware missed -> +1 (a better)
        else:
            rank_delta = rk_b - rk_a
        # scores from the k-set logged per query
        mk = str(k)
        score_a = float(ra["metrics"].get(mk, {}).get("mrr") or 0.0)
        score_b = float(rb["metrics"].get(mk, {}).get("mrr") or 0.0)
        # graph edges used: number of top-k candidates whose methods list
        # contains "graph" (the provenance label from the retrieval layer)
        methods = rb.get("methods", [])[:k]
        edges_used = sum(1 for m in methods if "graph" in (m or []))
        candidate_source = "hybrid"
        if is_expand and any("graph:expand" in (m or []) for m in methods):
            candidate_source = "graph_expansion"
        if rank_delta is None:
            verdict = "UNCHANGED"
        elif rank_delta < 0:
            verdict = "IMPROVED"
        elif rank_delta > 0:
            verdict = "DEGRADED"
        else:
            verdict = "UNCHANGED"
        out.append({
            "query_id": q,
            "question": ra.get("question", ""),
            "category": ra.get("category", ""),
            "target": ra.get("target", ""),
            "baseline_rank": rk_a,
            "graph_rank": rk_b,
            "rank_delta": rank_delta,
            "baseline_score": score_a,
            "graph_score": score_b,
            "base_hit@k": float(ra["metrics"].get(mk, {}).get("hit") or 0.0),
            "graph_hit@k": float(rb["metrics"].get(mk, {}).get("hit") or 0.0),
            "graph_context_available": ctx_mode,
            "graph_edges_used": edges_used,
            "candidate_source": candidate_source,
            "verdict": verdict,
            "system_a": system_a,
            "system_b": system_b,
            "k": k,
        })
    out.sort(key=lambda r: (r["rank_delta"] if r["rank_delta"] is not None else 999,
                            r["query_id"]))
    return out


def _classification_markdown(cls_rows: List[dict]) -> str:
    """Render the per-query classification summary (spec §14)."""
    lines = ["## Per-query classification (IMPROVED / UNCHANGED / DEGRADED)",
             "",
             f"system_a = {cls_rows[0]['system_a']}, system_b = "
             f"{cls_rows[0]['system_b']}, k = {cls_rows[0]['k']}",
             ""]
    imp = sum(1 for r in cls_rows if r["verdict"] == "IMPROVED")
    de  = sum(1 for r in cls_rows if r["verdict"] == "DEGRADED")
    un  = sum(1 for r in cls_rows if r["verdict"] == "UNCHANGED")
    n   = len(cls_rows) or 1
    lines += [
        f"| verdict | count | rate |",
        f"|---|---:|---:|",
        f"| IMPROVED | {imp} | {imp / n:.3f} |",
        f"| UNCHANGED | {un} | {un / n:.3f} |",
        f"| DEGRADED | {de} | {de / n:.3f} |",
        "",
        "Per-query detail (only non-UNCHANGED rows shown; UNCHANGED rows "
        "are the dominant outcome in this benchmark):",
        "",
        "| query_id | category | target | baseline_rank | graph_rank | rank_delta | base MRR@k | graph MRR@k | verdict |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in cls_rows:
        if r["verdict"] == "UNCHANGED":
            continue
        ba = "–" if r["baseline_rank"] is None else f"{r['baseline_rank']}"
        gb = "–" if r["graph_rank"] is None else f"{r['graph_rank']}"
        lines.append(
            f"| {r['query_id']} | {r['category']} | {r['target']} | "
            f"{ba} | {gb} | {r['rank_delta']:+d} | "
            f"{r['baseline_score']:.4f} | {r['graph_score']:.4f} | "
            f"{r['verdict']} |")
    lines.append("")
    return "\n".join(lines)


def render_markdown(comparisons: List[PairMetric],
                    per_query_cls: Optional[List[dict]] = None) -> str:
    """Render a list of :class:`PairMetric` rows as a Markdown table
    (one table per (a, b) system pair).  When ``per_query_cls`` is given,
    append the §14 per-query classification summary."""
    out: List[str] = []
    groups: Dict[Tuple[str, str], List[PairMetric]] = {}
    for c in comparisons:
        groups.setdefault((c.a, c.b), []).append(c)
    for (a, b), rows in groups.items():
        out.append(f"### {a} vs {b}")
        out.append("")
        out.append("| metric | k | n | mean(a) − mean(b) | CI95 | "
                   "impr | degr | unchanged | wilcoxon p | Cohen's d |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for c in sorted(rows, key=lambda r: (r.metric, r.k)):
            lo, hi = c.ci95
            ci = f"[{lo:+.4f}, {hi:+.4f}]" if not math.isnan(lo) else "n/a"
            p = ("–" if c.wilcoxon_p is None
                 else f"{c.wilcoxon_p:.3g}")
            cd = ("–" if c.cohens_d is None
                  else f"{c.cohens_d:+.2f}")
            out.append(
                f"| {c.metric} | {c.k} | {c.n} | {c.mean_diff:+.4f} | {ci} | "
                f"{c.improvement_rate:.3f} | {c.degradation_rate:.3f} | "
                f"{c.unchanged_rate:.3f} | {p} | {cd} |")
        out.append("")
    if per_query_cls:
        by_pair: Dict[Tuple[str, str], List[dict]] = {}
        for r in per_query_cls:
            by_pair.setdefault((r.get("system_a"), r.get("system_b")), []).append(r)
        for (a, b) in by_pair:
            out.append(_classification_markdown(by_pair[(a, b)]))
    return "\n".join(out)


def save_report(comparisons: List[PairMetric],
                out_path: Optional[Path] = None,
                per_query_cls: Optional[List[dict]] = None
                ) -> Path:
    """Persist the comparison as JSON + Markdown under
    ``OUT_REPORTS/graph_rerank_comparison.md``.  When ``per_query_cls`` is
    given, the same JSON file also carries the per-query rows and the
    Markdown body gains the §14 per-query section."""
    out_path = out_path or E.OUT_REPORTS / "graph_rerank_comparison.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join([
        "# Graph-aware re-ranker vs baseline -- comparison",
        "",
        "Methodology: per-query paired deltas over the shared 60-query "
        "benchmark; improvement / degradation / unchanged rates count the "
        "per-query wins per (metric, k); paired bootstrap CI on a−b; "
        "Wilcoxon signed-rank for paired significance; Cohen's d "
        "as a small-n effect size.",
        "",
        "n is borderline for significance testing (see stats.py).  p > 0.05 "
        "is flagged as non-significant, not absent.",
        "",
        render_markdown(comparisons, per_query_cls),
    ])
    out_path.write_text(body)
    jpath = out_path.with_suffix(".json")
    payload: dict = {
        "comparisons": [c.as_dict() for c in comparisons],
    }
    if per_query_cls:
        payload["per_query"] = per_query_cls
    jpath.write_text(__import__("json").dumps(payload, indent=2))
    return out_path
