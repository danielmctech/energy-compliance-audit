"""Thesis-quality figures (per the "Visualization" requirement).

Every figure is drawn ONLY from actually-computed aggregate rows
(``tables.load_aggregate``).  A system or metric that never executed is
omitted from the plot -- never filled with 0 or a placeholder (per "No Fabricated Results").

Figures:
  fig1_retrieval.png      -- bars: systems x {Recall@5, Recall@10, MRR}
  fig2_recall_curve.png   -- recall vs K per system
  fig3_ablation.png       -- full vs without-dense/sparse/graph (Recall@10)
  fig4_fine_tuning.png    -- dense base vs stage1 vs stage2 (Recall curve)
  fig5_category.png       -- per-category Recall@10 by system (if categories)
  fig6_overlap.png        -- Jaccard bar plot + unique-relevant by method

``matplotlib`` backend is Agg (headless), 300 dpi PNG.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as E
from .tables import load_aggregate

_PAL = {"dense": "#1f77b4", "sparse": "#d62728", "hybrid": "#2ca02c",
        "graph": "#ff7f0e", "neo4j": "#9467bd", "hybrid_graph": "#8c564b",
        "dense_ft_s1": "#17becf", "dense_ft_s2": "#bcbd22"}


def _color(name: str) -> str:
    return _PAL.get(name, "#7f7f7f")


def _save(fig, name: str, out_dir: Path = E.OUT_FIGURES) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_retrieval(agg: Optional[List[dict]] = None,
                  out_dir: Path = E.OUT_FIGURES) -> Path:
    agg = agg or load_aggregate()
    metrics = [("recall", 5, "Recall@5"), ("recall", 10, "Recall@10"),
               ("mrr", 10, "MRR@10")]
    systems = [s for s in E.EXPERIMENTS
               if any(r["system"] == s for r in agg)]

    def val(system, metric, k):
        r = next((row for row in agg
                  if row["system"] == system and row["metric"] == metric
                  and row["k"] == k), None)
        return r["value"] if r else 0.0

    import numpy as np
    xpos = np.arange(len(systems))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (metric, k, label) in enumerate(metrics):
        vals = [val(s, metric, k) for s in systems]
        ax.bar(xpos + i * w, vals, width=w, label=label,
               color=["#4c72b0", "#dd8452", "#55a868"][i])
    ax.set_xticks(xpos + w)
    ax.set_xticklabels(systems, rotation=20)
    ax.set_ylabel("score")
    ax.set_title("Retrieval performance by system")
    ax.legend(ncol=3, frameon=False)
    ax.set_ylim(0, 1.02)
    return _save(fig, "fig1_retrieval.png", out_dir)


def fig_recall_curve(agg: Optional[List[dict]] = None,
                     out_dir: Path = E.OUT_FIGURES) -> Path:
    agg = agg or load_aggregate()
    systems = [s for s in E.EXPERIMENTS
               if any(r["system"] == s and r["metric"] == "recall" for r in agg)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in systems:
        pts = sorted([(r["k"], r["value"]) for r in agg
                      if r["system"] == s and r["metric"] == "recall"])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker="o", color=_color(s), label=s)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("k (log scale)")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall vs k by retrieval system")
    ax.legend(ncol=2, frameon=False)
    ax.grid(True, alpha=0.3)
    return _save(fig, "fig2_recall_curve.png", out_dir)


def fig_ablation(agg: Optional[List[dict]] = None, k: int = 10,
                 out_dir: Path = E.OUT_FIGURES) -> Path:
    from .tables import table_c
    agg = agg or load_aggregate()
    rows = table_c(agg, "recall", k)
    labels = [r["configuration"] for r in rows]
    vals = [float(r["value"]) if r["value"] not in (None, "n/a") else 0
            for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals,
                  color=[_color(r["retrieval"]) for r in rows])
    ax.set_ylabel(f"Recall@{k}")
    ax.set_title(f"Ablation: recall@{k} vs full system")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{v:.3f}", ha="center")
    ax.set_ylim(0, 1.05)
    return _save(fig, "fig3_ablation.png", out_dir)


def fig_fine_tuning(agg: Optional[List[dict]] = None,
                    out_dir: Path = E.OUT_FIGURES) -> Path:
    agg = agg or load_aggregate()
    variants = [v for v in ("dense", "dense_ft_s1", "dense_ft_s2")
                if any(r["system"] == v for r in agg)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v in variants:
        pts = sorted([(r["k"], r["value"]) for r in agg
                      if r["system"] == v and r["metric"] == "recall"])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker="s", color=_color(v), label=v)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k (dense)")
    ax.set_title("Dense encoder: base vs fine-tuned LoRA stages")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    return _save(fig, "fig4_fine_tuning.png", out_dir)


def fig_overlap(overlap: Optional[Dict] = None,
                comp: Optional[Dict] = None,
                out_dir: Path = E.OUT_FIGURES) -> Path:
    from . import overlap as OV
    if overlap is None and comp is None:
        p = E.OUT_RETRIEVAL / "overlap.json"
        if p.exists():
            d = OV.load(p)
            overlap, comp = d["jaccard"], d["complementarity"]
        else:
            overlap, comp = {}, {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    if overlap:
        names = list(overlap.keys())
        ax.bar(names, list(overlap.values()), color="#4c72b0")
        ax.set_ylabel("Jaccard (mean over queries)")
    ax.set_title("Top-k overlap (Jaccard)")
    ax.tick_params(axis='x', labelrotation=25)
    ax = axes[1]
    if comp and "unique_relevant_only_by" in comp:
        ub = comp["unique_relevant_only_by"]
        ax.bar(ub.keys(), ub.values(),
               color=[_color(s) for s in ub.keys()])
        ax.set_ylabel("unique relevant targets")
    ax.set_title("Unique-relevant evidence by method")
    ax.tick_params(axis='x', labelrotation=20)
    fig.suptitle("Retrieval complementarity")
    return _save(fig, "fig6_overlap.png", out_dir)


def fig_category(per_query: Dict[str, list], items, k: int = 10,
                 out_dir: Path = E.OUT_FIGURES) -> Path:
    from collections import defaultdict
    systems = [s for s in E.EXPERIMENTS if s in per_query]
    by_cat: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        for s in systems:
            row = next((r for r in per_query.get(s, [])
                        if r["query_id"] == item.query_id), None)
            if row and row["metrics"].get(str(k)):
                v = row["metrics"][str(k)]["recall"]
                if v is not None:
                    by_cat[item.category][s].append(v)
    cats = list(by_cat)
    if not cats:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no category data", ha="center")
        return _save(fig, "fig5_category.png", out_dir)
    import numpy as np
    xpos = np.arange(len(cats))
    w = 0.8 / max(len(systems), 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, s in enumerate(systems):
        vals = [sum(by_cat[c].get(s, [0])) / len(by_cat[c].get(s, [0]))
                if by_cat[c].get(s) else 0 for c in cats]
        ax.bar(xpos + i * w, vals, width=w, label=s, color=_color(s))
    ax.set_xticks(xpos + (len(systems) - 1) * w / 2)
    ax.set_xticklabels(cats)
    ax.set_ylabel(f"Recall@{k}")
    ax.set_title(f"Recall@{k} by benchmark category")
    ax.legend(ncol=3, frameon=False)
    return _save(fig, "fig5_category.png", out_dir)


def make_all(agg: Optional[List[dict]] = None,
             per_query: Optional[Dict[str, list]] = None,
             items=None,
             out_dir: Path = E.OUT_FIGURES) -> Dict[str, Path]:
    out = {}
    out["retrieval"] = fig_retrieval(agg, out_dir)
    out["recall_curve"] = fig_recall_curve(agg, out_dir)
    out["ablation"] = fig_ablation(agg, out_dir=out_dir)
    out["fine_tuning"] = fig_fine_tuning(agg, out_dir)
    out["overlap"] = fig_overlap(out_dir=out_dir)
    if items is not None and per_query is not None:
        out["category"] = fig_category(per_query, items, out_dir=out_dir)
    return out
