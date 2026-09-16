"""Four retrieval baselines over the constructed scenario set (per the "Additional information" requirement).

Each baseline is a distinct way of using the SAME retrieval corpus --
none of them re-implements retrieval from scratch (per the "Repository-First Requirement" and "Evaluation Configuration Matrix" requirements):

  B1  rag_dense_fixed    base dense encoder, top-k from the fixed corpus.
  B2  rag_dense_late     late-stage segmentation: the audited *clause* is
                         segmented and each segment is searched dense;
                         results are fused by rank (RRF-style mean position).
  B3  rag_contextual     Anthropic contextual-retrieval: the target
                         provision's text is prepended to the clause (a
                         context-augmented query) before dense search.
  B4  rag_hybrid_graph   the full hybrid + graph configuration (KG2RAG-like).

Each dense baseline is run with BOTH the base encoder and the fine-tuned
LoRA encoder (per the "Fine-Tuning Evaluation" requirement).  The audit
LLM and metric set are identical across all of them -- only the retrieved
top-k differs, so per-baseline accuracy differences are attributable to
*retrieval* (per the "Groundedness / Faithfulness" separation).

The retrieval is decoupled from the audit: ``baseline_retrievals`` returns
``{scenario_id: {baseline_name: [top-k lineages]}}`` so the audit (LLM)
can be re-run for free offline and the retrieval can be swapped without
touching the judge.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import config as E
from .scenarios import Scenario
from .retrieval_run import RetrievalEngine


def _ranks_fuse(rank_lists: List[List[str]], k: int) -> List[str]:
    """Mean-rank fusion across several rank lists (late-stage segmentation)."""
    seen: Dict[str, List[int]] = {}
    for j, rl in enumerate(rank_lists):
        for rank, lid in enumerate(rl, start=1):
            seen.setdefault(lid, []).append(rank + j * 1e-9)
    scored = sorted(
        ((sum(rs) / len(rs), lid) for lid, rs in seen.items()), key=lambda x: x[0])
    return [lid for _s, lid in scored[:k]]


def baseline_retrievals(scenarios: Sequence[Scenario],
                        engine: Optional[RetrievalEngine] = None,
                        k: int = 5,
                        encoder: str = "base",
                        baselines: Optional[Sequence[str]] = None
                        ) -> Dict[str, Dict[str, list]]:
    """Return {scenario_id: {baseline: [top-k lids]}}.

    ``encoder`` in {"base", "finetuned_stage1", "finetuned_stage2"}.
    Baselines: rag_dense_fixed, rag_dense_late, rag_contextual,
    rag_hybrid_graph.  Only the requested subset is run.
    """
    engine = engine or RetrievalEngine()
    baselines = list(baselines) if baselines else [
        "rag_dense_fixed", "rag_dense_late", "rag_contextual",
        "rag_hybrid_graph"]
    # choose dense system variant by encoder
    if encoder == "base":
        dense_sys = "dense"
    elif encoder.startswith("finetuned_"):
        stage = encoder.split("_", 1)[1]  # stage1|stage2
        short = {"stage1": "s1", "stage2": "s2"}.get(stage)
        if short is None:
            raise ValueError(encoder)
        dense_sys = f"dense_ft_{short}"   # EXPERIMENTS key: dense_ft_s1 / s2
    else:
        raise ValueError(encoder)
    # B4 is the dedicated ``hybrid_graph`` mode: RRF fuses sparse+dense and a
    # graph-neighbour list, so graph-sourced chunks enter the top-k (distinct
    # from ``hybrid``, whose top-k is fixed and only carries a graph boost).
    out: Dict[str, Dict[str, list]] = {s.scenario_id: {} for s in scenarios}
    for s in scenarios:
        clause = s.clause_text
        # B1: dense fixed
        lids = [r["lineage_id"]
                for r in engine.retrieve(dense_sys, clause, k)]
        out[s.scenario_id]["rag_dense_fixed"] = lids
        if "rag_dense_late" in baselines:
            segs = [_seg for _seg in
                    [p.strip() for p in clause.replace(".", "\n").split("\n")
                     if 15 < len(p.strip()) < 400]][:3]
            rank_lists = [[r["lineage_id"] for r in engine.retrieve(
                dense_sys, seg, k)] for seg in segs] or \
                         [[r["lineage_id"] for r in engine.retrieve(
                         dense_sys, clause, k)]]
            out[s.scenario_id]["rag_dense_late"] = _ranks_fuse(rank_lists, k)
        if "rag_contextual" in baselines:
            ctx_query = (f"Context: governing provision ({s.provision_ref}): "
                         f"{s.provision_text[:1000]}\n\n"
                         f"Clause to audit: {clause}")
            lids = [r["lineage_id"]
                    for r in engine.retrieve(dense_sys, ctx_query, k)]
            out[s.scenario_id]["rag_contextual"] = lids
        if "rag_hybrid_graph" in baselines:
            lids = [r["lineage_id"]
                    for r in engine.retrieve("hybrid_graph", clause, k)]
            out[s.scenario_id]["rag_hybrid_graph"] = lids
    return out


def baseline_matrix(engine: Optional[RetrievalEngine] = None,
                    scenarios: Optional[Sequence[Scenario]] = None,
                    encoders: Sequence[str] = ("base",),
                    k: int = 5) -> Dict[tuple, Dict[str, list]]:
    """All (encoder) x (baseline) retrieval for a scenario set.

    Returns {(encoder, scenario_id): {baseline: [lids]}}.
    """
    from .scenarios import build_synthetic_scenarios
    from retrieval._corpus import load_corpus
    corpus = load_corpus()
    scenarios = list(scenarios) if scenarios is not None \
        else build_synthetic_scenarios(corpus, per_doc=3)
    out = {}
    for enc in encoders:
        br = baseline_retrievals(scenarios, engine=engine, k=k,
                                 encoder=enc)
        for sid, per_b in br.items():
            out[(enc, sid)] = per_b
    return out


def save(retrievals: Dict[str, Dict[str, list]],
         out_dir: Path = E.OUT_GRAPHS) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline_retrievals.json", "w") as f:
        json.dump(retrievals, f, indent=2)

