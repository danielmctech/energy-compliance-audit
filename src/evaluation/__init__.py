"""Evaluation framework for the hybrid + GraphRAG retrieval stack.

Two strictly separated layers (per the "Separate Retrieval from Generation" requirement):

* **Layer A -- retrieval quality** (``metrics``, ``retrieval_run``,
  ``overlap``) -- did we retrieve the correct evidence?
* **Layer B -- answer quality** (``generation``, ``answer_metrics``) --
  did the LLM use that evidence to produce a correct, grounded answer?

Plus triangulation helpers (``context``), statistics (``stats``),
compliance-scenario construction (``scenarios``), error analysis
(``error_analysis``), thesis tables/figures (``tables``, ``figures``) and the
reproducibility report (``report``).

The framework re-uses the project's existing retrieval stack
(``retrieval.Retriever``, ``graphrag_n4j.GraphRAG``) -- it does not
re-implement retrieval (per the "Repository-First Requirement").
"""
from __future__ import annotations

from .config import (
    BOOTSTRAP_SEED,
    ERROR_LABELS,
    EVAL_RESULTS,
    EXPERIMENTS,
    EvalConfig,
    GENERATION_SYSTEMS,
    RETRIEVAL_METRICS,
    SEED,
)
from .metrics import (
    dcg,
    exact_match,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    ndcg_per_query,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    token_f1,
)
from .stats import bootstrap_ci, paired_bootstrap_diff, wilcoxon

__all__ = [
    "BOOTSTRAP_SEED", "ERROR_LABELS", "EVAL_RESULTS", "EXPERIMENTS",
    "EvalConfig", "GENERATION_SYSTEMS", "RETRIEVAL_METRICS", "SEED",
    # metrics
    "dcg", "exact_match", "hit_rate_at_k", "mean_reciprocal_rank",
    "ndcg_at_k", "ndcg_per_query", "precision_at_k", "recall_at_k",
    "reciprocal_rank", "token_f1",
    # stats
    "bootstrap_ci", "paired_bootstrap_diff", "wilcoxon",
]
