"""Centralised, reproducible evaluation configuration.

Implements the "Reproducibility" and "Evaluation Configuration Matrix" sections.

Every tunable -- seeds, top-k, fusion/rrf parameters, retrieval cutoffs,
generation parameters, evaluation metric set -- lives here so that notebooks
and CLI share one source of truth and no experiment parameter is hard-coded
across cells.  Changing a value here changes every run identically.

NOTE on methodological neutrality (per the "Methodological Neutrality" requirement): the SAME k-set, benchmark and
relevance judgments are applied to every system.  No per-system cutoffs are
chased to flatter one configuration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


def _repo_root() -> Path:
    """Repo root: honour ENERGY_AUDIT_ROOT, else walk up from this file."""
    env = os.getenv("ENERGY_AUDIT_ROOT")
    if env and Path(env).exists():
        return Path(env).resolve()
    cur = Path(__file__).resolve().parent
    for cand in [cur, *cur.parents]:
        if (cand / "notebooks").is_dir() and (cand / "data").is_dir():
            return cand
    return cur.parent


ROOT = _repo_root()
DATA = ROOT / "notebooks" / "data"
EVAL_RESULTS = DATA / "evaluation"

# Output sub-directories (per the "Output Structure" requirement).  Created on demand, never reorganised.
OUT_AGGREGATE = EVAL_RESULTS / "aggregate"
OUT_PER_QUERY = EVAL_RESULTS / "per_query"
OUT_RETRIEVAL = EVAL_RESULTS / "retrieval"
OUT_GENERATION = EVAL_RESULTS / "generation"
OUT_GRAPHS = EVAL_RESULTS / "graph"
OUT_FIGURES = ROOT / "notebooks" / "data" / "figures"
OUT_REPORTS = EVAL_RESULTS / "reports"

#: Deterministic seeds.  Everything stochastic (bootstrap, scenario
#: construction, pair order, shuffle) reads a seed from here so results are
#: reproducible (per the "Bootstrap Confidence Intervals" and "Reproducibility" requirements).
SEED = 2024
BOOTSTRAP_SEED = 2024


@dataclass(frozen=True)
class EvalConfig:
    """All evaluation-time parameters.  ``frozen`` so a run's config can be
    hashed / embedded verbatim into the reproducibility report."""

    #: top-k values reported for Recall/Precision/Hit (per "Retrieval Metrics", "Precision@K", "Hit Rate / Success@K")
    k_values: Tuple[int, ...] = (1, 3, 5, 10, 20)
    #: retrieval cutoff used to fetch candidates before truncating to k.
    #: Larger than max(k) so that Precision@K is computed on a full top-k
    #: window ("Precision@K": do not divide by K if fewer than K were retrieved).
    retrieve_k: int = 50

    #: bootstrap resampling (per the "Bootstrap Confidence Intervals" requirement)
    n_bootstrap: int = 2000
    bootstrap_ci: float = 0.95

    #: LLM-as-judge / answer-model settings, shared by all systems so
    #: comparisons are on an identical generation footing (per "End-to-End RAG Evaluation" / "Methodological Neutrality").
    llm_model: str = "qwen3.8:27b"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    #: Primary answer judge -- a DIFFERENT model than the generator
    #: (spec: "gpt-oss:20b is the evaluator, not the authority").  ``latest``
    #: is the 20B-variant (12.85 GB) of gpt-oss on this box.  This breaks the
    #: generator==judge self-referentiality flagged in 06_Missing 6.10.
    judge_model: str = "gpt-oss:latest"
    #: Escalation judge -- invoked ONLY when the primary judge is uncertain
    #: (spec: "Nemotron is an escalation mechanism for ambiguous cases, not a
    #: replacement for source-derived ground truth").
    judge_escalator: str = "nemotron-3-nano:30b"
    #: confidence threshold below which (or overall_score in {2,3}) the row is
    #: re-judged by ``judge_escalator`` (spec §13 escalation policy).
    judge_confidence_threshold: float = 0.85
    #: overall_score values that ALSO trigger escalation even if confidence is
    #: high (spec: "overall_score in {2,3}").
    judge_escalate_scores: Tuple[int, ...] = (2, 3)
    judge_scale: int = 5               # 1..5 rubric

    #: Scenario construction protocol (per the "Additional information" requirement).
    #: Two architecturally independent drafters produce Candidate A / B;
    #: an anonymised adversarial adjudicator picks between them; a fourth,
    #: independent model escalates ONLY when the adjudicator rejects both.
    scenario_drafter_a: str = "gemma3:27b"
    scenario_drafter_b: str = "mistral-small3.1:24b"
    scenario_adjudicator: str = "qwen3.8:27b"
    scenario_escalator: str = "nemotron-3-nano:30b"

    #: --- graph-aware re-ranker feature flags (v2) ------------------------
    #: Each flag independently gates one axis of the graph-context pipeline.
    #: All default True; the baseline ``hybrid_rerank`` mode does not consult
    #: any of these -- they only affect ``hybrid_graph_rerank`` and ablation
    #: modes.  Toggling here is a *runtime* configuration, not a code path
    #: switch: the reranker prompt is identical (byte-for-byte) when the
    #: flags are all False.
    gr_enable: bool = True                 # master gate
    gr_1hop: bool = True                   # include 1-hop relations
    gr_2hop: bool = True                   # include 2-hop paths
    gr_typed: bool = True                  # use typed/directional labels (on by default)
    gr_snippets: bool = True               # include edge-evidence snippets (<=120 chars)
    gr_query_aware: bool = True            # order neighbours by query-token overlap
    gr_expand: bool = False                # A7: allow controlled graph expansion
    #: Budgets -- deliberately small (bounds the re-rank prompt size).
    gr_max_edges: int = 8                  # 1-hop lines per candidate
    gr_max_paths: int = 3                  # 2-hop "VIA" lines per candidate
    gr_token_limit: int = 500              # soft cap (~ chars = limit * 4)
    #: The candidate pool depth: how many chunks from the hybrid pool we
    #: actually re-rank AND render graph context for.  ``hybrid_rerank`` uses
    #: this same value (so its prompt is unchanged when ``gr_enable=False``).
    gr_candidate_k: int = 30

    def context_config(self, mode: str) -> "tuple":
        """Build the (GraphContextConfig or None, with_expand: bool) pair for
        a retriever mode from this config instance.

        Returns ``(None, False)`` for the A0 baseline ``hybrid_rerank`` or
        when ``gr_enable`` is off -- that is what preserves bit-identical
        ``hybrid_rerank`` results.
        """
        if not self.gr_enable:
            return None, False
        from reranking.graph_context import GraphContextConfig  # local: avoid circular
        if mode == "hybrid_gr_1hop":
            return GraphContextConfig(include_paths=False,
                                      max_edges=self.gr_max_edges,
                                      token_limit=self.gr_token_limit), False
        if mode == "hybrid_gr_relations":
            return GraphContextConfig(include_paths=False,
                                      include_snippets=False,
                                      max_edges=self.gr_max_edges,
                                      token_limit=self.gr_token_limit), False
        if mode == "hybrid_gr_expand":
            return GraphContextConfig(
                include_paths=self.gr_2hop,
                include_snippets=self.gr_snippets,
                query_aware=self.gr_query_aware,
                max_edges=self.gr_max_edges,
                max_paths=self.gr_max_paths,
                token_limit=self.gr_token_limit), self.gr_expand
        # hybrid_graph_rerank (main system)
        return GraphContextConfig(
            include_paths=self.gr_2hop,
            include_snippets=self.gr_snippets,
            query_aware=self.gr_query_aware,
            max_edges=self.gr_max_edges,
            max_paths=self.gr_max_paths,
            token_limit=self.gr_token_limit), False

    def as_dict(self) -> dict:
        d = {}
        for f in self.__dataclass_fields__:
            d[f] = getattr(self, f)
        return d


#: The retriever modes that render graph context; A0 is ``hybrid_rerank``
#: and is deliberately NOT in this set (its results are the pre-v2 baseline).
GRAPH_RERANK_MODES: Tuple[str, ...] = (
    "hybrid_graph_rerank",
    "hybrid_gr_1hop",
    "hybrid_gr_relations",
    "hybrid_gr_expand",
)


#: Evaluation Configuration Matrix (per the "Evaluation Configuration Matrix" requirement).  Keyed by a stable name used
#: verbatim in tables / figures / per-query logs.
#:
#:   retrieval : sparse | dense | hybrid | graph | neo4j_graph |
#:               hybrid_graph  (Retriever.MODES)
#:   graph     : whether the result is a graph-traversal system
#:   reranking : "Not implemented" in this repo -> the row is present but
#:               explicitly marked N/A (user decision), never a fabricated value
#:   dense_model : "base" | None
EXPERIMENTS: Dict[str, dict] = {
    # --- single-method baselines -----------------------------------------
    "dense":      {"retrieval": "dense",      "graph": False,
                   "reranking": "not_implemented", "dense_model": "base"},
    "sparse":     {"retrieval": "sparse",     "graph": False,
                   "reranking": "not_implemented", "dense_model": None},
    "hybrid":     {"retrieval": "hybrid",     "graph": False,
                   "reranking": "not_implemented", "dense_model": "base"},
    "graph":      {"retrieval": "graph",      "graph": True,
                   "reranking": "not_implemented", "dense_model": None},
    "neo4j":      {"retrieval": "neo4j_graph","graph": True,
                   "reranking": "not_implemented", "dense_model": None},
    # --- hybrid + graph composition (per "Graph + Hybrid Evaluation") ----
    # Distinct from ``hybrid``: this is the dedicated ``hybrid_graph`` retrieval
    # mode, which fuses sparse + dense + graph-neighbours via RRF so graph nodes
    # can enter the top-k and shift ordering, rather than the ``hybrid`` mode's
    # fixed top-k with only a graph-boost annotation on survivors.
    "hybrid_graph": {"retrieval": "hybrid_graph", "graph": True,
                     "reranking": "not_implemented", "dense_model": "base"},
    # --- listwise LLM re-ranker (per "Reranking Evaluation") ----------------
    # Same sparse+dense hybrid pool as ``hybrid``; a listwise LLM re-ranker
    # (src/reranking) re-orders the pool in place.  At k >= pool size the
    # top-k is set-identical to hybrid (Recall@20 = 0.983 both).  At
    # k < pool size it reorders within the same top-10, so Recall@1 /
    # Precision@1 / MRR / nDCG can change (measured: +0.30 on Recall@1,
    # +0.21 on MRR@10, +0.16 on nDCG@10).
    "hybrid_rerank": {"retrieval": "hybrid_rerank", "graph": False,
                       "reranking": "listwise_llm", "dense_model": "base"},
    # --- graph-aware re-ranker (v2) -- the *main* new system --------------
    # Same hybrid pool as ``hybrid_rerank``; per-candidate typed/directional
    # graph context (1/2-hop, CITES/AMENDS/SUPERSEDES/... and *_BY forms) is
    # rendered and injected into the re-ranker prompt as evidence.  Pool is
    # unchanged; only the LLM's judgement changes.
    "hybrid_graph_rerank": {
        "retrieval": "hybrid_graph_rerank", "graph": True,
        "reranking": "listwise_llm+graph_ctx", "dense_model": "base"},
    # --- ablations (task §15) ---------------------------------------------
    "hybrid_gr_1hop": {
        "retrieval": "hybrid_gr_1hop", "graph": True,
        "reranking": "listwise_llm+graph_ctx_1hop", "dense_model": "base"},
    "hybrid_gr_relations": {
        "retrieval": "hybrid_gr_relations", "graph": True,
        "reranking": "listwise_llm+graph_ctx_relations", "dense_model": "base"},
    "hybrid_gr_expand": {
        "retrieval": "hybrid_gr_expand", "graph": True,
        "reranking": "listwise_llm+graph_ctx+expand", "dense_model": "base"},
}

#: Which experiments the *answer* (end-to-end RAG) evaluation runs.
#:
#: Per ``Run answer generation on systems.md``, the PRIMARY answer-scoring set
#: is the re-ranker causal chain ``#3 hybrid -> #7 hybrid_rerank (A0) -> #8
#: hybrid_graph_rerank`` plus the two graph-context ablations (``#9 1hop``,
#: ``#10 relations``). The older ``sparse/dense/hybrid_graph/neo4j`` baseline
#: rows (still reproducible from their per-query + generation rows) and the
#: retrieval-only rows are NOT answer-scored by default in that document (Group A /
#: Tier 3-4) and are therefore NOT answer-scored by default -- they would add
#: ~2.6x the judge compute without advancing the core hypothesis.
GENERATION_SYSTEMS: Tuple[str, ...] = (
    "hybrid",                  # #3 retrieval baseline
    "hybrid_rerank",           # #7 A0 control (LLM reranking, no graph)
    "hybrid_graph_rerank",     # #8 main graph-conditioned hypothesis
    "hybrid_gr_1hop",          # #9 ablation: 1-hop context only
    "hybrid_gr_relations",     # #10 ablation: typed/directional relations
)

#: The *secondary* / historical baseline answer rows (sparse, dense,
#: hybrid_graph, neo4j).  Kept for the legacy Table B comparison in
#: ``04_Evaluation_and_Results.md``; NOT included in the primary set above.
LEGACY_GENERATION_SYSTEMS: Tuple[str, ...] = (
    "sparse", "dense", "hybrid_graph", "neo4j",
)

#: Metric names, in report order (Recall/Precision/Hit/MRR/nDCG).
RETRIEVAL_METRICS = ("recall", "precision", "hit", "mrr", "ndcg")

#: Failure taxonomy for error analysis (per the "Error Classification" requirement).  Stored as raw evidence +
#: a *suggested* label; the label is a hypothesis, not an assertion.
ERROR_LABELS = (
    "RETRIEVAL_FAILURE",
    "PARTIAL_RETRIEVAL",
    "CORRECT_RETRIEVAL_WRONG_ANSWER",
    "GRAPH_RETRIEVAL_FAILURE",
    "CONTEXT_FAILURE",
    "GENERATION_FAILURE",
    "HALLUCINATION",
    "CORRECT",
)
