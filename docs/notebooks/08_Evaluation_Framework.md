# 08 - Evaluation Framework

**Pipeline position:** final (`01 → … → 06`, then `07` as read-only, then
**`08`**).
**Thesis importance:** **The contribution** - *all* the thesis numbers,
tables, figures, statistics, and the honest measurement of negative results
come from this notebook.

## Purpose

Thesis-grade evaluation of the Hybrid Sparse-Dense-Graph + GraphRAG stack,
with the **retrieval** and **generation** layers **strictly separated** (per
the framework's "Separate Retrieval from Generation" requirement):

- **Layer A - retrieval quality** - did we retrieve the correct evidence?
- **Layer B - answer quality** - did the LLM use that evidence to produce a
  correct, grounded answer?

Plus overlap / context / triangulation, statistics, the four thesis tables,
figures, a reproducibility report, and constructed compliance scenarios
(dual-drafter → blind adjudicator → dispute-escalation) with four RAG
baselines.

All tunables live in `src/evaluation/config.py` (`EvalConfig`,
`EXPERIMENTS`). All outputs land under `notebooks/data/evaluation/` and
`notebooks/data/figures/`. All LLM calls are deterministic + cached.

## What it does (by section)

- **1. Benchmark** - a **60-query, single-target** benchmark
  (`per_query/benchmark.jsonl`); 58 article-type / 2 term-type.
- **2. Retrieval run** - 8 systems × 60 queries: `dense`, `sparse`,
  `hybrid`, `graph`, `neo4j`, `hybrid_graph`, `dense_ft_s1`, `dense_ft_s2`.
- **3. Generation** - 5 of the 8 arms (`dense`, `sparse`, `hybrid`,
  `hybrid_graph`, `neo4j`) generate answers via Ollama `qwen3.8:27b`
  (temperature 0.0, cached); the 3 graph/fine-tune arms are retrieval-only.
- **4. Answer metrics** - `token_f1`, `exact_match` (degenerate:
  `reference_answer` is the *target chunk text*, so EM = 0.000 by
  construction), `semantic_sim` (base encoder), `cites_target`, and an
  **LLM judge** (faithfulness / groundedness / completeness / relevance) -
  same model family as the generator (a self-referential risk, flagged).
- **5. Overlap & context** - Jaccard of dense/sparse/graph top-5,
  `context_coverage`, `leakage_checks` (the 49/60 LoRA target leak is
  caught here).
- **6. Statistics** - 3 pairwise comparisons, bootstrap 2000 (seed 2024),
  Wilcoxon signed-rank, Cohen's d. Only one pair clears the bar
  (hybrid > dense, p = 0.0143).
- **7. Tables** - `table_A.csv` (retrieval), `table_B.csv` (answer),
  `table_C.csv` (ablation - single-component, *not* additive), `table_D.csv`
  (fine-tune), `table_comparison.csv` (head-to-head `hybrid` vs `dense` /
  `sparse`, with absolute and relative deltas, `rel_vs_dense`,
  `rel_vs_sparse`).
- **8. Error analysis** - per `(system, query)` labels: `CORRECT`,
  `PARTIAL_RETRIEVAL`, `RETRIEVAL_FAILURE` (and a `GENERATION_FAILURE`
  class that, for answer arms, is *equivalent to*
  `RETRIEVAL_FAILURE` because `cites_target = 1.00` for every answer row -
  see `04` §4.8).
- **9. Figures** - `fig1–fig6.png` (retrieval, recall curve, ablation,
  fine-tuning, category, overlap).
- **10. Reproducibility** - a `reproducibility.json` + `.md` report
  asserting seed, cache hit-rate, and per-stage determinism.
- **11. Scenarios** - the four RAG baselines (B1–B4) and the
  dual-drafter / blind-adjudicator dispute-escalation protocol.

## Inputs / outputs

- **In:** the full `01`–`06` artifact set + `07` (for the live trace).
- **Out:** everything under `notebooks/data/evaluation/` (per-query,
  aggregate, tables, stats, error analysis, reports) and
  `notebooks/data/figures/fig1–fig6.png`.

## Where the logic lives

`src/evaluation/{config,benchmark,metrics,answer_metrics,stats,tables,
figures,error_analysis,generation,report,scenarios,overlap,context,
semantic, retrieval_run}.py` - the whole `src/evaluation/` package is
effectively *for this notebook*.

## Why it matters to the thesis

It is where the **claims become numbers**. More than that, it is where the
*honesty* lives - the negative results (the graph-fusion dilution, the LoRA
leak, the EM degeneracy, the same-family judge, the `neo4j` weakness),
the measured **not asserted** character of the "hybrid > dense" win, and
the *single-target scope limit* (flagged as the highest-severity gap) are
all *produced* here, not hidden. It is a **contribution in itself**: a
reusable, seeded, deterministic evaluation harness that a different thesis
can point at a new corpus and re-run.

## Key caveat

The **single-target benchmark** is the biggest limitation in the study - it
*cannot* detect the multi-instrument reasoning a real compliance audit
requires, and it is what hides the gap between the best channel (sparse
0.933) and the weakest (neo4j 0.233). The EM degeneracy and the
same-family judge are **measured limits, not bugs** - the harness *tells*
you where it can't be trusted, which is a defensible design.

**The one-sentence summary of `08`:** *it is where the thesis earns its
numbers - and its honesty.*
