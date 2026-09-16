# Notebook Guide - pipeline of `energy-audit`

Nine notebooks form a strict, ordered pipeline (`01 → 02 → 03a → 03b → 04 →
05 → 06`, with `07` as a read-only exploration off the live graph and `08` as
the end-to-end evaluation). Each file below is a short guide to **one**
notebook: what it does, what it takes in and emits, where the logic actually
lives in `src/`, and how central it is to the thesis.

## How to read the importance rating

Two different things are being ranked - don't conflate them:

- **Pipeline position** - the dependency order (you cannot run `05` without
  `04` without `03b` …). This is about *execution*.
- **Thesis importance** - how much the *contribution* depends on this
  notebook. This is about *claim*.

| Notebook | Pipeline position | Thesis importance | One-line role |
|---|---|---|---|
 | [`01`](01_corpus_normalisation_and_Metadata_Audit.md) Corpus audit | 1 (first) | Prerequisite | Validate the corpus before anything touches it |
 | [`02`](02_Parser_Evaluation.md) Parser evaluation | 2 | Method-critical | Decide which PDF parser feeds the whole stack |
 | [`03a`](03a_Structure_Extraction.md) Structure extraction | 3 | Foundation | Derive per-document structure maps (articles, definitions, obligations) |
 | [`03b`](03b_AST_Construction.md) AST construction | 4 | Core contract | Build the AST + chunks + the `lineage_id` RAG contract |
 | [`04`](04_Graph_Construction.md) Graph construction | 5 | Differentiator | Build the knowledge graph the GraphRAG arm expands over |
 | [`05`](05_Retrieval_Layer.md) Retrieval layer | 6 | The system | Hybrid sparse+dense(+graph) retrieval - `retrieve()` |
 | [`06`](06_GraphRAG_Query_API.md) GraphRAG query API | 7 | End-to-end | Neo4j GraphRAG: retrieval → provenance context → local LLM |
 | [`07`](07_Graph_Exploration.md) Graph exploration | off-pipeline | Demonstrative | Read-only Cypher + a live question→documents trace |
 | [`08`](08_Evaluation_Framework.md) Evaluation framework | final | The contribution | All thesis numbers, tables, figures, statistics |

**The four must-reads for the thesis claim are `03b`, `04`, `05`, and
`08`.** Everything else exists to make those four correct and defensible:
`01` ensures clean input, `02` chooses the parser, `03a` produces the
structure, and `07` is read-only demonstration (it changes nothing).

## The data contract that ties them together

The single thread every notebook depends on is the **`lineage_id`** introduced
in `03b` (e.g. `{doc}:article:{n}`, `{doc}:article:{n}:para:{p}`,
`{doc}:table:{seq}`). It is the join key so that:

- `04` attaches graph **edges to chunks** (any retrieved chunk lands on the
  graph - `04_Graph_Construction`),
- `05` reports **provenance** (text + lineage + methods + boosting edge),
- `06` renders **per-line provenance** in the LLM context, and
- `08` can cite *which chunk and which graph edge* supported an answer.

If any notebook is confusing, the question to ask is usually *what does it do
to the lineage* - that is the contract.

## Outputs at a glance

| Step | Key artifact(s) under `notebooks/data/` |
|---|---|
| 01 | `outputs/corpus_metadata.json` audit, `notebooks/data/evaluations/*` (audit CSVs) |
| 02 | `ground_truth/*.json`, `evaluations/parser_evaluation_*.csv`, `best_parser.json` |
| 03a | `structure_maps/*_structure.json`, `best_structure_map.json` |
| 03b | `ast/*_ast.json`, `chunks_*.json`, `ast_summary.csv` |
| 04 | `graph/nodes.jsonl`, `edges.jsonl`, `graph_summary.csv` (2,049 nodes / 4,139 edges) |
| 05 | `retrieval/` indexes + `retrieval_eval_*.jsonl` (per-mode) |
| 06 | `graphrag/graphrag_summary.csv`, `graphrag_logs.jsonl` |
| 07 | live Cypher (nothing written to disk by design) |
| 08 | `evaluation/` (per-query, aggregate, tables, stats, error analysis) + `figures/fig1–fig6.png` |

> **Reproducibility:** all LLM calls across the pipeline are deterministic
> (temperature 0.0) and cached under `notebooks/data/llm_cache/`, so re-runs
> are free and resumable. The 08 harness produces a `reproducibility.json`
> report. See `00_README_and_Index.md` for the anchor conventions.
