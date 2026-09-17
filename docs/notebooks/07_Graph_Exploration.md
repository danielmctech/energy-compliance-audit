# 07 - Graph Exploration & GraphRAG Trace

**Pipeline position:** off-pipeline (read-only over the live `04` graph + one
LLM trace through `06`).
**Thesis importance:** **Demonstrative** - it is the human-readable
evidence that the graph is real, rich, and queryable, and produces paste-ready
Cypher for the thesis and for a reader's own Neo4j Browser.

## Purpose

Useful, schema-true **Cypher over the live Neo4j graph** (2,076 nodes /
5,178 edges - read-only, no new indexes), plus a live **question → documents**
trace through `GraphRAG.query()` that yields a paste-ready Neo4j Browser
query. In short: it *demonstrates* the graph is a real, navigable
knowledge base, not an abstract claim.

## What it does (by section)

Ground rules: read-only against live Neo4j (bolt `:7687`, db `neo4j`),
no new indexes or constraints, visualisation = the built-in **Neo4j
Browser** at `http://localhost:7474` (every section prints copy-paste Cypher).
Section E is the only LLM-calling section (Ollama `qwen3.8:27b`); Section F
asserts *nothing changed*.

Typical sections (per the notebook):

- **A. Graph inventory & quality** - node/edge counts, label distributions,
  relationship-type histogram, `lineage_id` index check.
- **B. Highly connected entities** - the hub nodes; the regulatory entities
  that dominate the corpus (regulations, institutions, obligations).
- **C. Neighborhoods** - 1-hop and 2-hop expansions of a chosen article,
  with provenance (which sentence motivated each edge).
- **D. Cross-instrument cross-references** - edges that leave a document
  (`ext:article` stubs, `unresolved: true`).
- **E. Live GraphRAG trace** - one question, `GraphRAG.query()`, and the
  Cypher that the retrieval stage actually ran.
- **F. Read-only assertion** - verify no changes to the store.

## Inputs / outputs

- **In:** the live Neo4j store (`04` ingested via `src/graphrag_n4j/ingestion.py`),
  `06` for the E-section trace.
- **Out:** *nothing written to disk by design* (it is a demonstration, not an
  output); the Cypher printed to the cell output is the artefact.

## Where the logic lives

Mostly inline in the notebook (Cypher + small helpers); the trace section
re-uses `src/graphrag_n4j/rag.py`.

## Why it matters to the thesis

It is the *evidence the reader can run themselves*: paste a Cypher, watch the
graph respond. That is a much stronger "see, the graph is real" than any
figure - because the reader can *change* it. It also makes the
`04_Graph_Construction` claims **falsifiable on demand**. Combined with the
`07_Figures_Analysis_and_Critique` finding that the *figures are worth 6.5/10*
for a thesis, this notebook is worth 7/10 - it is the *one piece of
evidence* a reviewer can act on immediately.

## Key caveat

The **`neo4j` retrieval weakness (Recall@10 0.233)** that `06` and `08`
measured is *not explained* by `07`. The graph is rich and navigable -
that is *not* the same as *good for retrieval*. A reader who looks at the
hubs in B and the neighborhoods in C might conclude "the graph is clearly
useful," and be measuring the *wrong thing*. `07` is the **positive** side
of the graph evidence; `08` (via `06` §6.9) carries the **negative**
side. Read them as a pair.
