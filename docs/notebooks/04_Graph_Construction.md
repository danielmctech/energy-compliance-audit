# 04 - Graph Construction

**Pipeline position:** `01 → 02 → 03a → 03b → 04 → 05 → 06`
**Thesis importance:** **Differentiator** - this is what makes the
system "GraphRAG" rather than plain RAG. It builds the knowledge graph
(`05`/`06` expand over it).

## Purpose

Build the deterministic **knowledge graph** over the corpus that the GraphRAG
retrieval layer (`05`/`06`) expands over. Nodes and edges are **joined on the
`03b` lineage IDs**, so any retrieved chunk lands directly on the graph -
this is the GraphRAG *claim* made concrete.

## What it does

Node kinds: `document`, `preamble`, `article`, `external_article`,
`external_document`, `term`, `entity`.

Edge kinds - each carries `evidence = {doc_id, offset, snippet ≤ 120}`
(`unresolved: true` for external destinations):

- **`CROSS_REFERENCES`** - article → article. From `Article N` mentions in the
  prose (tables excluded). Lists/ranges (`Articles 2, 3 and 4`) expand into
  one edge per member. Instrument context after the mention (`of Regulation
  (EU) 2016/679`) resolves the target across documents inside the corpus;
  otherwise an `ext:article` stub.
- **`AMENDS`** - hosting node → amended node (the hosting article's amendment
  clause references a target article).
- Further typed edges (obligation, energy-unit, term, entity) per the
  `03a`/`03b` structure - the set is defined in `src/structure/graph_builder.py`.

The pipeline is fully **deterministic, regex-driven, with evidence spans** -
so any edge is *inspectable* and *falsifiable* against the source text.

## Inputs / outputs

- **In:** `03b` ASTs + chunks (lineage IDs), `03a` structure maps.
- **Out:** `notebooks/data/graph/nodes.jsonl`, `edges.jsonl`,
  `graph_summary.csv` (2,049 nodes / 4,139 edges).

## Where the logic lives

`src/structure/graph_builder.py` (edge builders, dedup); the Neo4j
persistence side is exercised by `06` (`src/graphrag_n4j/ingestion.py`).

## Why it matters to the thesis

This is where the "GraphRAG" in "Hybrid Sparse-Dense-**Graph**" earns its
name. It also *is the measured negative result*: `08` (`06` §6.5) finds the
graph, fused into the hybrid top-k, does **not** add recall on this corpus -
which is a *useful, reproducible* finding, not an absence of one. The
evidence-span design is what makes that finding defensible: you can open any
edge and see the exact sentence that motivated it.

## Key caveat

The graph is built from **regex cross-reference extraction**, so its quality
is bounded by the same boundary cases as `03a` (non-standard numbering,
amended articles, consolidated provisions). `06` §6.13 lists the absence of a
*graph-quality gold* as a known gap - the graph's own correctness is not
independently validated against a human-annotated reference.
