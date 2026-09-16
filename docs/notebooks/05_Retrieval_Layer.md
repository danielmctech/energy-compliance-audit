# 05 - Retrieval Layer (Hybrid Sparse-Dense-Graph)

**Pipeline position:** `01 → 02 → 03a → 03b → 04 → 05 → 06`
**Thesis importance:** **The system** - this notebook *is* the primary
contribution. Everything before it is feeding; everything after it is
demonstrating or measuring.

## Purpose

Implement the **retrieval layer** of the Hybrid Sparse-Dense-Graph RAG stack -
retrieval only, no generation. It exposes a single entry point:

```
retrieve(query, k=10, mode="hybrid")
# mode ∈ sparse | dense | graph | neo4j_graph | hybrid | hybrid_graph
```

returning ranked `RetrievedChunk` records with **full lineage**:
`lineage_id`, `source_methods` (which channels surfaced the chunk), `score`
(after fusion), and `graph_edge_type` (which graph edge boosted it, if any).

## What it does

- **sparse** - Okapi BM25 over an invariant ref-aware tokenizer, so
  `Article 5(2)(a)` tokenises identically in documents and in queries
  (a small but load-bearing detail: without it, references in user queries
  would never match references in the corpus).
- **dense** - sentence-encoder embeddings (base MiniLM) + FAISS cosine;
  fine-tuned LoRA variants plug in as additional encoders.
- **graph** - seed from BM25, expand 1–2 hops over the `04` graph.
- **`neo4j_graph`** - the same over the live Neo4j store.
- **hybrid / hybrid_graph** - **Reciprocal Rank Fusion**
  (`src/retrieval/fusion.py`) of sparse + dense, with or without the graph
  neighbour list.

## Inputs / outputs

- **In:** `03b` chunks, `04` graph, dense indexes.
- **Out:** the `retrieve()` callable + per-mode retrieval evaluations
  (`notebooks/data/retrieval/retrieval_eval_*.jsonl`).

## Where the logic lives

`src/retrieval/__init__.py` (the `retrieve()` entry), `src/retrieval/{
dense,sparse,graph}.py` (channels), `src/retrieval/fusion.py` (RRF),
`src/retrieval/fine_tune.py` (LoRA adapters).

## Why it matters to the thesis

This is the **hybrid** result: `08` finds `hybrid` (Recall@10 0.933) is
**statistically better than pure dense** (+0.100, p = 0.0143) and sits in
the top band with `sparse` and `graph`. It is also where the **measured
negative** is born: `hybrid_graph` - the *measured* attempt to fold the
graph in as a third RRF channel - ties on Recall@10 but is *below* `hybrid`
on MRR/nDCG (0.520 / 0.620 vs 0.710 / 0.766). The thesis's honesty is
*measured* here, not asserted.

## Key caveat

The fusion is **Reciprocal Rank Fusion** - a rank-level blend, not a
score-level one. That is the most common RAG fusion, but it is also why
`hybrid_graph` can underperform: injecting a third list at the rank level
dilutes without adding recall when the extra list's items are already
covered by the first two. A *score-aware* fusion (or a re-ranker) is the
Tier-2 future work in `08`.
