# 06 - GraphRAG Query API (Neo4j)

**Pipeline position:** `01 → 02 → 03a → 03b → 04 → 05 → 06`
**Thesis importance:** **End-to-end** - the first place retrieval meets
*generation* and provenance, the full pipeline as a question→answer.

## Purpose

Expose and exercise the **GraphRAG query API** built on `05`'s
`neo4j_graph` path: a single `rag.query(question)` returning a full
`result.answer` + `.sources` + `.chunks` + `.entities` + `.relationships` +
`.cypher` + `.retrieval_scores` + `.debug()`. This is the *usable* end of
the pipeline - the thing an auditor would actually call.

## What it does - three strictly separated stages

Implementation in `src/graphrag_n4j/rag.py`:

1. **Retrieval** - `Neo4jGraphRetriever.search` (vector seeds → bounded
   Cypher traversal), already verified in `05`.
2. **Context** - `graphrag_n4j.context.build_context` renders **5 sections**
   (`Documents`, `Chunks`, `Entities`, `Relationships`, `Provenance`) with
   **provenance on every line** - a hard design rule.
3. **Generation** - `neo4j_config.make_llm()` (Ollama, temperature 0.0,
   cached) → the answer.

## Inputs / outputs

- **In:** the `04` graph (loaded into Neo4j via `src/graphrag_n4j/ingestion.py`),
  the question(s).
- **Out:** `result.*` objects plus `notebooks/data/graphrag/
  graphrag_summary.csv` and `graphrag_logs.jsonl`.

## Where the logic lives

`src/graphrag_n4j/rag.py` (entry), `src/graphrag_n4j/retriever.py`
(search), `src/graphrag_n4j/context.py` (context), `src/graphrag_n4j/
schema.py` (Cypher), `src/neo4j_config.py` (LLM + connection).

## Why it matters to the thesis

This is where the *compliance audit* use case is *demonstrated*: query →
answer with **per-line provenance**, which is what makes a machine answer
citeable in a finding. It is also where the **negative result is confirmed
end-to-end**: `08` finds `neo4j` is the *weakest* retrieval
system (Recall@10 0.233, 41 retrieval failures). The GraphRAG arm is
therefore *measured, not assumed* - and the weakness is named and attributed.

## Key caveat

The **weak retrieval** is a real result, not a bug to hide - the question
"is the graph a better retrieval surface than the vector store for
regulatory text?" is answered *no, on this corpus, by a large margin*.
The failure is **not isolated to the storage layer**
(so it is a design question, not a bug), and `08` Tier 3 proposes the
repair (hop-aware context, seed expansion).
