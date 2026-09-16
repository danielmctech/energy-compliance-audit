# 02 - Parser Evaluation

**Pipeline position:** `01 → 02 → 03a → 03b → 04 → 05 → 06`
**Thesis importance:** **Method-critical** - it decides *which* parser
feeds the whole stack, so every choice after it inherits this result.

## Purpose

Compare **3 PDF-to-Markdown parsers** and **2 LLM structure extractors**
against a unified ground truth for the full EU energy-regulation corpus
(25 documents), and lock in the best one. The winner (`pymupdf4llm`,
structure score 1.0 in the recorded result) is what `03a`/`03b` consume - so
this is the fork in the road for the entire text pipeline.

## What it does

- Build a **ground truth** in two tiers: a heuristic v1
  (`ground_truth_all_docs.json`) and an **LLM-reviewed** v2
  (`ground_truth_all_docs_v2.json`).
- Run all 3 parsers over the same PDFs and score structure fidelity against
  the ground truth.
- Run both LLM extractors, but on the **pymupdf4llm markdown as the shared
  input text** - so the LLMs are compared on an identical source, not on
  different parsers' outputs (a fair comparison).
- Emit `best_parser.json`, which `03a`/`03b` read.

## Inputs / outputs

- **In:** `data/raw/eu/**/*.pdf`.
- **Out:** `notebooks/data/ground_truth/ground_truth_all_docs.json` (v1),
  `..._v2.json` (LLM-reviewed), `notebooks/data/evaluations/
  parser_evaluation_<ts>.csv`, `notebooks/data/evaluations/best_parser.json`.

## Where the logic lives

Ground-truth helpers in `src/parsing/ground_truth.py`; LLM extraction in
`src/parsing/llm_extractor.py`; parser implementations in
`src/parsing/parsers.py`.

## Why it matters to the thesis

This is where the thesis earns a **defensible parser choice** rather than an
arbitrary one. It also produces the **LLM-reviewed ground truth reused later**
(`04` graph-structure checks, `03a` map-vs-GT) - so one comparison feeds
multiple chapters. Because all LLM calls are cached
(`notebooks/data/llm_cache/`, keyed by `(model, sha1(prompt))`), this is
reproducible and resumable for free.

## Key caveat

The comparison is **deterministic + cached**; the only non-determinism is
where it comes from is the LLM, which is pinned (temperature 0.0) and cached.
The ground-truth v2 is LLM-*reviewed*, so it carries the same self-referential
risk that `06` flags for the judge - it is a strong heuristic, not a human
oracle.
