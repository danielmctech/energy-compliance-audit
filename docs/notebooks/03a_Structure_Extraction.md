# 03a - Structure Extraction

**Pipeline position:** `01 → 02 → 03a → 03b → 04 → 05 → 06`
**Thesis importance:** **Foundation** - produces the per-document
structure that `03b` tree-ifies and `04` graph-ifies.

## Purpose

Derive **per-document structure maps** from the shared processed markdown
(`data/processed/eu/**/*.md` - the `pymupdf4llm` output chosen in `02`):
structural article headings (number + title + paragraph count),
chapters/titles, recitals, tables, obligations, energy units, and a
chunking-strategy recommendation. It is the *reading* of a document's legal
anatomy, not its raw text.

## What it does (deterministic, regex only - no LLM, no cache, idempotent)

- `structure_maps.find_structural_articles` extracts the structural articles
  (a modelling decision: an article counts as *structural* only if its number
  matches the regulatory pattern).
- Reuses the **same helpers as `02`** for GT fields (title / instrument /
  tables / obligations), so *map vs ground truth* is a true data-quality check
  rather than a detector-mismatch check.
- Emits a structure map per document plus `best_structure_map.json`.

## Inputs / outputs

- **In:** `data/processed/eu/**/*.md`, and the GT fields from `02`.
- **Out:** `notebooks/data/structure_maps/*_structure.json` and
  `notebooks/data/structure_maps/best_structure_map.json`.

## Where the logic lives

`src/structure/structure_maps.py` (all extraction); `src/parsing/
ground_truth.py` for the shared GT field helpers.

## Why it matters to the thesis

Without this, the "regulatory-aware" claim has no substrate: structure maps
are what make a **chunk** a *legal chunk* (article, recital, table) instead of
an arbitrary sentence window, and what let `04` attach *meaningful* edges
(`CROSS_REFERENCES` to a specific article) rather than generic co-occurrence.
It is the source of the **deterministic, no-LLM** character of the middle
pipeline - a defensible, auditable choice.

## Key caveat

"Structural" is a **regex decision**, so its boundary cases (renumbered,
consolidated, or amended articles) are exactly the kind of edge case `06`
lists as a known gap. The recommendation to a chunking strategy is
heuristic - `03b` is where that choice is actually applied.
