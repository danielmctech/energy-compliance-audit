# 01 - Corpus Normalisation & Metadata Audit

**Pipeline position:** `01 (first) → 02 → 03a → 03b → 04 → 05 → 06`
**Thesis importance:** **Prerequisite** - it protects every downstream
result by catching corpus problems *before* they contaminate retrieval.

## Purpose

Validate the entire EU regulatory corpus **before** ingestion. The corpus
(canonically defined by `outputs/corpus_metadata.json`) is audited end to end,
so that a broken PDF, a missing expected document, or an incomplete metadata
record is detected here - not silently propagated into the graph and later
blamed on the retriever.

## What it does (per section)

The canonical source, `outputs/corpus_metadata.json`, lists every expected
document, its folder, filename, and required metadata fields. The audit then,
in order:

- **2 - Load corpus metadata** - load the manifest and the document inventory.
- **3 - Audit helpers** - a `DocumentAudit` dataclass that collects findings
  per document.
- **4 - Directory structure check** - verify the expected folder layout on
  disk.
- **5 - File presence & integrity** - PDF size, page count, SHA-256, and
  validity for each file.
- **6 - Metadata completeness** - required fields exist and are non-empty.
- **7 - Run full audit on all documents** - the main pass over the corpus.
- **8 - Cross-check manifest ↔ disk** - flag PDFs on disk missing from the
  manifest and manifest entries with no file; persist the full audit.

## Inputs / outputs

- **In:** `outputs/corpus_metadata.json`, `data/raw/eu/**/*.pdf`.
- **Out:** audit tables + a persisted audit report under `notebooks/data/`.

## Where the logic lives

`src/bootstrap_manifest.py` produces `outputs/corpus_metadata.json`; the
auditing helpers and pass logic are in the notebook itself (deterministic, no
LLM calls).

## Why it matters to the thesis

This is the **integrity gate**: every later number in `08` is only as good as
the corpus it was measured on. It also documents the `needs_review` flags on
manually-reviewed entries - an explicit, auditable admission of where the
input is human-maintained rather than machine-extracted.

## Key caveat

It audits *structure and presence*, not *legal correctness of content* - that
is a human-verification task, not a corpus-audit one.
