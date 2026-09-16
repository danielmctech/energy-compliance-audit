# 03b - AST Construction

**Pipeline position:** `01 → 02 → 03a → 03b → 04 → 05 → 06`
**Thesis importance:** **Core contract** - it defines the
**`lineage_id`** that is the join key for the whole GraphRAG + evaluation
stack. This is the single most load-bearing notebook for the *claim*.

## Purpose

Turn the `03a` structure maps into a per-document **Abstract Syntax Tree**
(document → preamble / chapters / articles → paragraphs / tables, every node
carrying a char-offset span into the source markdown) plus a **chunk list**
per the document's `03a` chunking strategy. The AST is the canonical
hierarchical view of a regulation; the chunks are the retrievable units.

## Key contribution: the `lineage_id` contract (Step-0 schema)

Every node and chunk carries a stable `lineage_id` - the **RAG input
contract**:

- `{doc}:document`, `{doc}:preamble`, `{doc}:chapter:{n}`
- `{doc}:article:{n}`, `{doc}:article:{n}:para:{p}`, `{doc}:table:{seq}`
- `{doc}:sentence:{start}-{end}` (sentence-strategy documents)

This is the single thread that ties `04 → 05 → 06 → 08` together: `04`
attaches graph **edges keyed by lineage**, `05` reports **provenance**
(chunk + lineage + methods + boosting edge), `06` renders **per-line
provenance**, and `08` can cite *which chunk and which edge* supported an
answer. If a later notebook is confusing, the question is usually *what does
it do to the lineage*.

## What it does

- Build the AST with `src/structure/ast_builder.py` (deterministic, regex
  only, no LLM).
- Re-derive spans from the same markdown through `structure_maps.
  find_structural_articles`, so the AST and the `03a` map are **always
  consistent** (no drift between the two).
- Apply the `03a` chunking strategy, emitting per-document chunk lists that
  obey the stated invariants.

## Inputs / outputs

- **In:** `03a` structure maps, shared processed markdown.
- **Out:** `notebooks/data/ast/*_ast.json`, `chunks_*.json`,
  `ast_summary.csv`.

## Where the logic lives

`src/structure/ast_builder.py`; `src/structure/structure_maps.py` (shared
helpers for consistency).

## Why it matters to the thesis

This is the **thesis's central artefact**: the auditable, lineaged,
deterministic representation that makes "retrieval of the right evidence"
even meaningful for legal text. It is the difference between a chunk that is
"paragraph 12 of something" and a chunk that is "Article N(2)(a) of
Regulation M, paragraph P, table T" - which is what a compliance auditor can
cite in a finding.

## Key caveat

The AST is only as good as the `03a` structure extraction it re-uses - its
boundary cases (renumbered/consolidated articles, non-standard numbering)
are inherited, not solved here. The span re-derivation prevents drift *against
`03a`*, but it cannot fix a wrong article boundary that `03a` itself
produced.
