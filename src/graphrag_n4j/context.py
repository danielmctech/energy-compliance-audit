"""Context construction for the ``neo4j_graph``-based GraphRAG layer (P4).

Builds a deterministic, provenance-preserving context from a ranked
``[(lineage_id, hop, [edge_kinds])]`` list (the exact shape
``Neo4jGraphRetriever.search`` produces) + a per-lineage chunk lookup.

Context construction sections:

    ## Relevant Documents
    ## Relevant Chunks
    ## Graph relationships
    ## Provenance

Rules:
  * no invented content -- every line is either a lookup result or a
    constant string we control;
  * provenance preserved -- every chunk line carries its doc_id;
  * bounded -- hard caps on items per section so the prompt stays small
    even for high-fanout queries.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: hard caps; keep the rendered context under ~500 tokens
MAX_SECTION_LINES = 20
MAX_CHAR_PER_LINE = 280
PROVENANCE_LIMIT = 40


def _truncate(text: Optional[str], n: int = MAX_CHAR_PER_LINE) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "\u2026"


def _edge_str(kinds: Sequence[str], hop: int) -> str:
    if hop == 0:
        return "seed"
    return (";".join(kinds) + f"@{hop}hop") if kinds else f"@{hop}hop"


def build_context(
    ranked: Sequence[Tuple[str, int, Sequence[str]]],
    chunk_lookup: Mapping[str, Mapping[str, Any]],
    *,
    entities: Optional[Sequence[str]] = None,
    entity_lookup: Optional[Mapping[str, Mapping[str, Any]]] = None,
    max_lines: int = MAX_SECTION_LINES,
) -> Dict[str, str]:
    """Render the ``Sections`` mapping (see module docstring).

    ``ranked``       ``[(lineage_id, hop, [edge_kinds])]`` in retrieval
                      order;
    ``chunk_lookup`` ``{lineage_id: dict with text, doc_id, node_type,
                      and any of number/title/term/search_text}``;
    ``max_lines``    cap on items per section (default 20).

    Empty sections render as ``(none)`` so the LLM sees the section was
    checked, not skipped.
    """
    ranked = list(ranked)
    entities = list(entities or [])
    entity_lookup = dict(entity_lookup or {})

    docs: List[str] = []
    for lid, _h, _k in ranked:
        doc = (chunk_lookup.get(lid) or {}).get("doc_id") or ""
        if doc and doc not in docs:
            docs.append(doc)
        if len(docs) >= max_lines:
            break

    chunk_lines: List[str] = []
    for lid, hop, kinds in ranked[:max_lines]:
        c = chunk_lookup.get(lid) or {}
        label = c.get("node_type") or "article"
        head = _truncate(c.get("title") or c.get("term") or c.get("search_text") or "", 120)
        body = _truncate(c.get("text") or "", 200)
        parts = [f"- [{label}] {lid}"]
        if head:
            parts.append(head)
        if body and body != head:
            parts.append(body)
        parts.append(f"({_edge_str(kinds, hop)})")
        chunk_lines.append(" ".join(parts).strip())

    entity_lines: List[str] = []
    for lid in entities[:max_lines]:
        c = entity_lookup.get(lid) or {}
        display = (
            c.get("term")
            or c.get("search_text")
            or c.get("lineage_id")
            or ""
        )
        doc = c.get("doc_id") or ""
        entity_lines.append(f"- {display}  (from {doc})" if doc else f"- {display}")

    rel_lines = [
        f"{lid}  ({_edge_str(kinds, hop)})"
        for lid, hop, kinds in ranked if hop >= 1
    ]

    prov_lines = [str(lid) for lid, _h, _k in ranked[:PROVENANCE_LIMIT]]

    def section(title: str, lines: List[str]) -> str:
        return "## " + title + "\n" + ("\n".join(lines) if lines else "(none)")

    return {
        "documents": section("Relevant Documents", docs),
        "chunks": section("Relevant Chunks", chunk_lines),
        "entities": section("Relevant Entities", entity_lines),
        "relationships": section("Graph relationships", rel_lines[:max_lines]),
        "provenance": section("Provenance", prov_lines),
    }


# ---------------------------------------------------------------------------
# prompt (dedicated to GraphRAG; kept in this module rather than
# embedded in notebooks)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a legal-research assistant over a local knowledge
graph of European energy-market legislation and agency guidance.

Answer ONLY from the context supplied below.

Rules:
1. If the context is insufficient to answer, say "The supplied context is
   insufficient to answer this question." and stop.  Do not guess.
2. Do not invent articles, relationships, entities, obligations, or facts
   that are not in the context.
3. Cite the lineage_id (e.g. `remit_1227_2011:article:4`) of every
   important claim you make.
4. Where a graph relationship path supports the answer, name the
   relationship (e.g. CROSS_REFERENCES, AMENDS, DEFINED_IN, APPLIES_TO).
5. If the context contains a conflict between sources, state the conflict
   explicitly.
6. Preserve uncertainty in the source (e.g. "may", "should", "shall")
   -- do not strengthen a "may" into a "must".
"""

QUESTION_TEMPLATE = """Context:
{context}

Question:
{question}

Answer:"""


def format_question(context: str, question: str) -> str:
    """Render the user-facing prompt ("answer only from context")."""
    return QUESTION_TEMPLATE.replace("{context}", context).replace(
        "{question}", question)

__all__ = [
    "MAX_SECTION_LINES",
    "MAX_CHAR_PER_LINE",
    "PROVENANCE_LIMIT",
    "SYSTEM_PROMPT",
    "QUESTION_TEMPLATE",
    "build_context",
    "format_question",
]
