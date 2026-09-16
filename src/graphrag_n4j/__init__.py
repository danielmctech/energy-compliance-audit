"""Neo4j GraphRAG ingestion, schema (P1) and -- later -- retrieval.

Renamed from an earlier ``src/neo4j`` package because that name shadows
the ``neo4j`` *driver* (when ``src/`` is on ``sys.path``, ``import neo4j``
resolved to this package and broke ``neo4j_graphrag.types``' reference to
``neo4j.Record``).

P1 scope (this package version):
  - schema: uniqueness constraints + vector + fulltext (Community-safe)
  - ingestion: idempotent MERGE of ``notebooks/data/graph/{nodes,edges}.jsonl``
  - embeddings: bge-m3 (1024-d) via Ollama /v1, cached on disk

P3: ``neo4j_graph`` retriever mode (uses same schema + VectorRetriever).
P4: ``GraphRAG`` query API (retrieval -> context -> local LLM generation)
    over the ``neo4j_graph`` path -- see :mod:`graphrag_n4j.rag` and
    :mod:`graphrag_n4j.context`.
"""
from __future__ import annotations

from .context import (
    SYSTEM_PROMPT,
    build_context,
    format_question,
)
from .ingestion import (
    EMBEDDABLE_KINDS,
    EMBEDDING_CACHE_DIR,
    KIND_TO_LABEL,
    KNOWN_EDGE_KINDS,
    LABEL_TO_KIND,
    counts,
    embed_search_texts,
    existing_embeddings_from_db,
    ingest,
    load_edges,
    load_nodes,
    search_text_for,
    upsert_edges,
    upsert_embeddings,
    upsert_nodes,
)
from .rag import GraphRAG, QueryResult
from .retriever import (
    EDGE_KINDS,
    Neo4jGraphRetriever,
    TARGET_LABELS,
)
from .schema import (
    EMBEDDING_PROPERTY,
    apply_schema,
    build_indexes,
)

__all__ = [
    # ingest / schema
    "KIND_TO_LABEL",
    "LABEL_TO_KIND",
    "EMBEDDABLE_KINDS",
    "EMBEDDING_CACHE_DIR",
    "EMBEDDING_PROPERTY",
    "EDGE_KINDS",
    "TARGET_LABELS",
    "apply_schema",
    "build_indexes",
    "counts",
    "embed_search_texts",
    "existing_embeddings_from_db",
    "ingest",
    "load_edges",
    "load_nodes",
    "search_text_for",
    "upsert_edges",
    "upsert_embeddings",
    "upsert_nodes",
    # retrieval + P4 query API
    "Neo4jGraphRetriever",
    "GraphRAG",
    "QueryResult",
    "SYSTEM_PROMPT",
    "build_context",
    "format_question",
]
