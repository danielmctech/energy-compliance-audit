"""Neo4j schema (Community-safe): uniqueness constraint + vector + fulltext.

Design constraints:
  * **NODE KEY is Enterprise-only** on Neo4j Community.  We therefore
    enforce uniqueness with a standard
    ``CREATE CONSTRAINT ... IF NOT EXISTS ... REQUIRE lineage_id IS UNIQUE``
    on each node label and look up nodes by ``lineage_id`` (never by
    elementId / node-key int id).  This is also why ``neo4j_graphrag``'s
    own ``upsert_vector(s)`` helpers (which assume node-key int ids) are
    NOT used -- ``ingestion.upsert_nodes`` MERGEs on ``lineage_id`` instead.

  * ``IF NOT EXISTS`` on CREATE (constraint / index) makes creation
    idempotent, so :func:`apply_schema` / :func:`build_indexes` are safe
    to re-run without first DROPPing anything.

Sibling-of-parent import: ``neo4j_config`` lives in ``src/``, one level up
from this package.  The repo-wide convention (see ``src/retrieval/_corpus.py``
for ``common``) is to try the top-level import first and fall back to
adding the parent dir to ``sys.path``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from neo4j_config import graphrag_settings
except ImportError:  # e.g. running as `python -c "import graphrag_n4j"`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from neo4j_config import graphrag_settings


# property holding the bge-m3 vector on embeddable nodes
EMBEDDING_PROPERTY = "emb"

# every node label used by the schema (matches KIND_TO_LABEL values in
# ingestion.py -- kept as a local tuple so `schema` has zero dependency on
# `ingestion`; the two files share this constant by convention).
NODE_LABELS: tuple[str, ...] = (
    "Document", "ExternalDocument", "Article", "ExternalArticle",
    "Preamble", "Term", "Entity",
)


def _vector_index_name(label: str, embedding_property: str) -> str:
    # neo4j index name rule: [a-zA-Z][a-zA-Z0-9_]*
    return f"v_{label}_{embedding_property}".lower()


def _fulltext_index_name(label: str) -> str:
    return f"ft_{label}".lower()


def apply_schema(driver, database: Optional[str] = None, log=print) -> None:
    """Create the ``lineage_id IS UNIQUE`` constraint on every node label.

    Idempotent (``IF NOT EXISTS``).
    """
    for label in NODE_LABELS:
        q = (
            f"CREATE CONSTRAINT lineage_unique_{label} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.lineage_id IS UNIQUE"
        )
        with driver.session(database=database) as s:
            s.run(q)
        log(f"constraint OK   {label}.lineage_id IS UNIQUE")


def build_indexes(driver, database: Optional[str] = None, log=print) -> None:
    """Create the vector indexes (Term + Article, bge-m3 dim) + fulltext index.

    Idempotent (``IF NOT EXISTS``).  The vector indexes are what the official
    ``neo4j_graphrag.retrievers.VectorRetriever`` (P3/P5) and the
    ``neo4j_graph`` retriever (P2) search.
    """
    # deferred import: ``neo4j_graphrag.indexes`` imports a lot (openai,
    # pydantic, etc.) and we don't want to force that on people who only
    # call :func:`apply_schema`.
    from neo4j_graphrag.indexes import (
        create_fulltext_index,
        create_vector_index,
    )

    g = graphrag_settings()
    dim = g.embedding_dim

    # vector indexes: one per embeddable node type that carries ``emb``.
    # Term = the strongest semantic seed (defined terms); Article = the
    # seed that directly maps to a retrievable chunk, so an article-titled
    # query can anchor expansion even when no single term matches.
    for label in ("Term", "Article"):
        create_vector_index(
            driver,
            name=_vector_index_name(label, EMBEDDING_PROPERTY),
            label=label,
            embedding_property=EMBEDDING_PROPERTY,
            dimensions=dim,
            similarity_fn="cosine",
            fail_if_exists=False,
            neo4j_database=database,
        )
        log(f"vector index OK   {label}.{EMBEDDING_PROPERTY} ({dim}-d, cosine)")

    create_fulltext_index(
        driver,
        name=_fulltext_index_name("Term"),
        label="Term",
        node_properties=["search_text"],
        fail_if_exists=False,
        neo4j_database=database,
    )
    log("fulltext index OK Term.search_text")


__all__ = [
    "EMBEDDING_PROPERTY",
    "NODE_LABELS",
    "apply_schema",
    "build_indexes",
]
