"""Small deterministic fixture graph + fakes for the test suite.

Requirement (testing):

    "Tests should not require a large production graph.
     Use a small fixture graph where possible."

The fixture lives in the real ``neo4j`` database (in this environment
``CREATE DATABASE`` is rejected -- ``Neo.ClientError.
Statement.UnsupportedAdministrationCommand``) but is kept separate from the
2,049-node / 4,139-edge production graph by a ``fixture:`` prefix on every
lineage_id.  :func:`wipe_fixture` removes all of it; tests never touch the
production nodes.

Vector seeding stays deterministic **and** offline:
the fixture Article nodes carry 1024-d basis vectors (same dimension as the
production bge-m3 indexes, so the official ``VectorRetriever`` happily
searches them), and :class:`FakeEmbedder` maps the *first token* of a query
to exactly that basis vector -- so known test queries seed known fixture
nodes with cosine 1.0, no Ollama call required.
"""
from __future__ import annotations

import zlib
from typing import Any, Dict, List

DB = "neo4j"
DIM = 1024


def token_vec(token: str) -> List[float]:
    """1024-d basis vector for ``token`` (shared by FakeEmbedder + fixtures)."""
    vec = [0.0] * DIM
    slot = zlib.crc32(token.lower().encode("utf-8")) % (DIM - 1)
    vec[slot] = 1.0
    vec[DIM - 1] = 0.0001  # make it non-axial, numerically distinct
    return vec


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

def _first_key(text: str) -> str:
    word = (text or "").strip().split()
    return word[0].lower().strip(",.;:") if word else ""


def _embed_text(text: str) -> List[float]:
    """Shared deterministic mapping used by FakeEmbedder (and test queries)."""
    key = _first_key(text)
    if key in ("publishing", "inside"):
        return token_vec(key)
    return token_vec("zzz-unknown-key")


class FakeEmbedder:
    """Deterministic query embedder satisfying the official ``Embedder`` ABC.

    ``embed_query(text)`` -> the basis vector of the *first* token of the
    text.  So the test query ``"publishing ..."`` seeds the fixture node
    that stores ``token_vec("publishing")`` at cosine 1.0 -- fully offline,
    no Ollama call.
    """

    def embed_query(self, text: str, **_kw: Any) -> List[float]:
        return _embed_text(text)


class FakeLLM:
    """Records every ``invoke``; returns a constant answer.

    Lets the GraphRAG generation path be asserted on (prompt content,
    answer, call count) with zero live-LLM calls.
    """

    def __init__(self, answer: str = "ANSWERED FROM CONTEXT") -> None:
        self.answer = answer
        self.calls: List[Dict[str, Any]] = []

    class _Res:
        def __init__(self, content: str) -> None:
            self.content = content

    def invoke(self, input: Any, *a, **kw) -> "_Res":
        self.calls.append({"input": input})
        return self._Res(self.answer)


# ---------------------------------------------------------------------------
# fixture graph
# ---------------------------------------------------------------------------
#
#   term:capacity        article A1        article A2        article B1
#        |                 |                 |
#        | DEFINED_IN      | CROSS_REFERERS  | DEFINED_IN
#        +---------------->+---------------->+  term:inside_information
#                                    |
#                                    | APPLIES_TO
#                                    +-> entity:operator
#
# A1 carries the "publishing" basis vector, A2 the "inside" basis vector.

FIXTURE_NODES: List[Dict[str, Any]] = [
    dict(kind="document", lineage_id="fixture:doc:a", title="Regulation A",
         doc_id="fixture:doc:a"),
    dict(kind="document", lineage_id="fixture:doc:b", title="Regulation B",
         doc_id="fixture:doc:b"),
    dict(kind="article", lineage_id="fixture:doc:a:article:1",
         title="Article 1", number=1, doc_id="fixture:doc:a",
         text="Article 1 defines the capacity of a market participant."),
    dict(kind="article", lineage_id="fixture:doc:a:article:2",
         title="Article 2", number=2, doc_id="fixture:doc:a",
         text="Article 2 says who must publish inside information."),
    dict(kind="article", lineage_id="fixture:doc:b:article:1",
         title="Article 1", number=1, doc_id="fixture:doc:b",
         text="Article 1 of regulation B describes reporting."),
    dict(kind="preamble", lineage_id="fixture:doc:a:preamble",
         doc_id="fixture:doc:a",
         text="Preamble of regulation A explains the purpose."),
    dict(kind="term", lineage_id="fixture:term:capacity", term="capacity",
         doc_id="fixture:doc:a", search_text="capacity"),
    dict(kind="term", lineage_id="fixture:term:inside_information",
         term="inside information", doc_id="fixture:doc:a",
         search_text="inside information"),
    dict(kind="entity", lineage_id="fixture:entity:operator",
         label="market operator", doc_id="fixture:doc:a",
         search_text="market operator"),
]

FIXTURE_EDGES: List[Dict[str, Any]] = [
    dict(src="fixture:term:capacity", dst="fixture:doc:a:article:1",
         kind="DEFINED_IN"),
    dict(src="fixture:term:inside_information",
         dst="fixture:doc:a:article:2", kind="DEFINED_IN"),
    dict(src="fixture:doc:a:article:1", dst="fixture:doc:a:article:2",
         kind="CROSS_REFERENCES", snippet="as referred to in Article 2"),
    dict(src="fixture:doc:a:article:2", dst="fixture:entity:operator",
         kind="APPLIES_TO"),
]

FIXTURE_EMBEDDINGS = {
    "fixture:doc:a:article:1": token_vec("publishing"),
    "fixture:doc:a:article:2": token_vec("inside"),
}


def _driver():
    from neo4j_config import make_driver
    return make_driver()


def wipe_fixture(driver=None) -> None:
    """Drop every fixture node (and its relationships). Idempotent."""
    d = driver or _driver()
    try:
        with d.session(database=DB) as s:
            s.run(
                "MATCH (n) WHERE n.lineage_id STARTS WITH 'fixture:' "
                "DETACH DELETE n"
            )
    finally:
        d.close()


def build_fixture(driver=None) -> None:
    """Create the fixture nodes + edges + basis embeddings.

    Reuses the production schema helpers (constraints are the same
    ``lineage_id IS UNIQUE`` shape) -- only labels/ids come from here.
    """
    from graphrag_n4j.ingestion import KIND_TO_LABEL, upsert_nodes, upsert_edges
    from graphrag_n4j.schema import apply_schema

    d = driver or _driver()
    try:
        apply_schema(d, DB, log=lambda *a: None)

        nodes = [dict(n) for n in FIXTURE_NODES]
        for n in nodes:
            emb = FIXTURE_EMBEDDINGS.get(n["lineage_id"])
            if emb is not None:
                n["emb"] = emb
        upsert_nodes(d, nodes, DB, log=lambda *a: None)

        upsert_edges(d, FIXTURE_EDGES, nodes=nodes, database=DB,
                     log=lambda *a: None)

        # store the embeddings directly (no Ollama)
        for lid, vec in FIXTURE_EMBEDDINGS.items():
            with d.session(database=DB) as s:
                s.run(
                    "MATCH (n) WHERE n.lineage_id = $lid SET n.emb = $v",
                    {"lid": lid, "v": vec},
                )
    finally:
        d.close()


__all__ = [
    "DB", "DIM", "token_vec",
    "FakeEmbedder", "FakeLLM",
    "FIXTURE_NODES", "FIXTURE_EDGES", "FIXTURE_EMBEDDINGS",
    "wipe_fixture", "build_fixture",
]
