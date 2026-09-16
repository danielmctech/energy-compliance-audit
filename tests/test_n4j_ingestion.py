"""Phase 6 -- Ingestion: entity/relationship creation, duplicate prevention,
malformed input.

Runs against the local Neo4j Community server using the small fixture graph
(:mod:`fixtures`), never the production dataset.
"""
from __future__ import annotations

import pytest

import graphrag_n4j.ingestion as ing
from fixtures import (
    FIXTURE_EDGES,
    FIXTURE_EMBEDDINGS,
    FIXTURE_NODES,
    build_fixture,
    wipe_fixture,
)

DB = "neo4j"


# ---------------------------------------------------------------------------
# entity creation
# ---------------------------------------------------------------------------

def test_entity_creation(fixture_graph, driver):
    with driver.session(database=DB) as s:
        n = s.run(
            "MATCH (x) WHERE x.lineage_id STARTS WITH 'fixture:' "
            "RETURN count(x) AS c"
        ).single()["c"]
    assert n == len(FIXTURE_NODES)

    with driver.session(database=DB) as s:
        a1 = s.run(
            "MATCH (a:Article {lineage_id: 'fixture:doc:a:article:1'}) "
            "RETURN a.title AS title, a.number AS number, a.doc_id AS doc"
        ).single()
    assert a1["title"] == "Article 1"
    assert a1["number"] == 1
    assert a1["doc"] == "fixture:doc:a"


# ---------------------------------------------------------------------------
# relationship creation
# ---------------------------------------------------------------------------

def test_relationship_creation(fixture_graph, driver):
    with driver.session(database=DB) as s:
        e = s.run(
            "MATCH (a)-[r]->(b) WHERE a.lineage_id STARTS WITH 'fixture:' "
            "AND b.lineage_id STARTS WITH 'fixture:' "
            "RETURN count(r) AS c"
        ).single()["c"]
    assert e == len(FIXTURE_EDGES)

    with driver.session(database=DB) as s:
        xref = s.run(
            "MATCH (a {lineage_id: 'fixture:doc:a:article:1'})"
            "-[r:CROSS_REFERENCES]->"
            "(b {lineage_id: 'fixture:doc:a:article:2'}) "
            "RETURN r.snippet AS snip"
        ).single()
    assert xref["snip"] == "as referred to in Article 2"


def test_defined_in_and_applies_to(fixture_graph, driver):
    with driver.session(database=DB) as s:
        di = s.run(
            "MATCH (t:Term {lineage_id: 'fixture:term:capacity'})"
            "-[r:DEFINED_IN]->(a) RETURN a.lineage_id AS lid"
        ).single()
        at = s.run(
            "MATCH (a {lineage_id: 'fixture:doc:a:article:2'})"
            "-[r:APPLIES_TO]->(en) RETURN en.lineage_id AS lid"
        ).single()
    assert di["lid"] == "fixture:doc:a:article:1"
    assert at["lid"] == "fixture:entity:operator"


# ---------------------------------------------------------------------------
# duplicate prevention (idempotent MERGE)
# ---------------------------------------------------------------------------

def test_node_idempotency(fixture_graph, driver):
    """Re-ingesting the same nodes produces no new rows and preserves counts."""
    nodes = [dict(n) for n in FIXTURE_NODES]
    for n in nodes:
        if n["lineage_id"] in FIXTURE_EMBEDDINGS:
            n["emb"] = FIXTURE_EMBEDDINGS[n["lineage_id"]]
    before = _fixture_node_count(driver)
    ing.upsert_nodes(driver, nodes, DB, log=lambda *a: None)
    ing.upsert_nodes(driver, nodes, DB, log=lambda *a: None)
    assert _fixture_node_count(driver) == before


def test_edge_idempotency(fixture_graph, driver):
    before = _fixture_edge_count(driver)
    ing.upsert_edges(driver, list(FIXTURE_EDGES), nodes=list(FIXTURE_NODES),
                     database=DB, log=lambda *a: None)
    ing.upsert_edges(driver, list(FIXTURE_EDGES), nodes=list(FIXTURE_NODES),
                     database=DB, log=lambda *a: None)
    assert _fixture_edge_count(driver) == before
    # and the property-set semantics did not duplicate the xref edge
    with driver.session(database=DB) as s:
        c = s.run(
            "MATCH (a {lineage_id: 'fixture:doc:a:article:1'})"
            "-[r:CROSS_REFERENCES]->() RETURN count(r) AS c"
        ).single()["c"]
    assert c == 1


def _fixture_node_count(driver) -> int:
    with driver.session(database=DB) as s:
        return s.run(
            "MATCH (x) WHERE x.lineage_id STARTS WITH 'fixture:' "
            "RETURN count(x) AS c"
        ).single()["c"]


def _fixture_edge_count(driver) -> int:
    with driver.session(database=DB) as s:
        return s.run(
            "MATCH (a)-[r]->(b) WHERE a.lineage_id STARTS WITH 'fixture:' "
            "AND b.lineage_id STARTS WITH 'fixture:' "
            "RETURN count(r) AS c"
        ).single()["c"]


# ---------------------------------------------------------------------------
# malformed input
# ---------------------------------------------------------------------------

def test_malformed_node_rejected(fixture_graph, driver):
    before = _fixture_node_count(driver)
    with pytest.raises(ValueError, match="malformed node"):
        ing.upsert_nodes(driver, [
            {"lineage_id": "fixture:doc:a:article:1", "title": "t"},
            {"kind": "bogus_kind", "lineage_id": "y"},
        ], DB, log=lambda *a: None)
    # validation happens before any write: graph state is untouched
    assert _fixture_node_count(driver) == before
    with driver.session(database=DB) as s:
        assert s.run(
            "MATCH (n) WHERE n.lineage_id = 'y' RETURN count(n) AS c"
        ).single()["c"] == 0


def test_malformed_edge_rejected_unknown_kind(fixture_graph, driver):
    with pytest.raises(ValueError, match="unknown kind"):
        ing.upsert_edges(driver, [
            {"src": "fixture:doc:a:article:1",
             "dst": "fixture:doc:a:article:2",
             "kind": "HACK; DROP INDEX v_term_emb"},
        ], nodes=list(FIXTURE_NODES), database=DB, log=lambda *a: None)


def test_malformed_edge_rejected_missing_endpoint(fixture_graph, driver):
    with pytest.raises(ValueError, match="endpoints"):
        ing.upsert_edges(driver, [
            {"src": None, "dst": "fixture:doc:a:article:1",
             "kind": "AMENDS"},
        ], nodes=list(FIXTURE_NODES), database=DB, log=lambda *a: None)
