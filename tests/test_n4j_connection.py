"""Phase 6 -- Neo4j: connection, constraints, indexes.

All tests run against the local Neo4j Community server (skipped if it is
not reachable).
"""
from __future__ import annotations

import graphrag_n4j.schema as sch


def test_connection(driver):
    """A readable query executes (driver + session + auth all work)."""
    with driver.session(database="neo4j") as s:
        assert s.run("RETURN 1 AS ok").single()["ok"] == 1


def test_constraints_exist(driver):
    """Every node label has the ``lineage_id IS UNIQUE`` constraint."""
    with driver.session(database="neo4j") as s:
        rows = list(s.run(
            "SHOW CONSTRAINTS "
            "YIELD name, labelsOrTypes, properties, type"
        ))
    # a ``lineage_id IS UNIQUE`` constraint shows as a UNIQUENESS predicate
    # keyed on linead_id.
    constrained = {
        label
        for r in rows
        if r["properties"] == ["lineage_id"]
        and (r["type"] or "").upper() in ("UNIQUENESS", "NODE_PROPERTY_UNIQUENESS")
        for label in r["labelsOrTypes"]
    }
    missing = set(sch.NODE_LABELS) - constrained
    assert not missing, f"labels missing lineage_id uniqueness: {missing}"


def test_vector_indexes_exist(driver):
    with driver.session(database="neo4j") as s:
        names = {
            r["name"]
            for r in s.run("SHOW VECTOR INDEXES YIELD name")
        }
    assert "v_term_emb" in names
    assert "v_article_emb" in names


def test_fulltext_index_exists(driver):
    with driver.session(database="neo4j") as s:
        names = {r["name"] for r in s.run("SHOW FULLTEXT INDEXES YIELD name")}
    assert "ft_term" in names


def test_apply_schema_is_idempotent(driver):
    """Running the schema creation twice raises nothing and leaves the
    constraint count unchanged."""
    with driver.session(database="neo4j") as s:
        n_before = len(list(s.run("SHOW CONSTRAINTS YIELD name")))

    sch.apply_schema(driver, "neo4j", log=lambda *a: None)
    sch.apply_schema(driver, "neo4j", log=lambda *a: None)

    with driver.session(database="neo4j") as s:
        n_after = len(list(s.run("SHOW CONSTRAINTS YIELD name")))
    assert n_after == n_before


def test_build_indexes_is_idempotent(driver):
    """``build_indexes`` can be re-run without errors (IF NOT EXISTS)."""
    sch.build_indexes(driver, "neo4j", log=lambda *a: None)
    sch.build_indexes(driver, "neo4j", log=lambda *a: None)
    with driver.session(database="neo4j") as s:
        names = {r["name"] for r in s.run("SHOW VECTOR INDEXES YIELD name")}
    assert {"v_term_emb", "v_article_emb"} <= names
