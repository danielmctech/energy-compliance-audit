"""Phase 6 -- Retrieval: vector, hybrid, graph traversal, provenance.

Uses the fixture graph + the *fake* embedder (deterministic, offline) so the
vector-seed -> traversal pipeline is fully exercised without Ollama.
"""
from __future__ import annotations

import graphrag_n4j.context as ctx
from graphrag_n4j.retriever import (
    EDGE_KINDS,
    Neo4jGraphRetriever,
)
from fixtures import FakeEmbedder


# ---------------------------------------------------------------------------
# vector retrieval (seed selection)
# ---------------------------------------------------------------------------

def test_vector_seed_selection(fixture_graph, driver):
    """A known query ('publishing ...') seeds the node with that vector."""
    r = Neo4jGraphRetriever(driver, max_hops=2, embedder=FakeEmbedder(), top_k=3, min_score=0.5)
    out = r.search("publishing requirements")
    assert r.last_debug is not None
    seed_ids = set(r.last_debug.seed_ids)
    assert "fixture:doc:a:article:1" in seed_ids, seed_ids


def test_vector_seed_is_deterministic(fixture_graph, driver):
    """Same query -> same seed (fake embedder is a pure function)."""
    r = Neo4jGraphRetriever(driver, max_hops=2, embedder=FakeEmbedder(), top_k=3, min_score=0.5)
    a = r.search("publishing requirements")
    b = r.search("publishing requirements")
    assert [x[0] for x in a] == [x[0] for x in b]


# ---------------------------------------------------------------------------
# graph traversal (bounded Cypher, hop/edge correctness)
# ---------------------------------------------------------------------------

def test_bounded_traversal_reachable(fixture_graph, driver):
    """From the 'publishing' seed, A2 is reachable at hop 1 (CROSS_REFERENCES)."""
    r = Neo4jGraphRetriever(driver, max_hops=2, embedder=FakeEmbedder(), top_k=3, min_score=0.5)
    out = dict((lid, (hop, kinds)) for lid, hop, kinds in r.search("publishing"))
    assert "fixture:doc:a:article:2" in out
    assert out["fixture:doc:a:article:2"][0] <= 2
    assert "CROSS_REFERENCES" in out["fixture:doc:a:article:2"][1]


def test_edge_kind_filter_excludes(fixture_graph, driver):
    """With only APPLIES_TO allowed (and a high min_score so the seed is
    exactly the 'publishing' node), A2 -- reachable only via
    CROSS_REFERENCES -- is excluded."""
    r = Neo4jGraphRetriever(driver, max_hops=1, embedder=FakeEmbedder(),
                            edge_kinds=("APPLIES_TO",),
                            top_k=3, min_score=0.5)
    out = {lid for lid, _h, _k in r.search("publishing requirements")}
    assert "fixture:doc:a:article:2" not in out


def test_unknown_edge_kind_rejected(driver):
    import pytest
    with pytest.raises(ValueError, match="not in known schema"):
        Neo4jGraphRetriever(driver, embedder=FakeEmbedder(), edge_kinds=("BOGUS",))


def test_max_hops_too_low_rejected(driver):
    import pytest
    with pytest.raises(ValueError, match="max_hops"):
        Neo4jGraphRetriever(driver, max_hops=0, embedder=FakeEmbedder())


# ---------------------------------------------------------------------------
# Cypher safety
# ---------------------------------------------------------------------------

def test_cypher_is_safe_and_readonly(fixture_graph, driver, capsys):
    r = Neo4jGraphRetriever(driver, max_hops=2, embedder=FakeEmbedder())
    q = r._traversal_query()
    # no user-controlled text is interleaved; relationship types are literals
    assert "MERGE" not in q.upper()
    assert "CREATE" not in q.upper()
    assert "DELETE" not in q.upper()
    assert "DROP" not in q.upper()
    # edge kinds appear only inside a whitelist literal
    for k in EDGE_KINDS:
        assert k in q


# ---------------------------------------------------------------------------
# provenance preservation
# ---------------------------------------------------------------------------

def test_provenance_preservation(fixture_graph, driver):
    """Every ranked lineage resolves back to a doc_id (provenance)."""
    r = Neo4jGraphRetriever(driver, max_hops=2, embedder=FakeEmbedder(), top_k=3, min_score=0.5)
    out = {lid for lid, _h, _k in r.search("publishing")}
    with driver.session(database="neo4j") as s:
        for lid in out:
            row = s.run(
                "MATCH (n) WHERE n.lineage_id = $lid "
                "RETURN n.lineage_id AS id, n.doc_id AS doc",
                {"lid": lid},
            ).single()
            assert row["doc"], f"no provenance doc_id for {lid}"


# ---------------------------------------------------------------------------
# context + provenance (no-live-LLM path, pure functions)
# ---------------------------------------------------------------------------

def test_build_context_sections():
    ranked = [
        ("fixture:doc:a:article:1", 1, ["CROSS_REFERENCES"]),
        ("fixture:doc:a:article:2", 0, []),
    ]
    chunk_lookup = {
        "fixture:doc:a:article:1": {
            "doc_id": "fixture:doc:a", "node_type": "article",
            "title": "Article 1", "text": "body one",
        },
        "fixture:doc:a:article:2": {
            "doc_id": "fixture:doc:a", "node_type": "article",
            "title": "Article 2", "text": "body two",
        },
    }
    c = ctx.build_context(ranked, chunk_lookup)
    assert "## Relevant Documents" in c["documents"]
    assert "fixture:doc:a" in c["documents"]
    assert "## Relevant Chunks" in c["chunks"]
    assert "## Provenance" in c["provenance"]
    # both lineage ids surface in provenance (order preserved)
    assert "fixture:doc:a:article:1" in c["provenance"]
    assert "fixture:doc:a:article:2" in c["provenance"]


def test_empty_context_renders_none():
    c = ctx.build_context([], {})
    assert "(none)" in c["documents"]
    assert "(none)" in c["chunks"]
    assert "(none)" in c["provenance"]


def test_format_question_renders():
    out = ctx.format_question("CTX", "Q?")
    assert "Context:\nCTX" in out
    assert "Question:\nQ?" in out


def test_system_prompt_forbids_invention():
    assert "Do not invent" in ctx.SYSTEM_PROMPT or "Do not guess" in ctx.SYSTEM_PROMPT
    assert "insufficient" in ctx.SYSTEM_PROMPT.lower()
