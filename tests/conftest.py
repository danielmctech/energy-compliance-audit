"""Shared fixtures for the Phase-6 test suite.

Two layers:
  * pure/unit tests -- no server, no Ollama (context builder, prompts,
    retriever validation, GraphRAG with a fake LLM + canned retrieval);
  * live tests -- the local Neo4j Community server with the *small fixture
    graph* from :mod:`fixtures` (not the 2,049-node production graph).

If Neo4j is not reachable, every live test is skipped (the suite stays
green on a box without the stack); pure tests still run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
_TESTS = str(ROOT / "tests")
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)


def _server_up() -> bool:
    try:
        from neo4j_config import make_driver
        d = make_driver()
        try:
            d.verify_connectivity()
            with d.session() as s:
                s.run("RETURN 1").single()
            return True
        finally:
            d.close()
    except Exception:
        return False


@pytest.fixture(scope="session")
def neo4j_live() -> bool:
    """``True`` when the local Neo4j server + credentials are usable."""
    return _server_up()


@pytest.fixture(autouse=True)
def _clean_fixture_between_tests(neo4j_live):
    """Yield, then wipe the fixture graph so tests stay isolated."""
    yield
    if neo4j_live:
        try:
            from fixtures import wipe_fixture
            wipe_fixture()
        except Exception:
            pass


@pytest.fixture
def fixture_graph(neo4j_live):
    """Build the small fixture graph (nodes+edges+basis embeddings)."""
    if not neo4j_live:
        pytest.skip("Neo4j not reachable")
    from fixtures import build_fixture
    build_fixture()
    return None


@pytest.fixture
def driver(neo4j_live):
    if not neo4j_live:
        pytest.skip("Neo4j not reachable")
    from neo4j_config import make_driver
    d = make_driver()
    yield d
    d.close()
