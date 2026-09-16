"""Phase 6 -- GraphRAG: context construction, answer generation, and
insufficient-context behaviour.

No live LLM is used: a ``FakeLLM`` (``llm=`` injection) answers, and a stub
retriever supplies deterministic ranked rows.  The one live dependency is the
small fixture graph in Neo4j (only to resolve chunk text), so these tests are
skipped when the server is down but never touch Ollama.
"""
from __future__ import annotations

import graphrag_n4j.context as ctx
from graphrag_n4j.rag import GraphRAG
from fixtures import FakeLLM


class StubRetriever:
    """Bypasses vector seeding / embeddings entirely; returns fixed rows."""

    def __init__(self, ranked, last_debug=None):
        self._ranked = ranked
        self.last_debug = last_debug or {
            "seed_ids": [],
            "score_of_seed": {},
            "cypher": "MATCH (n) ...",
        }

    def search(self, query, k=10):
        return self._ranked


def _rag(ranked, llm=None, min_chunks=1):
    rag = GraphRAG(llm=llm or FakeLLM(), min_chunks_for_answer=min_chunks)
    rag.retriever = StubRetriever(ranked)
    # make _merge_corpus_text a no-op that does not need the corpus present
    rag._corpus_cache = {"chunks": []}
    return rag


RANKED = [
    ("fixture:doc:a:article:2", 1, ["CROSS_REFERENCES"]),
    ("fixture:doc:a:article:1", 0, []),
]


# ---------------------------------------------------------------------------
# context construction
# ---------------------------------------------------------------------------

def test_context_has_all_sections():
    out = ctx.build_context(
        RANKED,
        {
            "fixture:doc:a:article:2": {
                "doc_id": "fixture:doc:a", "node_type": "article",
                "title": "Article 2", "text": "publish inside information",
            },
            "fixture:doc:a:article:1": {
                "doc_id": "fixture:doc:a", "node_type": "article",
                "title": "Article 1", "text": "capacity",
            },
        },
    )
    assert set(out.keys()) == {
        "documents", "chunks", "entities", "relationships", "provenance"
    }
    assert "## Relevant Documents" in out["documents"]
    assert "## Relevant Chunks" in out["chunks"]
    assert "## Provenance" in out["provenance"]


def test_context_caps_and_none():
    big = [
        (f"fixture:doc:a:article:{i}", 0, []) for i in range(50)
    ]
    lookup = {
        f"fixture:doc:a:article:{i}": {"doc_id": "fixture:doc:a", "text": "t"}
        for i in range(50)
    }
    out = ctx.build_context(big, lookup)
    lines = [l for l in out["chunks"].splitlines() if l.startswith("- ")]
    assert len(lines) <= 20  # MAX_SECTION_LINES cap applied


# ---------------------------------------------------------------------------
# answer generation (FakeLLM, offline)
# ---------------------------------------------------------------------------

def test_answer_generation_uses_llm(fixture_graph):
    llm = FakeLLM(answer="Article 2 requires publishing.")
    rag = _rag(RANKED, llm=llm)
    res = rag.query("Who must publish inside information?", debug=True)
    assert res.answer == "Article 2 requires publishing."
    assert llm.calls, "the injected LLM should have been invoked"
    # the prompt that reached the LLM carries the rendered context + question
    sent = llm.calls[-1]["input"]
    user_text = sent[-1]["content"] if isinstance(sent, list) else str(sent)
    assert "publish inside information" in user_text or "Context" in user_text
    assert "Who must publish inside information?" in user_text


def test_result_fields_populated(fixture_graph):
    rag = _rag(RANKED, llm=FakeLLM())
    res = rag.query("capacities?", debug=True)
    assert res.question == "capacities?"
    assert any("article" in c.get("lineage_id", "") for c in res.chunks)
    dbg = res.debug()
    assert "cypher" in dbg and "prompt" in dbg and "answer" in dbg
    # compatibility aliases
    assert res.retrieved_entities == res.entities
    assert res.retrieved_relationships == res.relationships


# ---------------------------------------------------------------------------
# insufficient-context behaviour
# ---------------------------------------------------------------------------

def test_insufficient_context_refuses(fixture_graph):
    llm = FakeLLM()
    rag = _rag([], llm=llm)  # no ranked rows at all
    res = rag.query("Unrelated question with no graph hits.")
    assert "insufficient" in res.answer.lower()
    assert "[INSUFFICIENT CONTEXT]" in res.answer
    assert llm.calls == [], "refusal must not call the LLM"


def test_min_chunks_threshold(fixture_graph):
    llm = FakeLLM()
    rag = _rag(
        [("fixture:doc:a:article:2", 1, ["CROSS_REFERENCES"])],
        llm=llm, min_chunks=5,
    )
    res = rag.query("too thin a context for a real answer.")
    assert "insufficient" in res.answer.lower()
    assert llm.calls == []
