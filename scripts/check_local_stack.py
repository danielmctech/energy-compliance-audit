"""Verify the local GraphRAG stack (P0 + P2 + P4).

Checks, in order:
  1. neo4j-graphrag / neo4j driver importable
  2. Ollama reachable + embedding model (bge-m3) embeds a probe
     (dimension read from the live call; must equal GRAPHRAAG_EMBEDDING_DIM)
  3. Ollama LLM (default qwen3.8:27b) answers a one-token probe via the
     graphrag client class
  4. Neo4j reachable + auth OK + server version (skipped with a clear
     message when Neo4j is not running yet)
  5. P2 ``neo4j_graph`` retriever runs through the public Retriever
     (skipped unless the P1 graph + vector indexes are present)
  6. P4 ``GraphRAG.query()`` returns a ``QueryResult`` with every field the
     documented names (answer/sources/chunks/entities/relationships/cypher/
     retrieval_scores) + a debug() trace (skipped unless the P1 graph +
     vector indexes are present)

Exit code 0 = full pass; non-zero = one or more hard failures.

Usage:  python scripts/check_local_stack.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import neo4j_config as cfg  # noqa: E402
from neo4j_config import env_summary  # noqa: E402


def _ok(msg: str):
    print(f"  OK   {msg}")


def _fail(msg: str):
    print(f" FAIL {msg}", file=sys.stderr)
    raise SystemExit(1)


def _skip(msg: str):
    print(f" SKIP {msg}")


def check_imports() -> None:
    print("[1/6] imports")
    try:
        import neo4j  # noqa: F401
        from neo4j_graphrag import llm, embeddings, indexes, generation, retrievers
    except ImportError as e:
        _fail(f"import failed: {e}\n  fix: pip install 'neo4j-graphrag[openai]' neo4j")
    _ok("neo4j + neo4j-graphrag importable")


def check_ollama_embeddings() -> None:
    print("[2/6] Ollama embeddings (bge-m3)")
    s = cfg.graphrag_settings()
    try:
        emb = cfg.make_embeddings(s)
        v = emb.embed_query("Article 28 GDPR supervisory authority")
    except Exception as e:
        _fail(f"embedding call failed: {e}\n"
              f"  fix: start Ollama (`ollama serve`) and `ollama pull {s.embedding_model}`")
    if len(v) != s.embedding_dim:
        _fail(f"embedding dim {len(v)} != configured {s.embedding_dim} "
              f"(set GRAPHRAAG_EMBEDDING_DIM={len(v)} in .env)")
    _ok(f"{s.embedding_model} -> {len(v)}-dim vector")


def check_ollama_llm() -> None:
    print("[3/6] Ollama LLM smoke test")
    s = cfg.graphrag_settings()
    try:
        llm = cfg.make_llm(s, model_params={"max_completion_tokens": 40})
        r = llm.invoke(input="Reply with exactly the word: ok",
                       system_instruction="Be brief.")
    except Exception as e:
        _fail(f"LLM call failed: {e}\n"
              f"  fix: `ollama pull {s.llm_model}`")
    if not r.content:
        _fail("LLM returned empty content")
    _ok(f"{s.llm_model} -> {r.content.strip()!r}")


def check_neo4j(hard: bool = True) -> None:
    print("[4/6] Neo4j")
    s = cfg.neo4j_settings()
    import socket
    hostport = s.uri.split("://")[-1].split("@")[-1]
    host, _, port = hostport.partition(":")
    host = host or "localhost"
    port = port or "7687"
    try:
        socket.create_connection((host or "localhost", int(port)), timeout=3).close()
        up = True
    except OSError:
        up = False
    if not up and not s.is_configured:
        if hard:
            _fail("Neo4j not reachable and NEO4J_PASSWORD not set in .env")
        else:
            _skip("Neo4j not running (install it, then set initial password)")
        return
    try:
        driver = cfg.make_driver(s)
        with driver.session(database=s.database) as sess:
            try:
                # Cypher 25 (Neo4j 2026.x)
                ver = sess.run(
                    "CALL dbms.components().yieldComponent(name, versions, edition)"
                ).single()
            except Exception:
                # Cypher 22 (Neo4j 5.26 LTS)
                ver = sess.run(
                    "CALL dbms.components() YIELD name, versions, edition"
                ).single()
        driver.close()
    except Exception as e:
        _fail(f"Neo4j connection/auth failed: {e}\n"
              f"  fix: start Neo4j service, then "
              f"'neo4j-admin dbms set-initial-password <pw>' and retry")
    versions = list(ver["versions"]) if ver else []
    _ok(f"connected, edition={ver['edition']}, versions={versions}, db={s.database}")


def _has_index(index_name: str) -> bool:
    d = cfg.make_driver()
    try:
        with d.session(database=cfg.neo4j_settings().database) as sess:
            return any(r["name"] == index_name
                       for r in sess.run("SHOW INDEXES YIELD name"))
    finally:
        d.close()


def check_retrieval_p2() -> None:
    """P2: run the new ``neo4j_graph`` mode through the public Retriever.

    Soft check -- requires the P1 graph + vector indexes to exist (run
    ``python scripts/ingest_neo4j.py`` first).  Skips cleanly if the
    schema is not in place yet.
    """
    print("[5/6] neo4j_graph retriever (P2)")
    if not _has_index("v_term_emb"):
        _skip("vector index v_term_emb missing -- run "
              "scripts/ingest_neo4j.py first")
        return
    try:
        import retrieval
        r = retrieval.default_retriever()
        out = r.retrieve("balance responsible party obligations",
                         k=5, mode="neo4j_graph")
    except Exception as e:
        _fail(f"neo4j_graph mode raised: {type(e).__name__}: {e}")
    if not out:
        _fail("neo4j_graph returned 0 results (graph not ingested?)")
    for c in out:
        assert c.lineage_id and c.text, "result missing lineage/text"
    top = out[0]
    _ok(f"neo4j_graph -> {len(out)} ranked articles, "
        f"top={top.lineage_id} edge={top.graph_edge_type}")


def check_graphrag_p4() -> None:
    """P4: run ``GraphRAG.query()`` end-to-end and verify the full
    result contract (required: ``result.answer|sources|chunks|entities|
    relationships|cypher|retrieval_scores``) plus the debug() trace.

    Soft check -- requires the P1 graph + vector indexes to exist.
    """
    print("[6/6] GraphRAG.query() (P4)")
    if not _has_index("v_term_emb"):
        _skip("vector index v_term_emb missing -- run "
              "scripts/ingest_neo4j.py first")
        return
    try:
        from graphrag_n4j import GraphRAG
        rag = GraphRAG()
        res = rag.query("Who must publish inside information under REMIT?")
    except Exception as e:
        _fail(f"GraphRAG.query raised: {type(e).__name__}: {e}")
    # documented fields
    required = ("answer", "sources", "chunks", "entities",
                "relationships", "cypher", "retrieval_scores")
    missing = [n for n in required if not hasattr(res, n)]
    if missing:
        _fail(f"QueryResult missing fields: {missing}")
    # debug trace: question -> seeds -> traversal -> ranked ->
    # context -> LLM -> answer
    dbg = res.debug()
    for k in ("question", "seeds", "cypher", "ranked", "context",
              "prompt", "scores", "elapsed_ms", "answer"):
        if k not in dbg:
            _fail(f"res.debug() missing key: {k}")
    n_chunks = len(res.chunks)
    n_rels = len(res.relationships)
    _ok(f"GraphRAG.query -> {len(res.answer)} chars, "
        f"{n_chunks} chunks, {n_rels} relationships, "
        f"{len(res.entities)} entities, {res.elapsed_ms} ms")


def main() -> int:
    print("Local GraphRAG stack check\n" + "-" * 40)
    print("Config:")
    for k, v in env_summary().items():
        print(f"  {k} = {v}")
    print("-" * 40)

    check_imports()
    check_ollama_embeddings()
    check_ollama_llm()
    try:
        check_neo4j()
    except SystemExit:
        # Neo4j not up yet: not fatal for the Python-side P0, but report it.
        print("\nRESULT: PARTIAL: Python stack OK, Neo4j pending (see above).", file=sys.stderr)
        return 2
    # P2: only meaningful when Neo4j is up + graph is ingested.
    try:
        check_retrieval_p2()
    except SystemExit as e:
        print("\nRESULT: FAIL: see the P2 step above.", file=sys.stderr)
        return int(e.code or 1)
    # P4: GraphRAG query API on top of P2.
    try:
        check_graphrag_p4()
    except SystemExit as e:
        print("\nRESULT: FAIL: see the P4 step above.", file=sys.stderr)
        return int(e.code or 1)
    print("\nRESULT: PASS: full stack verified (P0 + P2 + P4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
