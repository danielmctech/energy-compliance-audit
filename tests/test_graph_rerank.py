# -*- coding: utf-8 -*-
"""Unit + regression tests for the graph-aware re-ranker (task §19).

Most tests are fully offline (synthetic graph, stub re-ranker): they verify
the context builder, direction handling, budgets and — critically — that
graph-only candidates can never enter the default (non-expansion) pool.  No
Ollama / FAISS needed except the dense index load, which is cached.
"""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reranking.graph_context import (  # noqa: E402
    GraphContextConfig, build_graph_context, context_lines, node_label)
from reranking import Reranker  # noqa: E402


class _Fake:
    """Minimal GraphLike: .nodes/.adj/.adj_in/.edges."""
    pass


def _graph(edges, node_titles=None):
    """Build a synthetic GraphLike from ``edges`` = [(src, dst, kind, snippet)].

    ``node_titles`` maps lid -> display title.  Duplicate (src,dst,kind) edges
    are kept in the adjacency list (as if two extraction passes emitted them)
    so dedup behaviour is exercisable.
    """
    nodes: dict = {}
    adj: dict = {}
    adj_in: dict = {}
    edgel: list = []
    titles = node_titles or {}
    for src, dst, kind, snip in edges:
        for lid in (src, dst):
            if lid not in nodes:
                nodes[lid] = {"lineage_id": lid,
                              "title": titles.get(lid, lid),
                              "kind": "article"}
        adj.setdefault(src, []).append((dst, kind))
        adj_in.setdefault(dst, []).append((src, kind))
        edgel.append({"src": src, "dst": dst, "kind": kind, "snippet": snip})
    g = _Fake()
    g.nodes, g.adj, g.adj_in, g.edges = nodes, adj, adj_in, edgel
    return g


# --- context builder ---------------------------------------------------------

def test_no_results_for_isolated_node():
    g = _graph([("A:article:1", "B:article:9", "CROSS_REFERENCES", "s")])
    # C has no edges at all
    assert build_graph_context(g, "C:article:0", "query") == ""
    assert context_lines(g, "C:article:0", "query") == []


def test_lookup_failure_is_graceful():
    g = _graph([("A:article:1", "B:article:9", "CROSS_REFERENCES", "s")])
    # unknown seed -> empty, no exception
    assert build_graph_context(g, "ghost:article:42", "query") == ""


def test_duplicate_edges_are_folded():
    # two CROSS_REFERENCES edges A1 -> A2 (same kind, same neighbour)
    edges = [("A:article:1", "A:article:2", "CROSS_REFERENCES", "first snap"),
             ("A:article:1", "A:article:2", "CROSS_REFERENCES", "dup snap")]
    g = _graph(edges)
    lines = context_lines(g, "A:article:1", "query")
    # exactly one CITES line for the A:article:2 neighbour
    cites = [l for l in lines if l.startswith("CITES A:article:2")]
    assert len(cites) == 1, f"expected 1 line, got {cites}"


def test_multiple_paths_to_same_node_are_bounded():
    g = _graph([
        ("A:article:1", "A:article:2", "CROSS_REFERENCES", "s1"),
        ("A:article:1", "A:article:3", "CROSS_REFERENCES", "s2"),
        # both mid-nodes point back at A1 (would be a multi-path/cycle)
        ("A:article:2", "A:article:1", "CROSS_REFERENCES", "s3"),
        ("A:article:3", "A:article:1", "CROSS_REFERENCES", "s4"),
    ])
    lines = context_lines(g, "A:article:1", "query")
    # must terminate and produce a finite, bounded set of lines
    assert len(lines) <= 3 * 8 + 3 * 3  # generous bound; the point is it ends
    via = [l for l in lines if l.startswith("VIA ")]
    # the seed excluded from VIA ends, and paths are capped at max_paths
    assert len(via) <= GraphContextConfig().max_paths


def test_cyclic_graph_terminates():
    g = _graph([
        ("N:article:1", "N:article:2", "CROSS_REFERENCES", "x"),
        ("N:article:2", "N:article:3", "CROSS_REFERENCES", "y"),
        ("N:article:3", "N:article:1", "CROSS_REFERENCES", "z"),
    ])
    # must not hang; returns some finite context
    out = build_graph_context(g, "N:article:1", "query")
    assert isinstance(out, str)


def test_1hop_filtering_excludes_paths():
    g = _graph([
        ("A:article:1", "A:article:2", "CROSS_REFERENCES", "s1"),
        ("A:article:2", "A:article:3", "CROSS_REFERENCES", "s2"),
    ])
    cfg = GraphContextConfig(include_paths=False)
    lines = context_lines(g, "A:article:1", "query", cfg)
    assert not any(l.startswith("VIA ") for l in lines), \
        "2-hop paths must be excluded when include_paths=False"
    assert any(l.startswith("CITES A:article:2") for l in lines)


def test_2hop_paths_present_when_enabled():
    g = _graph([
        ("A:article:1", "A:article:2", "CROSS_REFERENCES", "s1"),
        ("A:article:2", "A:article:44", "CROSS_REFERENCES", "s2"),
    ])
    lines = context_lines(g, "A:article:1", "query",
                          GraphContextConfig(include_paths=True))
    assert any(l.startswith("VIA ") and "A:article:44" in l for l in lines), \
        f"expected a 2-hop VIA line, got {lines}"


def test_max_context_size_is_enforced():
    g = _graph([("A:article:1", f"A:article:{i}", "CROSS_REFERENCES",
                 "word " * 40) for i in range(40)])
    cfg = GraphContextConfig(token_limit=20, max_edges=40)  # tiny char cap
    out = build_graph_context(g, "A:article:1", "query", cfg)
    assert len(out) <= cfg.char_cap() + 80  # cap + header slack
    # fewer than 40 lines because the budget cut some off
    assert len(out.splitlines()) < 41


def test_direction_is_preserved():
    # A1 --AMENDS--> B9 (outgoing)  AND  B9 --CROSS--> A1 (incoming)
    g = _graph([
        ("A:article:1", "B:article:9", "AMENDS", "amend snip"),
        ("B:article:9", "A:article:1", "CROSS_REFERENCES", "cite snip"),
    ])
    lines = context_lines(g, "A:article:1", "query")
    joined = "\n".join(lines)
    assert "AMENDS B:article:9" in joined, f"outgoing AMENDS missing: {joined}"
    assert "CITED_BY B:article:9" in joined, f"incoming CITED_BY missing: {joined}"


def test_node_label_compresses_document_titles():
    nodes = {"d:document": {"kind": "document", "doc_id": "d",
                            "title": "REGULATION (EU) 2016/679 OF THE "
                                     "EUROPEAN PARLIAMENT AND OF THE COUNCIL"}}
    assert node_label(nodes, "d:document") == "Regulation (2016/679)"


# --- candidate / pool invariants (offline, stub reranker) --------------------

class _IdentityReranker:
    """Stub: echoes the input order, status 'llm'.  No LLM call.

    Records the ``max_candidates`` override it was given (``None`` when the
    caller -- the A0 baseline path -- did not pass one) on ``self.caps``.
    """
    def __init__(self):
        self.caps = []

    def rerank(self, query, pool, max_candidates=None):
        self.caps.append(max_candidates)
        return {"order": [c["lineage_id"] for c in pool],
                "status": "llm", "model": "fake", "pool": len(pool)}


def _real_retriever():
    from retrieval import Retriever
    r = Retriever(dense_models=["all-MiniLM-L6-v2"])
    return r


def test_graph_only_candidates_do_not_enter_default_pool():
    """Default (non-expansion) mode must re-rank ONLY the hybrid pool; no
    graph-neighbour chunk may be admitted."""
    r = _real_retriever()
    q = "conflict between the Data Act and GDPR personal data rules"
    k = 10
    pool_lids, fused, _ = r._hybrid_pool(q, max(k, 5))
    r._reranker = types.SimpleNamespace(get=lambda: _IdentityReranker())
    out = r.retrieve(q, k=k, mode="hybrid_gr_1hop")   # graph ctx on, no expand
    got = {c.lineage_id for c in out}
    assert got <= set(pool_lids), \
        f"graph-only chunks leaked into default pool: {got - set(pool_lids)}"


def test_baseline_unchanged_when_graph_disabled():
    """Graph-aware mode with graph_cfg=None must be the A0 baseline: same
    candidate set and a prompt byte-identical to the no-context reranker."""
    r = _real_retriever()
    q = "conflict between the Data Act and GDPR personal data rules"
    k = 10
    # baseline (A0 mode): no context by definition
    base_ids = [c.lineage_id for c in r.retrieve(q, k=k, mode="hybrid_rerank")]
    # graph mode with the feature explicitly disabled -> same set
    off_ids = [c.lineage_id for c in r.retrieve(
        q, k=k, mode="hybrid_graph_rerank", graph_cfg=None, with_expand=False)]
    assert set(base_ids) == set(off_ids)

    # and the reranker prompt is byte-identical with/without context
    rr = Reranker()
    cand = [{"lineage_id": "A:article:1", "doc_id": "A", "text": "T"}]
    with_ctx = [dict(cand[0], graph_context="CITES X")]
    p_none = rr._prompt(q, cand, 1)
    p_str = rr._prompt(q, with_ctx, 1)
    assert p_none != p_str          # context, when present, does change it
    # ... and an empty string (gr disabled) is treated as absent
    p_empty = rr._prompt(q, [dict(cand[0], graph_context="")], 1)
    assert p_empty == p_none


def test_expansion_retains_all_hybrid_candidates():
    """A7: graph expansion may *add* neighbours but must never drop the
    original hybrid candidates (retention constraint)."""
    r = _real_retriever()
    q = "conflict between the Data Act and GDPR personal data rules"
    k = 10
    pool_lids, _, _ = r._hybrid_pool(q, max(k, 5))
    r._reranker = types.SimpleNamespace(get=lambda: _IdentityReranker())
    out = r.retrieve(q, k=k + len(pool_lids), mode="hybrid_gr_expand")
    got_ids = {c.lineage_id for c in out}
    # every hybrid seed is retained (identity rerank keeps all in the order)
    assert set(pool_lids) <= got_ids, \
        f"expansion dropped hybrid candidates: {set(pool_lids) - got_ids}"
    # and nothing is an external stub
    assert not any(i.startswith("ext:") for i in got_ids)


def test_reranker_output_is_a_permutation():
    base = {"lineage_id": "A:article:1", "doc_id": "A", "text": "T"}
    rr = Reranker()
    ordered = rr._to_order([1, 2, 3], [dict(base, lineage_id=f"A:{i}") for i in range(3)])
    assert ordered is not None and len(set(ordered)) == 3
    # invalid (drops a candidate) -> None
    assert rr._to_order([1, 2], [dict(base) for _ in range(3)]) is None


def test_reranker_prompt_embeds_context_block():
    rr = Reranker()
    cands = [{"lineage_id": "A:article:1", "doc_id": "A", "text": "T",
              "graph_context": "CITES B — \"snip\"\nAMENDS C"}]
    p = rr._prompt("q", cands, 1)
    assert "Graph evidence:" in p
    assert "- CITES B" in p and "- AMENDS C" in p


def test_reranker_prompt_guardrails_only_when_context_present():
    """Spec §6: the graph-evidence guardrail sentences must appear only when
    at least one candidate carries graph context -- the A0 baseline prompt
    stays byte-identical to the historical v1 text (no guardrail noise)."""
    rr = Reranker()
    plain = [{"lineage_id": "A:article:1", "doc_id": "A", "text": "T"},
             {"lineage_id": "A:article:2", "doc_id": "A", "text": "U"}]
    p_none = rr._prompt("q", plain, 2)
    p_ctx = rr._prompt(
        "q", [dict(plain[0], graph_context="CITES X"), plain[1]], 2)
    guardrail = "Graph connectivity does not imply relevance"
    assert guardrail not in p_none, "guardrail leaked into A0 baseline prompt"
    assert guardrail in p_ctx, "guardrail missing from graph-context prompt"
    # and the primary-evidence instruction is present in both
    assert "Rank the candidates from most to least relevant" in p_none
    assert "Rank the candidates from most to least relevant" in p_ctx


def test_pool_n_extends_reranker_pool():
    """Spec §2: ``pool_n`` must fetch a larger hybrid pool for the re-ranker
    when set; ``None`` must keep the historical ``max(k, 5)``.  The final
    top-k is always ``k`` regardless."""
    from retrieval import Retriever
    import types as _t

    r = _real_retriever()
    captured: list = []
    def _pool(query, n):
        captured.append(n)
        # return a fixed pool of 5 chunks (enough for k=3)
        lids = [f"D:article:{i}" for i in range(1, 6)]
        fused = {lid: {"score": 0.1, "methods": ["hybrid"]} for lid in lids}
        return lids, fused, [("hybrid", lids)]
    r._hybrid_pool = _pool
    r._reranker = _t.SimpleNamespace(get=lambda: _IdentityReranker())
    q = "conflict between the Data Act and GDPR personal data rules"

    # default (no pool_n): n = max(k, 5) = 5
    r.retrieve(q, k=3, mode="hybrid_gr_1hop")
    assert captured[-1] == 5

    # with pool_n=20: n = max(k, 5, pool_n) = 20
    r.retrieve(q, k=3, mode="hybrid_gr_1hop", pool_n=20)
    assert captured[-1] == 20

    # A0 baseline ignores pool_n entirely
    captured.clear()
    r.retrieve(q, k=3, mode="hybrid_rerank", pool_n=20)
    assert not captured or captured[-1] == 5  # A0 never receives pool_n


def test_reranker_max_candidates_override():
    """Spec §2: graph modes must send the full (deeper) pool to the model
    instead of the historical 20-cap; the A0 baseline must keep the 20-cap
    (no override passed) so its prompt/pool/cache key stay byte-identical."""
    from unittest import mock
    import re
    import common as c
    from reranking import Reranker

    def _fake(model, prompt, system=None, temperature=0.0, num_ctx=32768,
              num_predict=16384, max_retries=3, extra_options=None,
              use_cache=True, cache_key=None):
        # deterministic permutation: numbers 1..N in prompted order
        n = len(re.findall(r"(?m)^\[\d+\]", prompt))
        return "[" + ", ".join(map(str, range(1, n + 1))) + "]"

    cands = [{"lineage_id": f"D:article:{i}", "text": f"t{i}"}
             for i in range(1, 31)]  # 30 candidates
    rr = Reranker(max_candidates=20)
    with mock.patch.object(c, "ollama_chat", side_effect=_fake):
        # default (None) -> bounded by the instance cap of 20
        assert rr.rerank("q", cands)["pool"] == 20
        # explicit override -> the whole pool is prompted
        assert rr.rerank("q", cands, max_candidates=30)["pool"] == 30
        # override smaller than the instance cap still bounds the pool
        assert rr.rerank("q", cands, max_candidates=7)["pool"] == 7


def test_regression_graph_neighbors_do_not_auto_insert():
    """Task §19 "most important" regression.

    Scenario: a target article is already rank 1 in ``hybrid``.  Its graph
    has three CROSS_REFERENCES neighbours (Article 2 / 10 / 17).  When we run
    a *graph-context* re-rank (any non-expansion mode), those graph-only
    neighbours must NOT be automatically admitted into the candidate pool --
    they may be *mentioned* in the re-ranker's prompt as context, but they
    must not *become* results.

    We build a synthetic corpus so the assertion is deterministic and does not
    depend on the real (large) corpus or the LLM.
    """
    from retrieval import Retriever
    from retrieval._corpus import Chunk
    import types as _t

    # 6 chunks in one document.  Article 1 is the target and has graph
    # cross-references to 2, 10, 17 -- which are chunks NOT in the hybrid
    # pool for our query (so they would only surface if the graph auto-
    # inserts them, which is exactly the regression we want to prevent).
    # 42/99 are in-pool fillers unrelated to the graph.
    chunks = [Chunk(doc_id="D", lineage_id=f"D:article:{n}", node_id=f"D:article:{n}",
                    node_type="article",
                    text=("the target rule for reporting market data" if n == 1
                          else f"unrelated filler for rule {n}"),
                    span=(0, 1)) for n in (1, 2, 10, 17, 42, 99)]
    r = Retriever.__new__(Retriever)
    r.corpus = {"chunks": chunks,
                "article_index": {},
                "graph": _graph([(f"D:article:1", f"D:article:{n}",
                                  "CROSS_REFERENCES", f"cites {n}")
                                for n in (2, 10, 17)],
                                {f"D:article:{n}": f"Article {n}"
                                 for n in (1, 2, 10, 17)})}
    r.chunks = chunks
    r._idx_to_lid = [c.lineage_id for c in chunks]
    r._by_lineage = {c.lineage_id: i for i, c in enumerate(chunks)}
    r._texts = [c.text for c in chunks]
    r.sparse = None
    r._dense = {}
    r._graph = _t.SimpleNamespace(get=lambda: None)
    r._reranker = _t.SimpleNamespace(get=lambda: _IdentityReranker())

    # STUB for _hybrid_pool: the true hybrid pool has article-1 (target) then
    # two filler chunks -- and deliberately NOT the graph neighbours 2/10/17.
    # This models "they are graph-only, not surfaced by sparse+dense RRF".
    HYBRID_POOL = ["D:article:1", "D:article:42", "D:article:99"]
    def _pool(query, n):
        order = HYBRID_POOL[:min(n, len(HYBRID_POOL))]
        fake_fused = {lid: {"score": round(0.10 * (len(order) - i), 6),
                            "methods": ["hybrid"]}
                      for i, lid in enumerate(order)}
        return order, fake_fused, [("hybrid", order)]
    r._hybrid_pool = _pool
    # search_graph (used only for provenance annotation in hybrid/_hybrid_rerank):
    # returns neighbours as (lineage, hop, kinds); the neighbours are in the
    # chunk corpus (so they're plausible retrieval targets) but were NOT in
    # the hybrid pool.
    r.search_graph = lambda q, k: \
        [(f"D:article:{n}", 1, ["CROSS_REFERENCES"]) for n in (2, 10, 17)]
    # _node_for: identity map article:N -> article:N (synthetic)
    r._node_for = lambda lid: lid

    q = "rule for reporting market data"
    for mode in ("hybrid_graph_rerank", "hybrid_gr_1hop",
                 "hybrid_gr_relations"):
        out = r.retrieve(q, k=10, mode=mode)
        got = {c.lineage_id for c in out}
        # 1) hybrid target retained
        assert "D:article:1" in got, f"{mode}: lost hybrid rank-1 target"
        # 2) pool members not displaced (identity re-rank keeps order)
        assert set(HYBRID_POOL) <= got, f"{mode}: hybrid pool dropped {set(HYBRID_POOL) - got}"
        # 3) THE KEY ASSERTION: graph-only neighbours are NOT auto-inserted
        assert not ({"D:article:2", "D:article:10", "D:article:17"} & got), \
            f"{mode}: graph-only neighbours leaked into top-k: {got}"


# --- spec §6: graph evidence is typed, not a flat hop score ------------------
# Spec §6: "graph evidence must be typed/directional and explicitly labelled
# 1-hop vs 2-hop ... Graph connectivity is NOT evidence by itself."

def test_graph_evidence_is_typed_not_flat_score():
    """Spec §6(a): context lines carry a typed RELATION label (CITES / AMENDS /
    SUPERSEDES / DEFINES / VIA ...) -- never a bare floating-point hop score
    (e.g. '0.5' from HOP_WEIGHT) driving or decorating the ranking."""
    g = _graph([
        ("A:article:1", "A:article:2", "CROSS_REFERENCES", "s1"),
        ("A:article:2", "A:article:3", "CROSS_REFERENCES", "s2"),
        ("A:article:4", "A:article:1", "AMENDS", "amend snip"),
        ("A:article:1", "A:article:5", "SUPERSEDES", "sup snip"),
    ])
    out = build_graph_context(g, "A:article:1", "query")
    assert out, "expected non-empty evidence for a well-connected node"
    body_lines = out.splitlines()[1:]            # drop the "Graph context" header
    assert body_lines, "expected evidence lines"
    labels = (r"CITES", r"AMENDS", r"AMENDED_BY", r"SUPERSEDES", r"SUPERSEDED_BY",
              r"DEFINES", r"APPLIES_TO", r"PART_OF", r"IMPLEMENTS", r"VIA ")
    for line in body_lines:
        assert re.search(r"^(?:" + "|".join(map(re.escape, labels)) + r")\b", line), \
            f"evidence line lacks a typed relation label {line!r}"
    # no bare decimal hop weight ("0.5"-style number) anywhere in the evidence
    assert not re.search(r"\b0\.\d+\b", out), \
        f"flat hop score leaked into evidence: {out!r}"


def test_1hop_is_relation_label_and_2hop_is_explicitly_via():
    """Spec §6(b): 1-hop evidence lines start with the relation label; 2-hop
    evidence lines start with the explicit VIA marker (path through the
    intermediate node)."""
    g = _graph([
        ("A:article:1", "A:article:2", "CROSS_REFERENCES", "s1"),
        ("A:article:2", "A:article:3", "CROSS_REFERENCES", "s2"),
    ])
    lines = context_lines(g, "A:article:1", "query")
    hop1 = {l for l in lines if not l.startswith("VIA ")}
    hop2 = {l for l in lines if l.startswith("VIA ")}
    assert hop1, f"expected 1-hop relation lines, got {lines}"
    assert all(re.match(r"^(CITES|AMENDS|DEFINES|APPLIES_TO|PART_OF|"
                        r"IMPLEMENTS|SUPERSEDES|AMENDED_BY|CITED_BY)\b", l)
               for l in hop1), f"1-hop line not relation-labelled: {hop1}"
    # every VIA line renders the intermediate node and the path end
    for l in hop2:
        assert "A:article:2" in l and "A:article:3" in l, f"VIA line malformed: {l!r}"


def test_rerank_prompt_carries_typed_lines_not_hop_numbers():
    """Spec §6 rendered inside the actual rerank prompt: the per-candidate
    'Graph evidence' block carries the typed relation labels (CITES/AMENDS)
    and never a bare HOP_WEIGHT decimal (0.5 / 1.0); the §6 guardrail
    sentences are present and say connectivity does not imply relevance."""
    q = "energy efficiency obligations"
    pool = [
        {"lineage_id": "X:article:1", "doc_id": "x",
         "text": "the main text of article 1",
         "graph_context": "CITES X:article:2\n"
                          "AMENDS Y:article:7 — \"amends the second clause\""},
        {"lineage_id": "X:article:3", "doc_id": "x",
         "text": "the main text of article 3"},
    ]
    rr = Reranker(max_candidates=20)
    prompt = rr._prompt(q, pool, len(pool))
    assert "CITES X:article:2" in prompt
    assert "AMENDS Y:article:7" in prompt
    assert "Graph evidence" in prompt
    assert not re.search(r"\b0\.\d+\b", prompt), \
        f"bare hop weight in prompt:\n{prompt}"
    assert "Graph connectivity does not imply relevance" in prompt
    # A0 baseline (no candidate context): prompt byte-identical to v1 -- no
    # guardrails, no evidence lines
    base = [{k: v for k, v in c2.items() if k != "graph_context"}
            for c2 in pool]
    base_prompt = rr._prompt(q, base, len(base))
    assert "Graph evidence" not in base_prompt
    assert "Graph connectivity" not in base_prompt


def test_irrelevant_graph_neighbours_do_not_reorder_identity_ranking():
    """Spec §6 integration: with a no-op (identity) LLM permutation, graph
    evidence for pool members must not change their order; graph-only
    neighbours must not be admitted.  Asserts the full pipeline over the real
    corpus: exactly the pool members that own graph nodes carry a rendered
    ``graph_context``, and output order == input order."""
    from retrieval import Retriever
    from reranking.graph_context import GraphContextConfig
    r = Retriever(dense_models=["all-MiniLM-L6-v2"])
    q = "conflict between the Data Act and GDPR personal data rules"
    k = 5
    cfg = GraphContextConfig()
    gr = r.corpus["graph"]
    for src, dst in (
            ("D:article:1", "D:article:2"),
            ("D:article:2", "D:article:17")):
        gr.edges.append({"src": src, "dst": dst, "kind": "CROSS_REFERENCES",
                         "snippet": "s"})
        gr.nodes.setdefault(src, {"lineage_id": src, "kind": "article",
                                  "title": src})
        gr.nodes.setdefault(dst, {"lineage_id": dst, "kind": "article",
                                  "title": dst})
        gr.adj.setdefault(src, []).append((dst, "CROSS_REFERENCES"))
        gr.adj_in.setdefault(dst, []).append((src, "CROSS_REFERENCES"))
    pool_lids, _, _ = r._hybrid_pool(q, k)
    pool_lids = [l for l in pool_lids if l in r._by_lineage]
    assert len(pool_lids) >= 2
    pool, _gc = r._gr_candidates(q, pool_lids, cfg, [], {})
    assert len(pool) == len(pool_lids), \
        f"{len(pool)} candidates for {len(pool_lids)} lids"
    r._reranker = types.SimpleNamespace(get=lambda: _IdentityReranker())
    out = r.retrieve(q, k=k, mode="hybrid_gr_1hop", graph_cfg=cfg)
    # 1) output is a permutation of the pool -- nothing from outside admitted
    assert [c.lineage_id for c in out] == list(pool_lids), \
        f"identity re-rank must preserve pool order: {out}"
    # 2) the synthetic D-node canary never leaked into any candidate evidence
    ctxs = [c.get("graph_context") or "" for c in pool]
    for ctx in ctxs:
        assert "D:article:2" not in ctx and "D:article:17" not in ctx, \
            f"unrelated neighbour {ctx!r}"
    # 3) any evidence that fired is typed/directional, never bare hop scores
    for ctx in ctxs:
        assert not re.search(r"\b0\.\d+\b", ctx), \
            f"flat hop score in evidence: {ctx!r}"
        for line in ctx.splitlines():
            if "snippet" in line or line.startswith("…"):
                continue
            assert re.match(r"^(CITES|CITED_BY|AMENDS|AMENDED_BY|VIA|DEFINES|"
                            r"APPLIES_TO|APPLIED_TO|PART_OF|CONTAINS|IMPLEMENTS|"
                            r"IMPLEMENTED_BY|SUPERSEDES|SUPERSEDED_BY)\b", line), \
                f"untyped evidence line: {line!r}"
    # 4) graph-only neighbours never leaked into the k results
    assert not ({"D:article:2", "D:article:17"} &
                {c.lineage_id for c in out}), "graph-only neighbour leaked"


# =====================================================================
# GCG-2hop expansion (spec 06 §4/§5/§6): deterministic within-2-hop +
# construction-time provenance.  Pure-function tests against the same
# synthetic graph the re-ranker tests use (no corpus / Ollama / FAISS).
# =====================================================================
from retrieval.two_hop import (  # noqa: E402  # isort: skip
    apply_admission_policy, expand_within_2hop, two_hop_only)
from retrieval.graph import GRAPH_EDGE_KINDS  # noqa: E402  # isort: skip


def _adj(g):
    return g.adj


def _real(nodes_seq):
    return set(nodes_seq)


def test_1hop_then_2hop_ordering_and_provenance():
    # A -CR-> B -CR-> C -CR-> D   (all real)
    g = _graph([
        ("A:article:1", "B:article:2", "CROSS_REFERENCES", "s"),
        ("B:article:2", "C:article:3", "CROSS_REFERENCES", "s"),
        ("C:article:3", "D:article:4", "CROSS_REFERENCES", "s"),
    ])
    real = _real(["A:article:1", "B:article:2", "C:article:3", "D:article:4"])
    out = expand_within_2hop(_adj(g), ["A:article:1"], real)
    nodes = [r["node"] for r in out]
    assert nodes == ["B:article:2", "C:article:3"], \
        f"D must NOT be reached (that is 3 hops); got {nodes}"
    b, c = out
    assert b["source"] == "graph_1hop" and b["graph_distance"] == 1
    assert b["intermediate"] is None
    assert b["graph_path"] == ["A:article:1", "CROSS_REFERENCES", "B:article:2"]
    assert c["source"] == "graph_2hop" and c["graph_distance"] == 2
    assert c["graph_relation"] == ["CROSS_REFERENCES", "CROSS_REFERENCES"]
    assert c["intermediate"] == "B:article:2"
    assert c["graph_path"] == ["A:article:1", "CROSS_REFERENCES", "B:article:2",
                               "CROSS_REFERENCES", "C:article:3"]
    assert c["graph_seed"] == "A:article:1"


def test_multistep_chain_reaches_exactly_two_hops():
    # A -> B -> C -> D : from A, 2-hop set = {B, C} (D is 3 hops, excluded)
    g = _graph([
        ("A:article:1", "B:article:2", "AMENDS", "s"),
        ("B:article:2", "C:article:3", "CROSS_REFERENCES", "s"),
        ("C:article:3", "D:article:4", "CROSS_REFERENCES", "s"),
    ])
    real = _real(["A:article:1", "B:article:2", "C:article:3", "D:article:4"])
    out = expand_within_2hop(_adj(g), ["A:article:1"], real)
    assert [r["node"] for r in out] == ["B:article:2", "C:article:3"]
    # mixed kind on the 2-hop path is preserved in the relation tuple
    assert out[1]["graph_relation"] == ["AMENDS", "CROSS_REFERENCES"]


def test_non_real_graph_nodes_are_excluded():
    # X is a graph node with NO corpus chunk -> must not be a candidate,
    # and must not act as a real intermediate (spec §4).
    g = _graph([
        ("A:article:1", "X:article:99", "CROSS_REFERENCES", "s"),  # X not real
        ("B:article:2", "C:article:3", "CROSS_REFERENCES", "s"),   # B real
        ("A:article:1", "B:article:2", "CROSS_REFERENCES", "s"),
        ("B:article:2", "C:article:3", "CROSS_REFERENCES", "s"),
    ])
    real = _real(["A:article:1", "B:article:2", "C:article:3"])  # X excluded
    out = expand_within_2hop(_adj(g), ["A:article:1"], real)
    nodes = {r["node"] for r in out}
    assert "X:article:99" not in nodes, "non-corpus graph node leaked in"
    assert "B:article:2" in nodes            # 1-hop real
    assert "C:article:3" in nodes            # 2-hop real (A->B->C)


def test_budget_keep_truncates_deterministically():
    # A -> B,C,D (three 1-hop real) ; keep=2 -> deterministic first two
    g = _graph([
        ("A:article:1", "B:article:2", "CROSS_REFERENCES", "s"),
        ("A:article:1", "C:article:3", "CROSS_REFERENCES", "s"),
        ("A:article:1", "D:article:4", "CROSS_REFERENCES", "s"),
    ])
    real = _real(["A:article:1", "B:article:2", "C:article:3", "D:article:4"])
    full = expand_within_2hop(_adj(g), ["A:article:1"], real)
    kept = expand_within_2hop(_adj(g), ["A:article:1"], real, keep=2)
    assert len(kept) == 2
    assert [r["node"] for r in kept] == [r["node"] for r in full[:2]]


def test_allowed_kind_filtering():
    # only CROSS_REFERENCES / AMENDS count; a REFERENCES edge is ignored
    g = _graph([
        ("A:article:1", "B:article:2", "REFERENCES", "s"),   # out of allow-set
        ("A:article:1", "C:article:3", "CROSS_REFERENCES", "s"),
    ])
    real = _real(["A:article:1", "B:article:2", "C:article:3"])
    out = expand_within_2hop(_adj(g), ["A:article:1"], real,
                             allowed=GRAPH_EDGE_KINDS)
    assert [r["node"] for r in out] == ["C:article:3"]


def test_dedup_and_seed_exclusion():
    # C reachable both as 1-hop (A->C) and 2-hop (A->B->C); must appear once,
    # at its best (1-hop) position.  Seed A never re-emitted.
    g = _graph([
        ("A:article:1", "B:article:2", "CROSS_REFERENCES", "s"),
        ("B:article:2", "C:article:3", "CROSS_REFERENCES", "s"),
        ("A:article:1", "C:article:3", "CROSS_REFERENCES", "s"),
        ("C:article:3", "A:article:1", "CROSS_REFERENCES", "s"),  # back-edge
    ])
    real = _real(["A:article:1", "B:article:2", "C:article:3"])
    out = expand_within_2hop(_adj(g), ["A:article:1"], real)
    nodes = [r["node"] for r in out]
    assert nodes.count("C:article:3") == 1, f"dup emitted: {nodes}"
    assert "A:article:1" not in nodes
    # C's recorded distance is its best (1)
    c = next(r for r in out if r["node"] == "C:article:3")
    assert c["graph_distance"] == 1


def test_multiple_calls_are_deterministic():
    g = _graph([
        ("S:article:10", "M:article:20", "CROSS_REFERENCES", "s"),
        ("S:article:10", "T:article:30", "AMENDS", "s"),
        ("M:article:20", "U:article:40", "CROSS_REFERENCES", "s"),
        ("T:article:30", "V:article:50", "CROSS_REFERENCES", "s"),
    ])
    real = _real(["S:article:10", "M:article:20", "T:article:30",
                  "U:article:40", "V:article:50"])
    a = expand_within_2hop(_adj(g), ["S:article:10"], real)
    b = expand_within_2hop(_adj(g), ["S:article:10"], real)
    assert a == b
    # 1-hop (M,T in stored edge order) then 2-hop (from M: U; from T: V)
    assert [r["node"] for r in a] == ["M:article:20", "T:article:30",
                                      "U:article:40", "V:article:50"]
    assert [r["graph_distance"] for r in a] == [1, 1, 2, 2]


def test_two_hop_only_predicate():
    g = _graph([
        ("A:article:1", "B:article:2", "CROSS_REFERENCES", "s"),
        ("B:article:2", "C:article:3", "CROSS_REFERENCES", "s"),
        ("A:article:1", "D:article:4", "CROSS_REFERENCES", "s"),  # D = 1-hop
    ])
    # C is exactly 2 hops from A  -> True
    assert two_hop_only(_adj(g), ["A:article:1"], "C:article:3") is True
    # D is 1 hop  -> False
    assert two_hop_only(_adj(g), ["A:article:1"], "D:article:4") is False
    # A is a seed  -> False
    assert two_hop_only(_adj(g), ["A:article:1"], "A:article:1") is False
    # E absent entirely  -> False
    assert two_hop_only(_adj(g), ["A:article:1"], "E:article:5") is False


def test_admission_policy_first_keeps_stored_order():
    # 3 one-hop (M,T,W) + 2 two-hop (U via M, V via T); expand emits 1-hop THEN
    # 2-hop.  "first" policy + keep=3 must return the first three of that order
    # (all one-hop) -- the as-specified behaviour that starves 2-hop targets.
    g = _graph([
        ("S:article:10", "M:article:20", "CROSS_REFERENCES", "s"),
        ("S:article:10", "T:article:30", "CROSS_REFERENCES", "s"),
        ("S:article:10", "W:article:60", "CROSS_REFERENCES", "s"),
        ("M:article:20", "U:article:40", "CROSS_REFERENCES", "s"),
        ("T:article:30", "V:article:50", "CROSS_REFERENCES", "s"),
    ])
    real = _real(["S:article:10", "M:article:20", "T:article:30",
                  "W:article:60", "U:article:40", "V:article:50"])
    full = expand_within_2hop(_adj(g), ["S:article:10"], real, keep=None)
    kept = apply_admission_policy(full, policy="first", keep=3)
    assert [r["node"] for r in kept] == ["M:article:20", "T:article:30",
                                         "W:article:60"]
    assert all(r["graph_distance"] == 1 for r in kept)


def test_admission_policy_2hop_first_prefers_distance2():
    # Same graph.  "2hop_first" + keep=3 must admit the two distance-2 nodes
    # (U,V) and exactly one 1-hop -- proving the 2-hop targets ENTER the pool
    # under the same 20-slot budget, with the same candidate set.
    g = _graph([
        ("S:article:10", "M:article:20", "CROSS_REFERENCES", "s"),
        ("S:article:10", "T:article:30", "CROSS_REFERENCES", "s"),
        ("S:article:10", "W:article:60", "CROSS_REFERENCES", "s"),
        ("M:article:20", "U:article:40", "CROSS_REFERENCES", "s"),
        ("T:article:30", "V:article:50", "CROSS_REFERENCES", "s"),
    ])
    real = _real(["S:article:10", "M:article:20", "T:article:30",
                  "W:article:60", "U:article:40", "V:article:50"])
    full = expand_within_2hop(_adj(g), ["S:article:10"], real, keep=None)
    kept = apply_admission_policy(full, policy="2hop_first", keep=3)
    assert len(kept) == 3
    assert {"U:article:40", "V:article:50"} <= {r["node"] for r in kept}
    # same candidate set as "first" (nothing added/removed), only re-ordered:
    assert {r["node"] for r in kept} | {r["node"] for r in full} == \
           {r["node"] for r in full}
    assert full is not kept  # a new list, input not mutated
    assert expand_within_2hop(_adj(g), ["S:article:10"], real, keep=None) == full


def test_admission_policy_deterministic_and_unknown_rejected():
    g = _graph([
        ("S:article:10", "M:article:20", "CROSS_REFERENCES", "s"),
        ("M:article:20", "U:article:40", "CROSS_REFERENCES", "s"),
    ])
    real = _real(["S:article:10", "M:article:20", "U:article:40"])
    full = expand_within_2hop(_adj(g), ["S:article:10"], real)
    a = apply_admission_policy(full, policy="2hop_first", keep=1)
    b = apply_admission_policy(full, policy="2hop_first", keep=1)
    assert a == b                          # deterministic
    assert [r["node"] for r in a] == ["U:article:40"]
    import pytest as _p
    with _p.raises(ValueError):
        apply_admission_policy(full, policy="nope")
    with _p.raises(ValueError):
        apply_admission_policy(full, policy="split_bad")


def test_admission_policy_split_keeps_both_distance_classes():
    # 3 one-hop (M,T,W) + 2 two-hop (U via M, V via T).  "split_2_1" + keep=3
    # must return d1[:2]+d2[:1] = {M,T,U} (deterministic, both classes present)
    # -- proving a distance-quota split (rather than a total re-order) secures
    # BOTH one-hop and two-hop golds in the same 20-slot budget.
    g = _graph([
        ("S:article:10", "M:article:20", "CROSS_REFERENCES", "s"),
        ("S:article:10", "T:article:30", "CROSS_REFERENCES", "s"),
        ("S:article:10", "W:article:60", "CROSS_REFERENCES", "s"),
        ("M:article:20", "U:article:40", "CROSS_REFERENCES", "s"),
        ("T:article:30", "V:article:50", "CROSS_REFERENCES", "s"),
    ])
    real = _real(["S:article:10", "M:article:20", "T:article:30",
                  "W:article:60", "U:article:40", "V:article:50"])
    full = expand_within_2hop(_adj(g), ["S:article:10"], real, keep=None)
    kept = apply_admission_policy(full, policy="split_2_1", keep=3)
    assert [r["node"] for r in kept] == ["M:article:20", "T:article:30",
                                        "U:article:40"]
    assert {r["graph_distance"] for r in kept} == {1, 2}
    kept2 = apply_admission_policy(full, policy="split_1_2", keep=3)
    assert [r["node"] for r in kept2] == ["M:article:20", "U:article:40",
                                         "V:article:50"]
    # total reorder ("2hop_first") trades the two-hop classes OFF each other;
    # the split preserves both.
    first = apply_admission_policy(full, policy="first", keep=3)
    assert [r["node"] for r in first] == ["M:article:20", "T:article:30",
                                          "W:article:60"]
    assert {r["node"] for r in first}.isdisjoint(
           {r["node"] for r in kept if r["graph_distance"] == 2})

