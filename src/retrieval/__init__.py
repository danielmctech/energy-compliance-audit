"""Hybrid Sparse-Dense-Graph RAG retrieval layer.

``retrieve(query, k, mode)`` returns ranked ``RetrievedChunk`` records with
full lineage (lineage_id + source text + methods that surfaced it + graph
edge provenance). Modes:
``sparse | dense | graph | neo4j_graph | hybrid | hybrid_graph``.

``graph``        = bounded BFS over CROSS_REFERENCES/AMENDS from sparse seeds.
``neo4j_graph``  = vector-seeded bounded Cypher traversal over all four edge
                   kinds, both directions (Neo4j-backed, P2).
``hybrid``       = RRF(sparse, dense); graph neighbours are annotated but do
                   not enter the candidate pool (fixed ordering).
``hybrid_graph`` = RRF(sparse, dense, graph-neighbours); graph-sourced chunks
                   are fused in alongside the channel lists, so they can enter
                   the top-k and shift relative ordering.

All internal search methods return lineage-keyed ranked lists:
``[(lineage_id, raw_score), ...]`` sorted by score desc.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import json

from ._corpus import load_corpus
from .fusion import GRAPH_BOOST, rrf_fuse
from .sparse import SparseIndex

__all__ = ["RetrievedChunk", "Retriever", "retrieve", "default_retriever"]


@dataclass
class RetrievedChunk:
    text: str
    lineage_id: str
    source_methods: List[str]
    score: float
    graph_edge_type: Optional[str] = None  # e.g. "CROSS_REFERENCES@1hop"
    model_variant: Optional[str] = None   # base | finetuned_stageN
    doc_id: Optional[str] = None
    node_type: Optional[str] = None
    extra: dict = field(default_factory=dict)


class _Lazy:
    def __init__(self, fn):
        self._fn = fn
        self._value = None
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            if self._value is None:
                self._value = self._fn()
            return self._value


class Retriever:
    MODES = ("sparse", "dense", "graph", "neo4j_graph", "hybrid",
              "hybrid_graph", "hybrid_rerank",
              "hybrid_graph_rerank", "hybrid_gr_1hop",
              "hybrid_gr_relations", "hybrid_gr_expand")

    def __init__(self, chunks_dir: Optional[Path] = None,
                 graph_dir: Optional[Path] = None,
                 dense_models: Optional[list] = None,
                 dense_specs: Optional[dict] = None):
        import os

        if dense_specs is None:
            dense_specs = {"all-MiniLM-L6-v2": "all-MiniLM-L6-v2"}
            raw = os.getenv("RETRIEVAL_EXTRA_DENSE")
            if raw:
                try:
                    for item in json.loads(raw):
                        if item.get("name") and item.get("model"):
                            dense_specs[item["name"]] = item["model"]
                except (ValueError, AttributeError) as exc:  # noqa: BLE001
                    print("RETRIEVAL_EXTRA_DENSE ignored:", exc)
        if dense_models is None:
            dense_models = list(dense_specs)
        self.corpus = load_corpus(chunks_dir, graph_dir)
        self.chunks = self.corpus["chunks"]
        self._texts = [c.text for c in self.chunks]
        self._idx_to_lid = [c.lineage_id for c in self.chunks]
        self._by_lineage = {c.lineage_id: i
                            for i, c in enumerate(self.chunks)}
        self.sparse = SparseIndex(self._texts)

        from .dense import DenseIndex
        from .graph import GraphRetriever

        self.dense_specs = dense_specs
        self.dense_models = dense_models
        self._dense = {
            name: _Lazy(lambda n=name: DenseIndex(
                self._texts, model_name=dense_specs.get(n, n)))
            for name in self.dense_models
        }
        self._graph = _Lazy(lambda: GraphRetriever(self.corpus["graph"]))
        # listwise LLM re-ranker (hybrid_rerank). Lazy so importing the
        # retrieval layer does not import the LLM / touch Ollama until the
        # mode is actually used.
        self._reranker = _Lazy(self._build_reranker)
        # importing it connects to Neo4j / pulls neo4j_graphrag, so we
        # must not touch it for callers who never select the mode.
        self._neo4j_graph = _Lazy(self._build_neo4j_graph)

    # -- single methods (lineage_id ranked lists) ------------------------
    def search_sparse(self, query: str, k: int = 50) -> List[tuple]:
        return [(self._idx_to_lid[i], s)
                for i, s in self.sparse.search(query, k)]

    def search_dense(self, query: str, k: int = 50) -> dict:
        return {name: [(self._idx_to_lid[i], s)
                       for i, s in idx.get().search(query, k)]
                for name, idx in self._dense.items()}

    def search_graph(self, query: str, k: int = 50,
                     n_seeds: int = 10) -> list:
        """Return [(lineage_id, hop, [edge_kinds])] for the top seeds
        (hop=0, kinds=[]) plus their 1-2 hop graph neighbors. All
        entries are chunks in the retrieval corpus."""
        from ._corpus import article_node_for

        graph = self._graph.get()
        resolved = []   # (seed_lid, node)
        seen_nodes = set()
        for lid, _s in self.search_sparse(query, n_seeds):
            c = self.chunks[self._by_lineage[lid]]
            node = article_node_for(c, self.corpus["article_index"], graph)
            if node and node not in seen_nodes:
                seen_nodes.add(node)
                resolved.append((lid, node))
        out = [(lid, 0, []) for lid, _node in resolved]
        for lid, _score, hop, kinds in graph.search(sorted(seen_nodes),
                                                    max_hops=2):
            if lid in self._by_lineage:
                out.append((lid, hop, kinds))
        return out[:k]

    #: sentinel -- distinguishes "caller did not pass graph_cfg" from "caller
    #: explicitly passed ``None``" (the latter is how ``gr_enable=False``
    #: disables the feature without changing the A0 baseline behaviour).
    _GR_DEFAULT = object()

    # -- public entry point -----------------------------------------------
    def retrieve(self, query: str, k: int = 10, mode: str = "hybrid",
                  graph_cfg: object = _GR_DEFAULT,
                  with_expand: Optional[bool] = None,
                  pool_n: Optional[int] = None
                  ) -> List[RetrievedChunk]:
        """``pool_n`` (spec §2) overrides the re-ranker pool depth for graph
        modes.  ``None`` (the default) preserves historical behaviour:
        ``n = max(k, 5)``.  Pass a larger value (e.g. ``gr_candidate_k=30``)
        to give the LLM more candidates to reorder.  The final top-k is
        always ``k`` regardless of pool depth."""
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if mode == "sparse":
            rank_lists = [("sparse", self.search_sparse(query, max(k, 5)))]
            fused = rrf_fuse(rank_lists)
            top = sorted(fused, key=lambda lid: (-fused[lid]["score"], lid))
            return self._from_top(top, fused, rank_lists, k)
        if mode == "dense":
            name, ranked = self.best_dense(query, max(k, 5))
            rank_lists = [(name, ranked)]
            fused = rrf_fuse(rank_lists)
            top = sorted(fused, key=lambda lid: (-fused[lid]["score"], lid))
            return self._from_top(top, fused, rank_lists, k,
                                   model_variant=name)
        if mode == "graph":
            return self._from_graph(query, k)
        if mode == "neo4j_graph":
            return self._from_neo4j_graph(query, k)
        if mode == "hybrid_graph":
            return self._hybrid_graph(query, k)
        if mode == "hybrid_rerank":
            # A0 baseline: no graph context, no expansion, no pool override --
            # byte-identical to v1 by construction (graph_cfg/with_expand/pool_n
            # are all ignored here so hybrid_rerank stays cache-key stable).
            return self._hybrid_rerank(query, k)
        if mode in self._GRAPH_RERANK_MODES:
            if graph_cfg is self._GR_DEFAULT:
                # direct API, no config provided -- apply the mode's preset.
                graph_cfg = self._gr_cfg(mode)
            # explicit ``None`` (gr_enable=False) is honoured: no context.
            if with_expand is None:
                with_expand = (mode == "hybrid_gr_expand")
            return self._hybrid_rerank(query, k,
                                       graph_cfg=graph_cfg,
                                       with_expand=bool(with_expand),
                                       pool_n=pool_n)
        return self._hybrid(query, k)

    #: graph-context ablation presets over the SAME hybrid pool (task §15).
    #: A0 = hybrid_rerank (no context); A2 = hybrid_gr_1hop; A4 =
    #: hybrid_gr_relations; A3 + A5/A6 = hybrid_graph_rerank (1/2-hop, typed
    #: relations + neighbour labels + snippets, query-aware); A7 =
    #: hybrid_gr_expand (controlled graph-candidate expansion).
    _GRAPH_RERANK_MODES = ("hybrid_graph_rerank", "hybrid_gr_1hop",
                           "hybrid_gr_relations", "hybrid_gr_expand")

    def _build_neo4j_graph(self):
        """Lazy P2 retriever -- built on first use; touches Neo4j only then."""
        import sys

        src = str(Path(__file__).resolve().parent.parent)
        if src not in sys.path:
            sys.path.insert(0, src)
        from graphrag_n4j.retriever import Neo4jGraphRetriever
        return Neo4jGraphRetriever()

    def search_neo4j_graph(self, query: str, k: int = 50) -> list:
        """``[(lineage_id, hop, [edge_kinds])]`` over the Neo4j graph --
        hybrid vector seeds + bounded bidirectional expansion (P2).
        Same result contract as :meth:`search_graph`; the seeding
        strategy (vector vs sparse-BM25) and traversal scope (4 edge
        kinds, both directions) differ, so the two are directly
        comparable but *not* subsets of each other."""
        return self._neo4j_graph.get().search(query, k)

    # -- modes --------------------------------------------------------------
    def best_dense(self, query: str, k: int):
        results = self.search_dense(query, k)
        cands = [(name, ranked) for name, ranked in results.items()
                 if ranked]
        if not cands:
            return "dense", []
        return max(cands, key=lambda nr: nr[1][0][1])

    def _build_reranker(self):
        """Lazy listwise LLM re-ranker -- built on first use of `hybrid_rerank`."""
        from reranking import Reranker
        return Reranker()

    def _hybrid_pool(self, query: str, n: int):
        """sparse + dense RRF fused pool, capped to ``n`` (the ``hybrid`` seed).

        Shared seeding for :meth:`_hybrid` and :meth:`_hybrid_rerank` so the
        two modes differ ONLY in what they do with this pool (hybrid: annotate
        + light rescore of survivors; hybrid_rerank: LLM reorder of the same pool).
        """
        rank_lists = [("sparse", self.search_sparse(query, n))]
        for name, ranked in self.search_dense(query, n).items():
            if ranked:
                rank_lists.append((name, ranked))
        fused = rrf_fuse(rank_lists)
        top_lids = sorted(fused, key=lambda lid: (-fused[lid]["score"], lid))[:n]
        return top_lids, fused, rank_lists

    def _hybrid(self, query: str, k: int) -> List[RetrievedChunk]:
        n = max(k, 5)
        top_lids, fused, rank_lists = self._hybrid_pool(query, n)
        # graph boost: regulatory 1-2 hop neighbours of the top seeds
        neigh = [g for g in self.search_graph(query, 30)
                 if g[1] >= 1 and g[0] in set(top_lids)]
        for g in neigh:
            fused[g[0]]["score"] += GRAPH_BOOST * (1.0 if g[1] == 1 else 0.5)
        return self._from_top(top_lids, fused, rank_lists, k, graph_edges=neigh)

    def _hybrid_rerank(self, query: str, k: int,
                       graph_cfg: Optional[object] = None,
                       with_expand: bool = False,
                       pool_n: Optional[int] = None) -> List[RetrievedChunk]:
        """``hybrid`` pool re-ranked by a listwise LLM re-ranker.

        The candidate pool is the SAME set :meth:`_hybrid` seeds from (sparse
        + dense RRF, capped to ``max(k,5)``); only the *order* inside that
        pool is allowed to change -- unless ``with_expand`` (A7), in which
        case graph-adjacent chunks (already in the fused pool's universe,
        1-2 hops from a seed's graph node) are admitted as *additional*
        controlled candidates.

        ``graph_cfg`` is optional (:class:`reranking.graph_context.
        GraphContextConfig`).  When it is ``None`` (the ``hybrid_rerank``
        baseline) no graph evidence is computed or sent to the LLM and the
        prompt is byte-identical to the v1 re-ranker -- the ``hybrid_rerank``
        results are preserved bit-for-bit.  When set, each candidate's local
        graph context (typed/directional relations) is rendered and injected
        into the re-rank prompt as evidence.

        Consequences for the benchmark (measured, 60 queries, A0 baseline):

        * ``k >= pool size`` (e.g. k=20): Recall / Precision / Hit are
          set-identical to ``hybrid`` (top-20 pool is unchanged).
        * ``k < pool size`` (e.g. k=1,3,5,10): Recall / Precision / Hit /
          MRR / nDCG shift because we are reordering within the same top-10
          and then truncating.  Measured delta:
          Recall@1 0.583 -> 0.883 (Δ +0.30),
          MRR@10  0.710 -> 0.917 (Δ +0.21),
          nDCG@10 0.766 -> 0.929 (Δ +0.16).

        If the LLM is unreachable or its output does not parse to a valid
        permutation, the pool order is kept as-is (a no-op, not a corrupted
        result).

        ``pool_n`` (spec §2) optionally overrides the pool depth: when set,
        a larger hybrid pool is fetched before the LLM re-orders it, giving
        the re-ranker more candidates to choose from.  ``None`` keeps the
        historical ``n = max(k, 5)`` so the A0 baseline is untouched.
        """
        n = max(k, 5)
        if pool_n is not None:
            n = max(k, 5, int(pool_n))
        top_lids, fused, rank_lists = self._hybrid_pool(query, n)
        # keep graph-boost provenance consistent with hybrid (annotation only)
        neigh = [g for g in self.search_graph(query, 30)
                 if g[1] >= 1 and g[0] in set(top_lids)]
        for g in neigh:
            fused[g[0]]["score"] += GRAPH_BOOST * (1.0 if g[1] == 1 else 0.5)

        expanded_lids: List[str] = []
        if with_expand:
            # A7: controlled expansion -- graph-adjacent chunks (CROSS_REFERENCES
            # / AMENDS, 1-2 hops from a seed's article node) that are still
            # corpus chunks get a place in the pool behind the hybrid seeds.
            # They are *enriched* by the graph, not invented: no ext: stubs,
            # no non-chunk nodes.  Bounded to 2 per seed (deterministic order).
            from ._corpus import article_node_for
            gr = self._graph.get()
            seed_nodes = set()
            for lid in top_lids:
                idx = self._by_lineage.get(lid)
                if idx is None:
                    continue
                node = article_node_for(self.chunks[idx],
                                        self.corpus["article_index"],
                                        self.corpus["graph"])
                if node:
                    seed_nodes.add(node)
            expanded: List[str] = []
            for node in sorted(seed_nodes):
                # search() returns [(dst, score, hop, kinds), ...] sorted by
                # hop weight -- take up to 2 per seed, only existing chunks.
                taken = 0
                for row in gr.search([node], max_hops=2):
                    dst = row[0]
                    if dst in self._by_lineage and dst not in top_lids \
                            and dst not in expanded:
                        expanded.append(dst)
                        taken += 1
                        if taken >= 2:
                            break
                    if len(expanded) >= 2 * len(seed_nodes):
                        break
            top_lids = list(top_lids) + expanded
            expanded_lids = expanded

        # -- ask the LLM for a permutation of top_lids by relevance ----------------
        candidates, gc_by_lid = self._gr_candidates(
            query, top_lids, graph_cfg, neigh, fused)
        pool = candidates or []
        if not pool:
            return self._from_top(top_lids, fused, rank_lists, k,
                                  graph_edges=neigh)
        rr = self._reranker.get()
        # A0 baseline (graph_cfg is None): do NOT pass max_candidates -- the
        # Reranker's historical cap of 20 applies, keeping the cache key
        # byte-identical to v1.  Graph modes: send the whole (deeper) pool so
        # gr_candidate_k/pool_n actually reaches the model instead of being
        # silently truncated to 20.
        if graph_cfg is None:
            result = rr.rerank(query, pool)
        else:
            result = rr.rerank(query, pool, max_candidates=len(pool))
        if {x for x in result["order"]} != set(top_lids):
            new_order = list(top_lids)
            # record that we fell back even though the LLM was reachable
            result = {**result, "status": "fallback", "note": "permutation"}
        else:
            new_order = list(result["order"])
        out = self._from_top(new_order, fused | {l: {"score": 0.0, "methods": ["graph:expand"]} for l in expanded_lids},
                             rank_lists, k, graph_edges=neigh,
                             rerank_meta=result)
        if gc_by_lid:
            for rec in out:
                if rec.lineage_id in gc_by_lid:
                    rec.extra["graph_context"] = gc_by_lid[rec.lineage_id]
        return out

    def _gr_candidates(self, query: str, top_lids: List[str],
                       graph_cfg, neigh, fused):
        """Build the candidate dicts for the re-ranker.

        When ``graph_cfg`` is set, each candidate also carries a
        ``graph_context`` string (typed/directional relations of the chunk's
        graph node) that the re-ranker injects into its prompt as evidence.
        The provenance lines are returned for ``RetrievedChunk.extra``.
        """
        candidates = []
        gc_by_lid: Dict[str, str] = {}
        for lid in top_lids:
            idx = self._by_lineage.get(lid)
            if idx is None:
                continue
            c = self.chunks[idx]
            d = {"lineage_id": lid, "doc_id": c.doc_id, "text": c.text}
            if graph_cfg is not None:
                node = self._node_for(lid)
                if node is not None:
                    from reranking.graph_context import build_graph_context
                    ctx = build_graph_context(self.corpus["graph"], node,
                                              query, graph_cfg)
                    if ctx:
                        lines = ctx.splitlines()
                        # drop the header line (we re-label in the prompt)
                        if lines and lines[0].startswith("Graph context"):
                            lines = lines[1:]
                        d["graph_context"] = "\n".join(lines)
                        gc_by_lid[lid] = ctx
            candidates.append(d)
        return candidates, gc_by_lid

    def _node_for(self, lid: str) -> Optional[str]:
        """Graph node id for a chunk lineage id (article/preamble/document)."""
        from ._corpus import article_node_for
        idx = self._by_lineage.get(lid)
        if idx is None:
            return None
        return article_node_for(self.chunks[idx],
                                self.corpus["article_index"],
                                self.corpus["graph"])

    # -- graph-context ablation presets (task §15) ------------------------
    def _gr_cfg(self, mode: str):
        from reranking.graph_context import GraphContextConfig
        if mode == "hybrid_gr_1hop":           # A2: 1-hop only
            return GraphContextConfig(include_paths=False)
        if mode == "hybrid_gr_relations":      # A4: typed relations, no snippets
            return GraphContextConfig(include_snippets=False,
                                      include_paths=False)
        # hybrid_graph_rerank (A3 + A5 + A6): 1/2-hop, typed + neighbour
        # labels + evidence snippets, query-aware ordering
        return GraphContextConfig()

    def _hybrid_graph_rerank(self, query: str, k: int,
                             mode: str) -> List[RetrievedChunk]:
        """Legacy direct caller; retained for any existing call sites."""
        return self._hybrid_rerank(
            query, k,
            graph_cfg=self._gr_cfg(mode),
            with_expand=(mode == "hybrid_gr_expand"))

    def _hybrid_graph(self, query: str, k: int) -> List[RetrievedChunk]:
        """RRF over the sparse, dense *and* graph channels.

        Unlike :meth:`_hybrid` -- which computes the top-k from sparse+dense
        and then only annotates (and cosmetically rescored) the survivors --
        here the 1-2 hop regulatory neighbours form a *third* ranked list fed
        into :func:`rrf_fuse`.  Consequences that make this structurally
        distinct from ``hybrid``:

        * a chunk reachable via CROSS_REFERENCES/AMENDS but *not* surfaced by
          either channel can enter the top-k (new-node admission);
        * a chunk appearing in more channels accumulates RRF credit, so the
          relative ordering of the fused set can shift (re-ranking).

        The neighbour list is the same ``search_graph`` set (ordered by hop
        weight, seeds excluded) used to *annotate* provenance, so the fused
        ranking and the reported ``graph_edge_type`` stay consistent.
        """
        from .graph import HOP_WEIGHT

        n = max(k, 5)
        rank_lists = [("sparse", self.search_sparse(query, n))]
        for name, ranked in self.search_dense(query, n).items():
            if ranked:
                rank_lists.append((name, ranked))
        # graph channel: 1-2 hop neighbours, ordered by hop weight (seeds out)
        graph_rows = [g for g in self.search_graph(query, max(3 * n, 30))
                      if g[1] >= 1]
        if graph_rows:
            rank_lists.append(
                ("graph", [(lid, HOP_WEIGHT.get(hop, 0.25))
                           for lid, hop, _kinds in graph_rows]))
        fused = rrf_fuse(rank_lists)
        top_lids = sorted(fused, key=lambda lid: (-fused[lid]["score"], lid))
        top_lids = top_lids[: max(k, 5)]
        top_set = set(top_lids)
        neigh = [g for g in graph_rows if g[0] in top_set]
        return self._from_top(top_lids, fused, rank_lists, k,
                              graph_edges=neigh)

    def _from_graph(self, query: str, k: int) -> List[RetrievedChunk]:
        """Graph mode = seeds + their 1-2 hop article neighbours.

        hop=0 is the seed's own containing article (the article node the
        sparse seed resolves to). hop>=1 are graph neighbours reached by
        CROSS_REFERENCES / AMENDS. Both are returned so the user sees the
        full regulatory context in one call."""
        return self._graph_render(query, self.search_graph(query, k * 3), k)

    def _from_neo4j_graph(self, query: str, k: int) -> List[RetrievedChunk]:
        """Neo4j graph mode (P2) = hybrid vector seeds + bounded 1-2 hop
        bidirectional expansion over all four edge kinds. Same rendering
        contract as :meth:`_from_graph`; seeding strategy and traversal
        scope differ, so the two are comparable but not nested."""
        return self._graph_render(query, self.search_neo4j_graph(query, k * 3), k)

    def _graph_render(self, query: str, ranked: list,
                      k: int) -> List[RetrievedChunk]:
        out: List[RetrievedChunk] = []
        for lid, hop, kinds in ranked[:k]:
            idx = self._by_lineage.get(lid)
            if idx is None:
                continue
            c = self.chunks[idx]
            if hop == 0:
                methods = ["graph:seed"]
                edge_str = "seed"
            else:
                methods = [f"graph:{kind}@{hop}hop" for kind in kinds]
                edge_str = f"{';'.join(kinds)}@{hop}hop"
            out.append(RetrievedChunk(
                text=c.text, lineage_id=lid, source_methods=methods,
                score=1.0 if hop <= 1 else 0.5,
                graph_edge_type=edge_str,
                doc_id=c.doc_id, node_type=c.node_type,
                extra={"graph": {"hop": hop, "edges": kinds, "boost": 0.0}}))
        return out[:k]

    def _from_top(self, top_lids: List[str], fused, rank_lists: List[tuple],
                  k: int, graph_edges: Optional[list] = None,
                  model_variant: Optional[str] = None,
                  rerank_meta: Optional[dict] = None,
                  ) -> List[RetrievedChunk]:
        if isinstance(top_lids, tuple):
            top_lids = list(top_lids)
        selected = [lid for lid in top_lids if lid in fused][:k]
        if not selected:
            selected = sorted(fused, key=lambda lid: (-fused[lid]["score"], lid))[:k]
        edge_by_lid = {g[0]: (g[1], g[2]) for g in graph_edges or []}
        out: List[RetrievedChunk] = []
        for lid in selected:
            idx = self._by_lineage.get(lid)
            if idx is None:
                continue
            c = self.chunks[idx]
            methods = list(fused[lid]["methods"])
            extra = {}
            if lid in edge_by_lid:
                hop, kinds = edge_by_lid[lid]
                extra["graph"] = {"hop": hop, "edges": kinds,
                                  "boost": GRAPH_BOOST * (1.0 if hop == 1 else 0.5)}
            if rerank_meta is not None:
                extra["rerank"] = {
                    "status": rerank_meta.get("status"),
                    "model": rerank_meta.get("model"),
                    "pool": rerank_meta.get("pool"),
                }
            out.append(RetrievedChunk(
                text=c.text, lineage_id=lid, source_methods=methods,
                score=round(fused[lid]["score"], 6),
                graph_edge_type=(f"{';'.join(edge_by_lid[lid][1])}@"
                                  f"{edge_by_lid[lid][0]}hop"
                                  if lid in edge_by_lid else None),
                model_variant=model_variant, doc_id=c.doc_id,
                node_type=c.node_type, extra=extra))
        return out


_default: Optional[Retriever] = None
_default_lock = threading.Lock()


def default_retriever() -> Retriever:
    """The single shared Retriever used by :func:`retrieve`.

    Notebooks should build ``r = retrieval.default_retriever()`` so the local
    ``r`` and the public ``retrieve()`` entry point operate on the *same*
    instance (identical ``dense_models``, chunk order, etc.). Building a
    separate ``Retriever()`` risks the two diverging, which breaks
    cross-validations that compare ``retrieve()`` results against ``r``.
    """
    global _default
    with _default_lock:
        if _default is None:
            _default = Retriever()
        return _default


def retrieve(query: str, k: int = 10, mode: str = "hybrid",
             graph_cfg: object = Retriever._GR_DEFAULT,
             with_expand: Optional[bool] = None
             ) -> List[RetrievedChunk]:
    return default_retriever().retrieve(
        query, k=k, mode=mode, graph_cfg=graph_cfg, with_expand=with_expand)
