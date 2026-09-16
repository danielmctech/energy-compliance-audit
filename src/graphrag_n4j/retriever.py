"""``neo4j_graph`` retriever (P2) -- bounded Cypher graph traversal.

Pipeline (avoids naive "question -> embedding -> top-k chunks";
the graph *is* the retrieval):

    query
      -> hybrid vector seed (vector top-k on :Term v_term_emb and
         :Article v_article_emb via the official ``neo4j-graphrag``
         ``VectorRetriever``),
      -> single bounded Cypher traversal
         ``(s)-[r*1..max_hops]-(t:Article|Preamble)`` restricted to the
         4 whitelisted edge kinds (CROSS_REFERENCES | AMENDS | DEFINED_IN
         | APPLIES_TO), bidirectional, per-target min length,
      -> rank:  by strongest anchoring-seed vector score, then hop,
      -> return ``[(lineage_id, hop, [edge_kinds])]`` in the exact shape
         the legacy ``graph`` mode produces, so
         ``Retriever._graph_render`` can render both identically.

Broader scope than the legacy ``graph`` mode (``src/retrieval/graph.py``) in
three ways:

* edges:  4 kinds, both directions, 1..max_hops     (legacy: 2 kinds,
  outgoing only, 2 hops)
* seeds:  semantic vector (term + article)          (legacy: sparse-BM25)
* rank:   by strongest anchoring seed + hop         (legacy: hop only)

The two are *comparable, not nested*: with different seeding strategies
each can surface results the other misses (see the Notebook 05 side-by-side
comparison).

Cypher safety:
  * the seed / target node labels and the edge kinds are **constants**
    drawn from :data:`EDGE_KINDS` / :data:`TARGET_LABELS` -- never
    user input.  We validate the constructor's ``edge_kinds`` against
    those known values before building the query string.
  * all other inputs (seed lineage_ids, max_hops) arrive as Cypher
    parameters; only ``max_hops`` is an integer literal in the query,
    which Cypher allows.
  * the traversal is read-only.

Observability: :attr:`last_debug` exposes the full trace -- seeds
per index (with vector scores), rendered Cypher (with redacted params),
the ranked result, and timing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from neo4j_config import (
        make_driver,
        make_embeddings,
        neo4j_settings,
    )
except ImportError:  # running without src/ already on sys.path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from neo4j_config import (
        make_driver,
        make_embeddings,
        neo4j_settings,
    )


# ---- schema-known constants (whitelist for Cypher literals) -------------
EDGE_KINDS: Tuple[str, ...] = (
    "CROSS_REFERENCES", "AMENDS", "DEFINED_IN", "APPLIES_TO",
)
TARGET_LABELS: Tuple[str, ...] = ("Article", "Preamble")  # retrievable nodes
SEED_INDEXES: Tuple[Tuple[str, str], ...] = (
    ("v_term_emb", "Term"),
    ("v_article_emb", "Article"),
)


# ---------------------------------------------------------------------------
# debug record
# ---------------------------------------------------------------------------

@dataclass
class _Debug:
    query: str
    k: int
    max_hops: int
    top_k: int
    edge_kinds: List[str]
    seeds: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    cypher: str = ""
    seed_ids: List[str] = field(default_factory=list)
    score_of_seed: Dict[str, float] = field(default_factory=dict)
    target_count: int = 0
    elapsed_ms: int = 0
    returned: List[Tuple[str, int, List[str]]] = field(default_factory=list)

    def dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "k": self.k,
            "max_hops": self.max_hops,
            "top_k": self.top_k,
            "edge_kinds": list(self.edge_kinds),
            "seeds": self.seeds,
            "cypher": self.cypher,
            "seed_ids": self.seed_ids,
            "score_of_seed": self.score_of_seed,
            "target_count": self.target_count,
            "elapsed_ms": self.elapsed_ms,
            "returned": self.returned,
        }


# ---------------------------------------------------------------------------
# retriever
# ---------------------------------------------------------------------------

class Neo4jGraphRetriever:
    """Hybrid vector seeds + one bounded Cypher traversal over the graph.

    Configurable (env-overridable) parameters, each bounded:
    max hops / max nodes / max relationships / relevance
    score / relationship-type-filter:

    * ``max_hops``    expansion depth        (RETRIEVAL_NEO4J_MAX_HOPS, 2)
    * ``top_k``       seeds fetched per index (RETRIEVAL_NEO4J_VTOPK,   15)
    * ``min_score``   keep seeds at or above this cosine (RETRIEVAL_NEO4J_MINS, 0.0)
    * ``max_results`` cap on returned rows  (RETRIEVAL_NEO4J_MAXRESULTS, 60)
    * ``edge_kinds``  whitelist of relationship types (default: all four)
    """

    def __init__(self,
                 driver=None, *,
                 embedder=None,
                 max_hops: Optional[int] = None,
                 top_k: Optional[int] = None,
                 min_score: Optional[float] = None,
                 max_results: Optional[int] = None,
                 edge_kinds: Optional[Tuple[str, ...]] = None):
        import os

        self.max_hops = max_hops if max_hops is not None else int(
            os.getenv("RETRIEVAL_NEO4J_MAX_HOPS", "2"))
        self.top_k = top_k if top_k is not None else int(
            os.getenv("RETRIEVAL_NEO4J_VTOPK", "15"))
        self.min_score = min_score if min_score is not None else float(
            os.getenv("RETRIEVAL_NEO4J_MINS", "0.0"))
        self.max_results = max_results if max_results is not None else int(
            os.getenv("RETRIEVAL_NEO4J_MAXRESULTS", "60"))
        self._edge_kinds = tuple(edge_kinds or EDGE_KINDS)
        unknown = [k for k in self._edge_kinds if k not in EDGE_KINDS]
        if unknown:
            raise ValueError(
                f"edge_kinds {unknown} not in known schema {list(EDGE_KINDS)}"
            )
        if self.max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        self._driver = driver
        self._embedder = embedder  # explicit (tests) or built from config
        self.last_debug: Optional[_Debug] = None

    # -- public API -------------------------------------------------------
    def search(
        self,
        query: str,
        k: int = 10,
    ) -> List[Tuple[str, int, List[str]]]:
        """Return ``[(lineage_id, hop, [edge_kinds])]`` where
        ``lineage_id`` resolves to a retrievable chunk (:Article or
        :Preamble) and ``hop`` is the minimum traversal length from a
        seed.  Sorted by (max anchoring-seed score desc, hop asc,
        lineage_id asc).  Only chunk-mapped targets are returned --
        Term / Entity hubs are used to anchor but not rendered.

        Shape-compatible with ``Retriever.search_graph`` so
        ``Retriever._graph_render`` can be shared across both graph
        modes.  Seeds themselves that are also :Article / :Preamble
        appear with ``hop=0`` (a retrievable chunk the user's query
        already matched semantically).
        """
        import time

        t0 = time.time()
        driver = self._driver or make_driver()
        embedder = self._embedder or make_embeddings()
        db = neo4j_settings().database
        dbg = _Debug(query=query, k=k, max_hops=self.max_hops,
                     top_k=self.top_k, edge_kinds=list(self._edge_kinds))

        # ---- 1) seeds ---------------------------------------------------
        seeds = self._select_seeds(driver, db, embedder, query, dbg)
        if not seeds:
            dbg.elapsed_ms = int((time.time() - t0) * 1000)
            self.last_debug = dbg
            return []
        seed_lids = list(seeds.keys())
        dbg.seed_ids = seed_lids
        dbg.score_of_seed = {lid: seeds[lid]["score"] for lid in seed_lids}

        # ---- 2) bounded traversal --------------------------------------
        cypher = self._traversal_query()
        params = {
            "seed_ids": seed_lids,
            "kinds": list(self._edge_kinds),
        }
        with driver.session(database=db) as s:
            raw = list(s.run(cypher, params))
        dbg.cypher = cypher
        dbg.target_count = len(raw)

        # ---- 3) rank -----------------------------------------------------
        score_of = dbg.score_of_seed
        best: Dict[str, Tuple[int, set, frozenset]] = {}

        def best_score(seed_id_set) -> float:
            return max((score_of.get(s, 0.0) for s in seed_id_set), default=0.0)

        for r in raw:
            lid = r["tid"]
            hop = int(r["hop"])
            kinds = frozenset(r["kinds"] or ())
            via = frozenset(r["seed_ids"] or [])
            cur = best.get(lid)
            # keep the record with the strongest anchoring seed (tie ->
            # shorter hop -> more edge coverage)
            if cur is None:
                best[lid] = (hop, kinds, via)
                continue
            cur_score, cur_hop = best_score(cur[2]), cur[0]
            new_score = best_score(via)
            if new_score > cur_score:
                best[lid] = (hop, kinds, via)

        # include seed nodes that are themselves retrievable (Article/Preamble)
        with driver.session(database=db) as s:
            seed_rows = list(s.run(
                "UNWIND $ids AS lid "
                "MATCH (n) WHERE n.lineage_id = lid "
                "  AND (n:Article OR n:Preamble) "
                "RETURN n.lineage_id AS lid",
                {"ids": seed_lids},
            ))
        for rec in seed_rows:
            s_lid = rec["lid"]
            if s_lid not in best:
                best[s_lid] = (0, frozenset(), frozenset({s_lid}))

        ranked = sorted(
            best.items(),
            key=lambda kv: (-best_score(kv[1][2]), kv[1][0], kv[0]),
        )[: self.max_results]

        out = [(lid, hop, sorted(kinds)) for lid, (hop, kinds, _via) in ranked]
        dbg.returned = out
        dbg.elapsed_ms = int((time.time() - t0) * 1000)
        self.last_debug = dbg
        return out

    # -- seed selection (official VectorRetriever) -----------------------
    def _select_seeds(self, driver, db, embedder, query, dbg):
        from neo4j_graphrag.retrievers import VectorRetriever
        from neo4j_graphrag.types import RetrieverResultItem

        def _fmt(record):
            node = record.get("node") or {}
            score = float(record.get("score") or 0.0)
            return RetrieverResultItem(
                content={
                    k: node.get(k)
                    for k in ("lineage_id", "search_text", "kind", "term",
                             "number", "title", "doc_id")
                    if node.get(k) is not None
                },
                metadata={"score": score,
                          "nodeLabels": list(record.get("nodeLabels") or [])},
            )

        per_index: Dict[str, List[Dict[str, Any]]] = {}
        seed_map: Dict[str, Dict[str, Any]] = {}
        for index_name, label in SEED_INDEXES:
            try:
                vr = VectorRetriever(
                    driver, index_name, embedder=embedder,
                    return_properties=["lineage_id", "search_text", "kind",
                                       "term", "number", "title", "doc_id"],
                    result_formatter=_fmt,
                    neo4j_database=db,
                )
                res = vr.search(query_text=query, top_k=self.top_k,
                                effective_search_ratio=1)
            except Exception as exc:  # e.g. index missing before P1
                per_index[f"{label}(error)"] = [{"error": str(exc)}]
                continue
            items = []
            for it in res.items:
                if not isinstance(it.content, dict):
                    continue
                lid = it.content.get("lineage_id")
                score = float((it.metadata or {}).get("score", 0.0))
                if not lid or score < self.min_score:
                    continue
                rec = {
                    "lineage_id": lid,
                    "score": score,
                    "label": label,
                    "search_text": it.content.get("search_text"),
                }
                items.append(rec)
                cur = seed_map.get(lid)
                if cur is None or score > cur["score"]:
                    seed_map[lid] = rec
            per_index[label] = items
        dbg.seeds = per_index
        return seed_map

    # -- Cypher -----------------------------------------------------------
    def _traversal_query(self) -> str:
        """Single bounded traversal from every seed at once.

        Returns rows ``(tid, hop, seed_ids, kinds)`` where:
          * ``tid``      target lineage (a :Article / :Preamble)
          * ``hop``      minimum path length (<= max_hops)
          * ``seed_ids`` distinct seed lineage_ids on a shortest path
          * ``kinds``    distinct relationship types encountered

        Relationship *types* are embedded as quoted string literals
        (they are validated against the known schema in the constructor);
        dynamic values (``seed_ids``) arrive as Cypher parameters.
        Read-only query.
        """
        import json

        target = "|".join(TARGET_LABELS)
        kinds_lit = ", ".join(json.dumps(k) for k in self._edge_kinds)
        return (
            "MATCH (s), p=(s)-[r*1.." + str(self.max_hops) + "]-(n:" + target + ") "
            "WHERE s.lineage_id IN $seed_ids "
            "  AND all(x IN r WHERE type(x) IN [" + kinds_lit + "]) "
            "  AND n <> s "
            "WITH n, "
            "     min(length(p)) AS hop, "
            "     collect(DISTINCT s.lineage_id) AS seed_ids, "
            "     collect([x IN r | type(x)]) AS per_path "
            "RETURN n.lineage_id AS tid, hop, seed_ids, "
            "       reduce(acc=[], p IN per_path | acc + p) AS kinds"
        )


def default() -> Neo4jGraphRetriever:
    """Build the shared instance (no caller-provided driver)."""
    return Neo4jGraphRetriever()


__all__ = [
    "EDGE_KINDS",
    "SEED_INDEXES",
    "TARGET_LABELS",
    "Neo4jGraphRetriever",
    "default",
]
