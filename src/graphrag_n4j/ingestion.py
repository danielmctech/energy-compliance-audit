"""Idempotent ingestion of ``notebooks/data/graph/{nodes,edges}.jsonl`` into Neo4j.

Verified invariants (from a scan of the JSONLs before writing this file):
  - ``lineage_id`` unique across all 2,049 node records  ->  MERGE on it is safe.
  - 0 edges with dangling ``src`` / ``dst``.
  - 144 nodes (entity + external_*) carry no ``doc_id``  ->  we OMIT absent
    properties (Cypher props cannot be null).
  - 4,139 distinct ``(src, dst, kind)`` triples, so one relationship per pair
    loses nothing.

Cypher choices:
  * ``MERGE (n:Label {lineage_id: $lid})`` -- Community-safe (no NODE KEY,
    no elementId lookup).
  * Labels are fixed per kind (KIND_TO_LABEL below); one MERGE per label
    makes the Cypher trivially predictable and lets the constraints below
    target a single label per constraint.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

KIND_TO_LABEL: Dict[str, str] = {
    "document":          "Document",
    "external_document": "ExternalDocument",
    "article":           "Article",
    "external_article":  "ExternalArticle",
    "preamble":          "Preamble",
    "term":              "Term",
    "entity":            "Entity",
}
LABEL_TO_KIND: Dict[str, str] = {v: k for k, v in KIND_TO_LABEL.items()}

# known relationship types (must match the extraction schema); used to
# reject malformed edge records before any Cypher is built (kind is
# otherwise spliced into the query text as a relationship-type literal).
KNOWN_EDGE_KINDS: Tuple[str, ...] = (
    "CROSS_REFERENCES", "AMENDS", "DEFINED_IN", "APPLIES_TO",
)

EMBEDDABLE_KINDS: Tuple[str, ...] = ("term", "article", "preamble", "entity")

# on-disk embedding cache (keyed by model+dim), shared across runs
EMBEDDING_CACHE_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "notebooks" / "data" / "neo4j"
)


def _graph_dir() -> Path:
    try:
        from common import repo_root
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from common import repo_root
    return repo_root() / "notebooks" / "data" / "graph"


# ---------------------------------------------------------------------------
# jsonl loaders
# ---------------------------------------------------------------------------

def load_nodes(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = Path(path) if path else _graph_dir() / "nodes.jsonl"
    with open(p, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_edges(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = Path(path) if path else _graph_dir() / "edges.jsonl"
    with open(p, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# search_text (fulltext + embedding seed)
# ---------------------------------------------------------------------------

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def search_text_for(node: Dict[str, Any]) -> str:
    kind = node["kind"]
    if kind == "term":
        return _norm(node.get("term")) or node["lineage_id"]
    if kind == "entity":
        return _norm(node.get("label")) or node["lineage_id"]
    if kind in ("article", "external_article"):
        num = node.get("number")
        title = (node.get("title") or "").strip()
        if title and _norm(title) != f"article {num}".strip():
            return f"Article {num} - {title}"
        return f"Article {num}"
    if kind in ("document", "external_document"):
        return _norm(node.get("title")) or node["lineage_id"]
    if kind == "preamble":
        return f"preamble of {node['doc_id']}"
    return node["lineage_id"]


def _node_props(node: Dict[str, Any]) -> Dict[str, Any]:
    """Property map for ``SET``.  Absent fields are omitted, not null."""
    out: Dict[str, Any] = {
        "lineage_id":  node["lineage_id"],
        "kind":        node["kind"],
        "search_text": search_text_for(node),
    }
    for key in ("doc_id", "title", "term", "label", "number"):
        if node.get(key) is not None:
            out[key] = node[key]
    return out


def _edge_props(e: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"kind": e["kind"]}
    for key in (
        "doc_id", "offset", "snippet", "ref_number", "instrument",
        "term", "entity", "confidence", "unresolved",
    ):
        if e.get(key) is not None:
            out[key] = e[key]
    return out


# ---------------------------------------------------------------------------
# upserts
# ---------------------------------------------------------------------------

def upsert_nodes(
    driver,
    nodes: List[Dict[str, Any]],
    database: Optional[str] = None,
    batch_size: int = 500,
    log=print,
) -> int:
    """MERGE nodes, grouped by label (one MERGE per label keeps the Cypher
    simple and matches the per-label uniqueness constraints).

    Raises ``ValueError`` on malformed records (missing ``kind`` /
    ``lineage_id`` or an unknown kind) instead of splicing bad labels into
    Cypher."""
    for n in nodes:
        kind = n.get("kind")
        if not n.get("lineage_id") or kind not in KIND_TO_LABEL:
            raise ValueError(f"malformed node record: {n!r}")
    total = 0
    for i in range(0, len(nodes), batch_size):
        chunk = nodes[i : i + batch_size]
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for n in chunk:
            by_label.setdefault(KIND_TO_LABEL[n["kind"]], []).append(n)
        for label, rows in by_label.items():
            q = (
                "UNWIND $rows AS r "
                f"MERGE (n:{label} {{lineage_id: r.lineage_id}}) "
                "SET n += r.props "
            )
            payload = [
                {"lineage_id": n["lineage_id"], "props": _node_props(n)}
                for n in rows
            ]
            with driver.session(database=database) as s:
                s.run(q, {"rows": payload, "label": label})
        total += len(chunk)
    log(f"upsert_nodes OK ({total} nodes)")
    return total


def upsert_edges(
    driver,
    edges: List[Dict[str, Any]],
    nodes: Optional[List[Dict[str, Any]]] = None,
    database: Optional[str] = None,
    batch_size: int = 500,
    log=print,
) -> int:
    """MERGE edges.

    Endpoint labels vary by edge kind (verified against the JSONL):
      - CROSS_REFERENCES : article -> article | external_article
      - AMENDS           : document | preamble | article -> external_document | document
      - DEFINED_IN       : term -> document | article | preamble
      - APPLIES_TO       : document | preamble | article -> entity
    so we group by ``(src_label, dst_label, kind)`` and run one Cypher query
    per group -- each group uses a fixed (labelled) MATCH which is both
    cheaper and easier to reason about than ``WHERE a:Article OR a:...``.
    """
    if nodes is None:
        nodes = load_nodes()
    lid_to_kind: Dict[str, str] = {n["lineage_id"]: n["kind"] for n in nodes}

    # -- validate before any Cypher is built (kind is spliced into the
    #    query text as a relationship-type literal) ---------------------
    for e in edges:
        if not e.get("src") or not e.get("dst"):
            raise ValueError(f"malformed edge record (endpoints): {e!r}")
        if e.get("kind") not in KNOWN_EDGE_KINDS:
            raise ValueError(
                f"malformed edge record (unknown kind): {e!r} -- known: "
                f"{list(KNOWN_EDGE_KINDS)}"
            )

    # -- group edges by (src_label, dst_label, kind) ---------------------
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for e in edges:
        sk = lid_to_kind.get(e["src"])
        dk = lid_to_kind.get(e["dst"])
        if sk is None or dk is None:
            # should not happen (0 dangling verified); log + skip
            log(f"  WARN: edge drops (missing endpoint kind): src={e['src']} dst={e['dst']}")
            continue
        key = (KIND_TO_LABEL[sk], KIND_TO_LABEL[dk], e["kind"])
        groups.setdefault(key, []).append(e)

    total = 0
    for (src_label, dst_label, kind), group_edges in groups.items():
        # each MERGE is one fixed-label MATCH; batched inside the group
        for i in range(0, len(group_edges), batch_size):
            chunk = group_edges[i : i + batch_size]
            payload = [
                {"src": e["src"], "dst": e["dst"], "props": _edge_props(e)}
                for e in chunk
            ]
            q = (
                "UNWIND $rows AS r "
                f"MATCH (a:{src_label} {{lineage_id: r.src}}) "
                f"MATCH (b:{dst_label} {{lineage_id: r.dst}}) "
                f"MERGE (a)-[e:{kind}]->(b) "
                "SET e = r.props"
            )
            with driver.session(database=database) as s:
                s.run(q, {"rows": payload})
            total += len(chunk)
    log(f"upsert_edges OK ({total} edges, {len(groups)} (src,dst,kind) groups)")
    return total


# ---------------------------------------------------------------------------
# embeddings (Ollama /v1 + on-disk cache)
# ---------------------------------------------------------------------------

def _cache_path(model: str, dim: int) -> Path:
    key = hashlib.sha1(f"{model}|{dim}".encode()).hexdigest()[:12]
    return EMBEDDING_CACHE_DIR / f"embed_cache_{key}.json"


def _load_cache(model: str, dim: int) -> Dict[str, List[float]]:
    p = _cache_path(model, dim)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    return {e["key"]: e["emb"] for e in obj.get("entries", [])}


def _save_cache(model: str, dim: int, entries: Dict[str, List[float]]) -> None:
    p = _cache_path(model, dim)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"key": k, "emb": v}
        for k, v in sorted(entries.items(), key=lambda kv: kv[0])
    ]
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"model": model, "dim": dim, "entries": payload}, fh)


def existing_embeddings_from_db(
    driver,
    database: Optional[str] = None,
    embedding_property: str = "emb",
) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    with driver.session(database=database) as s:
        for rec in s.run(
            "MATCH (n) WHERE n.emb IS NOT NULL "
            "RETURN n.lineage_id AS lid, n.emb AS v"
        ):
            out[rec["lid"]] = list(rec["v"])
    return out


def embed_search_texts(
    nodes: List[Dict[str, Any]],
    embeddings,
    existing_emb: Optional[Dict[str, List[float]]] = None,
    batch_size: int = 64,
    log=print,
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """Compute ``emb`` for embeddable nodes missing one in the DB.

    Deduplicates identical ``search_text`` strings (e.g. several articles
    titled "Article 1") and caches unique texts on disk, keyed by
     (model, dim)."""
    try:
        from neo4j_config import graphrag_settings
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from neo4j_config import graphrag_settings
    g = graphrag_settings()
    model, dim = g.embedding_model, g.embedding_dim

    embeddable = [n for n in nodes if n["kind"] in EMBEDDABLE_KINDS]
    already = set((existing_emb or {}).keys())

    need_pair: List[Tuple[str, str]] = [
        (n["lineage_id"], search_text_for(n)) for n in embeddable
        if n["lineage_id"] not in already
    ]
    if not need_pair:
        log("embed SKIPPED (all already embedded)")
        return dict(existing_emb or {}), {
            "total": len(embeddable), "new": 0,
            "reused": len(embeddable),
        }

    cache = _load_cache(model, dim)
    text_to_lids: Dict[str, List[str]] = {}
    for lid, text in need_pair:
        text_to_lids.setdefault(text, []).append(lid)

    t0 = time.time()
    fresh: Dict[str, List[float]] = {}
    texts = list(text_to_lids)
    to_fetch = [t for t in texts if t not in cache]
    for i in range(0, len(to_fetch), batch_size):
        chunk = to_fetch[i : i + batch_size]
        resp = embeddings.client.embeddings.create(input=chunk, model=model)
        # the API `index` field is relative to *this* `input`, so use the
        # batch-local index, not the global `to_fetch` index.
        for d in resp.data:
            cache[chunk[d.index]] = d.embedding
        if i + batch_size < len(to_fetch):
            log(f"  embed {i + batch_size:4d} / {len(to_fetch)}")
    _save_cache(model, dim, cache)

    for text, lids in text_to_lids.items():
        vec = cache.get(text)
        if vec is None:
            continue
        for lid in lids:
            fresh[lid] = vec
    if existing_emb:
        fresh.update(existing_emb)
    log(
        f"embed OK ({len(fresh) - len(existing_emb or {})} new, "
        f"{len(cache)} unique texts, {time.time() - t0:.1f}s)"
    )
    return fresh, {
        "total": len(embeddable),
        "new":    len(fresh) - len(existing_emb or {}),
        "reused": len(existing_emb or {}),
        "cache_unique_texts": len(cache),
    }


def upsert_embeddings(
    driver,
    emb_map: Dict[str, List[float]],
    database: Optional[str] = None,
    embedding_property: str = "emb",
    batch_size: int = 512,
    log=print,
) -> int:
    """SET ``emb`` on nodes that have it in ``emb_map`` but not yet in DB."""
    total = 0
    rows = [
        {"lid": lid, "v": vec} for lid, vec in emb_map.items()
        if vec is not None
    ]
    for i in range(0, len(rows), batch_size):
        payload = rows[i : i + batch_size]
        with driver.session(database=database) as s:
            s.run(
                "UNWIND $rows AS r "
                "MATCH (n) WHERE n.lineage_id = r.lid "
                "SET n.emb = r.v",
                {"rows": payload},
            )
        total += len(payload)
    log(f"upsert_embeddings OK ({total} vectors)")
    return total


# ---------------------------------------------------------------------------
# counts for assertions
# ---------------------------------------------------------------------------

def counts(driver, database: Optional[str] = None) -> Dict[str, int]:
    out: Dict[str, Any] = {}
    with driver.session(database=database) as s:
        out["nodes"]    = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        out["edges"]    = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        out["terms"]    = s.run("MATCH (n:Term) RETURN count(n) AS c").single()["c"]
        out["articles"] = s.run("MATCH (n:Article) RETURN count(n) AS c").single()["c"]
        out["embedded"] = s.run(
            "MATCH (n) WHERE n.emb IS NOT NULL RETURN count(n) AS c"
        ).single()["c"]
        out["indexes"] = {
            r["name"]: f"{r['type']}/{r['state']}"
            for r in s.run("SHOW INDEXES YIELD name, type, state")
        }
    return out


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

def ingest(
    driver,
    nodes_path: Optional[Path] = None,
    edges_path: Optional[Path] = None,
    database: Optional[str] = None,
    embeddings=None,
    embedding_property: str = "emb",
    log=print,
) -> Dict[str, Any]:
    from .schema import apply_schema, build_indexes
    try:
        from neo4j_config import make_embeddings
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from neo4j_config import make_embeddings

    if embeddings is None:
        embeddings = make_embeddings()

    t0 = time.time()
    nodes = load_nodes(nodes_path)
    edges = load_edges(edges_path)
    log(f"loaded nodes={len(nodes):>5d}   edges={len(edges):>5d}")

    apply_schema(driver, database, log)
    build_indexes(driver, database, log)

    upsert_nodes(driver, nodes, database, log=log)

    existing = existing_embeddings_from_db(driver, database, embedding_property)
    emb_map, stats = embed_search_texts(nodes, embeddings, existing, log=log)
    if emb_map:
        upsert_embeddings(
            driver, emb_map, database,
            embedding_property=embedding_property, log=log,
        )

    upsert_edges(driver, edges, nodes=nodes, database=database, log=log)

    c = counts(driver, database)
    elapsed = time.time() - t0
    log(f"ingest DONE ({elapsed:.1f}s)")
    return {
        "nodes":      c["nodes"],
        "edges":      c["edges"],
        "terms":      c["terms"],
        "articles":   c["articles"],
        "embedded":   c["embedded"],
        "embed_stats":stats,
        "indexes":    c["indexes"],
        "elapsed_seconds": round(elapsed, 2),
    }


__all__ = [
    "KIND_TO_LABEL", "LABEL_TO_KIND", "KNOWN_EDGE_KINDS",
    "EMBEDDABLE_KINDS", "EMBEDDING_CACHE_DIR",
    "search_text_for",
    "load_nodes", "load_edges",
    "upsert_nodes", "upsert_edges", "upsert_embeddings",
    "existing_embeddings_from_db",
    "embed_search_texts",
    "counts", "ingest",
]
