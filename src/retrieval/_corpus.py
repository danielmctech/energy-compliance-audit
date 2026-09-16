"""shared corpus model + tokenizer for the RAG retrieval layer."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import common as c
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import common as c


@dataclass
class Chunk:
    doc_id: str
    lineage_id: str
    node_id: str
    node_type: str
    text: str
    span: tuple
    context: dict = field(default_factory=dict)


@dataclass
class GraphLike:
    """Light dict-based graph: node lookup + outgoing edges by kind.

    ``adj``    -- src -> list[(dst, kind)]: the edge direction as stored.
    ``adj_in`` -- dst -> list[(src, kind)]: the reverse index, added so the
                  graph-aware reranker (v2) can present *incoming* relations
                  as their ``*_BY`` forms (CITED_BY / AMENDED_BY / ...).
                  ``adj_in`` is purely additive: every existing reader that
                  only touches ``nodes`` / ``adj`` is unaffected.
    """
    nodes: dict
    adj: dict = field(default_factory=dict)    # src -> list[(dst, kind)]
    adj_in: dict = field(default_factory=dict)  # dst -> list[(src, kind)]


def load_corpus(chunks_dir: Optional[Path] = None,
                graph_dir: Optional[Path] = None) -> dict:
    """Load all chunks (03b) + the graph (04) into retrieval-ready form.

    Returns {"chunks": List[Chunk], "by_doc": {doc_id: [idx..]},
    "doc_of": {lineage_id: doc_id}, "graph": GraphLike, "docs": [doc_id]}.
    """
    chunks_dir = chunks_dir or c.NOTEBOOKS_DATA / "ast"
    graph_dir = graph_dir or c.NOTEBOOKS_DATA / "graph"
    chunks: List[Chunk] = []
    by_doc: dict = {}
    for f in sorted(chunks_dir.glob("chunks_*.json")):
        doc_id = f.name.replace("chunks_", "", 1).replace(".json", "")
        data = json.loads(f.read_text())
        idx0 = len(chunks)
        for rec in data:
            chunks.append(Chunk(
                doc_id=doc_id,
                lineage_id=rec["lineage_id"],
                node_id=rec["node_id"],
                node_type=rec["node_type"],
                text=rec["text"],
                span=tuple(rec["span"]),
                context=rec.get("context", {}),
            ))
        by_doc[doc_id] = list(range(idx0, len(chunks)))

    nodes = {}
    if (graph_dir / "nodes.jsonl").exists():
        for line in (graph_dir / "nodes.jsonl").read_text().splitlines():
            if line.strip():
                n = json.loads(line)
                nodes[n["lineage_id"]] = n
    adj: dict = {}
    adj_in: dict = {}
    edges: List[dict] = []   # raw edge records (v1+v2) with evidence -- additive
    if (graph_dir / "edges.jsonl").exists():
        for line in (graph_dir / "edges.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            edges.append(e)
            adj.setdefault(e["src"], []).append((e["dst"], e["kind"]))
            adj_in.setdefault(e["dst"], []).append((e["src"], e["kind"]))
    graph = GraphLike(nodes=nodes, adj=adj, adj_in=adj_in)
    graph.edges = edges   # extra attr (additive); default consumers ignore it
    # article spans per doc (AST) -- used to map sentence chunks to the
    # containing article graph node
    article_index: dict = {}
    for f in sorted(chunks_dir.glob("*_ast.json")):
        article_index.update(_article_spans_of(f))
    return {"chunks": chunks, "by_doc": by_doc,
            "doc_of": {i: cix.doc_id for i, cix in enumerate(chunks)},
            "graph": graph, "docs": sorted(by_doc),
            "article_index": article_index}


def _article_spans_of(ast_file: "Path") -> dict:
    """doc_id -> sorted list of {"number": int, "start": int, "end": int}"""
    out: dict = {}
    try:
        root = json.loads(ast_file.read_text())
    except Exception:
        return out
    arts: list = []

    def walk(n: dict):
        if n.get("type") == "article":
            nid = n.get("identifier") or ""
            num = nid.rsplit("-", 1)[-1] if "-" in nid else nid
            try:
                num_i = int(num)
            except ValueError:
                return
            span = n.get("span") or [0, 0]
            arts.append({"number": num_i, "start": span[0], "end": span[1]})
        for c in n.get("children", []):
            walk(c)

    walk(root)
    arts.sort(key=lambda a: a["start"])
    if arts:
        out[    root.get("identifier") or ast_file.stem.replace("_ast", "")] = arts
    return out


def article_node_for(chunk: "Chunk", article_index: dict,
                     graph: "GraphLike") -> Optional[str]:
    """Resolve the graph node id that `chunk` belongs to.

    Article/preamble chunks that are themselves graph nodes map to
    themselves; sentence chunks map to the enclosing article node
    (by AST span containment). Documents map to their article when a
    sentence has no article (e.g. preamble sentences in guidance docs).
    """
    if chunk.lineage_id in graph.nodes:
        return chunk.lineage_id
    doc = chunk.doc_id
    arts = article_index.get(doc) or []
    if not arts:
        return None
    # find containing article by span: start <= chunk.start < end
    start = chunk.span[0]
    best = None
    for a in arts:
        if a["start"] <= start < a["end"]:
            best = a
            break
    if best is None:
        # nearest by distance (for preamble / front-matter sentences)
        best = min(arts, key=lambda a: min(abs(a["start"] - start),
                                           abs(a["end"] - start)))
    lid = f"{doc}:article:{best['number']}"
    return lid if lid in graph.nodes else None


# --- query-side tokenizer -----------------------------------------------
# Design goal: a reference like "Article 5(2)(a)" tokenizes *identically*
# in document text and in a user query, so exact-ref queries stay
# discriminative. Strategy:
#   1. protect instrument numbers (CELEX `2016/679`) and acronyms,
#   2. split the residue on non-alphanumerics -- so `Article 5(2)(a)` ->
#      ["article", "5", "2", "a"], the same in doc and query,
#   3. drop pure stop-words and <2-char noise.
# (Word-boundary ref capture was tried first but "Articles 8, 10 and 11"
# yields "article8", "10" and "11" from `Article N` in docs while a query
# yields no such tokens -- asymmetric, so we avoid it.)

_INSTRUMENT = re.compile(r"\b\d{4}/\d{2,5}(?:/\w{2,3})?\b")
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,}\b")


def tokenize(text: str) -> List[str]:
    # protected spans (instrument numbers + acronyms), non-overlapping
    mats = [m for m in list(_INSTRUMENT.finditer(text))
            + list(_ACRONYM.finditer(text)) if m.end() > m.start()]
    mats.sort(key=lambda m: m.start())
    kept: list = []
    last_end = 0
    for m in mats:
        if m.start() < last_end:
            continue
        kept.append((m.start(), m.end(), m.group(0)))
        last_end = m.end()

    def words(seg: str) -> List[str]:
        # split residue on non-alphanumerics: `5(2)(a)` -> 5, 2, a
        out: List[str] = []
        for w in re.findall(r"[A-Za-z]+|\d+", seg):
            lw = w.lower()
            if (w.isalpha() and len(w) < 2) or lw in _STOP:
                continue
            # digits always kept: they carry ref numbers (5(2) -> 5, 2)
            out.append(lw)
        return out

    out: List[str] = []
    pos = 0
    for s, e, tok in kept:
        out.extend(words(text[pos:s]))
        # acronyms case-sensitive (REMIT vs remit); instruments lower
        out.append(tok.upper() if re.fullmatch(r"[A-Z][A-Z0-9]+", tok)
                   else tok.lower())
        pos = e
    out.extend(words(text[pos:]))
    return out


_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "but",
    "is", "are", "be", "was", "were", "been", "shall", "must", "may", "can",
    "as", "by", "with", "without", "under", "where", "which", "that", "this",
    "these", "those", "its", "their", "any", "all", "each", "such", "not",
    "no", "nor", "should", "might", "would", "could", "will",
    "at", "from", "into", "between", "among", "upon", "per", "via",
    "does", "do", "have", "has", "had", "an", "the",
    "member", "states", "union", "european", "regulation", "directive",
    "decision", "commission", "applies", "shall",
}
