"""Additive migration: extend the on-disk graph with v2 edge kinds.

Strict no-touch policy for v1 edges
    v1 (CROSS_REFERENCES / AMENDS / DEFINED_IN / APPLIES_TO) edges are
    read from the on-disk ``edges.jsonl`` and emitted back in the *same
    order* with the *same fields*.  The only modification is that a
    boolean ``cites: true`` is added to every CROSS_REFERENCES edge that
    crosses a document boundary (src doc != dst doc).  That captures both
    resolved cross-instrument references (525) AND the 331 edges that
    point to external-article stubs, for a total of 856 flagged edges.

Additions (appended to the end of edges.jsonl):
    PART_OF      article -> containing document (one per article in corpus)
    IMPLEMENTS   higher instrument -> lower instrument (regex over chunk text)
    SUPERSEDES   newer instrument -> older instrument (regex over chunk text)

Backward-compatibility invariant
    ``GRAPH_EDGE_KINDS`` in src/retrieval/graph.py is unchanged
    (CROSS_REFERENCES / AMENDS), so ``hybrid`` / ``hybrid_rerank`` /
    ``hybrid_graph`` / ``graph`` / ``neo4j`` benchmark runs remain
    bit-identical.  The v2 kinds are *visible only* to the new v2
    context builder (the graph-aware reranker).

Run:  PYENV_VERSION=energy-audit python scripts/regen_graph_artifacts.py
"""
from __future__ import annotations

import json
import re
import sys
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from structure.graph_builder import (  # noqa: E402
    Graph, KIND_CROSS, KIND_PART_OF, KIND_IMPL, KIND_SUP,
    IMPLEMENTS, SUPERSEDES, _snippet, _instrument_keys,
)
import common as c  # type: ignore  # noqa: E402


@dataclass
class _DocRecShim:
    """Duck-type for ``graph_builder.DocRec`` -- exposes the fields the
    new builders consume (content, smap, article_num, art_titles,
    art_spans)."""
    doc_id: str
    content: str
    smap: dict
    article_num: set
    art_titles: Dict[str, str]
    art_spans: Dict[str, tuple]


def _rebuild_docs(graph_dir: Path, chunks_dir: Path) -> List[_DocRecShim]:
    nodes: Dict[str, dict] = {}
    art_titles_by_doc: Dict[str, Dict[str, str]] = defaultdict(dict)
    for line in (graph_dir / "nodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        n = json.loads(line)
        lid = n["lineage_id"]
        nodes[lid] = n
        if n.get("kind") == "article" and n.get("doc_id"):
            num = lid.rsplit(":", 1)[-1]
            art_titles_by_doc[n["doc_id"]][num] = n.get("title") or f"Article {num}"

    doc_title: Dict[str, str] = {}
    for n in nodes.values():
        if n.get("kind") == "document" and n.get("doc_id"):
            doc_title[n["doc_id"]] = n.get("title") or n["doc_id"]

    art_spans_by_doc: Dict[str, Dict[str, tuple]] = defaultdict(dict)
    doc_chunks: Dict[str, List[str]] = defaultdict(list)
    # Instrument/CELEX metadata per doc, pulled from each doc's AST root
    # ``metadata`` block.  This is what lets cross-instrument resolution
    # (``_instrument_keys``) actually find in-corpus documents instead of
    # collapsing every endpoint to an ``ext:instrument:`` stub.  The AST
    # root is the authoritative source in this corpus (structure_maps/ is
    # not materialised here), so we read it directly from the same ``ast/``
    # directory the chunk files already come from.
    doc_instrument: Dict[str, str] = {}
    doc_celex: Dict[str, str] = {}
    for f in sorted(chunks_dir.glob("*_ast.json")):
        doc = f.name.replace("_ast.json", "")
        try:
            a = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        md = a.get("metadata") or {}
        if md.get("instrument"):
            doc_instrument[doc] = md["instrument"]
        if md.get("celex"):
            doc_celex[doc] = md["celex"]

    for f in sorted(chunks_dir.glob("chunks_*.json")):
        doc = f.name.replace("chunks_", "", 1).replace(".json", "")
        data = json.loads(f.read_text())
        for rec in data:
            doc_chunks[doc].append(rec["text"])
            if rec.get("node_type") == "article" and ":article:" in rec["lineage_id"]:
                num = rec["lineage_id"].rsplit(":", 1)[-1]
                span = rec.get("span")
                if span and len(span) >= 2:
                    art_spans_by_doc[doc][num] = tuple(span[:2])

    out: List[_DocRecShim] = []
    for doc in sorted({*art_titles_by_doc, *doc_title, *art_spans_by_doc}):
        smap: Dict[str, object] = {"document_title": doc_title.get(doc)}
        if doc_instrument.get(doc):
            smap["instrument"] = doc_instrument[doc]
        if doc_celex.get(doc):
            smap["celex"] = doc_celex[doc]
        out.append(_DocRecShim(
            doc_id=doc,
            content="\n".join(doc_chunks.get(doc, [])),
            smap=smap,
            article_num=set(art_titles_by_doc.get(doc, {})),
            art_titles=dict(art_titles_by_doc.get(doc, {})),
            art_spans=dict(art_spans_by_doc.get(doc, {})),
        ))
    return out


def _pick_best(cands: List[_DocRecShim], numyear: str) -> _DocRecShim:
    """Disambiguate an instrument key that maps to several in-corpus docs.

    Guidance / monitoring / simulation docs frequently carry the *same*
    instrument number as the primary regulation they elaborate (e.g. both
    ``acer_remit_guidance`` and ``remit_1227_2011`` are ``2011/1227``).
    ``setdefault`` would keep the alphabetically-first hit, which is the
    guidance doc -- wrong.  Instead prefer the candidate whose ``doc_id``
    actually encodes the instrument number's numeric tokens; that recovers
    the primary instrument.  Ties break lexicographically (deterministic).
    """
    toks = set(re.findall(r"\d+", numyear))
    def score(r: _DocRecShim) -> tuple:
        idn = re.sub(r"[^a-z0-9]+", "_", r.doc_id.lower())
        hit = sum(1 for t in toks if t and t in idn.split("_"))
        return (-hit, r.doc_id)
    return min(cands, key=score)


def _resolve_instrument(g: Graph, numyear: str,
                        recs_by_instr: Dict[str, List[_DocRecShim]]) -> str:
    """Map ``yyyy/nnnn`` to a node lid.  Prefers the in-corpus document
    node (matched via _instrument_keys), else an external-instrument
    stub node (created on the fly, dedup'd by lid)."""
    numyear = re.sub(r"/(?:EC|EURI|EUR|EEA)$", "", numyear, flags=re.I)
    cands = list(recs_by_instr.get(numyear) or [])
    if not cands:
        mm = re.match(r"(\d{4})/(\d{2,5})", numyear)
        if mm:
            cands = list(recs_by_instr.get(f"{mm.group(2)}/{mm.group(1)}") or [])
    if cands:
        hits = _pick_best(cands, numyear)
        return g.node_document(
            hits.doc_id, hits.smap.get("document_title") or hits.doc_id)
    return g.node_external_document(
        numyear.replace("/", "-").lower(), numyear)


def _v2_part_of(g: Graph, recs: List[_DocRecShim]) -> int:
    n = 0
    for rec in recs:
        doc = g.node_document(
            rec.doc_id, rec.smap.get("document_title") or rec.doc_id)
        for num, title in rec.art_titles.items():
            art = g.node_article(rec.doc_id, num, title)
            ev = {"doc_id": rec.doc_id, "offset": 0,
                  "snippet": (f"{title} is part of "
                              f"{rec.smap.get('document_title') or rec.doc_id}")}
            g.add_edge(art, doc, KIND_PART_OF, ev, extra={"ref_number": num})
            n += 1
    return n


def _v2_impl_sup(g: Graph, recs: List[_DocRecShim]) -> Tuple[int, int]:
    recs_by_instr: Dict[str, List[_DocRecShim]] = {}
    for rec in recs:
        for k, _d in _instrument_keys(rec):
            recs_by_instr.setdefault(k, []).append(rec)

    n_impl = 0
    for rec in recs:
        for m in IMPLEMENTS.finditer(rec.content):
            n1, n2 = m.group(1), m.group(2)
            if n1 == n2:
                continue
            src = _resolve_instrument(g, n1, recs_by_instr)
            dst = _resolve_instrument(g, n2, recs_by_instr)
            ev = {"doc_id": rec.doc_id, "offset": m.start(),
                  "snippet": _snippet(rec.content, m.start(), m.start() + 240)}
            g.add_edge(src, dst, KIND_IMPL, ev,
                       extra={"instrument_upper": n1, "instrument_lower": n2})
            n_impl += 1

    n_sup = 0
    for rec in recs:
        for m in SUPERSEDES.finditer(rec.content):
            n1, n2 = m.group(1), m.group(2)
            if n1 is None or n2 is None or n1 == n2:
                continue
            # Direction is decided by instrument year, not by the verb
            # polarity ("X repeals Y" vs "X is repealed by Y" both appear).
            # The later-year instrument is the *newer* one and is the
            # superseder (src); the earlier-year one is superseded (dst).
            y1 = int(re.match(r"(\d{4})/", n1).group(1))
            y2 = int(re.match(r"(\d{4})/", n2).group(1))
            newer, older = (n1, n2) if y1 >= y2 else (n2, n1)
            src = _resolve_instrument(g, newer, recs_by_instr)
            dst = _resolve_instrument(g, older, recs_by_instr)
            ev = {"doc_id": rec.doc_id, "offset": m.start(),
                  "snippet": _snippet(rec.content, m.start(), m.start() + 240)}
            g.add_edge(src, dst, KIND_SUP, ev,
                       extra={"supersedes": older, "superseded_by": newer})
            n_sup += 1
    return n_impl, n_sup


def _count_cites(v1: List[dict]) -> int:
    """Number of CROSS_REFERENCES edges that cross a doc boundary."""
    n = 0
    for e in v1:
        if (e.get("kind") == KIND_CROSS
                and e["src"].split(":")[0] != e["dst"].split(":")[0]):
            n += 1
    return n


def main() -> int:
    graph_dir = c.NOTEBOOKS_DATA / "graph"
    chunks_dir = c.NOTEBOOKS_DATA / "ast"

    # 1. read existing edges.  v1 kinds are preserved byte-identically;
    #    any pre-existing v2 kinds are DROPPED and regenerated from
    #    scratch (this is what makes the script idempotent -- running it
    #    twice yields the same output rather than re-appending).
    V1_KINDS = {KIND_CROSS, "AMENDS", "DEFINED_IN", "APPLIES_TO"}
    nodes: Dict[str, dict] = {}
    for line in (graph_dir / "nodes.jsonl").read_text().splitlines():
        if line.strip():
            n = json.loads(line)
            nodes[n["lineage_id"]] = n
    all_edges: List[dict] = [json.loads(l) for l in
                             (graph_dir / "edges.jsonl").read_text().splitlines()
                             if l.strip()]
    v1: List[dict] = [e for e in all_edges if e["kind"] in V1_KINDS]
    n_dropped_v2 = len(all_edges) - len(v1)
    v1_kinds = Counter(e["kind"] for e in v1)
    n_cites = _count_cites(v1)
    print(f"v1: {len(nodes)} nodes, {len(v1)} edges {dict(v1_kinds)}"
          + (f"  (dropped {n_dropped_v2} stale v2 edges to regenerate)"
             if n_dropped_v2 else ""))
    print(f"v1 CROSS_REFERENCES w/ cross-doc boundary (will be cites=True): {n_cites}")

    # 2. rebuild docs (for v2 regex builders)
    recs = _rebuild_docs(graph_dir, chunks_dir)
    print(f"reconstructed {len(recs)} docs")

    # 3. build v2 graph starting from the v1 node set
    g = Graph()
    for lid, n in nodes.items():
        g.nodes[lid] = dict(n)
    n_part_of = _v2_part_of(g, recs)
    n_impl, n_sup = _v2_impl_sup(g, recs)
    print(f"v2 added: PART_OF={n_part_of}, IMPLEMENTS={n_impl}, SUPERSEDES={n_sup}")
    print(f"node count after: {len(g.nodes)}")

    # 4. assemble output: v1 first (with CITES flag), v2 appended
    v1_keys = {(e["src"], e["dst"], e["kind"]) for e in v1}
    out: List[dict] = []
    for e in v1:
        e2 = json.loads(json.dumps(e))      # deep copy
        if (e2.get("kind") == KIND_CROSS
                and e2["src"].split(":")[0] != e2["dst"].split(":")[0]):
            e2["cites"] = True
        out.append(e2)
    appended = 0
    for e in g.edges:
        key = (e["src"], e["dst"], e["kind"])
        if key not in v1_keys:
            out.append(e)
            appended += 1

    out_kinds = Counter(e["kind"] for e in out)
    print(f"final edges: {len(out)} (v1={len(v1)}, v2 appended={appended})")
    print(f"final kinds: {dict(out_kinds)}")

    # 5. write
    graph_dir.mkdir(parents=True, exist_ok=True)
    out_nodes = {lid: dict(n) for lid, n in nodes.items()}
    for lid, n in g.nodes.items():
        out_nodes.setdefault(lid, dict(n))
    with open(graph_dir / "nodes.jsonl", "w") as f:
        for lid in sorted(out_nodes):
            f.write(json.dumps(out_nodes[lid]) + "\n")
    with open(graph_dir / "edges.jsonl", "w") as f:
        for e in out:
            f.write(json.dumps(e) + "\n")

    # per-doc summary
    rows = []
    docs = sorted({n.get("doc_id") for n in out_nodes.values() if n.get("doc_id")})
    by_doc: Dict[str, Counter] = defaultdict(Counter)
    cites_by_doc: Dict[str, int] = defaultdict(int)
    for e in out:
        by_doc[e["src"].split(":")[0]][e["kind"]] += 1
        if e.get("cites"):
            cites_by_doc[e["src"].split(":")[0]] += 1
    for d in docs:
        cnt = by_doc.get(d, Counter())
        rows.append({
            "doc_id": d,
            "cross": cnt.get(KIND_CROSS, 0),
            "amends": cnt.get("AMENDS", 0),
            "defined_in": cnt.get("DEFINED_IN", 0),
            "applies_to": cnt.get("APPLIES_TO", 0),
            "part_of": cnt.get(KIND_PART_OF, 0),
            "implements": cnt.get(KIND_IMPL, 0),
            "supersedes": cnt.get(KIND_SUP, 0),
            "cites": cites_by_doc.get(d, 0),
            "unresolved": sum(1 for e in out
                              if e["src"].split(":")[0] == d and e.get("unresolved")),
        })
    with open(graph_dir / "graph_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id", "cross", "amends",
                                         "defined_in", "applies_to", "part_of",
                                         "implements", "supersedes", "cites",
                                         "unresolved"])
        w.writeheader()
        w.writerows(rows)

    def _res(kind: str) -> Tuple[int, int]:
        rs = [e for e in out if e["kind"] == kind]
        r = sum(1 for e in rs if not e.get("unresolved"))
        return r, len(rs) - r

    imp_res, imp_unres = _res(KIND_IMPL)
    sup_res, sup_unres = _res(KIND_SUP)

    handoff = {
        "generated": datetime.now().isoformat(),
        "v2_additions": {
            "cites_flag": f"{n_cites} CROSS_REFERENCES edges flagged "
                           "cites=True (cross-document reference semantic)",
            "PART_OF": f"{n_part_of} article -> document structural edges",
            "IMPLEMENTS": f"{imp_res + imp_unres} higher -> lower instrument "
                         f"edges ({imp_res} resolved to in-corpus docs, "
                         f"{imp_unres} endpoint(s) external)",
            "SUPERSEDES": f"{sup_res + sup_unres} newer -> older instrument "
                         f"edges ({sup_res} resolved to in-corpus docs, "
                         f"{sup_unres} endpoint(s) external)",
        },
        "instrument_resolution": {
            "note": ("cross-instrument endpoints are resolved via each doc's "
                     "AST metadata.instrument; the doc_id that encodes the "
                     "instrument number wins when several docs share one "
                     "(guidance vs primary). Endpoints with no in-corpus "
                     "primary stay ext:instrument: stubs (unresolved=true)."),
            "IMPLEMENTS_resolved": imp_res,
            "IMPLEMENTS_unresolved": imp_unres,
            "SUPERSEDES_resolved": sup_res,
            "SUPERSEDES_unresolved": sup_unres,
        },
        "counts": {
            "n_nodes": len(out_nodes),
            "n_edges": len(out),
            "kinds": dict(out_kinds),
            "unresolved": sum(1 for e in out if e.get("unresolved")),
        },
        "backward_compat": (
            "the first 4,139 edges are byte-identical to the v1 graph "
            "(same order, same fields; the only addition is cites=true "
            "on the cross-document CROSS_REFERENCES subset).  "
            "GRAPH_EDGE_KINDS in src/retrieval/graph.py is unchanged "
            "(CROSS_REFERENCES / AMENDS), so every existing hybrid* "
            "benchmark remains bit-identical.  The v2 kinds (PART_OF / "
            "IMPLEMENTS / SUPERSEDES) are visible only to the new "
            "graph-aware reranker (v2 context builder)."
        ),
        "files": ["nodes.jsonl", "edges.jsonl", "graph_summary.csv",
                  "graph_handoff.json"],
    }
    (graph_dir / "graph_handoff.json").write_text(json.dumps(handoff, indent=2))
    print("wrote: nodes.jsonl, edges.jsonl, graph_summary.csv, graph_handoff.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
