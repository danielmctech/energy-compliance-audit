"""
graph_builder -- the 04 graph (GraphRAG substrate, Step 1).

Consumes the 03a structure maps (structural truth + per-article mention
counts) and the 03b AST lineage schema (node IDs), and builds a
deterministic, serialisable knowledge graph over the corpus:

    nodes  -- documents, articles, preambles, defined terms, scope
              entities, and *external* article stubs (unresolved targets)
    edges  --
      CROSS_REFERENCES   src article --("Article N")--> tgt article
      AMENDS             src article --("amended by <instrument>")--> tgt doc
      DEFINED_IN         src term    --(<term> "means" ...)--> article/preamble
      APPLIES_TO         src article --("applies to X")--> entity (heuristic)

Design choices:
  * node IDs are the 03b **lineage IDs** (`{doc}:article:{n}`, ...) so a
    retrieved chunk can be joined straight onto the graph;
  * every edge carries `src`, `dst`, `kind`, and `evidence`
    (`{doc_id, offset, snippet<=120}`) -- provenance is non-negotiable;
  * cross-instrument resolution uses each doc's CELEX / instrument number
    inside the mention window; what cannot be resolved gets an `ext:`
    stub node and `unresolved: true` on the edge (kept, not dropped);
  * no LLM, deterministic, idempotent; ranges & lists ("Articles 1 to 5")
    expand into one edge per target (capped).

Outputs (under notebooks/data/graph/): nodes.jsonl, edges.jsonl,
graph_summary.csv, graph_handoff.json.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import common as c
    from structure import structure_maps as SM
    from parsing import ground_truth as GT
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    import common as c
    from structure import structure_maps as SM
    from parsing import ground_truth as GT


# ---------------------------------------------------------------------------
# vocab / constants
# ---------------------------------------------------------------------------

KIND_CROSS = "CROSS_REFERENCES"
KIND_AMENDS = "AMENDS"
KIND_DEFINED = "DEFINED_IN"
KIND_APPLIES = "APPLIES_TO"
KIND_PART_OF = "PART_OF"        # article -> containing document (structural)
KIND_IMPL = "IMPLEMENTS"        # higher instrument -> lower instrument (delegated acts)
KIND_SUP = "SUPERSEDES"         # successor instrument -> predecessor instrument

MAX_LIST_TARGETS = 25          # per mention group, cap range/list expansion
SNIPPET_CHARS = 120
ENTITY_SLUG = re.compile(r"[^a-z0-9]+")

ARTICLE_REF = re.compile(r"\bArticles?\s+(\d+[a-z]?)\b")
# what follows an `Article N` mention: range ("to 5") or list (", 3", "and 4")
FOLLOW = re.compile(r"^\s*(?:to|-|to,|--)\s*(\d+[a-z]?)\b"
                    r"|^\s*,\s*(\d+[a-z]?)\b"
                    r"|^\s*and\s+(?:the\s+)?(\d+[a-z]?)\b")
AMENDS = re.compile(
    r"\b(?:amend(?:s|ed|ing|ment)s?|amended by|amends)\b"
    r".{0,200}?"
    r"(?:Regulation|Directive|Decision|Commission\s+Implementation|Commission\s+Implementing)"
    r"(?:\s*\(EU?\))?\s*(?:No\.?\s*)?(\d{4}/\d{3,5}(?:/\w{0,3})?)",
    re.I)
# Higher instrument ("shall implement" / "implemented through" / "Commission Implementing Regulation")
# with a target instrument number nearby -- the target is *lower* in the
# hierarchy.  The `.{0,80}?` keeps the match local so a distant "Regulation"
# in the same clause is not grabbed.
_INSTR = (r"(?:Regulation|Directive|Decision|Commission)"
          r"(?:\s*\((?:EU|EC|EURI|EUR|EEA)\))?")
_IMPLY_VERB = (r"\b(?:implements?|implemented (?:through|by)\b"
               r"|is implemented (?:by|in)\b"
               r"|Commission\s+Implement(?:ing|ation)"
               r"\s+(?:Regulation|Decision))")
IMPLEMENTS = re.compile(
    _INSTR + r"\s*(?:No\.?\s*)?(\d{4}/\d{2,5})\b"
    r".{0,80}?"
    + _IMPLY_VERB
    + r".{0,180}?"
    + _INSTR + r"\s*(?:No\.?\s*)?(\d{4}/\d{3,5}(?:/\w{0,3})?)",
    re.I)
# "X repeals Y" / "X replaces Y" / "X has been replaced by Y (of date D)" --
# X (first instrument) is the *newer* one, Y (second) is the older one.
# Both instruments appear in the same clause, in order.
_SUPER_VERB = r"\b(?:repeals?|replaces?|has (?:replaced|superseded)\b)"
SUPERSEDES = re.compile(
    _INSTR + r"\s*(?:No\.?\s*)?(\d{4}/\d{2,5})\b"
    r".{0,160}?"
    + _SUPER_VERB
    + r".{0,160}?"
    + _INSTR + r"\s*(?:No\.?\s*)?(\d{4}/\d{2,5})",
    re.I)
# anchor words that open a scope phrase; the entity label is captured on the
# consumer side (bounded window, no unbounded-quantifier nesting)
APPLY_ANCHOR = re.compile(
    r"\b(?:applies? to|apply to|is addressed to|shall apply (?:to|in)|"
    r"applied to|is applicable to|addressed to the|addressed to)\b", re.I)


def _slug(s: str) -> str:
    return ENTITY_SLUG.sub("_", s.lower().strip()).strip("_")


def _snippet(content: str, start: int, end: int) -> str:
    seg = content[start:end]
    seg = re.sub(r"\s+", " ", seg).strip()
    return seg[:SNIPPET_CHARS]


# ---------------------------------------------------------------------------
# corpus loading
# ---------------------------------------------------------------------------

@dataclass
class DocRec:
    doc_id: str
    md_path: Path
    content: str
    smap: dict
    article_num: set
    art_titles: Dict[str, str]
    art_spans: Dict[str, tuple]
    table_regions: List[tuple] = None  # char spans of tables (mentions in tables are data, not text)


def _table_char_regions(content: str) -> List[tuple]:
    out = []
    lines = content.splitlines()
    off = [0]
    for l in lines:
        off.append(off[-1] + len(l) + 1)
    for blk in SM.find_tables(content):
        if blk["start_line"] < 1 or blk["end_line"] > len(lines):
            continue
        out.append((off[blk["start_line"] - 1], off[blk["end_line"]]))
    return out


def load_corpus(maps_dir: Optional[Path] = None) -> List[DocRec]:
    maps_dir = maps_dir or c.NOTEBOOKS_DATA / "structure_maps"
    recs: List[DocRec] = []
    for p in sorted(maps_dir.glob("*_structure.json")):
        smap = json.loads(p.read_text())
        doc_id = p.name[:-len("_structure.json")]
        md_path = c.resolve_repo_path(smap["source_md"])
        content = md_path.read_text(encoding="utf-8")
        found = SM.find_structural_articles(content)
        spans: Dict[str, tuple] = {}
        anchors = [a.line_start for a in found] + [len(content)]
        for k, a in enumerate(found):
            spans.setdefault(a.number, (a.line_start, anchors[k + 1]))
        titles = {a["number"]: a.get("title") or "" for a in smap["articles"]}
        recs.append(DocRec(
            doc_id=doc_id, md_path=md_path, content=content, smap=smap,
            article_num=set(titles), art_titles=titles, art_spans=spans,
            table_regions=_table_char_regions(content),
        ))
    return recs


def in_table(rec: DocRec, off: int) -> bool:
    return any(s <= off < e for s, e in (rec.table_regions or []))


def _instrument_keys(rec: DocRec) -> List[tuple]:
    """(match_key, doc_id) pairs for cross-instrument resolution.
    Prefer CELEX number, then bare yyyy/nnnn."""
    keys: List[tuple] = []
    cx = rec.smap.get("celex") or ""
    m = re.search(r"(\d{4}/\d{3,5})", cx)
    if m:
        keys.append((m.group(1), rec.doc_id))
    inst = rec.smap.get("instrument") or ""
    m2 = re.search(r"(\d{4})/(\d{3,5})", inst)
    if m2:
        keys.append((f"{m2.group(1)}/{m2.group(2)}", rec.doc_id))
    # also the CELEX-without-leading-digit (2011R1227 -> 2011/1227 not needed;
    # but "1227/2011" style instrument text -> number/year order)
    if m2:
        keys.append((f"{m2.group(2)}/{m2.group(1)}", rec.doc_id))
    # de-dup, keep order
    seen = set()
    out = []
    for k, d in keys:
        if k not in seen:
            seen.add(k)
            out.append((k, d))
    return out


# ---------------------------------------------------------------------------
# graph building
# ---------------------------------------------------------------------------

class Graph:
    def __init__(self) -> None:
        self.nodes: Dict[str, dict] = {}
        self.edges: List[dict] = []
        self._edge_set = set()

    # node helpers
    def node_document(self, doc_id: str, title: str) -> str:
        lid = f"{doc_id}:document"
        if lid not in self.nodes:
            self.nodes[lid] = {"lineage_id": lid, "kind": "document",
                               "doc_id": doc_id, "title": title or doc_id}
        return lid

    def node_external_document(self, key: str, title: str) -> str:
        lid = f"ext:instrument:{key}"
        if lid not in self.nodes:
            self.nodes[lid] = {"lineage_id": lid, "kind": "external_document",
                               "doc_id": None, "title": title or key}
        return lid

    def node_preamble(self, doc_id: str) -> str:
        lid = f"{doc_id}:preamble"
        if lid not in self.nodes:
            self.nodes[lid] = {"lineage_id": lid, "kind": "preamble",
                               "doc_id": doc_id, "title": "Preamble"}
        return lid

    def node_article(self, doc_id: str, num: str, title: str) -> str:
        lid = f"{doc_id}:article:{num}"
        if lid not in self.nodes:
            self.nodes[lid] = {"lineage_id": lid, "kind": "article",
                               "doc_id": doc_id, "number": num,
                               "title": title or f"Article {num}"}
        return lid

    def node_external_article(self, num: str) -> str:
        lid = f"ext:article:{num}"
        if lid not in self.nodes:
            self.nodes[lid] = {"lineage_id": lid, "kind": "external_article",
                               "doc_id": None, "number": num,
                               "title": f"Article {num} (external)"}
        return lid

    def node_term(self, doc_id: str, term: str) -> str:
        lid = f"{doc_id}:term:{_slug(term)}"
        if lid not in self.nodes:
            self.nodes[lid] = {"lineage_id": lid, "kind": "term",
                               "doc_id": doc_id, "term": term}
        return lid

    def node_entity(self, slug: str, label: str) -> str:
        lid = f"corpus:entity:{slug}"
        if lid not in self.nodes:
            self.nodes[lid] = {"lineage_id": lid, "kind": "entity",
                               "label": label}
        return lid

    # edge helper
    def add_edge(self, src: str, dst: str, kind: str,
                 evidence: dict, extra: Optional[dict] = None) -> None:
        key = (src, dst, kind)
        if key in self._edge_set:
            return
        if src not in self.nodes or dst not in self.nodes:
            raise ValueError(f"edge endpoint missing: {src} -> {dst}")
        self._edge_set.add(key)
        e = {"src": src, "dst": dst, "kind": kind, **evidence}
        if extra:
            e.update(extra)
        e["unresolved"] = dst.startswith("ext:")
        self.edges.append(e)

    def summary(self) -> dict:
        kinds = Counter(e["kind"] for e in self.edges)
        by_src = Counter()
        for e in self.edges:
            doc = e["src"].split(":")[0]
            by_src[doc] += 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "kinds": dict(kinds),
            "unresolved": sum(1 for e in self.edges if e["unresolved"]),
            "edges_per_src_doc": dict(by_src),
        }


# ---------------------------------------------------------------------------
# per-kind builders
# ---------------------------------------------------------------------------

def _expand_list(content: str, first_end: int,
                 max_targets: int = MAX_LIST_TARGETS) -> List[str]:
    """'Articles 1 to 5' / 'Articles 2, 3 and 4' -> [numbers]."""
    rest = content[first_end:first_end + 60]
    out: List[str] = []
    pos = 0
    while pos < len(rest) and len(out) < max_targets:
        m = FOLLOW.match(rest, pos)
        if not m:
            break
        num = next((g for g in m.groups() if g), None)
        if num is None:
            break
        if num not in out:
            out.append(num)
        pos = m.end()
        # a following comma/and continues the list; anything else stops
        if not re.match(r"\s*(,|and|to|-)", rest[pos:pos + 6]):
            break
    return out


def _instrument_after(content: str, pos: int) -> Optional[str]:
    """After an `Article N [...]` mention: find the nearest instrument
    connector (`of / under / pursuant to ...`) and take the `yyyy/nnnn`
    that follows within a short window (instrument names are short; this
    rejects unrelated clauses such as 'of a conflict between ...').
    Leading list members (` and 20`) and point modifiers (` (1)`) are
    tolerated before the connector."""
    win = content[pos:pos + 120]
    for m in re.finditer(
            r"\b(?:of|under|pursuant\s+to|according\s+to|provided\s+in|"
            r"laid\s+down\s+in|referred\s+to)\s", win, re.I):
        seg = win[m.end():m.end() + 30]
        im = re.match(r"^(?:\s|[A-Za-z()\u2019'-]){0,24}(?:No\.?\s)*"
                      r"(\d{4}/\d{2,5})", seg)
        if im:
            num = im.group(1)
            num = re.sub(r"/(?:EC|EURI|EUR|EEA)$", "", num, flags=re.I)
            return num
    return None


def _instrument_index(recs: List[DocRec]) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for rec in recs:
        for key, _doc in _instrument_keys(rec):
            idx.setdefault(key, rec.doc_id)
            m = re.match(r"(\d{4})/(\d{2,5})", key)
            if m:
                idx.setdefault(f"{m.group(2)}/{m.group(1)}", rec.doc_id)
    return idx


def build_cross_references(g: Graph, recs: List[DocRec],
                           instr_index: Dict[str, str]) -> int:
    """For every article in every doc, find `Article N` mentions in the
    prose (table regions excluded), expand ranges/lists, and resolve:
    - own-instrument article numbers  -> in-doc article node
    - other in-corpus instrument + number (instrument context after the
      mention)                      -> that doc's article node
    - everything else               -> ext:article stub (unresolved)
    The first mention's instrument context binds all list members."""
    doc_articles: Dict[str, set] = {r.doc_id: r.article_num for r in recs}
    doc_titles: Dict[str, Dict[str, str]] = {r.doc_id: r.art_titles for r in recs}
    n_edges = 0
    for rec in recs:
        local = rec.article_num
        for num in rec.art_titles:
            span = rec.art_spans.get(num)
            if not span:
                continue
            # end of the article's own heading line (absolute offset)
            line_end = rec.content.find("\n", span[0])
            if line_end == -1:
                line_end = span[1]
            body = rec.content[span[0]:span[1]]
            for m in ARTICLE_REF.finditer(body):
                ref = m.group(1)
                off = span[0] + m.start()
                # skip the heading self-match (`## _Article 5_` naming art 5)
                if ref == num and off <= line_end:
                    continue
                # skip mentions inside tables (grid rows are data, not prose)
                if in_table(rec, off):
                    continue
                ev = {"doc_id": rec.doc_id, "offset": off,
                      "snippet": _snippet(rec.content, off, off + 240)}
                # list/range continuation: "Articles 15 and 20"
                phrase_end = off + (m.end() - m.start())
                group = [ref]
                for extra in _expand_list(rec.content, phrase_end):
                    if extra not in group:
                        group.append(extra)
                # instrument context ("of Regulation (EU) 2016/679") binds the
                # whole group; without it, a number refers to THIS document
                ctx_inst = _instrument_after(rec.content, phrase_end)
                dst_doc = rec.doc_id
                if ctx_inst is not None:
                    hit = instr_index.get(ctx_inst)
                    if hit is None:
                        mm = re.match(r"(\d{4})/(\d{2,5})", ctx_inst)
                        if mm:
                            hit = instr_index.get(f"{mm.group(2)}/{mm.group(1)}")
                    if hit is None:
                        # instrument out of corpus -> external article stubs
                        for target in group:
                            dst = g.node_external_article(target)
                            g.add_edge(g.node_article(rec.doc_id, num,
                                                      rec.art_titles.get(num, "")),
                                       dst, KIND_CROSS, ev,
                                       extra={"ref_number": target,
                                              "instrument": ctx_inst})
                            n_edges += 1
                        continue
                    dst_doc = hit
                dst_title_map = doc_titles.get(dst_doc, {})
                src_lid = g.node_article(rec.doc_id, num,
                                         rec.art_titles.get(num, ""))
                for target in group:
                    if target in dst_title_map:
                        dst = g.node_article(dst_doc, target,
                                             dst_title_map.get(target, ""))
                    else:
                        dst = g.node_external_article(target)
                    # no self-loop edge without explicit instrument context
                    if dst == src_lid and ctx_inst is None:
                        continue
                    g.add_edge(src_lid, dst, KIND_CROSS, ev,
                               extra={"ref_number": target})
                    n_edges += 1
    return n_edges


def build_amends(g: Graph, recs: List[DocRec],
                 instr_index: Dict[str, str]) -> int:
    """'amended by Regulation (EU) 2024/1234' -> edge to that document node
    (in-corpus instrument) or an external-instrument stub (out of corpus)."""
    titles = {r.doc_id: (r.smap.get("document_title") or r.doc_id) for r in recs}
    corpus_docs = set(titles)
    n_edges = 0
    for rec in recs:
        for m in AMENDS.finditer(rec.content):
            numyear = m.group(1)
            tgt = instr_index.get(numyear)
            # also try the number/year swap
            if tgt is None:
                mm = re.match(r"(\d{4})/(\d{3,5})", numyear)
                if mm:
                    tgt = instr_index.get(f"{mm.group(2)}/{mm.group(1)}")
            if tgt in corpus_docs:
                dst = g.node_document(tgt, titles.get(tgt, tgt))
            else:
                if tgt is None:
                    tgt = numyear
                dst = g.node_external_document(
                    tgt.replace("/", "-").lower(), numyear)
            src_lid = _locate_article(g, rec, m.start())
            ev = {"doc_id": rec.doc_id, "offset": m.start(),
                  "snippet": _snippet(rec.content, m.start(), m.end())}
            g.add_edge(src_lid, dst, KIND_AMENDS, ev,
                       extra={"instrument": numyear})
            n_edges += 1
    return n_edges


def _locate_article(g: Graph, rec: DocRec, offset: int) -> str:
    for num in rec.art_titles:
        span = rec.art_spans.get(num)
        if span and span[0] <= offset < span[1]:
            return g.node_article(rec.doc_id, num, rec.art_titles.get(num, ""))
    first = rec.art_spans and min(rec.art_spans.values(), key=lambda s: s[0])[0]
    if first and offset < first:
        return g.node_preamble(rec.doc_id)
    # offset beyond all articles -> last article
    if rec.art_spans:
        last = max(rec.art_spans.values(), key=lambda s: s[0])
        num = [n for n, s in rec.art_spans.items() if s == last][0]
        return g.node_article(rec.doc_id, num, rec.art_titles.get(num, ""))
    return g.node_document(rec.doc_id, rec.doc_id)


def build_defined_terms(g: Graph, recs: List[DocRec]) -> int:
    """`"term" means ...` inside an article or the preamble -> the term
    node DEFINED_IN that hosting node. Definition-article (title matches
    'definition(s)') is the preferred scope for bare `X means` captures;
    quoted forms are collected corpus-wide and de-duplicated."""
    n_edges = 0
    for rec in recs:
        first_start = (min(rec.art_spans.values(), key=lambda s: s[0])[0]
                       if rec.art_spans else 0)

        def _host(off: int) -> Optional[str]:
            for num in rec.art_titles:
                span = rec.art_spans.get(num)
                if span and span[0] <= off < span[1]:
                    return g.node_article(rec.doc_id, num,
                                          rec.art_titles.get(num, ""))
            if rec.art_spans and off < first_start:
                return g.node_preamble(rec.doc_id)
            if g.node_document(rec.doc_id, rec.doc_id) not in g.nodes:
                g.node_document(rec.doc_id, rec.doc_id)
            return g.node_document(rec.doc_id, rec.doc_id)

        sites: List[tuple] = []
        for pat in (r'"([^"]{3,50})"\s+means\b',
                    r"\u2018([^\u2019]{3,50})\u2019\s+means\b"):
            for m in re.finditer(pat, rec.content, re.I):
                sites.append((m.group(1).strip(), m.start()))

        # `X means` scoped to the definitions article (or preamble if none)
        defnum = next((n for n in rec.art_titles
                       if rec.art_titles[n] and
                       re.search(r"definitions?", rec.art_titles[n], re.I)),
                      None)
        if defnum:
            scope = rec.art_spans.get(defnum) or (0, 1)
        else:
            scope = (0, first_start) if rec.art_spans else (0, len(rec.content))
        for m in re.finditer(r"\b([A-Z][A-Za-z0-9-]{2,40})\s+means\b",
                             rec.content[scope[0]:scope[1]]):
            sites.append((m.group(1).strip(), scope[0] + m.start()))

        seen = set()
        for t, off in sites:
            if t in seen:
                continue
            seen.add(t)
            host = _host(off)
            if host is None:
                continue
            ev = {"doc_id": rec.doc_id, "offset": off,
                  "snippet": _snippet(rec.content, off, off + 240)}
            g.add_edge(g.node_term(rec.doc_id, t), host, KIND_DEFINED, ev,
                       extra={"term": t})
            n_edges += 1
    return n_edges


# closed vocab of energy-sector scope entities, longest-first so a window
# match prefers the specific phrase over a generic fragment of it.
ENTITY_PATTERNS: List[tuple] = [
    ("wholesale energy market participants", r"wholesale\s+energy\s+market\s+participants"),
    ("balance responsible parties", r"balance[-\s]?responsible\s+parties"),
    ("balance managers", r"balance[-\s]?managers"),
    ("digital service providers", r"digital\s+service\s+providers"),
    ("providers of essential services", r"providers?\s+of\s+essential\s+services"),
    ("transmission system operators", r"transmission\s+system\s+operators?"),
    ("distribution system operators", r"distribution\s+system\s+operators?"),
    ("national regulatory authorities", r"national\s+regulator(?:y|ies)"),
    ("critical digital infrastructure entities", r"critical\s+digital\s+infrastructure"),
    ("market participants", r"market\s+participants"),
    ("energy service companies", r"energy\s+service\s+companies"),
    ("energy companies", r"energy\s+companies"),
    ("energy suppliers", r"energy\s+suppliers"),
    ("market operators", r"market\s+operators"),
    ("undertakings", r"undertakings"),
    ("aggregators", r"aggregators"),
    ("end users", r"end[-\s]?users"),
    ("consumers", r"consumers"),
    ("investors", r"investors"),
    ("traders", r"traders"),
    ("operators", r"[a-z]\s+operators?"),
    ("providers", r"[a-z]\s+providers?"),
    ("entities", r"\bentities\b"),
    ("companies", r"\bcompanies\b"),
    ("member states", r"member\s+states"),
]


def _match_entity(window: str) -> Optional[str]:
    """Return the first (by position, then longest) vocab hit in `window`."""
    hits: List[tuple] = []
    for label, pat in ENTITY_PATTERNS:
        for m in re.finditer(pat, window, re.I):
            hits.append((m.start(), -len(m.group(0)), label))
    if not hits:
        return None
    hits.sort()
    return hits[0][2]


def build_applies_to(g: Graph, recs: List[DocRec]) -> int:
    """Heuristic scope edges: hosting node --applies to--> entity concept.
    For each apply-anchoring phrase we take the next ~90-char window and
    match against the closed ENTITY_PATTERNS vocabulary (longest wins)."""
    n_edges = 0
    for rec in recs:
        for m in APPLY_ANCHOR.finditer(rec.content):
            win = rec.content[m.end():m.end() + 90]
            label = _match_entity(win)
            if not label:
                continue
            slug = _slug(label)
            host = _locate_article(g, rec, m.start())
            ev = {"doc_id": rec.doc_id, "offset": m.start(),
                  "snippet": _snippet(rec.content, m.start(), m.start() + 140)}
            g.add_edge(host, g.node_entity(slug, label), KIND_APPLIES, ev,
                       extra={"confidence": "heuristic", "entity": slug})
            n_edges += 1
    return n_edges


# ---------------------------------------------------------------------------
# extended vocab (v2): PART_OF + CITES flag + IMPLEMENTS + SUPERSEDES
# ---------------------------------------------------------------------------
# The extended vocab is a **strict superset** of the existing 4 kinds:
#   * PART_OF, IMPLEMENTS, SUPERSEDES are new *kinds* (new nodes are allowed);
#   * CITES is a **flag on CROSS_REFERENCES** edges that carry an instrument
#     number (i.e. they point outside the current document).  No new edge is
#     created for CITES -- we only set ``cites: true`` on the existing edge.
#
# This guarantees backward compatibility:
#   * edge count and per-document counts are *only* additive (PART_OF is
#     structural; IMPLEMENTS / SUPERSEDES are rare in this corpus);
#   * ``GRAPH_EDGE_KINDS`` whitelist in src/retrieval/graph.py is unchanged,
#     so all existing retrieval runs remain bit-identical;
#   * existing tests that assert ``KIND_CROSS == "CROSS_REFERENCES"`` pass;
#   * existing Cypher whitelists (graphrag_n4j / ingestion.py) unchanged.
# ---------------------------------------------------------------------------

def build_part_of(g: Graph, recs: List[DocRec]) -> int:
    """One structural edge per article: ``article --PART_OF--> its document``.

    This replaces an implicit "which doc contains this article" lookup with a
    first-class graph edge, which the context builder (v2) can traverse as a
    typed edge (e.g. ``Article 5 --PART_OF--> Regulation (EU) 2019/944``).
    """
    n = 0
    for rec in recs:
        doc = g.node_document(rec.doc_id, rec.smap.get("document_title") or rec.doc_id)
        for num, title in rec.art_titles.items():
            art = g.node_article(rec.doc_id, num, title)
            ev = {"doc_id": rec.doc_id, "offset": 0,
                  "snippet": f"{title} is part of "
                             f"{rec.smap.get('document_title') or rec.doc_id}"}
            g.add_edge(art, doc, KIND_PART_OF, ev,
                       extra={"ref_number": num})
            n += 1
    return n


def _instr_node_for(g: Graph, numyear: str,
                    recs_by_instr: Dict[str, DocRec]) -> str:
    """Resolve an instrument ``yyyy/nnnn`` to the document node if in-corpus,
    else an external-instrument stub node.  Returns the node lid."""
    tgt = None
    mm = re.match(r"(\d{4})/(\d{2,5})", numyear)
    if mm:
        for rec in recs_by_instr.values():
            keys = set(_instrument_keys(rec))
            if mm.group(0) in keys or f"{mm.group(2)}/{mm.group(1)}" in keys:
                tgt = rec.doc_id
                break
    if tgt is not None:
        # look up the title from the record
        recs_by_instr_t = {r.doc_id: r for r in recs_by_instr.values()}
        title = recs_by_instr_t[tgt].smap.get("document_title") or tgt
        return g.node_document(tgt, title)
    return g.node_external_document(
        numyear.replace("/", "-").lower(), numyear)


def build_impl_sup(g: Graph, recs: List[DocRec],
                   instr_index: Dict[str, str]) -> Tuple[int, int]:
    """Two rare builders over the corpus:

    ``build_impl(g, recs)`` -- ``Regulation NNNN --IMPLEMENTS--> MNNN`` for
        "shall implement", "implemented through", "Commission Implementing
        Regulation ...".  The src is the *higher-tier* instrument, the dst
        the *delegated* act.

    ``build_sup(g, recs)``  -- ``Regulation MNNN --SUPERSEDES--> NNNN`` for
        "X repeals Y", "X replaces Y", "is replaced by X".

    Both builders use :data:`IMPLEMENTS` and :data:`SUPERSEDES` regexes.
    They are *additive*: they do not modify existing CROSS_REFERENCES / AMENDS
    edges.  In the current energy-audit corpus they emit 0-2 edges each (the
    corpus is mostly regulations, not directives / decisions) but the
    *capability* is present for future corpora.

    Return ``(n_impl, n_sup)``.
    """
    recs_by_instr: Dict[str, List[DocRec]] = {}
    for rec in recs:
        for k, _d in _instrument_keys(rec):
            recs_by_instr.setdefault(k, []).append(rec)

    def _resolve(numyear: str) -> Optional[str]:
        numyear = re.sub(r"/(?:EC|EURI|EUR|EEA)$", "", numyear, flags=re.I)
        if numyear in recs_by_instr:
            rec = recs_by_instr[numyear][0]
            return g.node_document(
                rec.doc_id, rec.smap.get("document_title") or rec.doc_id)
        mm = re.match(r"(\d{4})/(\d{2,5})", numyear)
        if mm:
            swapped = f"{mm.group(2)}/{mm.group(1)}"
            if swapped in recs_by_instr:
                rec = recs_by_instr[swapped][0]
                return g.node_document(
                    rec.doc_id, rec.smap.get("document_title") or rec.doc_id)
        return g.node_external_document(
            numyear.replace("/", "-").lower(), numyear)

    n_impl = 0
    for rec in recs:
        for m in IMPLEMENTS.finditer(rec.content):
            num_src = m.group(2)
            num_dst = m.group(3)
            if num_src == num_dst:
                continue
            src = _resolve(num_src)
            dst = _resolve(num_dst)
            if src is None or dst is None:
                continue
            ev = {"doc_id": rec.doc_id, "offset": m.start(),
                  "snippet": _snippet(rec.content, m.start(), m.start() + 240)}
            g.add_edge(src, dst, KIND_IMPL, ev,
                       extra={"instrument_upper": num_src,
                              "instrument_lower": num_dst})
            n_impl += 1

    n_sup = 0
    # SUPERSEDES requires TWO different instrument numbers in the same clause:
    # group(1) = first (newer), group(2) = second (older).  The direction is
    # "newer --SUPERSEDES--> older".
    for rec in recs:
        for m in SUPERSEDES.finditer(rec.content):
            first = m.group(1)
            second = m.group(2)
            if first is None or second is None or first == second:
                continue
            src = _resolve(first)   # newer (superseder)
            dst = _resolve(second)  # older (superseded)
            if src is None or dst is None or src == dst:
                continue
            ev = {"doc_id": rec.doc_id, "offset": m.start(),
                  "snippet": _snippet(rec.content, m.start(), m.start() + 240)}
            g.add_edge(src, dst, KIND_SUP, ev,
                       extra={"supersedes": second,
                              "superseded_by": first})
            n_sup += 1

    return n_impl, n_sup


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------

def build_graph(recs: Optional[List[DocRec]] = None,
                with_extended: bool = True) -> dict:
    recs = recs or load_corpus()
    g = Graph()
    stats = {}
    for rec in recs:
        g.node_document(rec.doc_id, rec.smap.get("document_title") or rec.doc_id)
    instr_index = _instrument_index(recs)
    stats["cross"] = build_cross_references(g, recs, instr_index)
    stats["amends"] = build_amends(g, recs, instr_index)
    stats["defined"] = build_defined_terms(g, recs)
    stats["applies"] = build_applies_to(g, recs)
    if with_extended:
        stats["part_of"] = build_part_of(g, recs)
        impl, sup = build_impl_sup(g, recs, instr_index)
        stats["impl"] = impl
        stats["sup"] = sup
    # CITES flag -- post-process: a CROSS_REFERENCES edge that carries an
    # ``instrument`` extra field (i.e. it points to an instrument different
    # from the originating article's document) is a CITES edge.
    # This is additive -- it does NOT create new edges, only a boolean on
    # existing ones.
    for e in g.edges:
        if e.get("kind") == KIND_CROSS and e.get("instrument"):
            e["cites"] = True
    return {"nodes": g.nodes, "edges": g.edges, "stats": stats,
            "summary": g.summary()}


def save_graph(result: dict, out_dir: Optional[Path] = None) -> dict:
    out_dir = out_dir or c.NOTEBOOKS_DATA / "graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = result["nodes"]
    edges = result["edges"]

    n_lines = [json.dumps(v) for _, v in sorted(nodes.items())]
    e_lines = [json.dumps(e) for e in edges]
    (out_dir / "nodes.jsonl").write_text("\n".join(n_lines) + "\n")
    (out_dir / "edges.jsonl").write_text("\n".join(e_lines) + "\n")

    # per-doc summary
    rows = []
    docs = sorted({n.get("doc_id") for n in nodes.values() if n.get("doc_id")})
    by_doc_src: Dict[str, Counter] = {}
    for e in edges:
        d = e["src"].split(":")[0]
        by_doc_src.setdefault(d, Counter())[e["kind"]] += 1
    for d in docs:
        cnt = by_doc_src.get(d, Counter())
        rows.append({
            "doc_id": d,
            "cross": cnt.get(KIND_CROSS, 0),
            "amends": cnt.get(KIND_AMENDS, 0),
            "defined_in": cnt.get(KIND_DEFINED, 0),
            "applies_to": cnt.get(KIND_APPLIES, 0),
            "unresolved": sum(1 for e in edges
                              if e["src"].split(":")[0] == d and e["unresolved"]),
        })
    import csv
    with open(out_dir / "graph_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id", "cross", "amends",
                                          "defined_in", "applies_to",
                                          "unresolved"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    handoff = {
        "generated": datetime.now().isoformat(),
        "summary": result["summary"],
        "stats_by_builder": result["stats"],
        "edge_kinds": {
            "CROSS_REFERENCES": "article -> article (via 'Article N' mention; "
                                "lists/ranges expanded; unresolved -> ext: stub)",
            "AMENDS": "article -> target document (instrument match)",
            "DEFINED_IN": "term node -> hosting article/preamble",
            "APPLIES_TO": "article -> scope entity (heuristic)",
        },
        "node_kinds": ["document", "preamble", "article", "external_article",
                       "external_document", "term", "entity"],
        "files": {
            "nodes": "notebooks/data/graph/nodes.jsonl",
            "edges": "notebooks/data/graph/edges.jsonl",
            "summary": "notebooks/data/graph/graph_summary.csv",
        },
        "note": "every edge has evidence {doc_id, offset, snippet<=120}; "
                "unresolved edges carry ext: lineage stubs and unresolved=true",
    }
    (out_dir / "graph_handoff.json").write_text(json.dumps(handoff, indent=2))
    return {"out_dir": str(out_dir),
            "files": [str(p) for p in sorted(out_dir.iterdir())]}


if __name__ == "__main__":
    res = build_graph()
    out = save_graph(res)
    s = res["summary"]
    print("nodes:", s["nodes"], "edges:", s["edges"])
    print("kinds:", s["kinds"], "unresolved:", s["unresolved"])
    print("out:", out["out_dir"])
