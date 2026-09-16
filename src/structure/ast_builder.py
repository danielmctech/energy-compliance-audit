"""
ast_builder -- the 03b AST.

Consumes the 03a structure maps (notebooks/data/structure_maps/*.json) plus
the shared processed markdown (data/processed/eu/**.md) and builds a
canonical, serialisable Abstract Syntax Tree per doc:

    document
    ├── preamble (recitals, introductory prose)
    ├── article  (one node per structural heading from 03a)
    │    ├── paragraph  (numbered or lettered sub-items)
    │    ├── paragraph
    │    └── table      (attached to the span it falls in)
    └── chapter (for hierarchical docs: chapter/title/part wrappers)

Design choices inherited from 03a:
  * **structure is fixed by 03a** -- the article list and the strategy
    in the map are the single source of truth; 03b re-parses only within
    an article's span, never re-decides *where* articles start.
  * **no LLM** -- the AST is deterministic and idempotent.
  * **node spans** are character offsets into the *source markdown*, so
    chunk builders in a later step can slice the text directly without
    re-finding boundaries.

Output schema (per node):

    {
      "id": "doc_id:type:number",     -- stable for re-runs
      "type": "document|preamble|chapter|article|paragraph|table",
      "identifier": "ART-2",          -- display id, e.g. "ART-2", "P-1.2"
      "title": "Controller processing"|null,
      "span": [start, end],           -- into the source markdown
      "children": [...],
      "table": {...}|null,
      "lineage_id": "doc:article:2:para:1(b)",  -- retrieval-grade ID (Step 0)
      "metadata": {...}               -- optional, e.g. mentions of this article
    }

Lineage IDs (Step 0, consumed by the 04 graph and the RAG retrieval layer):
    {doc}:document
    {doc}:preamble
    {doc}:chapter:{n}
    {doc}:article:{n}
    {doc}:article:{n}:para:{p}
    {doc}:table:{seq}                 -- document-order sequence
    {doc}:sentence:{start}-{end}      -- sentence-strategy docs (no articles)

Chunk invariants (asserted by the 03b notebook):
    * every chunk carries a unique lineage_id
    * chunk text == source_md[span[0]:span[1]] (stripped)
    * chunk spans within a doc never overlap
    * "shell" paragraphs (< 200 chars) are folded into the previous chunk
      of the same article -- never across article boundaries
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
# node
# ---------------------------------------------------------------------------

@dataclass
class AstNode:
    node_type: str
    identifier: str
    span: tuple = (0, 0)
    title: Optional[str] = None
    children: List["AstNode"] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    lineage_id: Optional[str] = None

    def add(self, child: "AstNode") -> None:
        self.children.append(child)

    def to_dict(self) -> dict:
        d = {
            "type": self.node_type,
            "identifier": self.identifier,
            "span": list(self.span),
            "title": self.title,
            "children": [c.to_dict() for c in self.children],
        }
        if self.lineage_id:
            d["lineage_id"] = self.lineage_id
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    # traversal
    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def count_of(self, node_type: str) -> int:
        return sum(1 for n in self.walk() if n.node_type == node_type)


# ---------------------------------------------------------------------------
# per-doc builder
# ---------------------------------------------------------------------------

def _paragraphs_in_span(content: str, start: int, end: int, article_num: str) -> List[AstNode]:
    """split [start, end) into paragraph nodes using the same two regexes as
    03a. Returns empty when the body is one undivided block."""
    out: List[AstNode] = []

    # gather all paragraph-open marks in order, with their offsets
    marks: List[tuple] = []
    for m in SM.PARA_LETTERED.finditer(content, start, end):
        marks.append((m.start(), "letter", m.group(0)))
    for m in SM.PARA_NUMBERED.finditer(content, start, end):
        marks.append((m.start(), "number", m.group(0)))
    marks.sort(key=lambda x: x[0])

    # the "open" of a paragraph is its first mark. A paragraph's span runs
    # from its mark to just before the next mark (or the end of the article).
    for i, (pos, kind, _head) in enumerate(marks):
        p_end = marks[i + 1][0] if i + 1 < len(marks) else end
        text = content[pos:p_end].strip()
        # identifier: ART-2.3 style -- number the paragraph within its article
        # and use a stable letter/number from the first mark if available.
        seq_in_article = len(out) + 1
        label = _short_paragraph_label(kind, text)
        out.append(AstNode(
            node_type="paragraph",
            identifier=f"P-{article_num}.{seq_in_article}{label}",
            span=(pos, p_end),
            metadata={"opens": label} if label else {},
        ))
    return out


def _short_paragraph_label(kind: str, text: str) -> Optional[str]:
    """Return (a) / (1) / (iii) / 1. as a short label if the paragraph
    opens with one; else None. Used only to help humans read the AST."""
    if kind == "letter":
        m = re.match(r"^[ \t]*(?:[-*][ \t]*)?\(([a-z])\)", text)
        return f"({m.group(1)})" if m else None
    if kind == "number":
        m = re.match(r"^[ \t]*(\d+)[.)]", text)
        return f"{m.group(1)}" if m else None
    m = re.match(r"^[ \t]*\(([ivxlcdm]+)\)", text)
    return f"({m.group(1)})" if m else None


def _tables_in_span(content: str, start: int, end: int) -> List[AstNode]:
    """return a table node for every block in (start, end)."""
    out: List[AstNode] = []
    lines = content.splitlines()
    # prefix char offsets: off[i] = start offset of line i (0-based)
    off: List[int] = [0]
    for l in lines:
        off.append(off[-1] + len(l) + 1)
    for blk in SM.find_tables(content):
        if blk["start_line"] < 1 or blk["end_line"] > len(lines):
            continue
        char_start = off[blk["start_line"] - 1]
        char_end   = off[blk["end_line"]]
        if char_start < start or char_start >= end:
            continue
        out.append(AstNode(
            node_type="table",
            identifier="TAB",
            span=(char_start, char_end),
            metadata={"rows": blk["rows"], "cols": blk["cols"]},
        ))
    out.sort(key=lambda n: n.span[0])
    return out


def _preamble_node(content: str, first_article_start: int, doc_id: str) -> AstNode:
    """the span from the document start to the first article = preamble."""
    recitals = SM.find_recitals(content[:first_article_start])
    node = AstNode(
        node_type="preamble",
        identifier=f"{doc_id}:preamble",
        span=(0, first_article_start),
    )
    node.metadata = {"recitals_count": len(recitals),
                     "recitals": recitals[:40]}
    return node


def build_ast(md_path: Path, smap: dict) -> AstNode:
    """
    Build the AST for one doc using 03a's map (structural truth) + the
    markdown source.

    smap -- the dict that 03a produced for this doc. Read from
             notebooks/data/structure_maps/{doc_id}_structure.json.
    """
    content = md_path.read_text(encoding="utf-8")
    doc_id  = smap["doc_id"]
    strategy = smap["chunking_strategy"]
    arts = smap["articles"]

    root = AstNode(
        node_type="document",
        identifier=doc_id,
        span=(0, len(content)),
        title=smap.get("document_title"),
        metadata={
            "instrument": smap.get("instrument"),
            "celex": smap.get("celex"),
            "publication_date": smap.get("publication_date"),
            "category": smap.get("category"),
            "chunking_strategy": strategy,
            "strategy_targets": smap.get("recommended_chunk_targets", {}),
            "words": smap.get("words"),
            "content_chars": smap.get("content_chars"),
        },
    )

    # re-derive structural articles from the source (03a's map carries the
    # list + titles, the spans are derived here exactly as 03a did).
    found_arts = SM.find_structural_articles(content)
    anchors = [a.line_start for a in found_arts] + [len(content)]
    for k, a in enumerate(found_arts):
        a.span_end = anchors[k + 1]
    first_article_start = found_arts[0].line_start if found_arts else 0

    if first_article_start > 0:
        root.add(_preamble_node(content, first_article_start, doc_id))

    # --- chapters (03a recorded them but they are a *parallel* hierarchy
    #    for EU legislative docs; we include them as siblings if any). ---
    if smap.get("chapters"):
        chaps = sorted(smap["chapters"], key=lambda ch: ch["pos"])
        ch_anchors = [ch["pos"] for ch in chaps] + [len(content)]
        for i, ch in enumerate(chaps):
            n = AstNode(
                node_type="chapter",
                identifier=f"CHP-{ch['num']}",
                span=(ch["pos"], ch_anchors[i + 1]),
                title=ch.get("title"),
                metadata={"kind": ch["kind"]},
            )
            root.add(n)

    # --- articles ---
    # found_arts (above) carries the exact spans for each structural
    # article (in document order).
    # map by number (first occurrence); 03a list can include duplicates
    # (sub-paragraph repeals) -- only the first is structural.
    found_map: dict = {}
    for fa in found_arts:
        found_map.setdefault(fa.number, fa)

    for a in arts:
        fa = found_map.get(a["number"])
        if fa is None:
            # no span in the source; still record the node from 03a
            art = AstNode(
                node_type="article",
                identifier=f"ART-{a['number']}",
                title=a.get("title"),
                metadata={"mentions": a.get("mentions", 0)},
            )
            root.add(art)
            continue
        art = AstNode(
            node_type="article",
            identifier=f"ART-{a['number']}",
            span=(fa.line_start, fa.span_end),
            title=a.get("title"),
            metadata={"mentions": a.get("mentions", 0),
                     "para_count_03a": a.get("para_count")},
        )
        # paragraphs within this article's span
        paras = _paragraphs_in_span(content, fa.line_start, fa.span_end, a["number"])
        # tables attached inside the article
        tabs  = _tables_in_span(content, fa.line_start, fa.span_end)
        for p in paras:
            art.add(p)
        for t in tabs:
            art.add(t)
        root.add(art)

    # --- tables that fall into the preamble (no article span) ---
    for t in _tables_in_span(content, 0, first_article_start):
        t.node_type = "table"
        root.add(t)

    _assign_lineage(root, doc_id)
    return root


def _assign_lineage(root: AstNode, doc_id: str) -> None:
    """Walk the tree in document order and attach a retrieval-grade
    ``lineage_id`` to every node (the Step 0 schema in the module docstring).

    Tables are numbered across the document in span order so the sequence
    is stable even when a table sits in the preamble."""
    # collect everything first so table numbering follows source order
    tables: List[AstNode] = []
    def collect(node):
        for ch in node.children:
            if ch.node_type == "table":
                tables.append(ch)
            collect(ch)
    collect(root)
    tables.sort(key=lambda t: (t.span[0], t.span[1]))
    table_lin = {id(t): f"{doc_id}:table:{i + 1}"
                 for i, t in enumerate(tables) if t.span != (0, 0)}

    root.lineage_id = f"{doc_id}:document"
    def rec(node: AstNode):
        for ch in node.children:
            if ch.node_type == "preamble":
                ch.lineage_id = f"{doc_id}:preamble"
            elif ch.node_type == "chapter":
                chn = ch.identifier.split("-", 1)[1] if ch.identifier.startswith("CHP-") else "?"
                ch.lineage_id = f"{doc_id}:chapter:{chn}"
            elif ch.node_type == "article":
                an = ch.identifier.split("-", 1)[1] if ch.identifier.startswith("ART-") else "?"
                ch.lineage_id = f"{doc_id}:article:{an}"
            elif ch.node_type == "paragraph" and node.node_type == "article":
                # stable per-paragraph position within its article
                seq = 1
                for sib in node.children:
                    if sib is ch:
                        break
                    if sib.node_type == "paragraph":
                        seq += 1
                ch.lineage_id = f"{node.lineage_id}:para:{seq}"
            elif ch.node_type == "table":
                ch.lineage_id = table_lin.get(id(ch), f"{doc_id}:table:unspanned")
            else:
                ch.lineage_id = None
            rec(ch)
    rec(root)


# ---------------------------------------------------------------------------
# chunking helper (the next pipeline step uses this to materialise chunks)
# ---------------------------------------------------------------------------

MIN_CHUNK_CHARS = 200


@dataclass
class Chunk:
    node_id: str
    node_type: str
    text: str
    span: tuple
    context: dict
    lineage_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {"node_id": self.node_id, "node_type": self.node_type,
                "span": list(self.span), "text": self.text,
                "context": self.context, "lineage_id": self.lineage_id}


def _fold_article_shells(chunks: List[Chunk], content: str,
                         min_chars: int = MIN_CHUNK_CHARS) -> List[Chunk]:
    """"Shell" article chunks (heading only / no substantive body, e.g. a
    repealed placeholder) are useless for retrieval. Fold each one into a
    source-order neighbour (prefer the next article chunk; else the
    previous) and record the dropped lineage under ``context["absorbed"]``
    so every lineage_id stays resolvable to exactly one chunk."""
    out: List[Chunk] = []
    for i, ch in enumerate(chunks):
        if ch.node_type == "article" and len(ch.text) < min_chars:
            if out:
                _merge_into(out[-1], ch, content)
                continue
            if i + 1 < len(chunks):
                _merge_into(chunks[i + 1], ch, content)
                continue
        out.append(ch)
    return out


def _merge_into(pv: "Chunk", cur: "Chunk", content: str) -> None:
    """Absorb `cur` into the preceding chunk `pv`: the survivor's span
    becomes the union, its text is re-sliced from the source (so
    ``text == source[span].strip()`` keeps holding), and `cur`'s lineage
    is recorded under ``context["absorbed"]`` -- nothing is lost."""
    sp = (min(pv.span[0], cur.span[0]), max(pv.span[1], cur.span[1]))
    if sp[1] <= sp[0]:
        return
    pv.span = sp
    pv.text = content[sp[0]:sp[1]].strip()
    pv.context.setdefault("absorbed", []).append(cur.lineage_id)
    if cur.context.get("n_paras"):
        pv.context["n_paras"] = pv.context.get("n_paras", 0) + cur.context["n_paras"]


def _coalesce_thin_cs(cs: List["Chunk"], content: str,
                      min_chars: int = MIN_CHUNK_CHARS) -> List["Chunk"]:
    """Merge thin chunks within ONE article's paragraph chunk stream
    (spans are all (article_start, para_end), so union re-slicing never
    crosses an article boundary). A thin first chunk folds into its next
    sibling; a thin later chunk folds into the previous one."""
    out: List[Chunk] = []
    i = 0
    while i < len(cs):
        c = cs[i]
        if len(c.text) >= min_chars:
            out.append(c)
        elif out:
            _merge_into(out[-1], c, content)
        elif i + 1 < len(cs):
            sp = (min(cs[i + 1].span[0], c.span[0]),
                  max(cs[i + 1].span[1], c.span[1]))
            nxt = cs[i + 1]
            nxt.span = sp
            nxt.text = content[sp[0]:sp[1]].strip()
            nxt.context.setdefault("absorbed", []).append(c.lineage_id)
        else:
            out.append(c)
        i += 1
    return out


def _article_text(content: str, node: AstNode) -> str:
    """article node text = the raw source slice (heading + body).
    ``text == content[span[0]:span[1]].strip()`` holds for every chunk we
    emit, so downstream steps can re-verify lineage against source."""
    s, e = node.span
    return content[s:e].strip()


def chunk_article_based(content: str, root: AstNode,
                        max_chars: int = 8000,
                        ) -> List[Chunk]:
    """Each article (with its paragraphs and tables) is a natural chunk.
    For articles that exceed max_chars we fall back to their paragraph
    sub-nodes. Preamble is its own chunk. Chapters are *context*, not
    separate chunks (their text is already inside the article spans)."""
    chunks: List[Chunk] = []
    thin: List[Chunk] = []   # (article_start, para_end) per-article stream

    def chapter_for(pos: int) -> Optional[AstNode]:
        # innermost enclosing chapter
        best: Optional[AstNode] = None
        for ch in [n for n in root.children if n.node_type == "chapter"]:
            if ch.span[0] <= pos < ch.span[1]:
                if best is None or ch.span[0] > best.span[0]:
                    best = ch
        return best

    for n in list(root.walk()):
        if n.node_type == "article":
            lin = n.lineage_id or f"{n.identifier}"
            text = _article_text(content, n)
            ch = chapter_for(n.span[0])
            if len(text) <= max_chars:
                ctx = {"identifier": n.identifier, "title": n.title,
                       "n_paras": n.count_of("paragraph"),
                       "n_tables": n.count_of("table")}
                if ch:
                    ctx["chapter"] = ch.identifier
                chunks.append(Chunk(node_id=n.identifier,
                                    node_type="article", text=text,
                                    span=n.span, context=ctx,
                                    lineage_id=lin))
            else:
                # contiguous partition of the article span: chunk 1 =
                # heading + first paragraph, the rest = their own slice.
                # No overlap, every char of the article covered.
                paras = [p for p in n.children if p.node_type == "paragraph"]
                a_cs: List[Chunk] = []
                if not paras:
                    a_cs.append(Chunk(node_id=n.identifier, node_type="article",
                                      text=content[n.span[0]:n.span[1]].strip(),
                                      span=n.span,
                                      context={"identifier": n.identifier,
                                               "title": n.title,
                                               **({"chapter": ch.identifier} if ch else {})},
                                      lineage_id=lin))
                else:
                    for j, p in enumerate(paras):
                        start = n.span[0] if j == 0 else paras[j - 1].span[1]
                        ctx = {"parent": n.identifier, "title": n.title,
                               **({"chapter": ch.identifier} if ch else {})}
                        a_cs.append(Chunk(node_id=p.identifier,
                                          node_type="paragraph",
                                          text=content[start:p.span[1]].strip(),
                                          span=(start, p.span[1]),
                                          context=ctx,
                                          lineage_id=p.lineage_id))
                thin.extend(_coalesce_thin_cs(a_cs, content))
        elif n.node_type in ("preamble", "chapter"):
            if n.node_type == "chapter":
                continue
            text = content[n.span[0]:n.span[1]].strip()
            chunks.append(Chunk(node_id=n.identifier,
                                node_type=n.node_type, text=text,
                                span=n.span,
                                context={k: n.metadata.get(k)
                                         for k in ("recitals_count", "kind")
                                         if n.metadata.get(k)},
                                 lineage_id=n.lineage_id))
    chunks = _fold_article_shells(chunks, content)
    return chunks


def chunk_sentence_512(content: str, root: AstNode,
                       max_chars: int = 1200,
                       min_chars: int = MIN_CHUNK_CHARS) -> List[Chunk]:
    """Fallback for docs with no structural articles (guidance / NC).
    Sentence-aware fixed-size chunks over the full text; every chunk span
    is a contiguous source slice and thin chunks (< min_chars) are merged
    into their source-order neighbour."""
    parts: List[tuple] = []
    for m in re.finditer(r".+?[.!?](?:\s+|$)|.+?(?=\Z)", content, re.DOTALL):
        parts.append((m.start(), content[m.start():m.end()].strip()))

    out: List[Chunk] = []
    cs: List[Chunk] = []

    def flush() -> None:
        nonlocal cs
        if not cs:
            return
        sp = (cs[0].span[0], cs[-1].span[1])
        t = content[sp[0]:sp[1]].strip()
        if len(t) < min_chars and len(cs) > 1 and sp[1] - sp[0] > min_chars:
            # union slice still thin (huge blank gaps): keep the densest
            # contiguous tail, dropping nothing semantic
            for k in range(1, len(cs)):
                sub_sp = (cs[k].span[0], cs[-1].span[1])
                sub_t = content[sub_sp[0]:sub_sp[1]].strip()
                if len(sub_t) >= min_chars:
                    sp, t, cs = sub_sp, sub_t, cs[k:]
                    break
        absorbed = [c.lineage_id for c in cs[1:]]
        out.append(Chunk(node_id=f"SENT-{sp[0]}", node_type="sentence",
                         text=t, span=sp,
                         context={"doc": root.identifier,
                                  **({"absorbed": absorbed} if absorbed else {}),
                                  **({"thin": True} if len(t) < min_chars else {})},
                         lineage_id=f"{root.identifier}:sentence:{sp[0]}-{sp[1]}"))
        cs = []

    for pos, ptext in parts:
        if len(ptext) > max_chars:
            flush()
            out.append(Chunk(node_id=f"SENT-{pos}", node_type="sentence",
                             text=ptext, span=(pos, pos + len(ptext)),
                             context={"doc": root.identifier},
                             lineage_id=f"{root.identifier}:sentence:{pos}-{pos + len(ptext)}"))
            continue
        if cs and cs[-1].span[1] + 1 + len(ptext) > max_chars:
            flush()
        cs.append(Chunk(node_id=f"SENT-{pos}", node_type="sentence",
                        text=ptext, span=(pos, pos + len(ptext)),
                        context={"doc": root.identifier},
                        lineage_id=f"{root.identifier}:sentence:{pos}-{pos + len(ptext)}"))
    flush()
    return out


def build_chunks(content: str, root: AstNode,
                 strategy: str, max_chars: int = 8000) -> List[Chunk]:
    if strategy == "article_based":
        return chunk_article_based(content, root, max_chars=max_chars)
    if strategy == "hierarchical":
        return chunk_article_based(content, root, max_chars=max_chars)
    # table_then_sentence / sentence_512
    return chunk_sentence_512(content, root)


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

def build_all(md_maps: dict, out_dir: Optional[Path] = None) -> dict:
    """md_maps: {doc_id: {"md_path": Path, "map": dict}} -> {doc_id: AstNode}."""
    out_dir = out_dir or c.NOTEBOOKS_DATA / "ast"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for doc_id, spec in md_maps.items():
        root = build_ast(spec["md_path"], spec["map"])
        out[doc_id] = root
        (out_dir / f"{doc_id}_ast.json").write_text(
            json.dumps(root.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    OUT = c.NOTEBOOKS_DATA / "structure_maps"
    files = sorted(OUT.glob("*_structure.json"))
    if only is not None:
        files = [f for f in files if f.stem.startswith(only)]
    AST_OUT = c.NOTEBOOKS_DATA / "ast"
    AST_OUT.mkdir(parents=True, exist_ok=True)
    for p in files:
        smap = json.loads(p.read_text())
        md_path = c.resolve_repo_path(smap["source_md"])
        root = build_ast(md_path, smap)
        doc_id = p.name[:-len("_structure.json")]
        (AST_OUT / f"{doc_id}_ast.json").write_text(
            json.dumps(root.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        print("=" * 60)
        print(doc_id, "  strategy =",
              smap["chunking_strategy"], "  nodes =", len(list(root.walk())))
        for n in list(root.walk())[:10]:
            print("   ", n.node_type, n.identifier,
                  f"span={n.span[0]}..{n.span[1]}",
                  (n.title or "")[:40])
