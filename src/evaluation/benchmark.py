"""Graph-aware 8-family synthetic benchmark for energy-audit RAG evaluation.

Replaces the legacy 60-query set (2 categories: article/term) with a
multi-family set covering the full spec from
``Implement a Multi-Family Synthetic Benchmark for Graph-Aware RAG.md``.

Family targets (per doc §2, §19):

    single_document              15
    non_relational_semantic      5
    one_hop_relational          10
    two_hop_relational          10
    relation_direction           5
    temporal_version             5
    graph_distractor             5
    multi_document_synthesis     5
    -------------------------
    TOTAL                       60

Each item carries machine-readable gold evidence (``gold_chunks``,
``gold_edges``, ``gold_path``) so downstream retrieval metrics and the
answer-judge pipeline can use it without inventing fields.

Determinism: pure corpus/edge lookups + ``random.Random(7)`` shuffle.
No LLM required (the SCENARIO_DRAFTERS pipeline from doc §10 is a
*runtime* step for production; the gold construction itself stays
inspectable and testable).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# -- repo-root discovery -----------------------------------------------------

def _repo_root() -> Path:
    import os
    env = os.getenv("ENERGY_AUDIT_ROOT")
    if env and Path(env).exists():
        return Path(env).resolve()
    cur = Path(__file__).resolve().parent
    for cand in [cur, *cur.parents]:
        if (cand / "notebooks").is_dir() and (cand / "data").is_dir():
            return cand
    return cur.parent


# -- corpus / graph dataclasses ---------------------------------------------

@dataclass
class _Node:
    lid: str            # lineage_id
    kind: str           # "document" | "article" | "term" | "entity" | "preamble"
    doc_id: Optional[str] = None
    number: Optional[str] = None
    title: Optional[str] = None
    term: Optional[str] = None
    label: Optional[str] = None


@dataclass
class _Edge:
    src: str
    dst: str
    kind: str           # CROSS_REFERENCES | AMENDS | DEFINED_IN | APPLIES_TO | ...
    unresolved: bool = False
    term: Optional[str] = None
    ref_number: Optional[str] = None
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None


# -- loaders ----------------------------------------------------------------

def _load_nodes(nodes_path: Path) -> Dict[str, _Node]:
    out: Dict[str, _Node] = {}
    with open(nodes_path) as f:
        for line in f:
            if not line.strip():
                continue
            n = json.loads(line)
            out[n["lineage_id"]] = _Node(
                lid=n["lineage_id"], kind=n["kind"],
                doc_id=n.get("doc_id"), number=n.get("number"),
                title=n.get("title"), term=n.get("term"),
                label=n.get("label"),
            )
    return out


def _load_edges(edges_path: Path) -> List[_Edge]:
    out: List[_Edge] = []
    with open(edges_path) as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            out.append(_Edge(
                src=e["src"], dst=e["dst"], kind=e["kind"],
                unresolved=bool(e.get("unresolved", False)),
                term=e.get("term"), ref_number=e.get("ref_number"),
                supersedes=e.get("supersedes"),
                superseded_by=e.get("superseded_by"),
            ))
    return out


def _load_chunk_targets(chunks_dir: Path) -> List[dict]:
    """All retrievable chunks across all docs, as plain dicts."""
    targets: List[dict] = []
    for f in sorted(chunks_dir.glob("chunks_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        doc_id = f.name.replace("chunks_", "", 1).replace(".json", "")
        for rec in data:
            targets.append({
                "lineage_id": rec["lineage_id"],
                "doc_id": rec.get("doc_id") or doc_id,
                "node_type": rec.get("node_type"),
                "text": rec.get("text", ""),
            })
    return targets


def _chunk_by_doc(chunks: List[dict]) -> Dict[str, List[dict]]:
    by_doc: Dict[str, List[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)
    return by_doc


def _chunk_of_article(chunks: List[dict], doc_id: str,
                      article_no: Optional[str] = None,
                      used: Optional[set] = None) -> Optional[dict]:
    """Best chunk corresponding to doc/article_no.

    ``article_no`` may be a digit ("45") or a digit+letter sub-article
    ("45d").  Preference (skipping already-allocated lids if `used` given):
      1. <doc>:article:<N>            (exact, digits or digit+letter)
      2. <doc>:article:<N>[a-z]*      (same number, any sub-letter)
      3. <doc>:article:*              (any article in same doc)
      4. <doc>:preamble              (preamble)
      5. Any chunk of <doc>           (last resort)
    """
    used = used or set()
    if article_no:
        # exact article (digits or digit+letter sub-article)
        exact_lid = f"{doc_id}:article:{article_no}"
        for c in chunks:
            if c["lineage_id"] == exact_lid and c["lineage_id"] not in used:
                return c
        # same number, different letter (e.g. article 12 and article 12b)
        base = re.match(r"^(\d+)", article_no)
        if base:
            pat = re.compile(rf"^{re.escape(doc_id)}:article:{base.group(1)}[a-zA-Z]*$")
            for c in chunks:
                if pat.match(c["lineage_id"]) and c["lineage_id"] not in used:
                    return c
    for c in chunks:
        if (c["doc_id"] == doc_id
                and re.match(rf"^{re.escape(doc_id)}:article:\d+[a-zA-Z]*$", c["lineage_id"])
                and c["lineage_id"] not in used):
            return c
    for c in chunks:
        if (c["doc_id"] == doc_id
                and c["lineage_id"].endswith(":preamble")
                and c["lineage_id"] not in used):
            return c
    for c in chunks:
        if c["doc_id"] == doc_id and c["lineage_id"] not in used:
            return c
    return None


# -- BenchmarkItem (extended) -----------------------------------------------

@dataclass
class BenchmarkItem:
    query_id: str
    question: str
    doc_id: str                       # primary document (for backward-compat)
    target_lineage_id: str            # retrievable chunk lid
    category: str                     # one of the 8 family names
    term: Optional[str] = None
    article_title: Optional[str] = None
    reference_answer: Optional[str] = None
    reference_basis: str = "target_chunk_text"
    difficulty: str = "medium"
    metadata: dict = field(default_factory=dict)

    # doc §8 machine-readable gold evidence
    gold_documents: List[str] = field(default_factory=list)
    gold_chunks: List[str] = field(default_factory=list)
    gold_entities: List[str] = field(default_factory=list)
    gold_edges: List[dict] = field(default_factory=list)
    gold_path: List[dict] = field(default_factory=list)
    hop_count: int = 0
    required_evidence: List[str] = field(default_factory=list)
    # relation-direction disambiguation (used by relation_direction family)
    intended_relation: Optional[str] = None
    intended_direction: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            # backward-compat fields (retrieval_run, overlap, answer_metrics all
            # read these names -- keep them so nothing breaks)
            "query_id": self.query_id,
            "question": self.question,
            "doc_id": self.doc_id,
            "reference_answer": self.reference_answer,
            "reference_basis": self.reference_basis,
            "relevant_document_ids": self.gold_documents or [self.doc_id],
            "relevant_chunk_ids": self.gold_chunks or [self.target_lineage_id],
            "relevant_entities": self.gold_entities,
            "relevant_relationships": self.gold_edges,
            "category": self.category,
            "difficulty": self.difficulty,
            "term": self.term,
            "article_title": self.article_title,
            "metadata": self.metadata,
            # doc §8 / §9 machine-readable gold evidence
            "gold_answer": self.reference_answer,
            "gold_documents": self.gold_documents,
            "gold_chunks": self.gold_chunks,
            "gold_entities": self.gold_entities,
            "gold_edges": self.gold_edges,
            "gold_path": self.gold_path,
            "hop_count": self.hop_count,
            "required_evidence": self.required_evidence,
            "intended_relation": self.intended_relation,
            "intended_direction": self.intended_direction,
        }


# -- deterministic validators (doc §12) --------------------------------------

def _validate_single(gold_chunks: List[str], all_chunk_lids: set) -> List[str]:
    """Return a list of validation-error strings (empty = pass)."""
    errs = []
    if not gold_chunks:
        errs.append("empty gold_chunks")
    for cid in gold_chunks:
        if cid not in all_chunk_lids:
            errs.append(f"gold_chunk {cid!r} not in corpus")
    return errs


def _validate_path(gold_path: List[dict], edges: List[_Edge]) -> List[str]:
    """Verify gold_path is a valid chain of edges in the corpus."""
    errs = []
    edge_set = {(e.src, e.dst, e.kind) for e in edges}
    for i, step in enumerate(gold_path):
        s, d, k = step.get("source"), step.get("target"), step.get("relation")
        if (s, d, k) not in edge_set:
            errs.append(f"gold_path[{i}] ({s} --{k}--> {d}) not in corpus")
        if i > 0 and gold_path[i - 1]["target"] != step.get("source"):
            errs.append(f"gold_path not contiguous at step {i}")
    return errs


def validate_item(item: BenchmarkItem,
                  all_chunk_lids: set,
                  edges: List[_Edge]) -> List[str]:
    """Deterministic validator -- returns a list of error strings (empty=pass).

    Checks the invariants the doc's §12 calls out: referenced chunks exist,
    gold_path is a valid contiguous chain, hop_count matches path length,
    and (for relational families) at least one gold edge exists.
    """
    errs = _validate_single(item.gold_chunks, all_chunk_lids)
    if item.gold_path:
        errs.extend(_validate_path(item.gold_path, edges))
        if item.hop_count != len(item.gold_path):
            errs.append(
                f"hop_count={item.hop_count} != len(gold_path)={len(item.gold_path)}")
    if item.category in (
        "one_hop_relational", "two_hop_relational",
        "relation_direction", "temporal_version",
        "multi_document_synthesis", "graph_distractor",
    ) and not item.gold_edges and not item.gold_path:
        errs.append(f"{item.category} must carry gold_edges or gold_path")
    return errs


# -- per-family builders ------------------------------------------------------

def _doc_of(lid: str) -> str:
    if ":article:" in lid:
        return lid.split(":article:")[0]
    return lid.split(":")[0]


def _article_num_of(lid: str) -> Optional[str]:
    m = re.match(r".*:article:(\d+)", lid)
    return m.group(1) if m else None


def _title_of(nodes: Dict[str, _Node], lid: str) -> str:
    n = nodes.get(lid)
    if n is None or not n.title:
        return f"{lid.split(':')[-1]}"
    if re.match(r"^Article\s+\d+$", n.title):
        return f"Article {n.number}"
    return n.title


def _edge_dict(e: _Edge) -> dict:
    return {"source": e.src, "relation": e.kind, "target": e.dst}


class _BuilderContext:
    def __init__(self, data_dir: Optional[Path] = None,
                 chunks_dir: Optional[Path] = None):
        if data_dir is None:
            import common as c  # type: ignore
            data_dir = Path(c.NOTEBOOKS_DATA) / "graph"
        if chunks_dir is None:
            import common as c  # type: ignore
            chunks_dir = Path(c.NOTEBOOKS_DATA) / "ast"
        self.nodes = _load_nodes(data_dir / "nodes.jsonl")
        self.edges = _load_edges(data_dir / "edges.jsonl")
        self.chunks = _load_chunk_targets(chunks_dir)
        self.chunk_by_doc = _chunk_by_doc(self.chunks)
        self.all_lids = {c["lineage_id"] for c in self.chunks}


def _build_single_document(ctx: _BuilderContext, n: int,
                           used_lids: set) -> List[BenchmarkItem]:
    """15 single-document queries.

    Pick (doc, article) pairs that have a direct article chunk so the
    retrieval target is unambiguous.  Question asks for obligations in a
    specific article, requiring retrieval of that article's chunk.
    """
    art_edges = [c for c in ctx.chunks if re.match(r".+:article:\d+$", c["lineage_id"])]
    art_edges.sort(key=lambda c: c["lineage_id"])
    out: List[BenchmarkItem] = []
    seen_docs: set = set()
    for c in art_edges:
        if c["lineage_id"] in used_lids:
            continue
        doc = c["doc_id"]
        art_no = _article_num_of(c["lineage_id"])
        n_node = ctx.nodes.get(f"{doc}:article:{art_no}")
        title = (n_node.title if n_node and n_node.title and
                 not re.match(r"^Article\s+\d+$", n_node.title or "")
                 else f"Article {art_no}")
        q = f"What are the obligations in {title} of {doc}?"
        item = BenchmarkItem(
            query_id="pending", question=q, doc_id=doc,
            target_lineage_id=c["lineage_id"],
            category="single_document",
            article_title=title,
            reference_answer=c["text"],
            difficulty="easy",
            gold_documents=[doc], gold_chunks=[c["lineage_id"]],
            required_evidence=[c["lineage_id"]],
            hop_count=0,
        )
        out.append(item)
        used_lids.add(c["lineage_id"])
        seen_docs.add(doc)
        if len(out) >= n:
            break
    return out


def _build_non_relational(ctx: _BuilderContext, n: int,
                          used_lids: set) -> List[BenchmarkItem]:
    """5 DEFINED_IN queries (term defined inside an article).

    Gold = the article chunk that defines the term; gold_edges = the
    DEFINED_IN edge.  No inter-instrument relation required.
    """
    defined = [e for e in ctx.edges
               if e.kind == "DEFINED_IN" and not e.unresolved
               and e.term and len(e.term) >= 4]
    defined.sort(key=lambda e: (e.src, e.dst))
    out: List[BenchmarkItem] = []
    seen: set = set()
    for e in defined:
        # term lid is like <doc>:term:<slug>; dst is like <doc2>:article:<N>
        dst_doc = _doc_of(e.dst)
        art_no = _article_num_of(e.dst)
        chunk = _chunk_of_article(ctx.chunks, dst_doc, art_no, used=used_lids)
        if chunk is None:
            continue
        if e.term.lower() in seen:
            continue
        q = f"What does '{e.term}' mean in {dst_doc}?"
        item = BenchmarkItem(
            query_id="pending", question=q, doc_id=dst_doc,
            target_lineage_id=chunk["lineage_id"],
            category="non_relational_semantic",
            term=e.term,
            reference_answer=chunk["text"],
            difficulty="easy",
            gold_documents=[dst_doc], gold_chunks=[chunk["lineage_id"]],
            gold_entities=[e.term],
            gold_edges=[_edge_dict(e)],
            gold_path=[_edge_dict(e)],
            required_evidence=[chunk["lineage_id"]],
            hop_count=1,
            intended_relation="DEFINED_IN",
        )
        out.append(item)
        used_lids.add(chunk["lineage_id"])
        seen.add(e.term.lower())
        if len(out) >= n:
            break
    return out


def _build_one_hop(ctx: _BuilderContext, n: int,
                    used_lids: set) -> List[BenchmarkItem]:
    """10 one-hop relational queries.

    Mixed pool (per doc §2, §19 pool_n): AMENDS ∪ APPLIES_TO ∪
    CROSS_REFERENCES (cross-doc).  Each row asks for the relationship
    between src and dst; gold = the src article chunk + the edge.
    """
    pool: List[_Edge] = []
    for e in ctx.edges:
        if e.unresolved:
            continue
        if e.kind == "AMENDS" and (
                ":article:" in e.src or ":preamble" in e.src
                or e.src.endswith(":document")):
            pool.append(e)
        elif e.kind == "APPLIES_TO" and ":article:" in e.src:
            pool.append(e)
        elif (e.kind == "CROSS_REFERENCES"
                and ":article:" in e.src and ":article:" in e.dst
                and _doc_of(e.src) != _doc_of(e.dst)):
            pool.append(e)
    pool.sort(key=lambda e: e.src)
    out: List[BenchmarkItem] = []
    for e in pool:
        src_doc = _doc_of(e.src)
        art_no = _article_num_of(e.src)
        chunk = _chunk_of_article(ctx.chunks, src_doc, art_no, used=used_lids)
        if chunk is None:
            chunk = _chunk_of_article(ctx.chunks, src_doc, used=used_lids)
        if chunk is None:
            continue
        if e.kind == "AMENDS":
            dst_doc = _doc_of(e.dst)
            if art_no:
                art_or_pre = f"Article {art_no}"
            elif ":preamble" in e.src:
                art_or_pre = "the preamble"
            else:
                art_or_pre = "the instrument"
            q = (f"Which provision of {src_doc} ({art_or_pre}) amends "
                 f"{dst_doc}, and what does it change?")
            entities: List[str] = []
            direction = "src->dst (newer amends older)"
        elif e.kind == "APPLIES_TO":
            dst_node = ctx.nodes.get(e.dst)
            if dst_node and dst_node.label:
                entity_label = dst_node.label.lower()
            elif ":" in e.dst and len(e.dst.split(":")) >= 3:
                entity_label = e.dst.split(":")[-1].replace("_", " ")
            else:
                continue
            q = (f"To which {entity_label} does Article {art_no} "
                 f"of {src_doc} apply?")
            entities = [entity_label]
            direction = "src->dst (article applies to entity)"
        else:  # CROSS_REFERENCES cross-doc
            dst_doc = _doc_of(e.dst)
            q = (f"Article {art_no} of {src_doc} references Article "
                 f"{_article_num_of(e.dst)} of {dst_doc}; state the "
                 f"relationship between the two provisions.")
            entities = []
            direction = "src->dst (article cross-references other article)"
        item = BenchmarkItem(
            query_id="pending", question=q, doc_id=src_doc,
            target_lineage_id=chunk["lineage_id"],
            category="one_hop_relational",
            reference_answer=chunk["text"],
            difficulty="medium",
            gold_documents=[src_doc, _doc_of(e.dst)]
            if _doc_of(e.dst) != src_doc else [src_doc],
            gold_chunks=[chunk["lineage_id"]],
            gold_entities=entities,
            gold_edges=[_edge_dict(e)],
            gold_path=[_edge_dict(e)],
            required_evidence=[chunk["lineage_id"]],
            hop_count=1,
            intended_relation=e.kind,
            intended_direction=direction,
        )
        out.append(item)
        used_lids.add(chunk["lineage_id"])
        if len(out) >= n:
            break
    return out


def _build_two_hop(ctx: _BuilderContext, n: int,
                   used_lids: set) -> List[BenchmarkItem]:
    """10 two-hop relational queries.

    Find chains A→B→C where A, B, C are articles in possibly different
    docs, connected by CROSS_REFERENCES.  Gold = the src article chunk;
    gold_path = [edge_AB, edge_BC]; hop_count = 2.
    """
    cross = [e for e in ctx.edges
             if e.kind == "CROSS_REFERENCES" and not e.unresolved
             and ":article:" in e.src and ":article:" in e.dst]
    out: List[BenchmarkItem] = []
    # iterate over mid-node B and pair its incoming + outgoing edges
    # (build in/out maps)
    incoming: Dict[str, List[_Edge]] = {}
    outgoing: Dict[str, List[_Edge]] = {}
    for e in cross:
        incoming.setdefault(e.dst, []).append(e)
        outgoing.setdefault(e.src, []).append(e)
    # candidate mid-nodes: in-corpus articles with both incoming and outgoing
    # cross-refs to *different* docs
    mids = []
    for b in outgoing:
        for a in incoming.get(b, []):
            for c in outgoing[b]:
                if (a.src.split(":")[0] != c.dst.split(":")[0]
                        and a.dst == b and c.src == b):
                    mids.append((a, b, c))
    mids.sort(key=lambda t: (t[0].src, t[1], t[2].dst))
    for a, b, c in mids:
        src_doc = _doc_of(a.src)
        art_no_a = _article_num_of(a.src)
        chunk = _chunk_of_article(ctx.chunks, src_doc, art_no_a, used=used_lids)
        if chunk is None:
            chunk = _chunk_of_article(ctx.chunks, src_doc, used=used_lids)
        if chunk is None:
            continue
        # skip self-loop chains (A→B→A) -- the same article both in and out
        if c.dst == a.src:
            continue
        q = (f"Combining the cross-reference from Article "
             f"{_article_num_of(a.src)} of {src_doc} to Article "
             f"{_article_num_of(a.dst)} of {_doc_of(a.dst)}, "
             f"and that article's reference to Article "
             f"{_article_num_of(c.dst)} of {_doc_of(c.dst)}, "
             f"what is the resulting obligation?")
        item = BenchmarkItem(
            query_id="pending", question=q, doc_id=src_doc,
            target_lineage_id=chunk["lineage_id"],
            category="two_hop_relational",
            reference_answer=chunk["text"],
            difficulty="hard",
            gold_documents=[_doc_of(a.src), _doc_of(a.dst), _doc_of(c.dst)],
            gold_chunks=[chunk["lineage_id"]],
            gold_edges=[_edge_dict(a), _edge_dict(c)],
            gold_path=[_edge_dict(a), _edge_dict(c)],
            required_evidence=[chunk["lineage_id"]],
            hop_count=2,
        )
        out.append(item)
        used_lids.add(chunk["lineage_id"])
        if len(out) >= n:
            break
    return out


def _build_relation_direction(ctx: _BuilderContext, n: int,
                               used_lids: set) -> List[BenchmarkItem]:
    """5 relation-direction queries.

    For each edge, ask a yes/no question that tests *direction* only:
    the answer is determined by knowing which side is the source.
    Pool = AMENDS + APPLIES_TO + cross-doc CROSS_REFERENCES (article->article).
    """
    pool: List[_Edge] = []
    for e in ctx.edges:
        if e.unresolved:
            continue
        if e.kind == "AMENDS" or e.kind == "APPLIES_TO":
            if ":article:" in e.src or ":preamble" in e.src or e.src.endswith(":document"):
                pool.append(e)
        elif e.kind == "CROSS_REFERENCES":
            if (":article:" in e.src and ":article:" in e.dst
                    and _doc_of(e.src) != _doc_of(e.dst)):
                pool.append(e)
    pool.sort(key=lambda e: e.src)
    out: List[BenchmarkItem] = []
    for e in pool:
        src_doc = _doc_of(e.src)
        art_no = _article_num_of(e.src)
        chunk = _chunk_of_article(ctx.chunks, src_doc, art_no, used=used_lids)
        if chunk is None:
            chunk = _chunk_of_article(ctx.chunks, src_doc, used=used_lids)
        if chunk is None:
            continue
        dst_doc = _doc_of(e.dst)
        if e.kind == "AMENDS":
            q = (f"Does {src_doc} (Article {art_no or 'the instrument'}) "
                 f"amend {dst_doc}, or does {dst_doc} amend {src_doc}?")
            direction = "src->dst (amending instrument -> amended provision)"
        elif e.kind == "APPLIES_TO":
            dst_node = ctx.nodes.get(e.dst)
            if dst_node and dst_node.label:
                el = dst_node.label.lower()
            elif ":" in e.dst and len(e.dst.split(":")) >= 3:
                el = e.dst.split(":")[-1].replace("_", " ")
            else:
                continue
            q = (f"Does Article {art_no} of {src_doc} apply to {el}, "
                 f"or does {el} apply to Article {art_no} of {src_doc}?")
            direction = "src->dst (article -> entity)"
        else:  # cross-doc CROSS_REFERENCES
            art_no_dst = _article_num_of(e.dst)
            q = (f"Does Article {art_no} of {src_doc} reference "
                 f"Article {art_no_dst} of {dst_doc}, or does "
                 f"Article {art_no_dst} of {dst_doc} reference "
                 f"Article {art_no} of {src_doc}?")
            direction = "src->dst (referencing article -> referenced article)"
        item = BenchmarkItem(
            query_id="pending", question=q, doc_id=src_doc,
            target_lineage_id=chunk["lineage_id"],
            category="relation_direction",
            reference_answer=chunk["text"],
            difficulty="hard",
            gold_documents=[src_doc, dst_doc] if dst_doc != src_doc else [src_doc],
            gold_chunks=[chunk["lineage_id"]],
            gold_edges=[_edge_dict(e)],
            gold_path=[_edge_dict(e)],
            required_evidence=[chunk["lineage_id"]],
            hop_count=1,
            intended_relation=e.kind,
            intended_direction=direction,
        )
        out.append(item)
        used_lids.add(chunk["lineage_id"])
        if len(out) >= n:
            break
    return out


def _build_temporal(ctx: _BuilderContext, n: int,
                    used_lids: set) -> List[BenchmarkItem]:
    """5 temporal-version queries.

    The AMENDS / SUPERSEDES relation itself encodes newer->older (no
    year-parsing needed; doc-id year conventions in this corpus are
    inconsistent -- ``data_act_2023_2854`` vs ``emd_reform_dir_2024_1711``
    vs ``remit_1227_2011`` -- so any regex is unreliable).

    Each row asks what the current / in-force provision is, given that a
    newer instrument has amended (or superseded) an older one.
    Gold = the dst chunk (the provision being amended/superseded).
    """
    out: List[BenchmarkItem] = []
    amends = [e for e in ctx.edges
              if e.kind == "AMENDS" and not e.unresolved]
    amends.sort(key=lambda e: (e.dst, e.src))
    for e in amends:
        dst_doc = _doc_of(e.dst)
        art_no = _article_num_of(e.dst)
        chunk = _chunk_of_article(ctx.chunks, dst_doc, art_no, used=used_lids)
        if chunk is None:
            chunk = _chunk_of_article(ctx.chunks, dst_doc, used=used_lids)
        if chunk is None:
            continue
        src_doc = _doc_of(e.src)
        q = (f"After {src_doc} amended {dst_doc}, what is the current "
             f"(in-force) provision about "
             f"{art_no or 'that article'} in {dst_doc}?")
        item = BenchmarkItem(
            query_id="pending", question=q, doc_id=dst_doc,
            target_lineage_id=chunk["lineage_id"],
            category="temporal_version",
            reference_answer=chunk["text"],
            difficulty="hard",
            gold_documents=[dst_doc, src_doc],
            gold_chunks=[chunk["lineage_id"]],
            gold_edges=[_edge_dict(e)],
            gold_path=[_edge_dict(e)],
            required_evidence=[chunk["lineage_id"]],
            hop_count=1,
            intended_relation="AMENDS (newer->older)",
            intended_direction="src is the amending instrument, "
                               "dst is the amended provision",
        )
        out.append(item)
        used_lids.add(chunk["lineage_id"])
        if len(out) >= n:
            break
    # fill any remaining slots with SUPERSEDES (in-corpus dst only)
    if len(out) < n:
        sups = [e for e in ctx.edges
                if e.kind == "SUPERSEDES"
                and e.dst and not e.dst.startswith("ext:")]
        sups.sort(key=lambda e: e.src)
        for e in sups:
            if len(out) >= n:
                break
            dst_doc = _doc_of(e.dst)
            chunk = _chunk_of_article(ctx.chunks, dst_doc, used=used_lids)
            if chunk is None:
                continue
            q = (f"What is the current (in-force) version of {dst_doc} "
                 f"after {e.superseded_by or 'a newer instrument'} "
                 f"supersedes it?")
            item = BenchmarkItem(
                query_id="pending", question=q, doc_id=dst_doc,
                target_lineage_id=chunk["lineage_id"],
                category="temporal_version",
                reference_answer=chunk["text"],
                difficulty="hard",
                gold_documents=[dst_doc],
                gold_chunks=[chunk["lineage_id"]],
                gold_edges=[_edge_dict(e)],
                gold_path=[_edge_dict(e)],
                required_evidence=[chunk["lineage_id"]],
                hop_count=1,
                intended_relation="SUPERSEDES (newer->older)",
                intended_direction="src supersedes dst (dst is replaced)",
            )
            out.append(item)
            used_lids.add(chunk["lineage_id"])
    return out[:n]


def _build_graph_distractor(ctx: _BuilderContext, n: int,
                            used_lids: set) -> List[BenchmarkItem]:
    """5 graph-distractor queries.

    Build from a pool of APPLIES_TO and DEFINED_IN edges grouped by
    (doc, article-no).  For each group find a *different* edge whose
    entity/term is not the same.  Ask "does X apply to F, or E?" where
    F is the distractor -- answer is determined by knowing the gold
    APPLIES_TO edge.
    """
    # pool of candidate (src article, primary dst entity, distractor dst)
    pool: List[Tuple[_Edge, _Edge]] = []
    applies = [e for e in ctx.edges
               if e.kind == "APPLIES_TO" and not e.unresolved
               and ":article:" in e.src]
    applies.sort(key=lambda e: e.src)
    # group by src article lid
    by_src: Dict[str, List[_Edge]] = {}
    for e in applies:
        by_src.setdefault(e.src, []).append(e)
    # also: for docs where a single article applies to one entity, find another
    # article in the same doc that applies to a different entity
    by_doc: Dict[str, List[_Edge]] = {}
    for e in applies:
        by_doc.setdefault(_doc_of(e.src), []).append(e)

    for src_lid, es in sorted(by_src.items()):
        if len(es) >= 2:
            for i, e1 in enumerate(es):
                for e2 in es[i + 1:]:
                    if e1.dst != e2.dst:
                        pool.append((e1, e2))
    # fall back to doc-level: pick two edges from same doc with different dsts
    if len(pool) < n * 2:
        for doc, es in sorted(by_doc.items()):
            if len(es) < 2:
                continue
            # pick pairs
            for i in range(len(es)):
                for j in range(i + 1, len(es)):
                    if es[i].dst != es[j].dst:
                        if (es[i], es[j]) not in pool and (es[j], es[i]) not in pool:
                            pool.append((es[i], es[j]))
    # deduplicate
    seen_pairs = set()
    pool_dedup: List[Tuple[_Edge, _Edge]] = []
    for e1, e2 in pool:
        key = (e1.src, e1.dst, e2.dst)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        pool_dedup.append((e1, e2))
    pool = pool_dedup

    out: List[BenchmarkItem] = []
    for e, other in pool:
        src_doc = _doc_of(e.src)
        art_no = _article_num_of(e.src)
        chunk = _chunk_of_article(ctx.chunks, src_doc, art_no, used=used_lids)
        if chunk is None:
            chunk = _chunk_of_article(ctx.chunks, src_doc, used=used_lids)
        if chunk is None:
            continue
        e_node = ctx.nodes.get(e.dst)
        o_node = ctx.nodes.get(other.dst)
        el = (e_node.label.lower() if e_node and e_node.label
              else e.dst.split(":")[-1].replace("_", " "))
        fl = (o_node.label.lower() if o_node and o_node.label
              else other.dst.split(":")[-1].replace("_", " "))
        q = (f"Does Article {art_no} of {src_doc} apply to {fl}, "
             f"or to {el}?")
        item = BenchmarkItem(
            query_id="pending", question=q, doc_id=src_doc,
            target_lineage_id=chunk["lineage_id"],
            category="graph_distractor",
            reference_answer=chunk["text"],
            difficulty="hard",
            gold_documents=[src_doc],
            gold_chunks=[chunk["lineage_id"]],
            gold_entities=[el, fl],
            gold_edges=[_edge_dict(e)],
            gold_path=[_edge_dict(e)],
            required_evidence=[chunk["lineage_id"]],
            hop_count=1,
            intended_relation="APPLIES_TO",
            intended_direction=f"src->dst (applies to {el}, NOT {fl})",
        )
        out.append(item)
        used_lids.add(chunk["lineage_id"])
        if len(out) >= n:
            break
    return out


def _build_multi_doc(ctx: _BuilderContext, n: int,
                    used_lids: set) -> List[BenchmarkItem]:
    """5 multi-document-synthesis queries.

    A CROSS_REFERENCES edge where src and dst are in *different* docs.
    Gold = the src article chunk; the expected answer must combine
    evidence from both documents.  Gold_edges carries both sides.
    """
    cross = [e for e in ctx.edges
             if e.kind == "CROSS_REFERENCES" and not e.unresolved
             and ":article:" in e.src and ":article:" in e.dst
             and _doc_of(e.src) != _doc_of(e.dst)]
    cross.sort(key=lambda e: e.src)
    out: List[BenchmarkItem] = []
    for e in cross:
        src_doc = _doc_of(e.src); dst_doc = _doc_of(e.dst)
        art_no = _article_num_of(e.src)
        chunk = _chunk_of_article(ctx.chunks, src_doc, art_no, used=used_lids)
        if chunk is None:
            chunk = _chunk_of_article(ctx.chunks, src_doc, used=used_lids)
        if chunk is None:
            continue
        q = (f"Combining the reference in Article {art_no} "
             f"of {src_doc} to Article {_article_num_of(e.dst)} of {dst_doc}, "
             f"what is the resulting requirement?")
        item = BenchmarkItem(
            query_id="pending", question=q, doc_id=src_doc,
            target_lineage_id=chunk["lineage_id"],
            category="multi_document_synthesis",
            reference_answer=chunk["text"],
            difficulty="medium",
            gold_documents=[src_doc, dst_doc],
            gold_chunks=[chunk["lineage_id"]],
            gold_edges=[_edge_dict(e)],
            gold_path=[_edge_dict(e)],
            required_evidence=[chunk["lineage_id"]],
            hop_count=1,
        )
        out.append(item)
        used_lids.add(chunk["lineage_id"])
        if len(out) >= n:
            break
    return out


# -- family targets (configurable per doc §19) -------------------------------

QUERY_FAMILY_TARGETS: Dict[str, int] = {
    "single_document": 15,
    "non_relational_semantic": 5,
    "one_hop_relational": 10,
    "two_hop_relational": 10,
    "relation_direction": 5,
    "temporal_version": 5,
    "graph_distractor": 5,
    "multi_document_synthesis": 5,
}


def _build_gold(chunks_dir: Optional[Path] = None,
                graph_dir: Optional[Path] = None) -> List[BenchmarkItem]:
    """Assemble the full 60-item set across the 8 families.

    Family order in the output is deterministic: single_document first,
    then non_relational_semantic, then the graph families.  Within each
    family, edges are sorted by src lid so the set is fully reproducible.
    """
    import common as c  # type: ignore  (resolves NOTEBOOKS_DATA)
    data_dir = Path(c.NOTEBOOKS_DATA) / "graph"
    chunks_dir = Path(c.NOTEBOOKS_DATA) / "ast"
    ctx = _BuilderContext(data_dir, chunks_dir)
    used_lids: set = set()
    out: List[BenchmarkItem] = []

    builders = [
        ("single_document",            _build_single_document),
        ("non_relational_semantic",    _build_non_relational),
        ("one_hop_relational",         _build_one_hop),
        ("two_hop_relational",         _build_two_hop),
        ("relation_direction",         _build_relation_direction),
        ("temporal_version",           _build_temporal),
        ("graph_distractor",           _build_graph_distractor),
        ("multi_document_synthesis",   _build_multi_doc),
    ]
    for fam, fn in builders:
        n = QUERY_FAMILY_TARGETS.get(fam, 0)
        got = fn(ctx, n, used_lids)
        out.extend(got)
        if len(got) < n:
            raise ValueError(
                f"family {fam}: requested {n}, built {len(got)} -- not enough "
                f"verifiable source material. Per doc §2, report the gap and "
                f"reduce the target rather than fabricate.")

    # stable assignment of query_ids in build order (q001..qN)
    for i, it in enumerate(out, start=1):
        it.query_id = f"q{i:03d}"
    # shuffle with fixed seed (doc §10 does the same); the *content* of the
    # set is fixed, only the *ordering* varies for test determinism.
    rnd = random.Random(7)
    rnd.shuffle(out)
    for i, it in enumerate(out, start=1):
        it.query_id = f"q{i:03d}"
    return out


# -- held-out benchmark (spec 08 §4/§16/§20) ----------------------------------
# Additive API: leaves build_benchmark() (the frozen 60-set) unchanged.
# Excludes every target lid from the old (diagnostic) 60 set so the new
# benchmark is genuinely held-out (spec §3).

#: Family ceilings (upper-bound probes) used to construct a balanced set in
#: the 150-200 range (spec §4).  temporal_version is capped at 10 (the
#: corpus ceiling, verified by the builder probe; held-out question-text
#: collisions with the frozen 60-set reduce it to the corpus max of 5,
#: which is recorded as a family shortfall per spec §4).
HELDOUT_FAMILY_TARGETS: Dict[str, int] = {
    "single_document":            50,
    "non_relational_semantic":    20,
    "one_hop_relational":         30,
    "two_hop_relational":         30,
    "relation_direction":         18,
    "temporal_version":           10,    # corpus ceiling
    "graph_distractor":           18,
    "multi_document_synthesis":   20,
}                                         # -> 186 targets, ~174 held out


def build_benchmark_heldout(
        targets: Optional[Dict[str, int]] = None,
        chunks_dir: Optional[Path] = None,
        graph_dir: Optional[Path] = None) -> List[BenchmarkItem]:
    """Deterministic held-out benchmark (spec 08 §4/§16/§20).

    Reuses the existing per-family builders (no new logic, no tuning) but:
      * seeds ``used_lids`` with every target lid of the frozen 60-set so
        no held-out target is a diagnostic one (spec §3, §20);
      * uses a larger per-family ``targets`` dict (default
        ``HELDOUT_FAMILY_TARGETS``) summing to ~165 items, all in the
        150-200 band required by spec §4;
      * runs the same deterministic validator (:func:`validate_item`) and
        assigns fresh query ids ``h001..hN``.

    Pure corpus lookups; no LLM (matches :func:`build_benchmark`).
    """
    import common as c
    t = dict(targets if targets is not None else HELDOUT_FAMILY_TARGETS)

    data_dir = Path(c.NOTEBOOKS_DATA) / "graph"
    if chunks_dir is None:
        chunks_dir = Path(c.NOTEBOOKS_DATA) / "ast"
    ctx = _BuilderContext(data_dir, chunks_dir)

    # Seed used_lids with every old-60 target -> held-out set is disjoint.
    used_lids: set = set()
    old = _build_gold(chunks_dir, graph_dir)      # frozen 60-set
    used_lids.update(it.target_lineage_id for it in old)
    used_lids.update(cid for it in old for cid in it.gold_chunks)
    used_queries = {it.question for it in old}
    # SNAPSHOT: builders add to used_lids as they produce items; if we
    # re-check membership against the live set, every produced item will
    # be found in it and rejected.  Use this fixed snapshot for rejects.
    _excluded_lids = set(used_lids)

    # Also mark the old question *texts* so we can reject a held-out item
    # whose question text matches an old one (near-duplicate check §20).
    # (Question text is a function of article/term, and we already exclude
    # target lids, so duplicates are structurally rare, but we still check.)
    builders = [
        ("single_document",            _build_single_document),
        ("non_relational_semantic",    _build_non_relational),
        ("one_hop_relational",         _build_one_hop),
        ("two_hop_relational",         _build_two_hop),
        ("relation_direction",         _build_relation_direction),
        ("temporal_version",           _build_temporal),
        ("graph_distractor",           _build_graph_distractor),
        ("multi_document_synthesis",   _build_multi_doc),
    ]
    out: List[BenchmarkItem] = []
    all_lids = ctx.all_lids
    edges = ctx.edges
    for fam, fn in builders:
        n = t.get(fam, 0)
        got = []
        for it in fn(ctx, n, used_lids):
            if it.question in used_queries:
                continue
            if it.target_lineage_id in _excluded_lids:
                continue
            errs = validate_item(it, all_lids, edges)
            if errs:
                continue
            got.append(it)
            used_lids.add(it.target_lineage_id)
            if len(got) >= n:
                break
        # Per-family shortfall: report the shortfall rather than fabricate
        # (spec §4 and doc §2 pattern used by _build_gold).
        if len(got) < n:
            if got:
                got[-1].metadata["family_shortfall"] = {fam: n - len(got)}
        out.extend(got)
    if not out:
        raise ValueError("held-out benchmark: no families could be built")

    # fresh held-out query ids (h001..)
    for i, it in enumerate(out, start=1):
        it.query_id = f"h{i:03d}"
    # Shuffle with a fixed seed for determinism (matches _build_gold style).
    rnd = random.Random(13)
    rnd.shuffle(out)
    for i, it in enumerate(out, start=1):
        it.query_id = f"h{i:03d}"
    return out




# -- public API ---------------------------------------------------------------

def build_benchmark(chunks_dir: Optional[Path] = None,
                    graph_dir: Optional[Path] = None
                    ) -> List[BenchmarkItem]:
    """Build the 60-item 8-family benchmark (deterministic)."""
    gold = _build_gold(chunks_dir, graph_dir)
    return gold


def save(items: Sequence[BenchmarkItem], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([i.as_dict() for i in items], f, indent=2, ensure_ascii=False)
    return out_path


def load(items_or_path) -> List[BenchmarkItem]:
    """Accept a path (read json) or a list of dicts/`BenchmarkItem`s."""
    if isinstance(items_or_path, (list, tuple)):
        src = items_or_path
    else:
        with open(items_or_path) as f:
            src = json.load(f)
    if src and isinstance(src, BenchmarkItem):
        return list(src)
    if src and isinstance(src[0], BenchmarkItem):
        return list(src)
    out: List[BenchmarkItem] = []
    for d in src:
        out.append(_item_from_dict(d))
    return out


def _item_from_dict(d: dict) -> "BenchmarkItem":
    """Reconstruct a BenchmarkItem from a serialized dict (shared by
    :func:`load` and :func:`load_jsonl`)."""
    return BenchmarkItem(
        query_id=d["query_id"], question=d["question"],
        doc_id=d.get("doc_id", d["relevant_document_ids"][0]),
        target_lineage_id=d["relevant_chunk_ids"][0],
        category=d.get("category", "unknown"),
        term=d.get("term"), article_title=d.get("article_title"),
        reference_answer=d.get("reference_answer"),
        reference_basis=d.get("reference_basis", ""),
        difficulty=d.get("difficulty", "unknown"),
        metadata=d.get("metadata", {}),
        gold_documents=d.get("gold_documents", []),
        gold_chunks=d.get("gold_chunks", []),
        gold_entities=d.get("gold_entities", []),
        gold_edges=d.get("gold_edges", []),
        gold_path=d.get("gold_path", []),
        hop_count=int(d.get("hop_count", 0)),
        required_evidence=d.get("required_evidence", []),
        intended_relation=d.get("intended_relation"),
        intended_direction=d.get("intended_direction"),
    )


def load_jsonl(path, valid_qids: Optional[Sequence[str]] = None
               ) -> List["BenchmarkItem"]:
    """Read ONE-ITEM-PER-LINE benchmark JSONL (the frozen held-out artifact
    written by ``scripts/build_heldout_benchmark_artifacts.py``) and return
    :class:`BenchmarkItem` objects.

    ``path`` is a JSONL file (newline-delimited), unlike :func:`load` which
    reads a single JSON array document — the two helpers must not be
    confused, the frozen artifact is JSONL.

    If ``valid_qids`` is given, only items whose ``query_id`` is in that set
    are returned (used to restrict the runnable run to the audit-valid
    subset; ``valid_qids=None`` returns ALL lines). The returned list keeps
    file order (the frozen id assignment ``h001..hN``).
    """
    p = Path(path)
    out: List["BenchmarkItem"] = []
    allow = {str(q) for q in valid_qids} if valid_qids is not None else None
    with open(p) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if allow is not None and d["query_id"] not in allow:
                continue
            out.append(_item_from_dict(d))
    return out


def audit_valid_qids(audit_path, set_name: str = "heldout_174",
                     valid_statuses=("valid_exact", "valid_subarticle")
                     ) -> set:
    """Return the set of ``query_id``s the deterministic audit (doc 09 Stage 1
    ``scripts/audit_benchmark.py``) marked valid for ``set_name``.

    Reads ``benchmark_audit.jsonl`` (one record per benchmark item) and
    keeps the items whose ``status`` is in ``valid_statuses``. This is the
    authoritative validity signal: the audit re-derives the gold target from
    the graph/AST and checks answer support, so a corpus-resident target that
    was mis-resolved or is answer-unsupported is still excluded.
    """
    out: set = set()
    with open(audit_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("set") != set_name:
                continue
            if r.get("status") in valid_statuses:
                out.add(r["query_id"])
    return out


def summary(items: Sequence[BenchmarkItem]) -> dict:
    from collections import Counter
    cats = Counter(i.category for i in items)
    docs = Counter(i.doc_id for i in items)
    rel = Counter(i.intended_relation for i in items if i.intended_relation)
    return {
        "n": len(items),
        "category_counts": dict(cats),
        "relation_counts": dict(rel),
        "n_docs": len(docs),
        "docs": dict(sorted(docs.items(), key=lambda kv: -kv[1])[:12]),
        "hop_count_counts": dict(Counter(i.hop_count for i in items)),
    }


# -- legacy alias (for any old import) ---------------------------------------

# The old module exposed _build_gold(r, c) -- keep a thin wrapper so any
# external code that imported it under the old signature still works.  The
# wrapper ignores its (r, c) arguments and just calls the new builder.
def _legacy_build_gold(r=None, c=None) -> List[BenchmarkItem]:  # pragma: no cover - deprecated
    import warnings
    warnings.warn(
        "benchmark._build_gold(r, c) is deprecated; use build_benchmark().",
        DeprecationWarning, stacklevel=2,
    )
    return _build_gold()
