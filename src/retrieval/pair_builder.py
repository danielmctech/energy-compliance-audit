"""Compliance-pair builder for the dense-encoder LoRA stages.

For every titled article we emit ONE training pair:

  query      = templated compliance question (rotating across templates)
  positive   = the article's chunk text (03b)
  negatives  = sibling article chunks from the SAME document (hard: same
               instrument & vocabulary, different section), topped up with
               random article chunks from OTHER documents when needed

Plus, for every DEFINED_IN edge whose target chunk we can anchor:

  query      = "What does '<term>' mean in <instrument>?"
  positive   = the chunk carrying the definition
  negatives  = sibling article chunks of that instrument

Stage 1 trains on these. Stage 2 re-uses them and additionally mines
hard negatives from the base encoder's own wrong top-k answers (see
finetune_notebook).

Schema (one line per entry):

  {
    "pair_id": "remit_1227_2011:article:4",
    "query": "...",
    "positive": {"lineage_id": "...", "text": "..."},
    "negatives": [{"lineage_id": "...", "text": "..."}, ...],
    "kind": "article" | "term",
    "doc_id": "remit_1227_2011"
  }
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ._corpus import load_corpus
from . import _corpus as _c

_TEMPLATES = (
    "What are the obligations in {title} of {doc}?",
    "Summarise the requirements of {title} under {doc}.",
    "Who must comply with {title} in {doc}?",
    "Which rules apply per {title} of {doc}?",
    "{title} in {doc}, what exactly does it require?",
    "Give the compliance checklist for {title} ({doc}).",
)

_SEED = 2024
# MiniLM is a 256-token encoder (~1000 chars) -- keep texts within budget
_MAX_TEXT = 1000


@dataclass
class Pair:
    pair_id: str
    query: str
    positive: dict
    negatives: List[dict]
    kind: str
    doc_id: str

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "query": self.query,
            "positive": self.positive,
            "negatives": self.negatives,
            "kind": self.kind,
            "doc_id": self.doc_id,
        }


def _article_number(chunk) -> Optional[int]:
    if ":article:" in chunk.lineage_id:
        try:
            return int(chunk.lineage_id.rsplit(":", 1)[-1])
        except ValueError:
            return None
    return None


def stable_template_index(lineage_id: str) -> int:
    """Deterministic template pick (independent of PYTHONHASHSEED)."""
    h = 0
    for ch in lineage_id:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h % len(_TEMPLATES)


def _make_negatives(article_pool: List, own_lid: str, n_neg: int,
                    corpus, doc_id: str, rng: random.Random) -> List[dict]:
    out: List[dict] = []
    used = {own_lid}
    for sib in article_pool:
        if sib.lineage_id in used:
            continue
        out.append({"lineage_id": sib.lineage_id,
                    "text": sib.text[:_MAX_TEXT]})
        used.add(sib.lineage_id)
        if len(out) >= n_neg:
            return out
    # top up with cross-doc articles (different instrument => easier negs)
    others = [ch for ch in corpus["chunks"]
              if ch.doc_id != doc_id and ch.node_type == "article"]
    rng.shuffle(others)
    for ch in others:
        if ch.lineage_id in used:
            continue
        out.append({"lineage_id": ch.lineage_id,
                    "text": ch.text[:_MAX_TEXT]})
        used.add(ch.lineage_id)
        if len(out) >= n_neg:
            break
    return out


def build_pairs(
    chunks_dir: Optional[Path] = None,
    graph_dir: Optional[Path] = None,
    n_neg: int = 4,
    min_title_len: int = 10,
    max_per_doc: int = 25,
    seed: int = _SEED,
) -> List[Pair]:
    rng = random.Random(seed)
    corpus = load_corpus(chunks_dir, graph_dir)
    graph = corpus["graph"]
    nodes: Dict[str, dict] = graph.nodes

    # article chunks grouped by doc, in article-number order
    by_doc: Dict[str, list] = defaultdict(list)
    for c in corpus["chunks"]:
        if c.node_type == "article":
            by_doc[c.doc_id].append(c)
    for doc in by_doc:
        by_doc[doc].sort(key=lambda x: _article_number(x) or 0)

    pairs: List[Pair] = []

    # --- article pairs (title from the graph node) ----------------------
    for doc, arts in by_doc.items():
        arts = arts[: max_per_doc * 2]
        pool = arts
        for c in arts:
            node = nodes.get(c.lineage_id)
            if not node:
                continue
            title = (node.get("title") or "").strip()
            if len(title) < min_title_len:
                continue  # "Article 1" style bare heading -> skip
            tpl_idx = stable_template_index(c.lineage_id)
            query = _TEMPLATES[tpl_idx].format(title=title, doc=doc)
            negs = _make_negatives(pool, c.lineage_id, n_neg, corpus, doc, rng)
            pairs.append(Pair(
                pair_id=c.lineage_id,
                query=query,
                positive={"lineage_id": c.lineage_id,
                          "text": c.text[:_MAX_TEXT]},
                negatives=negs,
                kind="article",
                doc_id=doc,
            ))

    # --- term-definition pairs (from DEFINED_IN edges on 04) ------------
    gdir = graph_dir or _c.c.NOTEBOOKS_DATA / "graph"
    if (gdir / "edges.jsonl").exists():
        edges = [json.loads(l) for l in
                 (gdir / "edges.jsonl").read_text().splitlines()
                 if l.strip()]
    else:
        edges = []

    for e in edges:
        if e["kind"] != "DEFINED_IN" or e.get("unresolved"):
            continue
        term = e.get("term")
        if not term or len(term) < 4:
            continue
        doc = e["doc_id"]
        target = _anchored_chunk(corpus, doc, term)
        if target is None:
            continue
        negs = _make_negatives(by_doc.get(doc, []),
                               target.lineage_id, n_neg, corpus, doc, rng)
        pairs.append(Pair(
            pair_id=f"{doc}:term:{term}",
            query=f"What does '{term}' mean in {doc}?",
            positive={"lineage_id": target.lineage_id,
                      "text": target.text[:_MAX_TEXT]},
            negatives=negs,
            kind="term",
            doc_id=doc,
        ))

    rng.shuffle(pairs)
    return pairs


def _anchored_chunk(corpus, doc_id: str, term: str):
    """Best chunk carrying the definition of `term` in `doc_id`."""
    tlow = term.lower()

    def _is_def(ch) -> bool:
        tl = ch.text.lower()
        if tlow not in tl:
            return False
        return ("means " in tl or "definition of " in tl
                or "shall be defined" in tl)

    for ch in corpus["chunks"]:
        if ch.doc_id != doc_id or ch.node_type not in ("sentence", "article"):
            continue
        if _is_def(ch) and len(ch.text) < 1500:
            return ch
    cands = [ch for ch in corpus["chunks"]
             if ch.doc_id == doc_id and tlow in ch.text.lower()
             and ch.node_type in ("sentence", "article")]
    if not cands:
        return None
    return min(cands, key=lambda ch: len(ch.text))


def save_pairs(pairs: List[Pair], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "pairs_stage1.jsonl"
    with p.open("w") as f:
        for pr in pairs:
            f.write(json.dumps(pr.to_dict()) + "\n")
    return p


def summary(pairs: List[Pair]) -> dict:
    by_kind = defaultdict(int)
    by_doc = defaultdict(int)
    for p in pairs:
        by_kind[p.kind] += 1
        by_doc[p.doc_id] += 1
    return {
        "pairs": len(pairs),
        "by_kind": dict(by_kind),
        "by_doc_top10": dict(sorted(by_doc.items(), key=lambda kv: -kv[1])[:10]),
        "mean_negs": sum(len(p.negatives) for p in pairs) / max(1, len(pairs)),
    }
