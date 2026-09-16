"""Materialize the 12 thin-dropped article-level chunks from the source MDs.

Root cause: `chunk_article_based` builds the `thin` paragraph stream for
articles >8000 chars (ast_builder.py:504) but never appends it to the
returned `chunks` (line 517), so every >8k-char article had NO chunk.

Fix (corpus repair): for each of the 12 affected articles, slice
`source_md[span[0]:span[1]]` (span comes from the existing AST) and emit an
article-level chunk with the article-level lineage the frozen benchmark
targets.  Text == source[span].strip()  (the corpus invariant).

The 4 `mica_2023_1114:article:149` items are NOT touched: MiCA ends at
article 81, so that provision does not exist.

Deterministic, no LLM.  Appends to the existing chunks_<doc>.json only if the
lineage is not already present (idempotent).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
AST_DIR = REPO / "notebooks/data/ast"


def find_doc_md(doc_id: str) -> Path:
    for root, _, files in os.walk(REPO / "notebooks/data/processed/eu"):
        for f in files:
            if f == f"{doc_id}.md":
                return Path(root) / f
    raise FileNotFoundError(doc_id)


def find_article(ast: dict, art: str) -> Optional[dict]:
    def walk(n: dict) -> Optional[dict]:
        if n.get("type") == "article":
            m = re.search(r"ART-?([a-zA-Z0-9]+)", n.get("identifier", ""))
            if m and m.group(1) == art:
                return n
        for c in n.get("children", []) or []:
            r = walk(c)
            if r:
                return r
        return None
    return walk(ast)


def materialize(doc_id: str, art: str) -> dict:
    md = find_doc_md(doc_id)
    content = md.read_text(encoding="utf-8")
    ast = json.loads((AST_DIR / f"{doc_id}_ast.json").read_text(encoding="utf-8"))
    node = find_article(ast, art)
    if node is None:
        raise ValueError(f"{doc_id}:article:{art} not in AST")
    s, e = node["span"]
    text = content[s:e].strip()
    if not text:
        raise ValueError(f"{doc_id}:article:{art} slice is empty")
    identifier = node["identifier"]
    n_paras = sum(1 for c in node.get("children", []) or [] if c.get("type") == "paragraph")
    n_tables = sum(1 for c in node.get("children", []) or [] if c.get("type") == "table")
    return {
        "node_id": identifier,
        "node_type": "article",
        "span": [s, e],
        "context": {
            "identifier": identifier,
            "title": node.get("title"),
            "n_paras": n_paras,
            "n_tables": n_tables,
            "recovered": "source_span",   # flag for the audit/report
        },
        "lineage_id": f"{doc_id}:article:{art}",
        "text": text,
    }


def main() -> int:
    targets: List[Tuple[str, str]] = [
        ("data_act_2023_2854", "2"),
        ("dora_2022_2554", "3"),
        ("dora_2022_2554", "35"),
        ("eidas_2_2024_1183", "1"),
        ("eidas_2_2024_1183", "16"),
        ("eidas_2_2024_1183", "5a"),
        ("eidas_2_2024_1183", "46e"),
        ("elec_dir_2019_944", "2"),
        ("elec_dir_2019_944", "40"),
        ("nis2_dir_2022_2555", "46"),
        ("emd_reform_reg_2024_1747", "2"),
        ("remit_ii_2024_1106", "1"),
    ]
    # group by doc so each file is read/written once
    by_doc: Dict[str, List[str]] = {}
    for d, a in targets:
        by_doc.setdefault(d, []).append(a)

    for doc_id, arts in sorted(by_doc.items()):
        path = AST_DIR / f"chunks_{doc_id}.json"
        existing = json.loads(path.read_text(encoding="utf-8"))
        present = {c.get("lineage_id") for c in existing}
        added = 0
        print(f"== {doc_id} ==")
        for art in arts:
            lid = f"{doc_id}:article:{art}"
            if lid in present:
                print(f"  skip {lid} (already present)")
                continue
            ch = materialize(doc_id, art)
            existing.append(ch)
            added += 1
            print(f"  +  {lid}  {len(ch['text']):>6} chars  (paras={ch['context']['n_paras']}, tables={ch['context']['n_tables']})")
        if added:
            path.write_text(json.dumps(existing, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"  -> {path.name} total {len(existing)} chunks")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
