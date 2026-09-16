"""Fetch 12 thin-dropped article bodies from EUR-Lex and synthesize missing chunks.

Root cause: ``chunk_article_based`` builds the ``thin`` list (line 504) but
never merges it into the returned ``chunks`` (line 517). Articles >8 000 chars
are silently dropped. This script fetches the article text from EUR-Lex
(Official Journal) and writes correct chunk JSON entries into the corpus.

12 articles, 35 benchmark items → all become valid.
The 4 ``mica_2023_1114:article:149`` items reference a non-existent article
(MiCA max ≈ 81) and remain excluded.
"""
from __future__ import annotations

import json
import re
import socket
import sys
import html as H
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

REPO   = Path(__file__).resolve().parent.parent
AST_DIR   = REPO / "notebooks/data/ast"
CHANKS = AST_DIR

# doc_id → (eli kind, regulation number, AST file)
DOCS = {
    "data_act_2023_2854":       ("reg", "2023/2854"),
    "dora_2022_2554":           ("reg", "2022/2554"),
    "eidas_2_2024_1183":        ("reg", "2024/1183"),
    "elec_dir_2019_944":        ("dir", "2019/944"),
    "nis2_dir_2022_2555":       ("dir", "2022/2555"),
    "emd_reform_reg_2024_1747": ("reg", "2024/1747"),
    "remit_ii_2024_1106":       ("reg", "2024/1106"),
}

# 12 (doc_id, article_number) pairs that are thin-dropped but exist in the AST
MISSING = [
    ("data_act_2023_2854",       "2"),
    ("dora_2022_2554",           "3"),
    ("dora_2022_2554",           "35"),
    ("eidas_2_2024_1183",        "1"),
    ("eidas_2_2024_1183",        "16"),
    ("eidas_2_2024_1183",        "5a"),
    ("eidas_2_2024_1183",        "46e"),
    ("elec_dir_2019_944",        "2"),
    ("elec_dir_2019_944",        "40"),
    ("nis2_dir_2022_2555",       "46"),
    ("emd_reform_reg_2024_1747", "2"),
    ("remit_ii_2024_1106",       "1"),
]


def slug(art: str) -> str:
    """article number → EUR-Lex URL slug (digits only, no letter)."""
    # EUR-Lex uses art_5a, art_46e etc. — keep the letter
    return art


def clean_html_to_text(html: str) -> str:
    """Convert EUR-Lex article HTML to plain text matching corpus format."""
    # Remove all tags except <br> → newline
    txt = re.sub(r"<br\s*/?>", "\n", html)
    txt = re.sub(r"<p[^>]*>", "", txt)
    txt = re.sub(r"</p>", "\n", txt)
    txt = re.sub(r"<li[^>]*>", "- ", txt)
    txt = re.sub(r"</li>", "\n", txt)
    txt = re.sub(r"<[^>]+>", "", txt)
    txt = H.unescape(txt)
    # Collapse 3+ consecutive newlines
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def fetch_article(doc_id: str, art: str) -> Dict[str, object]:
    kind, reg = DOCS[doc_id]
    art_slug = slug(art).replace(".", "").replace(" ", "")
    url = f"https://eur-lex.europa.eu/eli/{kind}/{reg}/art_{art_slug}/oj/eng"
    print(f"  fetching {url}", file=sys.stderr)
    socket.setdefaulttimeout(30)
    req = urllib.request.Request(url, headers={"User-Agent": "audit/1.0"})
    raw = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")

    # Find <div id="art_N"> ... up to the next <div id="art_ or </section>
    # EUR-Lex uses art_2, art_16, art_5a, art_46e
    art_id = f"art_{art_slug}"
    m = re.search(
        rf'<div[^>]*id="{re.escape(art_id)}"[^>]*>(.*?)(?=<div[^>]*id="art_\w+"|</section>)',
        raw, re.S)
    if m is None:
        # fallback: search for the "oj-ti-art" heading
        m2 = re.search(
            rf'<p[^>]*class="oj-ti-art"[^>]*>\s*Article.{1,20}</p>(.*?)(?=<p[^>]*class="oj-ti-art"|</section>)',
            raw, re.S)
        if m2 is None:
            raise ValueError(f"article {art_slug} not found in {url}")
        art_html = m2.group(0)
    else:
        art_html = m.group(0)

    # extract title (eli-title)
    t = art_html
    title_m = re.search(r'<div class="eli-title"[^>]*>\s*<p[^>]*>(.*?)</p>', t, re.S)
    title = H.unescape(re.sub(r"<[^>]+>", "", title_m.group(1))).strip() if title_m else ""

    body_text = clean_html_to_text(art_html)
    return {"title": title, "body": body_text}


def get_ast_span(doc_id: str, art: str) -> tuple:
    ast = json.loads((AST_DIR / f"{doc_id}_ast.json").read_text(encoding="utf-8"))
    found = None
    def walk(n):
        nonlocal found
        if n.get("type") == "article":
            m = re.search(r"ART-?([a-zA-Z0-9]+)", n.get("identifier", ""))
            if m and m.group(1) == art and found is None:
                found = n
        for c in n.get("children", []) or []:
            walk(c)
    walk(ast)
    if found is None:
        raise ValueError(f"ART-{art} not found in {doc_id} AST")
    return tuple(found["span"]), (found.get("title") or "")


def build_chunk(doc_id: str, art: str, elapsed: Dict[str, dict]) -> dict:
    span, ast_title = get_ast_span(doc_id, art)
    fetched = elapsed.get((doc_id, art))
    if fetched is None:
        fetched = fetch_article(doc_id, art)
    title = fetched["title"] or ast_title
    body  = fetched["body"]

    # match corpus text format: ## _Article N_ \n\n## **Title** \n\n body
    text = f"## _Article {art}_\n\n## **{title}**\n\n{body}"
    lineage = f"{doc_id}:article:{art}"
    node_id = f"ART-{art}"
    return {
        "node_id":    node_id,
        "node_type":  "article",
        "span":       list(span),
        "context": {
            "identifier": node_id,
            "title":      title,
            "n_paras": 1,
            "n_tables": 0,
            "recovered": True,
            "source":  "EUR-Lex",
        },
        "lineage_id": lineage,
        "text":       text,
    }


def main() -> int:
    import collections
    # group by doc for file writes
    chunks_by_doc: Dict[str, List[dict]] = collections.defaultdict(list)
    cached: Dict[tuple, dict] = {}

    for doc_id, art in MISSING:
        try:
            ch = build_chunk(doc_id, art, cached)
            chunks_by_doc[doc_id].append(ch)
            print(f"  OK  {doc_id}:article:{art}  len={len(ch['text'])} chars")
        except Exception as e:
            print(f"  FAIL {doc_id}:article:{art}  {e}", file=sys.stderr)

    # --- write chunks to existing chunk files (append new entries) ---
    for doc_id, chunks in chunks_by_doc.items():
        path = CHANKS / f"chunks_{doc_id}.json"
        if not path.exists():
            print(f"WARNING: {path} not found, creating", file=sys.stderr)
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        existing_lids = {c.get("lineage_id") for c in existing}
        added = 0
        for ch in chunks:
            if ch["lineage_id"] not in existing_lids:
                existing.append(ch)
                added += 1
        path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {added} chunks -> {path}  (total {len(existing)})")

    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
