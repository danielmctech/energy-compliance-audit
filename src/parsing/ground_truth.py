"""
ground-truth generation for the 02 parser evaluation.

two layers:
  1. heuristic  -- fast, deterministic, no model calls. one window per field
                   group per doc. every generated value is tagged
                   "confidence": "heuristic".
  2. llm review -- the configured reasoner model reviews each doc's ground
                   truth and fixes obvious heuristic errors (wrong titles,
                   missing article numbers, wrong dates), producing
                   ground_truth_all_docs_v2.json. cached per (model, doc).

schema (per doc):
{
  "doc_id": str, "category": str, "source_md": str,
  "windows": [
     {"window": 0, "text_offset": [start,end], "fields_found": {...}, "tables": {...}, "structure_elements": {...}}
  ],
  "global": {"document_title": str|null, "instrument": str|null, "article_numbers": [..],
             "publication_date": str|null, "celex": str|null},
  "review": {"model": str|null, "applied": bool, "changes": [..]}
}
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import common as c
    from parsing import llm_extractor
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    import common as c
    from parsing import llm_extractor

DATE_PATTERNS = [
    r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"(?<!\d)\d{1,2}\.\d{1,2}\.\d{4}(?!\d)",
]


# ---------------------------------------------------------------------------
# window sampling -- replaces the broken content.split('\f') logic
# ---------------------------------------------------------------------------

def sample_windows(content: str, n_windows: int = 6, window_chars: int = 2500) -> list[dict]:
    """
    n_windows non-overlapping text windows: first two (header/title/celex zone),
    rest evenly spaced, last one (final articles / annexes).
    """
    L = len(content)
    if L == 0:
        return []
    positions = []
    if n_windows <= 2:
        positions = [0, max(1, L - window_chars)]
    else:
        # first 2 windows near the head
        positions.append(0)
        positions.append(min(window_chars, max(1, L // max(1, n_windows // 2))))
        # last window near the tail
        positions.append(max(0, L - window_chars))
        # fill the middle evenly
        n_mid = n_windows - 3
        if n_mid > 0:
            lo, hi = positions[1], L - window_chars
            for k in range(1, n_mid + 1):
                positions.append(int(lo + (hi - lo) * k / (n_mid + 1)))
        positions = sorted(set(max(0, min(p, L - window_chars)) for p in positions))[:n_windows]
        if positions and positions[-1] != max(0, L - window_chars):
            positions[-1] = max(0, L - window_chars)

    out = []
    for p in positions:
        seg = content[p : p + window_chars]
        out.append({
            "window": len(out),
            "text_offset": [p, min(L, p + window_chars)],
            "text_preview": seg[:120].replace("\n", " "),
        })
    return out


# ---------------------------------------------------------------------------
# heuristic field extraction (same heuristics the old notebook had)
# ---------------------------------------------------------------------------

def _extract_title_heuristic(text: str) -> str:
    # title = short-ish line (<=160 chars) naming the doc, not its summary
    # skip OJ front-matter lines (Official Journal, page refs, language codes).
    skip = re.compile(
        r"official journal|^\s*L\s*\d+/\d+\b|\bEN\b|\bFR\b|\bDE\b|legislative acts"
        r"|^\s*[IIVX]+\s*$|^\d{1,2}[./]\d{1,2}[./]\d{4}\b|abstract|index terms"
        r"|^\s*(REGULATIONS|DIRECTIVES|DIRECTIVE ACTS|NETWORK CODES|ANNEXES?)\s*$"
        r"|^\s*of\s+\d{1,2}\s", re.I)
    # a heading that names an instrument (contains yyyy/nnnn) is the strongest title signal
    instrument_heading = re.compile(
        r"\b(REGULATION|DIRECTIVE|DECISION)[\s*_]?\(?[ ]?(EU|EC|EURATOM)?[ )_]*"
        r"(?:No[.\s_]*)?(\d{4})\s*/\s*(\d+)", re.I)
    fallback_heading = None
    for line in text.splitlines()[:20]:
        hm = re.match(r"^#{1,6}\s+(.+)$", line)
        if not hm:
            continue
        stripped = hm.group(1).strip().strip("*_ |").strip()
        if not (10 <= len(stripped) <= 200) or skip.search(stripped):
            continue
        if instrument_heading.search(stripped):
            return stripped
        if fallback_heading is None:
            fallback_heading = stripped
    for line in text.splitlines()[:20]:
        stripped = line.strip().strip("#* _|").strip()
        if not (15 <= len(stripped) <= 160):
            continue
        if skip.search(stripped):
            continue
        if re.search(r"\b(REGULATION|DIRECTIVE|DECISION)\s*\(?(EU|EC|EURATOM)\)?", stripped, re.I):
            return stripped
    if fallback_heading:
        return fallback_heading
    # any decent short line that isn't OJ boilerplate
    for line in text.splitlines()[:20]:
        stripped = line.strip().strip("#* _|").strip()
        if 25 <= len(stripped) <= 160 and not re.search(r"\.\s", stripped) and not skip.search(stripped):
            return stripped
    return None


def _extract_article_numbers(text: str) -> list[str]:
    return sorted(set(re.findall(r"\bArticle\s+(\d+[a-z]?)\b", text)), key=lambda x: (len(x), x))


def _extract_dates(text: str) -> list[str]:
    dates = []
    for pat in DATE_PATTERNS:
        dates.extend(re.findall(pat, text))
    return list(dict.fromkeys(dates))[:8]


def _extract_definitions(text: str) -> list[str]:
    defs = []
    for pat in (r'"([^"]+)"\s+means\b', r"'([^']+)'[\s]*means\b", r"\u2018([^\u2019]+)\u2019\s+means\b"):
        defs.extend(re.findall(pat, text, re.I))
    # also unquoted " <term> means " style definitions
    defs.extend(re.findall(r"\b([A-Z][\w\- ]{2,40})\s+means\b", text))
    out = [d.strip() for d in defs if 3 < len(d) < 60]
    return list(dict.fromkeys(out))[:6]


def _count_tables(text: str) -> int:
    seps = re.findall(r"^\|[\s\-:|]+\|\s*$", text, re.M)
    pipeblocks = sum(1 for ln in text.splitlines() if ln.count("|") >= 3)
    return max(len(seps), pipeblocks // 3)


def _table_headers(text: str) -> list[str]:
    out = []
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.count("|") >= 3 and i + 1 < len(lines) and re.match(r"^\s*\|[\s\-:|]+\|\s*$", lines[i + 1]):
            cells = [c.strip() for c in ln.strip("|").split("|") if c.strip()]
            if cells:
                out.append(" | ".join(cells))
    return out[:3]


def _headings(text: str) -> list[str]:
    return [m.group(2).strip() for m in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.M)][:10]


def _numbered_items(text: str) -> list[str]:
    return re.findall(r"^\s*(?:\d+\.|\d+|[a-z]\)|[A-Z]\))\s.+$", text, re.M)[:10]


def _paragraph_count(text: str) -> int:
    return len([p for p in text.split("\n\n") if len(p.strip()) > 50])


def _obligation_count(text: str) -> int:
    return len(re.findall(r"\b(shall|must|is\s+required\s+to)\b", text, re.I))


def _derogation_count(text: str) -> int:
    return len(re.findall(r"\b(notwithstanding|except\s+(where|when|if|as)|without\s+prejudice\s+to)\b", text, re.I))


def _instrument(text: str) -> str | None:
    """Find the primary EU instrument (Regulation/Directive/Decision + number).
    Tolerates markdown decoration (##, **, __, underscores, spaces)."""
    # primary instrument: kind (region) yyyy/nnnn   (optionally yyyy/nnnn/EC)
    m = re.search(
        r"(Regulation|Directive|Decision)[*_ ]*\(?[ ]*(EU|EC|EURATOM)[*_ ]*\)?[*_ ]*"
        r"(?:No[._ ]*)?[*_ ]*(\d{4})[ ]*/[ ]*(\d+)(?:(\s*/\s*(EC|EEC)))?",
        text, re.I)
    if m:
        kind, region, year, num, _, tail = m.groups()
        tail_str = f"/{tail}" if tail else ""
        region_str = f" ({region})" if region else ""
        return f"{kind}{region_str} {year}/{num}{tail_str}"
    # fallback: bare "Regulation No 910/2014"
    m = re.search(r"(Regulation|Directive|Decision)[*_ ]*No?[._ ]*[*_ ]*(\d{4})[\s/]+(\d+)", text, re.I)
    if m:
        return f"{m.group(1)} {m.group(2)}/{m.group(3)}"
    return None


def _publication_date(text: str) -> str | None:
    dates = _extract_dates(text)
    return dates[0] if dates else None


def _celex(text: str) -> str | None:
    m = re.search(r"\b3\d{10}\b", text)
    return m.group(0) if m else None


def heuristic_ground_truth_for_doc(md_path: Path, category: str,
                                   n_windows: int = 6, window_chars: int = 2500) -> dict:
    content = md_path.read_text(encoding="utf-8")
    doc = {
        "doc_id": md_path.stem,
        "category": category,
        "source_md": c.rel_to_root(md_path),
        "generated_by": "heuristic v2",
        "generated": datetime.now().isoformat(),
        "confidence": "heuristic",
        "content_chars": len(content),
        "windows": [],
        "global": {
            "document_title": _extract_title_heuristic(content[:6000])
            or _extract_title_heuristic(content[:1200]),
            "instrument": _instrument(content[:8000]),
            "article_numbers": _extract_article_numbers(content),
            "publication_date": _publication_date(content[:8000]),
            "celex": _celex(content[:8000]),
            "obligations_count": _obligation_count(content),
            "tables_count": _count_tables(content),
        },
    }
    for w in sample_windows(content, n_windows, window_chars):
        seg = content[w["text_offset"][0] : w["text_offset"][1]]
        doc["windows"].append({
            "window": w["window"],
            "text_offset": w["text_offset"],
            "fields_found": {
                "document_title": _extract_title_heuristic(seg),
                "article_numbers": _extract_article_numbers(seg),
                "dates": _extract_dates(seg),
                "definitions": _extract_definitions(seg),
            },
            "tables": {"count": _count_tables(seg), "headers": _table_headers(seg)},
            "structure_elements": {
                "headings": _headings(seg),
                "numbered_lists": _numbered_items(seg),
                "paragraphs_count": _paragraph_count(seg),
                "obligation_count": _obligation_count(seg),
                "derogation_count": _derogation_count(seg),
            },
        })
    return doc


def build_ground_truth(docs: list[Path] | None = None,
                       n_windows: int = 6) -> dict:
    docs = docs or c.discover_markdown()
    ground = {
        "_metadata": {
            "generated": datetime.now().isoformat(),
            "generator": "heuristic v2 (window sampling)",
            "source_md_files": len(docs),
            "n_windows_per_doc": n_windows,
            "note": "values are regex-derived, tagged confidence=heuristic",
        },
        "documents": {},
    }
    for md in sorted(docs):
        cat = c.doc_type_for(md)
        g = heuristic_ground_truth_for_doc(md, cat, n_windows=n_windows)
        ground["documents"][g["doc_id"]] = g
    return ground


# ---------------------------------------------------------------------------
# LLM review layer
# ---------------------------------------------------------------------------

REVIEW_PROMPT = """You are reviewing a machine-generated ground-truth record
for an EU legal document. The record was made by regex heuristics and may be
partially wrong (titles, article numbers, dates, CELEX).

Using ONLY the document excerpt provided, produce a CORRECTED version of the
"global" block as JSON with exactly these keys:
  document_title (string|null), instrument (string|null), celex (string|null),
  publication_date (YYYY-MM-DD|null), article_numbers (list of strings),
  changes (list of strings, each describing one correction)

Rules: only values present in the excerpt; null when absent; valid JSON only.
Note: "article_numbers" in the original record was derived from the FULL document;
the excerpt below is only the first part. Keep article_numbers unless they are
clearly wrong. Keep other original keys (obligations_count, tables_count) unchanged.

ORIGINAL GLOBAL:
@@ORIGINAL_GLOBAL@@

DOCUMENT EXCERPT:
@@EXCERPT@@"""

REVIEW_SYSTEM = "You are an EU legal-document data-quality reviewer. Respond with valid JSON only."


def llm_review_ground_truth(ground: dict, model: str = None, max_docs: int = None) -> dict:
    """
    run llm review over every doc in `ground`, updating each doc's
    "global" block and recording the reviewer + changes list.
    returns a NEW dict (does not mutate input) with all fields from input plus
    review metadata.
    """
    model = model or c.env_models()["reasoner_default"]
    docs = ground.get("documents", {})
    items = list(docs.items())
    if max_docs:
        items = items[:max_docs]

    reviewed = {
        "_metadata": dict(ground.get("_metadata", {}),
                          review_model=model,
                          review_generated=datetime.now().isoformat()),
        "documents": {},
    }

    for doc_id, doc in items:
        new_doc = dict(doc)
        excerpt = c.resolve_repo_path(doc["source_md"]).read_text(encoding="utf-8")[:9000]
        try:
            parsed = _review_call(excerpt, doc.get("global", {}), model)
            changes = parsed.pop("changes", [])
            new_global = dict(doc.get("global", {}))
            # merge: the model's non-empty values win, but empty/null values
            # fall back to the heuristic original (guards against the model
            # dropping keys it considered out of scope)
            for key, value in parsed.items():
                if value in (None, "", []):
                    continue
                new_global[key] = value
            new_doc["global"] = new_global
            new_doc["confidence"] = "llm_reviewed"
            new_doc["review"] = {"model": model, "applied": True, "changes": changes}
        except Exception as e:  # noqa: BLE001
            new_doc["confidence"] = "heuristic"
            new_doc["review"] = {"model": model, "applied": False, "error": str(e)}
        reviewed["documents"][doc_id] = new_doc

    return reviewed


def _review_call(excerpt: str, original_global: dict, model: str) -> dict:
    prompt = REVIEW_PROMPT.replace(
        "@@ORIGINAL_GLOBAL@@", json.dumps(original_global, ensure_ascii=False)
    ).replace("@@EXCERPT@@", excerpt)
    raw = c.ollama_chat(
        model,
        prompt,
        system=REVIEW_SYSTEM,
        temperature=0.0,
        num_ctx=32768,
        num_predict=8192,
        max_retries=3,
        use_cache=True,
        cache_key=c.stable_hash("rev", model, excerpt[:4000], json.dumps(original_global, sort_keys=True)),
    )
    return c.extract_json(raw)


def save_ground_truth(ground: dict, name: str) -> Path:
    c.ensure_dirs()
    path = c.GROUND_TRUTH_PATH / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ground, f, indent=2, ensure_ascii=False)
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=6)
    ap.add_argument("--review-model", default=None)
    ap.add_argument("--no-review", action="store_true")
    args = ap.parse_args()

    m = c.env_models()
    print("building heuristic ground truth for all docs ...")
    g = build_ground_truth(n_windows=args.n_windows)
    p1 = save_ground_truth(g, "ground_truth_all_docs.json")
    print("saved:", p1, f"({len(g['documents'])} docs)")

    if not args.no_review:
        model = args.review_model or m["reasoner_default"]
        print(f"running llm review with {model} ...")
        g2 = llm_review_ground_truth(g, model=model)
        p2 = save_ground_truth(g2, "ground_truth_all_docs_v2.json")
        print("saved:", p2)
