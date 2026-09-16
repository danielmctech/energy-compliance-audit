"""
03a structure maps.

Deterministic, regex-only structure extraction over the shared processed
markdown (data/processed/eu/**/*.md). No LLM, no cache, idempotent.

Design (inherited from notebook 02):
- the markdown comes from the *best text parser* (pymupdf4llm), which scored
  1.0 on structure preservation in 02's evaluation;
- a `## _Article N_` heading is a *structural* article; `Article 14 of
  Directive ...` inside prose is a *mention* (counted separately);
- GT helper regexes (title / instrument / tables / obligations) come from
  src/ground_truth.py so 03a and 02 can never disagree on those fields.

Outputs per doc: articles (number, title, paragraph count, mentions),
chapters/titles/parts, recitals, tables, energy units, obligations,
derogations, and a chunking-strategy recommendation for 03b.
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
    from parsing import ground_truth as GT
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    import common as c
    from parsing import ground_truth as GT

MAX_CHARS_MAP = 400_000

# --- structure patterns (all MULTILINE, line-anchored) ----------------------

# structural article heading: `## _Article 2_`, `## **Article 2 - Controller
# processing**`, `_Article 5_`, `## _Article 30(3)_`, `**Article 9**`.
# Optional `(n)` after the number = a sub-paragraph repeal of the SAME article.
# A candidate is kept only if it is *marked* (`#` heading or `*`/`_` wrap);
# bare prose ("Article 3." / "Article 40 – ...") is rejected.
ARTICLE_HEAD = re.compile(
    r"^[ \t]*"
    r"(?P<hash>#{1,6})?[ \t]*"
    r"(?P<open>[*_`]{1,4}[ \t]*)?"
    r"[Aa]rticle[ \t]+(?P<num>\d+[a-z]?)(?:\(\d+\))?(?![0-9a-z])"
    r"(?:[ \t]*[-\u2013\u2014:][ \t]+(?P<title>[^\n]{3,160}?))?"
    r"(?P<close>(?:\*{1,3}|_{1,3})(?!\w)[ \t]*)?$",
    re.IGNORECASE | re.MULTILINE,
)


def _is_marked(m) -> bool:
    """a candidate article line is structural only if markdown-marked."""
    return bool(m.group("hash")) or bool(m.group("open")) or bool(m.group("close"))

HEADING = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

PARA_LETTERED = re.compile(r"^[ \t]*(?:[-*][ \t]*)?\([a-z]\)[ \t]*\S", re.MULTILINE)
PARA_NUMBERED = re.compile(r"^[ \t]*\d+[.)][ \t]*\S", re.MULTILINE)

CHAP_OR_TITLE = re.compile(
    r"^[ \t]*#{0,6}[ \t]*(CHAPTER|TITLE|PART)[ \t]+([IVX]+|\d+|[A-Z])\b",
    re.IGNORECASE | re.MULTILINE,
)

RECITAL_START = re.compile(r"^[ \t]*(?:[-*][ \t]*)?\((\d+)\)[ \t]+[A-Z\"'\u2018(]", re.MULTILINE)

TABLE_ROW = re.compile(r"^\|.+\|[ \t]*$")


# --- dataclasses -------------------------------------------------------------

@dataclass
class ArticleSpan:
    number: str
    title: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    span_end: int = 0
    level: int = 1
    para_count: int = 0
    mentions: int = 0


@dataclass
class StructureMap:
    doc_id: str
    category: str
    source_md: str
    generated: str
    generator: str
    content_chars: int
    words: int
    document_title: Optional[str]
    instrument: Optional[str]
    celex: Optional[str]
    publication_date: Optional[str]
    article_count: int
    articles: List[dict] = field(default_factory=list)
    chapters: List[dict] = field(default_factory=list)
    recitals_count: int = 0
    tables: dict = field(default_factory=dict)
    paragraph_stats: dict = field(default_factory=dict)
    obligations: dict = field(default_factory=dict)
    derogations: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    chunking_strategy: str = "sentence_512"
    recommended_chunk_targets: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --- detectors ---------------------------------------------------------------

def find_structural_articles(content: str) -> List[ArticleSpan]:
    out: List[ArticleSpan] = []
    for m in ARTICLE_HEAD.finditer(content):
        if not _is_marked(m):
            continue
        out.append(ArticleSpan(
            number=m.group("num"),
            title=(m.group("title") or "").strip() or None,
            line_start=content.rfind("\n", 0, m.start()) + 1,
            line_end=m.end(),
            level=len(m.group("hash")) if m.group("hash") else 1,
        ))
    return out


def find_chapters(content: str) -> List[dict]:
    out = []
    for m in CHAP_OR_TITLE.finditer(content):
        tail = " ".join(content[m.end(): m.end() + 160].split())[:140]
        out.append({
            "kind": m.group(1).upper(),
            "num": m.group(2),
            "title": tail,
            "pos": m.start(),
        })
    return out


def find_recitals(content: str, cap: int = 500) -> List[int]:
    nums = [int(m.group(1)) for m in RECITAL_START.finditer(content)]
    return nums[:cap]


def find_tables(content: str) -> List[dict]:
    """group consecutive `|...|` blocks into table spans."""
    blocks = []
    i, lines = 0, content.splitlines()
    while i < len(lines):
        if TABLE_ROW.match(lines[i]) and i + 1 < len(lines) and re.match(r"^\|[\s\-:|]+\|[ \t]*$", lines[i + 1]):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            blocks.append({
                "start_line": i + 1,
                "end_line": j,
                "rows": j - i - 1,
                "cols": max(len(x.strip("| ").split("|")) for x in lines[i:j]),
            })
            i = j
        else:
            i += 1
    return blocks


UNIT_PATTERNS = {
    "power": re.compile(r"\b(MW|GW|kW)\b"),
    "energy": re.compile(r"\b(MWh|GWh|GJ)\b"),
    "price": re.compile(r"\b(EUR|€)\s*/\s*(MWh|GJ|kWh)\b"),
}


def find_units(content: str) -> dict:
    return {cat: len(p.findall(content)) for cat, p in UNIT_PATTERNS.items()}


# --- map builder -------------------------------------------------------------

def _article_title_from_headings(content: str, span_from: int, span_to: int, lvl: int) -> Optional[str]:
    """first non-structural heading inside the article span -> its text."""
    if span_to - span_from > 4000:  # don't hunt across a huge annex
        return None
    for m in HEADING.finditer(content, span_from, span_to):
        if len(m.group(1)) < lvl:
            continue
        text = m.group(2).strip()
        if re.search(r"\bArticle\s+\d+|\bCHAPTER\s+[IVX]+|\bTITLE\s+[IVX]+|\bANNEX\b|\(\d+\)\s", text, re.I):
            continue
        clean = re.sub(r"[\*_`#]", " ", text)
        clean = " ".join(clean.split())
        return clean[:160] if clean else None
    return None


def count_paragraphs(seg: str) -> int:
    return len(PARA_LETTERED.findall(seg)) + len(PARA_NUMBERED.findall(seg))


def build_structure_map(md_path: Path, category: Optional[str] = None) -> StructureMap:
    content = md_path.read_text(encoding="utf-8")[:MAX_CHARS_MAP]
    n_words = max(len(content.split()), 1)
    category = category or c.doc_type_for(md_path)

    arts = find_structural_articles(content)
    anchors = [a.line_start for a in arts] + [len(content)]
    for k, a in enumerate(arts):
        a.span_end = anchors[k + 1]
        a.para_count = count_paragraphs(content[a.line_start:a.span_end])
        a.mentions = len(re.findall(
            r"\bArticle[s]?\s+(?:\d+[a-z]?\s*[,;]\s*)*" + re.escape(a.number) + r"(?![0-9a-z])",
            content,
        ))
        if not a.title:
            a.title = _article_title_from_headings(content, a.line_end, a.span_end, a.level)

    tables = find_tables(content)
    recitals = find_recitals(content)
    obligation_count = GT._obligation_count(content)
    derogation_count = GT._derogation_count(content)
    chapters = find_chapters(content)

    strategy = (
        "hierarchical" if (arts and chapters)
        else "article_based" if arts
        else "table_then_sentence" if tables
        else "sentence_512"
    )
    para_total = sum(a.para_count for a in arts)

    return StructureMap(
        doc_id=md_path.stem,
        category=category,
        source_md=c.rel_to_root(md_path),
        generated=datetime.now().isoformat(),
        generator="03a_v1 (regex on best-parser markdown, shared GT helpers)",
        content_chars=len(content),
        words=n_words,
        document_title=GT._extract_title_heuristic(content),
        instrument=GT._instrument(content),
        celex=GT._celex(content),
        publication_date=GT._publication_date(content),
        article_count=len(arts),
        articles=[
            {"number": a.number, "title": a.title,
             "para_count": a.para_count, "mentions": a.mentions}
            for a in arts
        ],
        chapters=chapters,
        recitals_count=len(recitals),
        # headline count uses the GT definition (02's tables_count) so map vs GT
        # can never disagree; `blocks` are the strict |header|+|---|+|rows| spans
        tables={"count": GT._count_tables(content),
                "blocks_found": len(tables),
                "total_rows": sum(t["rows"] for t in tables),
                "blocks": tables},
        paragraph_stats={
            "total": para_total,
            "min_per_article": min((a.para_count for a in arts), default=0),
            "max_per_article": max((a.para_count for a in arts), default=0),
            "mean_per_article": round(para_total / max(len(arts), 1), 2),
        },
        obligations={
            "count": obligation_count,
            "per_1000_words": round(obligation_count / n_words * 1000, 2),
        },
        derogations={
            "count": derogation_count,
            "per_1000_words": round(derogation_count / n_words * 1000, 2),
        },
        units=find_units(content),
        chunking_strategy=strategy,
        recommended_chunk_targets={
            "article": len(arts),
            "chapter": len(chapters),
            "table": len(tables),
        },
    )


def build_all_maps(md_paths: Optional[List[Path]] = None, out_dir: Optional[Path] = None) -> dict:
    """Build + write one map per doc. Returns {doc_id: map_dict}."""
    out_dir = out_dir or c.NOTEBOOKS_DATA / "structure_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_paths = md_paths or c.discover_markdown()
    maps = {}
    for p in sorted(md_paths):
        maps[p.stem] = build_structure_map(p).to_dict()
        (out_dir / f"{p.stem}_structure.json").write_text(
            json.dumps(maps[p.stem], indent=2, ensure_ascii=False), encoding="utf-8")
    return maps


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    paths = [p for p in c.discover_markdown() if (only is None or p.stem.startswith(only))]
    for p in sorted(paths)[:int(sys.argv[2]) if len(sys.argv) > 2 else -1]:
        sm = build_structure_map(p).to_dict()
        print("=" * 64)
        print(p.stem, "|", sm["chunking_strategy"], "| arts:", sm["article_count"])
        for a in sm["articles"][:6]:
            print("   [%-4s] p=%-3s %s" % (a["number"], a["para_count"], (a["title"] or "«-»")[:60]))
