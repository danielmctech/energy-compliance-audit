# -*- coding: utf-8 -*-
"""Unit tests for src/structure_maps.py detectors (offline, regex-only).

Note: a few helpers in structure_maps have looser/tighter regexes than their
docstrings suggest. These tests pin the *observed* behavior so any change is
forced to make an intentional choice rather than silently regressing."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from structure import structure_maps as SM  # noqa: E402

DOC = """# Some Regulation (EU) 2019/944

## **Article 2 - Scope**

This Regulation applies to energy.

## **Article 7 - Obligations**

Providers shall comply with the rules in Article 2.

| Parameter | Value |
|-----------|-------|
| alpha     | 1     |
| beta      | 2     |

### Heading Level 3
detail follows
"""


def test_find_structural_articles_marked_bold_form():
    arts = SM.find_structural_articles(DOC)
    nums = [a.number for a in arts]
    assert nums == ["2", "7"]
    assert arts[0].title == "Scope"
    assert arts[1].title == "Obligations"
    assert arts[0].level == 2                       # `## ` heading
    # anchored to a real position in the content
    assert DOC[arts[0].line_start - 1:arts[0].line_end].strip().startswith("#")


@pytest.mark.parametrize(
    "line",
    ["## **Article 12 - Duties**", "_Article 5_", "## **Article 30**"],
)
def test_marked_article_variants(line):
    arts = SM.find_structural_articles(line)
    assert len(arts) == 1
    assert arts[0].number in ("12", "5", "30")


def test_prose_article_mentions_are_not_structural():
    prose = ("As set out in Article 2 of Regulation 2019/944, "
             "providers shall comply with the duty in Article 7.")
    assert SM.find_structural_articles(prose) == []


def test_find_chapters_kinds():
    chaps = SM.find_chapters("## Chapter I - General\nbody\n### Title 2 Scope\ntext")
    kinds = [c["kind"] for c in chaps]
    assert "CHAPTER" in kinds
    assert "TITLE" in kinds


def test_find_recitals_needs_capitalized_opening():
    # the regex requires the word after `(n)` to start capitalized
    rec = "Preamble\n\n(1) Whereas the market matters.\n(2) Furthermore it applies.\n"
    assert SM.find_recitals(rec) == [1, 2]
    # lowercase continuations are intentionally not matched
    assert SM.find_recitals("(1) first sentence\n(2) second sentence\n") == []


def test_find_tables_fields():
    tables = SM.find_tables(DOC)
    assert len(tables) == 1
    t = tables[0]
    assert t["start_line"] <= t["end_line"]
    assert t["rows"] == 3        # data rows (header excluded)
    assert t["cols"] == 2
    lines = DOC.splitlines()
    assert "Parameter" in lines[t["start_line"] - 1]


def test_article_title_from_headings():
    title = SM._article_title_from_headings(DOC, 30, len(DOC), 2)
    assert title == "Heading Level 3"


def test_structure_map_to_dict_roundtrip():
    s = SM.StructureMap(
        doc_id="d1", category="regulations", source_md="x.md",
        generated="now", generator="heuristic", content_chars=10, words=3,
        document_title=None, instrument=None, celex=None,
        publication_date=None, article_count=1,
        articles=[{"number": "2", "title": "Scope"}],
        recitals_count=2,
    )
    d = s.to_dict()
    assert d["article_count"] == 1
    assert d["articles"][0]["number"] == "2"
    assert d["recitals_count"] == 2
    assert d["chunking_strategy"] == "sentence_512"


def test_count_paragraphs_letters_and_numbers():
    assert SM.count_paragraphs("(a) first lettered\n(b) second") == 2
    assert SM.count_paragraphs("1. first\n2. second") == 2
    assert SM.count_paragraphs("a bare sentence\nplain text") == 0


def test_find_units_taxonomy():
    assert SM.find_units("500 MW") == {"power": 1, "energy": 0, "price": 0}
    assert SM.find_units("10 GWh") == {"power": 0, "energy": 1, "price": 0}
    assert SM.find_units("30 EUR/MWh price") == {"power": 0, "energy": 1, "price": 1}
    assert SM.find_units("no units here") == {"power": 0, "energy": 0, "price": 0}
