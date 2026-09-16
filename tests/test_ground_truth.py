# -*- coding: utf-8 -*-
"""Unit tests for src/ground_truth.py heuristic extractors (offline, no LLM)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from parsing import ground_truth as GT  # noqa: E402


# --- windows -----------------------------------------------------------------

def test_sample_windows_empty_content():
    assert GT.sample_windows("", n_windows=6) == []


def test_sample_windows_layout():
    text = "x" * 50_000
    wins = GT.sample_windows(text, n_windows=6, window_chars=2500)
    assert len(wins) == 6
    off = [w["text_offset"] for w in wins]
    assert off[0][0] == 0                       # first window at head
    assert off[-1] == [47_500, 50_000]           # last window anchored at tail
    # non-overlapping, in order
    for a, b in zip(off, off[1:]):
        assert a[1] <= b[0]


def test_sample_windows_n2():
    text = "y" * 10_000
    wins = GT.sample_windows(text, n_windows=2, window_chars=2500)
    assert [w["text_offset"] for w in wins] == [[0, 2500], [7500, 10000]]


# --- instrument / celex / dates ---------------------------------------------

def test_instrument_eu_form():
    assert GT._instrument("Regulation (EU) 2019/944 of the European Parliament") == \
        "Regulation (EU) 2019/944"


def test_instrument_markdown_decoration():
    assert GT._instrument("**Directive (EU) 2018/2001**") == "Directive (EU) 2018/2001"
    assert GT._instrument("_Decision (EC) 2007/61/EC_") == "Decision (EC) 2007/61/EC"


def test_instrument_missing():
    assert GT._instrument("just some prose without any law number") is None


def test_celex_matches_all_digit_reference():
    # documented regex is \b3\d{10}\b -> an 11-char all-digit token starting
    # with 3 (CELEX-style). Pin the actual behavior.
    assert GT._celex("see 32019094400 for details") == "32019094400"
    assert GT._celex("no reference in this sentence") is None


def test_publication_date():
    assert GT._publication_date("published on 11 June 2019 as a thing") == "11 June 2019"
    assert GT._publication_date("no dates here") is None


# --- structural counters ------------------------------------------------------

def test_count_tables():
    md = "intro\n| Name | Id |\n|------|----|\n| a | 1 |\n| b | 2 |\n"
    assert GT._count_tables(md) == 1
    assert GT._count_tables("no tables at all") == 0


def test_table_headers():
    assert GT._table_headers("| Name | Id |\n|------|----|\n| a | 1 |") == ["Name | Id"]
    assert GT._table_headers("plain text") == []


def test_paragraph_count():
    long_p = "word " * 40
    assert GT._paragraph_count("short\n\n" + long_p) == 1
    assert GT._paragraph_count("tiny\n\nalso tiny") == 0


def test_obligation_and_derogation_counts():
    txt = "Each provider shall comply.\nConsumers must do their part."
    assert GT._obligation_count(txt) == 2
    d = "Notwithstanding Article 5, this is without prejudice to X."
    assert GT._derogation_count(d) == 2
    assert GT._obligation_count(d) == 0


def test_extract_article_numbers():
    assert GT._extract_article_numbers("Article 2, Article 14a, then Article 2 again") == \
        ["2", "14a"]
    assert GT._extract_article_numbers("no articles here") == []


def test_extract_definitions():
    txt = ("\"market operator\" means an entity.\n"
           "\"load follows\" means the thing.\n"
           "plain text means nothing")
    defs = GT._extract_definitions(txt)
    assert "market operator" in defs
    assert "load follows" in defs


def test_numbered_items():
    txt = "intro\n1. first item\n2. second item\na) third item"
    items = GT._numbered_items(txt)
    assert items == ["1. first item", "2. second item", "a) third item"]


def test_headings():
    assert GT._headings("# One\n## Two\nbody\n### Deep") == ["One", "Two", "Deep"]
    assert GT._headings("no headings") == []
