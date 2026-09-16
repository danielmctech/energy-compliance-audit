# -*- coding: utf-8 -*-
"""Unit tests for src/graph_builder.py helpers -- slugs, snippets, and the
key relationship-detection regexes (offline, deterministic)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from structure import graph_builder as GB  # noqa: E402


# --- slug / snippet ------------------------------------------------------------

def test_slug_slashes_punctuation_collapse():
    assert GB._slug("  Energy System Operator  (ESO)!! ") == "energy_system_operator_eso"
    assert GB._slug("ACME-CORP / 2019") == "acme_corp_2019"
    assert GB._slug("already_clean") == "already_clean"


def test_snippet_collapses_whitespace_and_caps_length():
    content = "before   \n\nafter text here"
    seg = GB._snippet(content, len("before   \n\n"), len(content))
    assert seg == "after text here"
    long = "a" * (GB.SNIPPET_CHARS + 50)
    assert len(GB._snippet(long, 0, len(long))) == GB.SNIPPET_CHARS


# --- relationship regexes ------------------------------------------------------

def test_amends_captures_instrument_number():
    assert GB.AMENDS.search("It amends Regulation (EU) 2019/944 as listed").group(1) == "2019/944"
    assert GB.AMENDS.search("It was amended by Directive (EU) 2018/2001").group(1) == "2018/2001"
    assert GB.AMENDS.search("no amending here") is None


def test_article_ref_finds_numbers_and_subparagraphs():
    found = GB.ARTICLE_REF.findall("Article 2 and Articles 5(2) to 9")
    assert "2" in found and "5" in found


def test_apply_anchor_matches_scope_phrases():
    text = ("This Directive applies to the market. "
            "It shall apply in all states, addressed to the providers.")
    hits = [m.group(0).lower() for m in GB.APPLY_ANCHOR.finditer(text)]
    assert any("applies to" in h for h in hits)
    assert any("shall apply in" in h for h in hits)
    assert any("addressed to" in h for h in hits)
    # no false positives in plain prose
    assert list(GB.APPLY_ANCHOR.finditer("a completely neutral sentence")) == []


def test_entity_matching_returns_best_label():
    assert GB._match_entity("is addressed to the market operators") == "market operators"
    assert GB._match_entity("the aggregators and end users") is not None
    assert GB._match_entity("no recognisable entity here") is None


def test_kind_constants_are_stable():
    assert GB.KIND_CROSS == "CROSS_REFERENCES"
    assert GB.KIND_AMENDS == "AMENDS"
    assert GB.KIND_DEFINED == "DEFINED_IN"
    assert GB.KIND_APPLIES == "APPLIES_TO"
    # v2 extended vocabulary (additive; the first four are unchanged)
    assert GB.KIND_PART_OF == "PART_OF"
    assert GB.KIND_IMPL == "IMPLEMENTS"
    assert GB.KIND_SUP == "SUPERSEDES"


def test_table_char_regions_no_false_positive_on_prose():
    content = "no tables here, just prose.\n\nmore prose line\n"
    regions = GB._table_char_regions(content)
    assert regions == []


# --- v2 extended vocabulary (additive) --------------------------------------

def test_supersedes_matches_newer_and_older_instruments():
    # "newer repeals older" -> group(1)=newer, group(2)=older
    m = GB.SUPERSEDES.search("Directive (EU) 2019/944 repeals Regulation (EC) 2003/54/EC")
    assert m is not None
    assert m.group(1) == "2019/944"
    assert m.group(2) == "2003/54"
    # EC jurisdiction code is handled (not just EU)
    assert GB.SUPERSEDES.search("Regulation (EU) 2014/95 has replaced Directive 2004/39/EC").group(2) == "2004/39"
    # self-reference (same number twice) is present -> caller dedupes
    assert GB.SUPERSEDES.search("Regulation 2003/54/EC repeals Regulation 2003/54/EC") is not None


def test_implements_matches_upper_and_lower_instruments():
    m = GB.IMPLEMENTS.search(
        "Framework Regulation (EU) 2023/1162 implemented through Regulation (EU) 2022/868")
    assert m is not None
    assert m.group(1) == "2023/1162"     # upper (higher tier)
    assert m.group(2) == "2022/868"      # lower (delegated)


def test_graph_add_edge_dedupes_on_src_dst_kind():
    """The idempotency of the v2 migration rests on Graph.add_edge dropping an
    edge whose (src, dst, kind) triple is already present -- so re-running the
    builders cannot duplicate v2 edges."""
    g = GB.Graph()
    a = g.node_document("d", "D")
    b = g.node_external_document("1234-567", "2034/567")
    ev = {"doc_id": "d", "offset": 0, "snippet": "x"}
    g.add_edge(a, b, GB.KIND_SUP, ev, extra={"x": 1})
    g.add_edge(a, b, GB.KIND_SUP, ev, extra={"x": 2})   # same triple -> dropped
    assert len(g.edges) == 1
    # a *different* kind on the same pair is allowed (distinct edge)
    g.add_edge(a, b, GB.KIND_IMPL, ev)
    assert len(g.edges) == 2


def test_part_of_emits_one_edge_per_article():
    """build_part_of is structural: one article->document edge per article,
    independent of any prose.  Verified against an in-memory Graph."""
    rec = _make_docrec(doc_id="doc_a",
                       art_titles={"1": "Article 1", "3": "Article 3"})
    g = GB.Graph()
    n = GB.build_part_of(g, [rec])
    assert n == 2
    kinds = {e["kind"] for e in g.edges}
    assert kinds == {"PART_OF"}
    # every edge points article -> its own document
    for e in g.edges:
        assert e["src"].startswith("doc_a:article:")
        assert e["dst"] == "doc_a:document"


def _make_docrec(doc_id, art_titles):
    """Minimal DocRec-shaped stand-in exposing only the fields build_part_of
    and the regex builders read (content / art_titles / smap / ...)."""
    from collections import namedtuple
    Shim = namedtuple("Shim",
                      ["doc_id", "content", "smap", "article_num",
                       "art_titles", "art_spans"])
    return Shim(
        doc_id=doc_id,
        content="",
        smap={"document_title": f"Regulation {doc_id}"},
        article_num=set(art_titles),
        art_titles=dict(art_titles),
        art_spans={},
    )
