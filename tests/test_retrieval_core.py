# -*- coding: utf-8 -*-
"""Unit tests for src/retrieval: fusion (RRF), sparse (BM25), tokenizer
(corpus helpers) -- all offline, no FAISS/Ollama."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retrieval.fusion import rrf_fuse, RRF_K  # noqa: E402
from retrieval.sparse import SparseIndex  # noqa: E402
from retrieval._corpus import tokenize  # noqa: E402


# --- fusion ------------------------------------------------------------------

def test_rrf_fuse_scores_and_metadata():
    fused = rrf_fuse([
        ("dense",  [("doc_a", 0.9), ("doc_b", 0.8)]),
        ("sparse", [("doc_b", 1.0), ("doc_c", 0.5)]),
    ])
    # doc_b appears in both lists -> highest fused score
    assert set(fused) == {"doc_a", "doc_b", "doc_c"}
    assert fused["doc_b"]["score"] > fused["doc_a"]["score"]
    assert fused["doc_b"]["methods"] == ["dense", "sparse"]
    assert fused["doc_b"]["ranks"] == {"dense": 2, "sparse": 1}
    # reciprocal-rank check: rank 1 in one list
    expect = 1.0 / (RRF_K + 1)
    assert abs(fused["doc_a"]["score"] - expect) < 1e-12


def test_rrf_fuse_empty():
    assert rrf_fuse([]) == {}


# --- sparse (BM25) -----------------------------------------------------------

def test_sparse_index_ranks_relevant_doc_first():
    docs = [
        "the market operator shall report to the national regulator",
        "banana smoothie recipes with tropical fruit",
        "the national regulator shall sanction market operators",
        "quantum physics introductory notes",
    ]
    idx = SparseIndex(docs)
    hits = idx.search("market operator regulator", k=3)
    assert hits, "expected at least one hit"
    assert hits[0][0] in (0, 2)          # one of the two relevant docs
    assert hits[0][0] != 1 and hits[0][0] != 3
    # scores descending
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_sparse_index_no_match():
    idx = SparseIndex(["energy market rules"])
    assert idx.search("zzzqqq xkcd") == []


def test_sparse_index_all_empty_corpus():
    idx = SparseIndex(["", "  ", "\n"])
    assert idx.search("anything") == []


# --- tokenizer ---------------------------------------------------------------

def test_tokenize_keeps_ref_numbers_and_acronyms():
    toks = tokenize("REMIT (2019/944/EU) applies to the 2019/944 operator")
    assert "REMIT" in toks           # acronym kept case-sensitive
    assert "2019/944/eu" in toks     # instrument ref kept
    assert "2019/944" in toks        # bare year/number kept
    assert "operator" in toks        # content word kept
    assert "the" not in toks         # stopword dropped


def test_tokenize_splits_list_markers():
    # ref numbers inside parentheses are split out; single letters are dropped
    # as noise, so the signal kept is the numeric components.
    toks = tokenize("5(2) of 12 and section 34")
    assert {"5", "2", "12", "34", "section"} <= set(toks)
    for t in toks:
        assert " " not in t
        assert t == t.lower() or t.isupper()
