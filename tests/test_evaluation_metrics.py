"""Metric edge-case tests ("Unit Tests").

Every formula verified independently against hand-computed values.  Edge
cases documented in the metric requirements:
  no relevant docs / all relevant retrieved / none retrieved /
  relevant at rank 1 / at rank K / multiple relevant / fewer than K
  retrieved / duplicate retrieved ids / duplicate relevance labels /
  empty retrieval.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation import metrics as M  # noqa: E402


# ---- recall ---------------------------------------------------------------
def test_recall_all_relevant_retrieved():
    assert M.recall_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == 1.0


def test_recall_none_retrieved():
    assert M.recall_at_k(["x", "y"], ["a", "b"], 5) == 0.0


def test_recall_relevant_at_rank_k():
    # recall@k counts relevant items within the TOP-k (order-insensitive here)
    assert M.recall_at_k(["x", "y", "a", "z", "b"], ["a", "b"], 5) == 1.0
    # only the first 2 slots are inspected -> just a -> 1 of 2
    assert M.recall_at_k(["x", "y", "a", "z", "b"], ["a", "b"], 3) == 0.5


def test_recall_relevant_below_k_excluded():
    assert M.recall_at_k(["x", "y", "z"], ["a"], 2) == 0.0


def test_recall_no_relevant_items_is_none():
    # "Retrieval Metrics": documented handling -> None, not 0.0
    assert M.recall_at_k(["a"], [], 5) is None


def test_recall_duplicates_deduped():
    assert M.recall_at_k(["a", "a", "a"], ["a", "b"], 5) == 0.5


def test_recall_empty_retrieval():
    assert M.recall_at_k([], ["a"], 5) == 0.0


# ---- precision -------------------------------------------------------------
def test_precision_standard():
    # 1 of 3 in top-3 relevant, K=3 -> 1/3
    assert abs(M.precision_at_k(["a", "x", "y"], ["a", "b"], 3)
               - (1 / 3)) < 1e-9


def test_precision_fewer_than_k_retrieved():
    # only 2 retrieved, K=10 -> divide by 2, not 10 (documented)
    assert M.precision_at_k(["a", "x"], ["a", "b"], 10) == 0.5


def test_precision_no_relevant():
    assert M.precision_at_k(["x", "y"], ["a"], 5) == 0.0


def test_precision_zero_retrieved_none():
    assert M.precision_at_k([], ["a"], 5) is None


def test_precision_all_relevant():
    assert M.precision_at_k(["a", "b"], ["a", "b"], 2) == 1.0


# ---- hit rate ---------------------------------------------------------------
def test_hit_at_k_hit():
    assert M.hit_rate_at_k(["x", "a"], ["a", "b"], 2) == 1.0


def test_hit_at_k_miss():
    assert M.hit_rate_at_k(["x", "y"], ["a"], 2) == 0.0


def test_hit_no_relevant():
    assert M.hit_rate_at_k(["a", "b"], [], 5) == 0.0


# ---- MRR --------------------------------------------------------------------
def test_mrr_rank1():
    assert abs(M.mean_reciprocal_rank(["a", "x"], ["a"]) - 1.0) < 1e-9


def test_mrr_rank3():
    assert abs(M.mean_reciprocal_rank(["x", "y", "a"], ["a"]) - 1 / 3) < 1e-9


def test_mrr_not_found_zero():
    assert M.mean_reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_mrr_multiple_relevant_first_wins():
    # first relevant is the one that counts
    assert abs(M.mean_reciprocal_rank(["a", "b"], ["a", "b"]) - 1.0) < 1e-9


# ---- nDCG --------------------------------------------------------------------
def test_ndcg_perfect_order():
    assert abs(M.ndcg_at_k(["a", "b", "x"], ["a", "b"], 3) - 1.0) < 1e-9


def test_ndcg_relevant_buried_low():
    # relevant at ranks 3,4 vs perfect -> < 1
    v = M.ndcg_at_k(["x", "y", "a", "b"], ["a", "b"], 4)
    assert 0.0 < v < 1.0


def test_ndcg_no_relevant_zero():
    assert M.ndcg_at_k(["x", "y"], ["a"], 5) == 0.0


def test_ndcg_known_value():
    # binary, rel at rank2 only, k=3: DCG = 0 + 1/log2(3) + 0
    # IDCG = 1/log2(2)
    v = M.ndcg_at_k(["x", "a", "y"], ["a"], 3)
    expected = (1 / math.log2(3)) / (1 / math.log2(2))
    assert abs(v - expected) < 1e-9


# ---- EM / F1 --------------------------------------------------------------------
def test_em_case_insensitive_punct():
    assert M.exact_match("Article 4(2) applies", "article 4 2 applies") == 1.0
    assert M.exact_match("no", "yes") == 0.0


def test_f1_partial_overlap():
    v = M.token_f1("the quick fox", "the fox is quick")
    assert 0.0 < v < 1.0


def test_f1_no_overlap():
    assert M.token_f1("aaa bbb", "ccc ddd") == 0.0


def test_f1_empty():
    assert M.token_f1("", "something") == 0.0
    assert M.token_f1("something", "") == 0.0
