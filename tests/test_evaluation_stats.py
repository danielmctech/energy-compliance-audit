"""Deterministic framework tests (no LLM / no Neo4j) for stats, benchmark,
overlap and the synthetic-scenario / correction logic ("Unit Tests" / "Error Classification")."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation import stats  # noqa: E402


# ---- bootstrap CIs ----------------------------------------------------------
def test_bootstrap_ci_point_is_mean():
    vals = [0.0, 1.0, 0.5, 0.5, 1.0]
    point, lo, hi = stats.bootstrap_ci(vals, n_bootstrap=200, seed=1)
    assert abs(point - (0.0 + 1.0 + 0.5 + 0.5 + 1.0) / 5) < 1e-9
    assert lo <= point <= hi


def test_bootstrap_ci_single_value_degenerate():
    point, lo, hi = stats.bootstrap_ci([0.7], n_bootstrap=10)
    assert point == lo == hi == 0.7


def test_bootstrap_ci_empty_raises():
    with pytest.raises(ValueError):
        stats.bootstrap_ci([], n_bootstrap=10)


def test_bootstrap_ci_deterministic():
    vals = [0.2, 0.4, 0.6, 0.8]
    a = stats.bootstrap_ci(vals, n_bootstrap=100, seed=42)
    b = stats.bootstrap_ci(vals, n_bootstrap=100, seed=42)
    assert a == b


def test_paired_bootstrap_diff_matches_mean_diff():
    a = [1.0, 0.8, 0.6]
    b = [0.9, 0.7, 0.5]
    diff, lo, hi = stats.paired_bootstrap_diff(a, b, n_bootstrap=200, seed=3)
    assert abs(diff - (0.1 + 0.1 + 0.1) / 3) < 1e-9


def test_paired_bootstrap_length_mismatch():
    with pytest.raises(ValueError):
        stats.paired_bootstrap_diff([1.0, 2.0], [1.0])


# ---- wilcoxon ----------------------------------------------------------------
def test_wilcoxon_clear_difference_small_p():
    a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    p, _ = stats.wilcoxon(a, b)
    assert p is not None and p < 0.05


def test_wilcoxon_all_equal_none():
    p, _ = stats.wilcoxon([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert p is None


def test_wilcoxon_too_few_nz_none():
    p, _ = stats.wilcoxon([1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0])
    assert p is None  # only one non-zero difference


def test_cohens_d_direction():
    # differences [1, 2, 0] -> positive mean, non-zero variance
    d = stats.cohen_d_paired([1.0, 2.0, 1.0], [0.0, 0.0, 1.0])
    assert d is not None and d > 0


def test_cohens_d_degenerate_zero_variance_none():
    # constant differences -> zero variance -> undefined, not a fake value
    assert stats.cohen_d_paired([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]) is None


def test_summarize_comparison_keys():
    s = stats.summarize_comparison([1, 1, 1], [0.5, 0.5, 0.5], "a", "b",
                                   n_bootstrap=50, seed=1)
    for k in ("mean_a", "mean_b", "diff_a_minus_b", "ci95", "wilcoxon_p",
              "cohens_d", "n"):
        assert k in s
    assert s["mean_a"] == 1.0 and s["n"] == 3
