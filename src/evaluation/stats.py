"""Paired statistical inference for system comparisons.

Per "Statistical Analysis" + "Bootstrap Confidence Intervals".

Two estimators, both seeded/reproducible:

* :func:`bootstrap_ci` -- non-parametric 95% CI on a single mean, resampling
  per-query metric values with replacement.
* :func:`wilcoxon`     -- Wilcoxon signed-rank test (scipy) on *paired*
  per-query metric differences, the correct test when the same queries are
  scored by two systems (per "Statistical Analysis": "appropriate paired tests").

The framework reports mean/median/std/CI/p-value/effect-size (Cohen's d on
the paired differences, a small-n effect size) per comparison, and states
the small-n caveat when ``n`` is low (per "Statistical Analysis": "If the dataset is too small
for meaningful statistical inference, state this limitation").  Here n=60 is
borderline; we compute but flag p>0.05 as non-significant, not absent.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def bootstrap_ci(values: Sequence[float], n_bootstrap: int = 2000,
                 ci: float = 0.95, seed: int = 2024) -> Tuple[float, float, float]:
    """Return ``(point_est, lo, hi)`` for the mean of ``values``.

    Point estimate is the ordinary mean; lo/hi are the
    ``((1-ci)/2, 1-(1-ci)/2)`` quantiles of the bootstrap mean
    distribution.  Requires at least 2 observations (else the mean itself
    with degenerate CI, documented as no-variance).
    """
    arr = np.asarray(list(values), dtype="float64")
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("bootstrap_ci: empty values")
    if arr.size == 1:
        v = float(arr[0])
        return v, v, v
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype="float64")
    for b in range(n_bootstrap):
        sample = arr[rng.integers(0, arr.size, size=arr.size)]
        means[b] = sample.mean()
    lo_alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(means, lo_alpha))
    hi = float(np.quantile(means, 1.0 - lo_alpha))
    return float(arr.mean()), lo, hi


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float],
                          n_bootstrap: int = 2000, ci: float = 0.95,
                          seed: int = 2024,
                          ) -> Tuple[float, float, float]:
    """95% CI on the *paired* mean difference ``mean(a) - mean(b)``.

    Resamples paired indices (preserving the a/b pairing) rather than
    resampling each vector independently -- that pairing is what makes it a
    paired test (the "Statistical Analysis" requirement).  Returns ``(diff, lo, hi)``.
    """
    if len(a) != len(b):
        raise ValueError("paired_bootstrap_diff: length mismatch")
    aa = np.asarray(list(a), dtype="float64")
    bb = np.asarray(list(b), dtype="float64")
    d = aa - bb
    if d.size <= 1:
        v = float(d.mean()) if d.size else 0.0
        return v, v, v
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype="float64")
    idx = np.arange(d.size)
    for _ in range(n_bootstrap):
        sel = idx[rng.integers(0, d.size, size=d.size)]
        diffs[_] = d[sel].mean()
    lo_alpha = (1.0 - ci) / 2.0
    return (float(d.mean()),
            float(np.quantile(diffs, lo_alpha)),
            float(np.quantile(diffs, 1.0 - lo_alpha)))


def wilcoxon(a: Sequence[float], b: Sequence[float]
             ) -> Tuple[Optional[float], Optional[float]]:
    """Wilcoxon signed-rank test on paired per-query scores.

    Returns ``(p_value, p_value)`` where the first is the test p-value and
    the second is repeated for convenience (some tables want both two-sided
    and the same).  ``None`` when the test cannot be run (all differences
    zero, or < 2 non-zero ranks).
    """
    from scipy.stats import wilcoxon as _w

    aa = np.asarray(list(a), dtype="float64")
    bb = np.asarray(list(b), dtype="float64")
    d = (aa - bb)[~np.isnan(aa - bb)]
    nz = np.count_nonzero(d)
    if nz < 2:
        return None, None
    try:
        res = _w(aa, bb, zero_method="wilcox", correction=False)
        return float(res.pvalue), float(res.pvalue)
    except ValueError:
        return None, None


def cohen_d_paired(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Effect size on paired differences (Cohen's d on ``a - b``)."""
    d = np.asarray([x - y for x, y in zip(a, b)], dtype="float64")
    d = d[~np.isnan(d)]
    if d.size < 2 or float(d.std(ddof=1)) == 0.0:
        return None
    return float(d.mean() / d.std(ddof=1))


def summarize_comparison(a: Sequence[float], b: Sequence[float], name_a: str,
                         name_b: str, n_bootstrap: int = 2000,
                         seed: int = 2024) -> Dict:
    """Full per-pairing comparison dict for the reproducibility report."""
    mean_a = float(np.mean([x for x in a if not math.isnan(x)]))
    mean_b = float(np.mean([x for x in b if not math.isnan(x)]))
    diff, lo, hi = paired_bootstrap_diff(a, b, n_bootstrap=n_bootstrap, seed=seed)
    p, _ = wilcoxon(a, b)
    return {
        "a": name_a, "b": name_b,
        "mean_a": mean_a, "mean_b": mean_b,
        "diff_a_minus_b": diff, "ci95": [lo, hi],
        "wilcoxon_p": p,
        "cohens_d": cohen_d_paired(a, b),
        "n": len(a),
    }
