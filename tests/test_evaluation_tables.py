"""Tests for ``tables.table_b`` (spec §12 judge columns).

No LLM.  Verifies the spec judge columns are present, correctly formatted,
that missing metrics render as ``n/a`` (never fabricated), and that the
legacy base / 4-axis columns are retained for backward compatibility.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation import tables as T  # noqa: E402


def _agg(row_list):
    return row_list


def test_table_b_spec_columns_present():
    rows = T.table_b([
        {"system": "hybrid", "metric": "exact_match", "value": 0.0, "n": 60},
        {"system": "hybrid", "metric": "judge__correct", "value": 0.5, "n": 60},
    ], systems=["hybrid"])
    cols = set(rows[0].keys())
    expected = {
        "correct", "faithful", "complete", "evidence_supported",
        "overall_score", "confidence", "graph_reasoning_correct",
        "unsupported_claim_rate", "escalation_rate",
        # legacy retained
        "em", "f1", "semantic_sim", "cites_target",
        "judge_relevance", "judge_faithfulness",
        "judge_groundedness", "judge_completeness",
    }
    assert expected <= cols, f"missing: {expected - cols}"


def test_table_b_spec_values_formatted():
    rows = T.table_b([
        {"system": "h", "metric": "judge__correct", "value": 0.555, "n": 60},
        {"system": "h", "metric": "judge__confidence", "value": 0.833, "n": 60},
        {"system": "h", "metric": "judge__graph_reasoning_correct",
         "value": 0.6, "n": 60},
        {"system": "h", "metric": "judge__escalation_rate",
         "value": 0.2, "n": 60},
        {"system": "h", "metric": "judge__unsupported_claim_rate",
         "value": 0.1, "n": 60},
        {"system": "h", "metric": "judge__overall_score", "value": 4.1, "n": 60},
    ], systems=["h"])
    r = rows[0]
    assert r["correct"] == "0.555"
    assert r["confidence"] == "0.833"
    assert r["graph_reasoning_correct"] == "0.600"
    assert r["escalation_rate"] == "0.200"
    assert r["unsupported_claim_rate"] == "0.100"
    assert r["overall_score"] == "4.10"


def test_table_b_missing_metrics_are_na():
    rows = T.table_b([
        {"system": "h", "metric": "exact_match", "value": 0.0, "n": 60},
    ], systems=["h"])
    r = rows[0]
    assert r["correct"] == "n/a"
    assert r["faithful"] == "n/a"
    assert r["overall_score"] == "n/a"
    assert r["confidence"] == "n/a"
    assert r["em"] == "0.000"


def test_table_b_multiple_systems():
    rows = T.table_b([
        {"system": "a", "metric": "judge__correct", "value": 0.9, "n": 10},
        {"system": "b", "metric": "judge__correct", "value": 0.3, "n": 10},
    ], systems=["a", "b"])
    by_s = {r["system"]: r for r in rows}
    assert by_s["a"]["correct"] == "0.900"
    assert by_s["b"]["correct"] == "0.300"
