"""Deterministic tests for benchmark construction + overlap ("Benchmark Interface" / "Retrieval Overlap Analysis").

No LLM.  Validates the gold-set is exactly 60, categories, deterministic
order, and the leakage probe, plus Jaccard / complementarity edge cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation import benchmark as B  # noqa: E402
from evaluation import overlap as O  # noqa: E402


# ---- benchmark ---------------------------------------------------------------
def _items():
    return B.build_benchmark()


def test_benchmark_size_60():
    assert len(_items()) == 60


def test_benchmark_categories_8_families():
    cats = {i.category for i in _items()}
    assert cats == {
        "single_document",
        "non_relational_semantic",
        "one_hop_relational",
        "two_hop_relational",
        "relation_direction",
        "temporal_version",
        "graph_distractor",
        "multi_document_synthesis",
    }


def test_benchmark_family_target_counts():
    from collections import Counter
    cats = Counter(i.category for i in _items())
    assert cats == {
        "single_document": 15,
        "non_relational_semantic": 5,
        "one_hop_relational": 10,
        "two_hop_relational": 10,
        "relation_direction": 5,
        "temporal_version": 5,
        "graph_distractor": 5,
        "multi_document_synthesis": 5,
    }


def test_benchmark_relationals_carry_gold_edges_and_path():
    items = _items()
    rel = [i for i in items if i.category in (
        "one_hop_relational", "two_hop_relational",
        "relation_direction", "temporal_version")]
    assert rel, "expected relational items"
    for i in rel:
        assert i.gold_edges, f"{i.query_id} missing gold_edges"
        assert i.gold_path, f"{i.query_id} missing gold_path"
        assert i.hop_count == len(i.gold_path)


def test_benchmark_two_hop_is_two_edges():
    items = [i for i in _items() if i.category == "two_hop_relational"]
    assert items
    for i in items:
        assert i.hop_count == 2
        assert len(i.gold_path) == 2


def test_benchmark_deterministic_ids():
    a = [i.query_id for i in _items()]
    b = [i.query_id for i in _items()]
    assert a == b
    assert a == [f"q{i:03d}" for i in range(1, 61)]


def test_benchmark_each_has_unique_target():
    items = _items()
    assert len({i.target_lineage_id for i in items}) == len(items)


def test_benchmark_reference_answer_present():
    items = _items()
    assert all(i.reference_answer for i in items)
    assert all(i.reference_basis == "target_chunk_text" for i in items)


# ---- leakage probe -----------------------------------------------------------
def _pairs_path():
    return Path(__file__).resolve().parent.parent / \
        "notebooks/data/retrieval/pairs_stage1.jsonl"


def test_leakage_probes_shape():
    items = _items()
    probes = B.leakage_probes(_pairs_path(), items)
    assert "positive_targets" in probes
    assert "queries" in probes


def test_leakage_flag_consistency():
    """Each item's leakage flag must agree with the probe sets."""
    items = _items()
    probes = B.leakage_probes(_pairs_path(), items)
    pos, qs = probes["positive_targets"], probes["queries"]
    for i in items:
        assert i.leakage["target_is_training_positive"] == \
            (i.target_lineage_id in pos)
        assert i.leakage["question_matches_training_query"] == (i.question in qs)


def test_summary_reports_leakage():
    s = B.summary(_items())
    assert "leakage" in s
    assert "targets_that_are_lora_training_positives" in s["leakage"]
    assert "questions_that_match_training_queries" in s["leakage"]


# ---- overlap -----------------------------------------------------------------
def test_jaccard_identical():
    assert O.jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint():
    assert O.jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_empty_union():
    assert O.jaccard(set(), set()) == 0.0


def test_jaccard_partial():
    assert abs(O.jaccard({"a", "b", "c"}, {"b", "c", "d"}) - 2 / 4) < 1e-9


def test_overlap_matrix_pairs():
    pq = {"dense": {"q1": [{"lineage_id": "a"}, {"lineage_id": "b"}]},
          "sparse": {"q1": [{"lineage_id": "b"}, {"lineage_id": "c"}]}}
    res = O.overlap_matrix(pq, ["dense", "sparse"], k=2)
    assert "dense~sparse" in res
    # top-2 dense {a,b}, sparse {b,c} -> inter {b}=1, union {a,b,c}=3
    # (overlap_matrix rounds to 4 dp -> 0.3333)
    assert res["dense~sparse"] == 0.3333


def test_overlap_skips_missing_systems():
    pq = {"dense": {"q1": [{"lineage_id": "a"}]}}
    res = O.overlap_matrix(pq, ["dense", "sparse"], k=1)
    assert "dense~sparse" not in res


def _item(qid, target):
    return B.BenchmarkItem(
        query_id=qid, question="q", doc_id="doc", target_lineage_id=target,
        category="article")


def test_complementarity_unique_only_by():
    items = [_item("q1", "a"), _item("q2", "d")]
    pq = {
        "dense":  {"q1": [{"lineage_id": "a"}], "q2": [{"lineage_id": "x"}]},
        "sparse": {"q1": [{"lineage_id": "x"}], "q2": [{"lineage_id": "d"}]},
        "graph":  {"q1": [{"lineage_id": "b"}], "q2": [{"lineage_id": "x"}]},
    }
    res = O.complementarity(pq, items, ["dense", "sparse", "graph"], k=1)
    ub = res["unique_relevant_only_by"]
    # q1 target a: only dense hits it; q2 target d: only sparse hits it
    assert ub["dense"] == 1
    assert ub["sparse"] == 1
    assert ub["graph"] == 0
    assert res["core_systems"] == ["dense", "sparse", "graph"]


def test_complementarity_fusion_only():
    items = [_item("q1", "c")]
    pq = {
        "dense": {"q1": [{"lineage_id": "a"}]},
        "sparse": {"q1": [{"lineage_id": "b"}]},
        "hybrid": {"q1": [{"lineage_id": "c"}, {"lineage_id": "a"}]},
    }
    res = O.complementarity(pq, items, ["dense", "sparse", "hybrid"], k=2)
    assert res["targets_found_only_after_fusion"] == 1
    assert res["queries_with_no_method_hitting_target"] == 1
