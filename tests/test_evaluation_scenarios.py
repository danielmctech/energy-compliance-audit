"""Tests for the two-drafter / blind-adjudicator / dispute-escalation protocol.

All LLM helpers are stubbed, so no network or model is needed.  These pin the
pure decision logic: which draft is accepted, when escalation is invoked, and
that no decision is ever fabricated (a failed/invalid escalation is dropped).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation import scenarios as S  # noqa: E402


PROVISION = ("The operator shall maintain at least five percent reserve "
             "and report the result to the agency within ten business days.")


def _chunk():
    class C:
        node_type = "article"
        text = PROVISION
        doc_id = "remit_2009"
        lineage_id = "remit:art:27"
    return C()


def _clause(model, text, want):
    base = ("The operator shall maintain at least five percent reserve "
            "and submit to the agency a report of that reserve position.")
    return {"clause": f"[{model}] " + base,
            "deviation_if_any": ("value set to two percent"
                                 if want == "non_compliant" else None)}


@pytest.fixture
def corpus():
    return {"chunks": [_chunk()]}


def _adj(want_bool, score_a=8, score_b=6):
    # adjudicator agrees with intent for candidate A only, unless overridden
    return {
        "a": {"compliant": True, "score": score_a, "critical_errors": []},
        "b": {"compliant": want_bool, "score": score_b, "critical_errors": []},
        "better": "A", "notes": "x",
    }


# ---- acceptance paths -------------------------------------------------------
def test_both_qualify_picks_higher_score(corpus):
    S._llm_clause = _clause
    S._adjudicate = lambda m, prov, a, b: _adj(True, score_a=8, score_b=7)
    S._escalate = raise_not_called
    out = S.build_voted_scenarios(corpus, max_scenarios=2, seed=7)
    assert out and out[0].metadata["decision_path"] == "both_qualify"
    # A scores 8 > B scores 7 -> A chosen
    assert out[0].metadata["chosen"] == "A"


def test_single_qualify_no_escalation(corpus):
    S._llm_clause = _clause
    # only A compliant, intent compliant -> A qualifies alone, B does not
    S._adjudicate = lambda m, prov, a, b: {
        "a": {"compliant": True, "score": 4, "critical_errors": []},
        "b": {"compliant": False, "score": 9, "critical_errors": ["wrong value"]},
        "better": "B", "notes": "x"}
    called = {}
    def esc(*a, **k):
        called["yes"] = True
    S._escalate = esc
    out = S.build_voted_scenarios(corpus, max_scenarios=2, seed=7)
    assert out, "single qualifying draft must be accepted"
    # only A is compliant & intent is compliant -> A must be the accept
    assert out[0].metadata["chosen"] == "A"
    assert out[0].metadata["decision_path"] == "single_qualify"
    assert "escalation" not in out[0].metadata


def raise_not_called(*a, **k):
    raise AssertionError("escalation must not run when a draft qualifies")


# ---- dispute / escalation ---------------------------------------------------
def _clause_no_second_round(model, text, want):
    # neutralise the non-compliant round (single-chunk corpus would let the
    # same fixed verdict be accepted under that label instead) by returning
    # an under-length clause; the compliant round is tested below.
    if want != "compliant":
        return {"clause": "too short", "deviation_if_any": None}
    base = ("The operator shall maintain at least five percent reserve "
            "and submit to the agency a report of that reserve position.")
    return {"clause": f"[{model}] " + base, "deviation_if_any": None}


def _adj_neither():
    return {
        "a": {"compliant": False, "score": 0, "critical_errors": []},
        "b": {"compliant": False, "score": 0, "critical_errors": []},
        "better": None, "notes": ""}


def test_dispute_escalation_confirmed(corpus):
    S._llm_clause = _clause_no_second_round
    # intent compliant; NEITHER draft judged compliant -> dispute
    S._adjudicate = lambda m, prov, a, b: _adj_neither()
    S._escalate = lambda m, prov, a, b, want, adj: {
        "choice": "B", "compliance_of_choice": True, "reason": "ok"}
    out = S.build_voted_scenarios(corpus, max_scenarios=2, seed=7)
    assert out
    assert out[0].metadata["decision_path"] == "escalation_confirmed"
    assert out[0].metadata["chosen"] == "B"
    assert out[0].metadata["escalation"]["choice"] == "B"


def test_dispute_escalation_rejects_both(corpus):
    S._llm_clause = _clause_no_second_round
    S._adjudicate = lambda m, prov, a, b: _adj_neither()
    S._escalate = lambda m, prov, a, b, want, adj: {"choice": "neither",
                                                    "compliance_of_choice": False}
    assert S.build_voted_scenarios(corpus, max_scenarios=2, seed=7) == []


def test_dispute_escalation_failure_never_fabricated(corpus):
    S._llm_clause = _clause_no_second_round
    S._adjudicate = lambda m, prov, a, b: _adj_neither()
    def boom(*a, **k):
        raise RuntimeError("ollama down")
    S._escalate = boom
    assert S.build_voted_scenarios(corpus, max_scenarios=2, seed=7) == []


# ---- invariants -------------------------------------------------------------
def test_drafter_never_judges_own_draft(corpus):
    S._llm_clause = _clause
    adjudicated = []
    def adj(model, prov, a, b):
        adjudicated.append(model)
        return _adj(True)
    S._adjudicate = adj
    S._escalate = raise_not_called
    S.build_voted_scenarios(corpus, max_scenarios=2, seed=7)
    cfg = __import__("evaluation.config", fromlist=["EvalConfig"]).EvalConfig()
    # the adjudicator must be the dedicated third model, i.e. NEITHER drafter
    for m in adjudicated:
        assert m == cfg.scenario_adjudicator
        assert m not in (cfg.scenario_drafter_a, cfg.scenario_drafter_b)


def test_short_clause_skipped(corpus):
    S._llm_clause = lambda m, text, want: {"clause": "too short", "deviation_if_any": None}
    S._adjudicate = raise_not_called
    assert S.build_voted_scenarios(corpus, max_scenarios=2, seed=7) == []
