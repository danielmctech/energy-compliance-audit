"""Tests for the spec-aligned LLM answer judge (``answer_metrics.py``).

Covers:
* schema parsing (spec §12 fields + legacy 4 axes)
* the confidence gate -> ``judge_escalator`` re-score (spec §13 escalation)
* escalator-failure path (primary kept, error recorded)
* unsupported-claim-rate derivation from the ``claims`` list
* ``score_generation`` packet assembly (all 7 spec §12 sections present)
* ``aggregate`` exposes the new metric rows

The Ollama backend is replaced with a deterministic fake returning canned
JSON strings -- this exercises the judge's *parse, gate, and adopt* code
without a model call or cache miss.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import common as c  # noqa: E402
from evaluation import answer_metrics as AM  # noqa: E402
from evaluation import config as E  # noqa: E402


# ---------------------------------------------------------------------------
# canned JSON responses
# ---------------------------------------------------------------------------

def _canned(ok: bool = True, conf: float = 0.95,
            score: int = 5, complete: bool = True,
            graph: object = None) -> str:
    """Return a judge-JSON string with the required fields populated.

    ``ok=False`` -> the response will include a *low* overall_score so the
    escalation gate should fire.
    ``graph`` -> bool or "null" or None (JSON null) for
    graph_reasoning_correct.
    """
    g = "null" if graph is None else str(bool(graph)).lower()
    return json.dumps({
        "correct": True, "faithful": True, "complete": complete,
        "evidence_supported": True, "unsupported_claims": not complete,
        "graph_reasoning_correct": (True if graph is None else graph),
        "overall_score": score, "confidence": conf,
        "needs_escalation": False,
        "claims": [
            {"claim": "Article 12 amends Regulation B", "supported": True},
            {"claim": "Article 12 was adopted in 2021",
             "supported": False},
        ],
        "relevance": 4, "faithfulness": 4, "groundedness": 4,
        "completeness": 4,
    })


class _FakeBackend:
    """Patch ``common.ollama_chat`` with a deterministic fake.

    ``script`` = list of (model, response) pairs, consumed in order.  Every
    call appends the model id to :attr:`calls`.  When the model is unknown
    ``fallback`` is used.
    """

    def __init__(self, script, fallback=None):
        self.script = list(script)
        self.fallback = fallback
        self.calls: list = []

    def __enter__(self):
        def fake(model, prompt, system=None, temperature=0.0,
                 num_ctx=32768, num_predict=16384, max_retries=3,
                 extra_options=None, use_cache=True, cache_key=None):
            self.calls.append(model)
            if self.script:
                expected, body = self.script[0]
                assert model == expected, \
                    f"expected {expected!r}, got {model!r}"
                self.script.pop(0)
                return body
            if self.fallback is not None:
                return self.fallback
            raise AssertionError("no scripted response and no fallback")
        self._patch = mock.patch.object(c, "ollama_chat", side_effect=fake)
        return self._patch.__enter__()

    def __exit__(self, *exc):
        self._patch.__exit__(*exc)


def test_extract_judge_json_parses_spec_fields():
    raw = _canned()
    obj = AM._extract_judge_json(raw)
    assert obj["correct"] is True
    assert obj["faithful"] is True
    assert obj["complete"] is True
    assert obj["evidence_supported"] is True
    assert obj["unsupported_claims"] is False
    assert obj["overall_score"] == 5
    assert obj["confidence"] == 0.95
    # legacy axes are still present for downstream consumers
    for ax in AM._JUDGE_AXES:
        assert isinstance(obj[ax], int)
    # spec §12: two claims, half unsupported -> 0.5 rate
    assert len(obj["_claims"]) == 2
    assert obj["unsupported_claim_rate"] == pytest.approx(0.5)


def test_should_escalate_on_low_confidence():
    cfg = E.EvalConfig()
    assert AM._should_escalate({"confidence": 0.5, "overall_score": 5},
                               cfg) is True
    assert AM._should_escalate({"confidence": 0.95, "overall_score": 5},
                               cfg) is False
    assert AM._should_escalate({"confidence": 0.95, "overall_score": 3},
                               cfg) is True
    assert AM._should_escalate({"confidence": None, "overall_score": None},
                               cfg) is False
    assert AM._should_escalate(
        {"confidence": 0.95, "overall_score": 5, "needs_escalation": True},
        cfg) is True


def test_llm_judge_escalation_invokes_judge_escalator():
    """Spec §13: primary confidence < 0.85 -> escalator re-scores; the
    escalator's fields win and are what the caller sees."""
    primary = _canned(conf=0.50)            # low confidence -> gate fires
    escalator = _canned(conf=0.90, complete=False)  # escalator's answer
    with _FakeBackend([(E.EvalConfig().judge_model, primary),
                       (E.EvalConfig().judge_escalator, escalator)]):
        j = AM.llm_judge(answer="ans", question="q", evidence="ev",
                         gold_answer="gold")
    assert j["escalated"] is True
    assert j["final_model"] == E.EvalConfig().judge_escalator
    assert j["primary_model"] == E.EvalConfig().judge_model
    # the escalator's ``complete=False`` won out
    assert j["complete"] is False


def test_llm_judge_keeps_primary_when_confident():
    """Spec §13: confident primary is accepted; ``judge_escalator`` is not
    invoked (no second LLM call)."""
    with _FakeBackend([(E.EvalConfig().judge_model, _canned(conf=0.95))]):
        j = AM.llm_judge(answer="ans", question="q", evidence="ev",
                         gold_answer="gold")
    assert j["escalated"] is False
    assert j["final_model"] == E.EvalConfig().judge_model


def test_llm_judge_escalation_failure_records_error():
    """Spec §13: when the escalator is unreachable the primary's output
    is kept and the failure is recorded on the row -- never fabricated."""
    primary = _canned(conf=0.50)
    def _raise(model, prompt, system=None, temperature=0.0,
               num_ctx=32768, num_predict=16384, max_retries=3,
               extra_options=None, use_cache=True, cache_key=None):
        if model == E.EvalConfig().judge_model:
            return primary
        raise RuntimeError("escalator 502")
    with mock.patch.object(c, "ollama_chat", side_effect=_raise):
        j = AM.llm_judge(answer="ans", question="q",
                         gold_answer="gold")
    assert j["escalated"] is True
    assert "502" in j.get("escalation_error", "")
    # primary's fields preserved
    assert j["final_model"] == E.EvalConfig().judge_model


def test_llm_judge_unavailable_returns_zero_not_fabricated():
    """Spec: no fabricated scores -- all zeros + error key."""
    def _down(model, prompt, **kw):
        raise RuntimeError("ollama down")
    with mock.patch.object(c, "ollama_chat", side_effect=_down):
        j = AM.llm_judge(answer="ans", question="q",
                         gold_answer="gold")
    assert j["error"] is not None
    for f in AM._JUDGE_AXES:
        assert j[f] == 0
    assert j["overall_score"] is None
    assert j["escalated"] is False


# ---------------------------------------------------------------------------
# score_generation packet assembly
# ---------------------------------------------------------------------------

def test_score_generation_builds_spec_packet():
    """The 7-section prompt must be sent to the LLM and include the row's
    gold answer, retrieved evidence, and (when present) graph evidence."""
    row = {
        "query_id": "Q1",
        "question": "What are the Article 12 obligations?",
        "target": "REGX:article:12",
        "category": "article",
        "answer": "Article 12 requires X.",
        "reference_answer": "REGX Article 12 states Y.",       # gold
        "user_prompt": (
            "Relevant regulatory evidence (top 5 retrieved chunks):\n\n"
            "[1] REGX  (REGX:article:12)\nchunk text A\n\n"
            "Question: What are the Article 12 obligations?"),
        "graph_evidence": "REGX --AMENDS--> ART12 snippet",
    }
    seen: dict = {}
    def fake(model, prompt, system=None, temperature=0.0,
             num_ctx=32768, num_predict=16384, max_retries=3,
             extra_options=None, use_cache=True, cache_key=None):
        seen["model"] = model
        seen["prompt"] = prompt
        return _canned()
    with mock.patch.object(c, "ollama_chat", side_effect=fake):
        res = AM.score_generation(row, run_judge=True)
    for sec in ("[1] QUERY", "[2] GOLD ANSWER", "[3] GOLD FACTS",
                "[4] GOLD EVIDENCE", "[5] RETRIEVED TEXTUAL EVIDENCE",
                "[6] GRAPH EVIDENCE", "[7] GENERATED RAG ANSWER"):
        assert sec in seen["prompt"], f"missing packet section {sec}"
    # content assertions
    assert "Article 12 requires X." in seen["prompt"]
    assert "REGX Article 12 states Y." in seen["prompt"]
    assert "chunk text A" in seen["prompt"]
    assert "REGX --AMENDS--> ART12 snippet" in seen["prompt"]
    # determinism (same call twice produces the same prompt)
    seen2: dict = {}
    def fake2(model, prompt, **kw):
        seen2["prompt"] = prompt
        return _canned()
    with mock.patch.object(c, "ollama_chat", side_effect=fake2):
        AM.score_generation(row, run_judge=True)
    assert seen["prompt"] == seen2["prompt"]


def test_score_generation_no_claim_list_means_no_ucr():
    """When the judge returns no claims, ``unsupported_claim_rate`` is None
    (not 0.0 -- absence is distinct from 'all supported')."""
    row = {
        "query_id": "Q1", "question": "q", "target": "D:article:1",
        "category": "article", "answer": "a",
        "reference_answer": "r", "user_prompt": "prompt",
    }
    no_claims = json.dumps({
        "correct": True, "faithful": True, "complete": True,
        "evidence_supported": True, "unsupported_claims": False,
        "overall_score": 5, "confidence": 0.95, "needs_escalation": False,
        "claims": [], "relevance": 5, "faithfulness": 5,
        "groundedness": 5, "completeness": 5,
    })
    def fake(model, prompt, **kw):
        return no_claims
    with mock.patch.object(c, "ollama_chat", side_effect=fake):
        res = AM.score_generation(row, run_judge=True)
    assert res["judge"]["unsupported_claim_rate"] is None
    assert res["judge"]["_claims"] == []


def test_aggregate_exposes_new_metric_rows():
    """aggregate() must emit rows for the spec §12 fields, the derived
    rates, and the escalation ratio so the report can surface them."""
    scored = {"hybrid": [
        {"judge": {"correct": True, "faithful": True, "complete": True,
                   "evidence_supported": True, "unsupported_claims": False,
                   "graph_reasoning_correct": True,
                   "overall_score": 5, "confidence": 0.95,
                   "needs_escalation": False,
                   "unsupported_claim_rate": 0.0,
                   "relevance": 5, "faithfulness": 4,
                   "groundedness": 4, "completeness": 4,
                   "escalated": False,
                   "final_model": E.EvalConfig().judge_model}},
        {"judge": {"correct": True, "faithful": True, "complete": False,
                   "evidence_supported": True, "unsupported_claims": True,
                   "graph_reasoning_correct": None,   # N/A
                   "overall_score": 4, "confidence": 0.70,
                   "needs_escalation": True,
                   "unsupported_claim_rate": 1.0,
                   "relevance": 4, "faithfulness": 3,
                   "groundedness": 3, "completeness": 3,
                   "escalated": True,
                   "final_model": E.EvalConfig().judge_escalator}},
    ]}
    rows = AM.aggregate(scored)
    by_name = {r["metric"]: r for r in rows}
    for expected in ("judge__correct", "judge__faithful",
                     "judge__complete", "judge__evidence_supported",
                     "judge__overall_score", "judge__confidence",
                     "judge__unsupported_claim_rate",
                     "judge__graph_reasoning_correct",
                     "judge__escalation_rate"):
        assert expected in by_name, f"missing {expected}"
    # escalation rate: 1 of 2 rows
    assert by_name["judge__escalation_rate"]["value"] == pytest.approx(0.5)
    # overall_score mean: (5 + 4) / 2 = 4.5
    assert by_name["judge__overall_score"]["value"] == pytest.approx(4.5)
    # unsupported_claim_rate mean: (0.0 + 1.0) / 2 = 0.5
    assert by_name["judge__unsupported_claim_rate"]["value"] == pytest.approx(0.5)
    # graph_reasoning: only the first row is applicable (True), second N/A
    assert by_name["judge__graph_reasoning_correct"]["value"] == pytest.approx(1.0)
    # legacy 4 axes still exposed
    assert by_name["judge__relevance"]["value"] == pytest.approx(4.5)
