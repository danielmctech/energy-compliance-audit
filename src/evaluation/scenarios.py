"""Constructed compliance-scenario evaluation (per the "Additional information" section).

Compensates for the absence of expert legal review with two construction
strategies:

1. **Formal constraint synthesis** (core set) -- built directly from
   obligation sentences in the canonical article corpus.  Ground truth is
   *correct by construction*: a clause faithful to the provision is labelled
   ``compliant``; the same clause with a KNOWN, recorded deviation (numeric
   value changed, obligation strength `shall`->`may`) is labelled
   ``non_compliant`` and we know exactly which provision is responsible and
   what the correct value is.  No model judgment is involved in the labels.

2. **LLM-generated, adversarially adjudicated** (larger set) -- two
   architecturally independent local drafters (Gemma 3 27B -> candidate A,
   Mistral Small 3.1 24B -> candidate B) each propose a clause for a real
   provision.  The candidates are ANONYMISED (the adjudicator never sees
   which model wrote which, reducing model-name bias) and judged by an
   independent adversarial adjudicator (Qwen 3.8 27B) on a fixed checklist:
   per-candidate compliance with the provision, score, critical errors.
   The adjudicator never sees the intended label, so it cannot pander to
   it.  A candidate is ACCEPTED when the adjudicator's independent
   compliance verdict matches the intended label.  If NO candidate matches,
   the pair is escalated to a fourth, architecturally independent model
   (Nemotron-3-nano 30B) whose verdict is final; still unmatched ->
   EXCLUDED (never resolved by re-rolling intent).  The drafting model
   never adjudicates its own draft by construction.  Full adjudication +
   escalation transcripts are stored in metadata for auditability.

What the evaluation measures for every (scenario, configuration):
* **retrieval accuracy**  -- does the config's retriever surface the
  responsible provision for the clause (hit + rank + top-k completeness)?
* **judgment accuracy**    -- does the auditor LLM call the clause
  compliant / non-compliant matching the ground truth?
* **correction quality**   -- (non-compliant only) does the proposed
  correction resolve the known deviation and cite the responsible provision?

Four baselines are configurations of the SAME retrieval stack:
  B1 rag_dense_fixed        dense retrieval over the fixed-size chunk corpus
  B2 rag_dense_late         clause is segmented at query time, each sub-clause
                            retrieved, results fused (late-stage segmentation)
  B3 rag_contextual         Anthropic-style contextual retrieval: the source
                            provision's context is prepended to the clause
                            before dense embedding
  B4 rag_hybrid_graph       the full hybrid + graph configuration (KG2RAG-like)
Each dense configuration is run with BOTH the base encoder and the fine-tuned
LoRA encoder so the fine-tuning contribution is isolated from the
architecture contribution (per the "Additional information" section, last paragraph).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import config as E


# ---------------------------------------------------------------------------
# scenario record
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    scenario_id: str
    source_doc: str
    provision_lineage_id: str      # responsible provision (ground truth)
    provision_ref: str             # human ref e.g. "REMIT Art. 27"
    provision_text: str            # full provision body (canonical)
    clause_text: str               # the "contract clause" under audit
    label: str                     # "compliant" | "non_compliant"
    deviation: Optional[dict] = None   # recorded known deviation (synthetic)
    correct_value: Optional[str] = None  # what the clause must say (synthetic)
    construction: str = "synthetic"     # synthetic | llm_voted
    query: str = ""
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "source_doc": self.source_doc,
            "provision_lineage_id": self.provision_lineage_id,
            "provision_ref": self.provision_ref,
            "provision_text": self.provision_text,
            "clause_text": self.clause_text,
            "label": self.label,
            "deviation": self.deviation,
            "correct_value": self.correct_value,
            "construction": self.construction,
            "query": self.query,
            "metadata": self.metadata,
        }


def _audit_query(s: Scenario) -> str:
    return (
        "You are an independent compliance auditor. Assess whether the "
        "contract clause below complies with the governing provision.\n\n"
        f"GOVERNING PROVISION ({s.provision_ref}):\n{s.provision_text}\n\n"
        f"CONTRACT CLAUSE UNDER AUDIT:\n{s.clause_text}\n\n"
        "Return ONLY this JSON:\n"
        '{"compliant": bool,\n'
        ' "responsible_provision": "instrument + article ref",\n'
        ' "issues": ["..."],'
        ' "correction": "the exact corrected clause text (only if non-compliant)"\n}'
    )


# ---------------------------------------------------------------------------
# 1. synthetic constraint construction (truth by construction)
# ---------------------------------------------------------------------------
_NUM = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(hours?|business\s+days?|days?|"
                  r"calendar\s+days?|months?|working\s+days?|percent|%)")


def _obligation_sentences(text: str) -> List[str]:
    """Sentences stating a duty ('shall', 'must')."""
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        s = sent.strip()
        if 60 < len(s) < 600 and ("shall" in s.lower() or "must" in s.lower()):
            out.append(s)
    return out


def _mutate_number(sent: str, rng: random.Random) -> Tuple[str, dict, str]:
    m = _NUM.search(sent)
    if not m:
        return None, None, None
    val = float(m.group(1).replace(",", "."))
    unit = m.group(2)
    alt = round(val * (0.5 if rng.random() < 0.5 else 2.0))
    if alt == val:
        alt = val + (1 if val < 100 else 10)
    new_sent = sent.replace(m.group(0), f"{int(alt)} {unit}", 1)
    if new_sent == sent:
        return None, None, None
    dev = {"type": "numeric_value", "field": unit,
           "original": m.group(0), "altered": f"{int(alt)} {unit}"}
    return new_sent, dev, m.group(0)


def _mutate_strength(sent: str) -> Tuple[str, dict, str]:
    if "shall" in sent:
        new_sent = sent.replace("shall", "may", 1)
    elif "must" in sent:
        new_sent = sent.replace("must", "should", 1)
    else:
        return None, None, None
    dev = {"type": "obligation_strength",
           "original": "shall" if "shall" in sent else "must",
           "altered": "may" if "shall" in sent else "should"}
    return new_sent, dev, None


def build_synthetic_scenarios(corpus, per_doc: int = 3,
                              seed: int = E.SEED
                              ) -> List[Scenario]:
    """Deterministic: pick obligation sentences per document, label them
    compliant (as-is) or non-compliant (with a recorded deviation)."""
    rng = random.Random(seed)
    from collections import defaultdict
    arts: Dict[str, list] = defaultdict(list)
    for c in corpus["chunks"]:
        if c.node_type == "article" and len(c.text) > 200:
            arts[c.doc_id].append(c)
    doc_ids = sorted(arts)
    out: List[Scenario] = []
    i = 0
    for doc in doc_ids:
        cands = []
        for c in arts[doc]:
            for sent in _obligation_sentences(c.text):
                cands.append((c, sent))
        rng.shuffle(cands)
        made = 0
        for c, sent in cands:
            new_sent, dev, correct = _mutate_number(sent, rng)
            if new_sent is None:
                new_sent, dev, correct = _mutate_strength(sent)
                if new_sent is None:
                    continue
            i += 1
            ref = f"{doc.replace('_', ' ').split()[0].upper()} " \
                  f"Art. {c.lineage_id.rsplit(':', 1)[-1]}"
            s = Scenario(
                scenario_id=f"syn_{i:03d}",
                source_doc=doc,
                provision_lineage_id=c.lineage_id,
                provision_ref=ref,
                provision_text=c.text[:2000],
                clause_text=f"Obligation clause: {new_sent}",
                label="non_compliant",
                deviation=dev, correct_value=correct,
                construction="synthetic",
            )
            s.query = _audit_query(s)
            out.append(s)
            # matching compliant twin (the faithful original sentence)
            s2 = Scenario(
                scenario_id=f"syn_{i:03d}c",
                source_doc=doc,
                provision_lineage_id=c.lineage_id,
                provision_ref=ref,
                provision_text=c.text[:2000],
                clause_text=f"Obligation clause: {sent}",
                label="compliant",
                construction="synthetic",
            )
            s2.query = _audit_query(s2)
            out.append(s2)
            made += 1
            if made >= per_doc:
                break
    return out


# ---------------------------------------------------------------------------
# 2. LLM-drafted, adversarially adjudicated protocol
# ---------------------------------------------------------------------------
def _llm_clause(model: str, scenario_seed_text: str,
                want: str) -> Dict:
    """One drafter proposes a single clause.  ``want`` tells the drafter
    WHAT to build (compliant / deviating); the adjudicator never sees it."""
    import common as c
    prompt = (
        "You draft a single contract clause relevant to the provision.\n"
        f"Provision:\n{scenario_seed_text[:1500]}\n\n"
        f"Draft ONE clause that is {want} with this provision.\n"
        "If non-compliant it must deviate in one specific, checkable way "
        "(number, deadline, or actor) but stay plausible.\n"
        "Return ONLY JSON:\n"
        '{"clause": "...", "deviation_if_any": "one sentence describing the deviation or null"}'
    )
    raw = c.ollama_chat(model=model, prompt=prompt, temperature=0.2,
                        use_cache=True,
                        cache_key=f"scen_draft_{model}_{want}_{c.stable_hash(scenario_seed_text)}")
    return c.extract_json(raw)


_ADJ_CHECKLIST = (
    "For EACH candidate independently, check:\n"
    "1. Hard-constraint compliance with the provision (every obligation, "
    "value, deadline, actor)\n"
    "2. Internal consistency (no self-contradiction)\n"
    "3. Factual / legal plausibility (nothing invented that the provision "
    "does not support)\n"
    "4. A single, verifiable deviation instead of a vague one (if it deviates)\n"
    "5. Duplication of boilerplate / no filler\n"
    "6. Edge cases and unintended assumptions\n"
    "Do NOT assume either candidate is correct.  Decide the COMPLIANCE of "
    "each candidate against the provision on its own merits."
)


def _adjudicate(model: str, provision: str,
                clause_a: str, clause_b: str) -> Dict:
    """Independent adversarial adjudicator over the two ANONYMISED drafts.

    The model name, drafter identities and intended label are withheld; the
    judge sees only Candidate A / Candidate B.  Returns a per-candidate
    verdict + scores (never a pass/fail on intent).
    """
    import common as c
    prompt = (
        "You are the adjudicator in an independent legal review.\n"
        f"Provision under audit:\n{provision[:1500]}\n\n"
        "Candidate A:\n{a}\n\nCandidate B:\n{b}\n\n" +
        _ADJ_CHECKLIST +
        "\nReturn ONLY JSON:\n"
        '{"a": {"compliant": bool, "score": 0-10, "critical_errors": []},'
        ' "b": {"compliant": bool, "score": 0-10, "critical_errors": []},'
        ' "better": "A"|"B"|"tie"|null, "notes": "one sentence"}'
    ).replace("{a}", clause_a).replace("{b}", clause_b)
    raw = c.ollama_chat(model=model, prompt=prompt, temperature=0.0,
                        use_cache=True,
                        cache_key=f"scen_adj_{c.stable_hash(model + provision + clause_a + clause_b)}")
    d = c.extract_json(raw)
    return {
        "a": {"compliant": bool((d.get("a") or {}).get("compliant")),
              "score": (d.get("a") or {}).get("score"),
              "critical_errors": (d.get("a") or {}).get("critical_errors") or []},
        "b": {"compliant": bool((d.get("b") or {}).get("compliant")),
              "score": (d.get("b") or {}).get("score"),
              "critical_errors": (d.get("b") or {}).get("critical_errors") or []},
        "better": d.get("better"),
        "notes": str(d.get("notes") or "")[:200],
    }


def _escalate(model: str, provision: str,
              clause_a: str, clause_b: str,
              intent: str, adj: Dict) -> Dict:
    """Dispute path: an architecturally independent FOURTH model makes the
    final decision on whether the better candidate satisfies the intended
    requirement (given the intent, unlike the blind adjudicator).  Runs only
    when the blind adjudicator could not confirm the intended label."""
    import common as c
    intent_text = ("COMPLIANT with the provision" if intent == "compliant"
                   else "NON-COMPLIANT: it must deviate in ONE specific, "
                        "checkable way (number, deadline, or actor)")
    prompt = (
        "You are the final reviewer in an independent legal review, "
        "adjudicating two drafted candidates.\n"
        f"Provision:\n{provision[:1500]}\n\n"
        f"Requirement for the acceptable candidate: {intent_text}.\n\n"
        f"Adjudicator's independent findings (not binding):\n"
        f"  A: {adj['a']}\n  B: {adj['b']}\n\n"
        "Candidate A:\n{a}\n\nCandidate B:\n{b}\n\n"
        "Decide: does candidate A or candidate B satisfy the requirement?\n"
        "Return ONLY JSON:\n"
        '{"choice": "A"|"B"|"neither", "compliance_of_choice": bool,'
        ' "reason": "one sentence"}'
    ).replace("{a}", clause_a).replace("{b}", clause_b)
    raw = c.ollama_chat(model=model, prompt=prompt, temperature=0.0,
                        use_cache=True,
                        cache_key=f"scen_esc_{c.stable_hash(model + intent + provision + clause_a + clause_b)}")
    return c.extract_json(raw)


def build_voted_scenarios(corpus, max_scenarios: int = 24,
                          drafters: Optional[Tuple[str, str]] = None,
                          seed: int = E.SEED) -> List[Scenario]:
    """Two-drafter -> blind-adjudicator -> dispute-escalation protocol.

    Per (provision, intended label):
      1. drafter A (cfg.scenario_drafter_a) drafts candidate A;
         drafter B (cfg.scenario_drafter_b) drafts candidate B.
      2. the BLIND adjudicator (cfg.scenario_adjudicator) judges the
         ANONYMISED candidates; it never sees the intended label, so it
         cannot pander to it.  A candidate qualifies when the adjudicator's
         independent compliance verdict equals the intended label.  If one
         or both qualify, the highest-scoring qualifying candidate is
         accepted (no escalation).  NOTE: a drafter never adjudicates its
         own draft; each draft is judged only by the blind adjudicator.
      3. DISPUTE (neither draft qualifies): the architecturally independent
         fourth model (cfg.scenario_escalator) makes the final call, invoked
         only on disputes.  "neither" / inconsistent -> EXCLUDED.
         Disagreements are never silently re-labelled.
    Acceptance detail (both drafts, adjudication transcript + escalation
    when used, decision path) is stored in ``metadata`` for auditability.
    """
    cfg = E.EvalConfig()
    drafter_a = (drafters[0] if drafters else cfg.scenario_drafter_a)
    drafter_b = (drafters[1] if drafters else cfg.scenario_drafter_b)
    adjudicator = cfg.scenario_adjudicator
    escalator = cfg.scenario_escalator

    rng = random.Random(seed)
    from collections import defaultdict
    arts: Dict[str, list] = defaultdict(list)
    for cc in corpus["chunks"]:
        if cc.node_type == "article" and _obligation_sentences(cc.text):
            arts[cc.doc_id].append(cc)
    pool = [c for _, arr in arts.items() for c in arr]
    rng.shuffle(pool)

    out: List[Scenario] = []
    i = 0
    drafts_per_want = max_scenarios // 2
    for want in ("compliant", "non_compliant"):
        made = 0
        for c in pool:
            if made >= drafts_per_want:
                break
            ref = f"{c.doc_id.replace('_', ' ').split()[0].upper()} " \
                  f"Art. {c.lineage_id.rsplit(':', 1)[-1]}"
            want_bool = (want == "compliant")
            # -- 1. two independent drafts ---------------------------------
            try:
                da = _llm_clause(drafter_a, c.text, want)
                db = _llm_clause(drafter_b, c.text, want)
            except Exception:
                continue
            clause_a = (da.get("clause") or "").strip()
            clause_b = (db.get("clause") or "").strip()
            if len(clause_a) < 40 or len(clause_b) < 40:
                continue
            # -- 2. blind adjudication (no model judged its own draft) ------
            try:
                adj = _adjudicate(adjudicator, c.text, clause_a, clause_b)
            except Exception:
                continue
            verdict = {"A": adj["a"]["compliant"], "B": adj["b"]["compliant"]}
            qualifying = [k for k, v in verdict.items() if v == want_bool]
            decision = None
            path = None
            esc = None
            if len(qualifying) >= 1:
                # at least one draft satisfies the intended requirement ->
                # accept the best-scoring among the qualifying candidates
                # (no model ever adjudicates its own draft: A/drafter_a and
                # B/drafter_b are both judged only by the blind adjudicator)
                def _score(k):
                    v = adj["a" if k == "A" else "b"]["score"]
                    return v if isinstance(v, (int, float)) else -1
                decision = max(qualifying, key=_score)
                path = "both_qualify" if len(qualifying) == 2 else "single_qualify"
            else:
                # DISPUTE: the blind adjudicator confirms NEITHER draft meets
                # the intended requirement -> the independent fourth model
                # (escalator) makes the final call (runs only on disputes).
                try:
                    esc = _escalate(escalator, c.text, clause_a, clause_b,
                                    want, adj)
                except Exception:
                    esc = {"error": "escalation_failed"}
                if esc.get("error"):
                    continue  # escalation failed -> excluded, never fabricated
                if esc.get("choice") in ("A", "B") \
                        and bool(esc.get("compliance_of_choice")) == want_bool:
                    decision = esc["choice"]
                    path = "escalation_confirmed"
                else:
                    continue  # escalated reviewer rejected -> excluded
            if decision is None:
                continue
            i += 1
            clause = clause_a if decision == "A" else clause_b
            deviation = da if decision == "A" else db
            s = Scenario(
                scenario_id=f"vote_{i:03d}",
                source_doc=c.doc_id,
                provision_lineage_id=c.lineage_id,
                provision_ref=ref,
                provision_text=c.text[:2000],
                clause_text=clause,
                label=want,
                deviation={"type": "llm_drafted",
                           "description": deviation.get("deviation_if_any")}
                ,
                construction="llm_voted",
            )
            s.metadata = {
                "protocol": "two_drafters_blind_adjudicator_dispute_escalation",
                "drafts": {"A": {"model": drafter_a, "clause": clause_a},
                           "B": {"model": drafter_b, "clause": clause_b}},
                "adjudicator": {"model": adjudicator, **adj},
                "chosen": decision,
                "decision_path": path,   # both_qualify|single_qualify|escalation_confirmed
            }
            if esc is not None:
                s.metadata["escalation"] = {
                    "model": escalator,
                    "choice": esc.get("choice"),
                    "compliance_of_choice": esc.get("compliance_of_choice"),
                    "reason": esc.get("reason"),
                }
            s.query = _audit_query(s)
            out.append(s)
            made += 1
    return out


# ---------------------------------------------------------------------------
# baselines + audit
# ---------------------------------------------------------------------------
def audit_scenario(s: Scenario,
                   judge_model: Optional[str] = None) -> Dict:
    """Run the auditor LLM on one scenario scenario; parse its decision.

    Cached via common.ollama_chat -> deterministic & free on re-run.  Parse
    failures -> {"compliant": None, "error": ...} (never a fabricated vote).
    """
    import common as c
    judge_model = judge_model or E.EvalConfig().llm_model
    try:
        raw = c.ollama_chat(model=judge_model, prompt=s.query,
                            system="You audit clauses against the provision. "
                                  "Be strict. JSON only.",
                            temperature=0.0, use_cache=True,
                            cache_key=f"audit_{judge_model}_{c.stable_hash(s.scenario_id)}")
        d = c.extract_json(raw)
        return {
            "compliant": bool(d.get("compliant")),
            "responsible_provision": str(d.get("responsible_provision"))[:120],
            "issues": d.get("issues") or [],
            "correction": d.get("correction") or "",
            "model": judge_model,
        }
    except Exception as exc:  # noqa: BLE001
        return {"compliant": None, "correction": "",
                "model": judge_model,
                "error": f"{type(exc).__name__}: {exc}"}


def correction_resolved(s: Scenario, correction: str) -> Dict:
    """Check a proposed correction against the KNOWN deviation (synthetic).

    A correction 'resolves' the deviation when the clause after correction is
    no longer the mutated string and, for numeric deviations, restores the
    correct value.  Returns booleans -- no LLM judgment, by construction.
    """
    if s.deviation is None:
        return {"resolved": None, "reason": "no recorded deviation"}
    cor = (correction or "").strip()
    if not cor:
        return {"resolved": False, "reason": "no correction text"}
    resolved = cor != s.clause_text
    if s.deviation.get("type") == "numeric_value" and s.correct_value:
        if s.correct_value not in cor:
            resolved = False
    if s.deviation.get("type") == "obligation_strength":
        want = "shall" if s.deviation.get("original") == "shall" else "must"
        if want not in cor:
            resolved = False
    return {"resolved": resolved,
            "reason": "correction restores the canonical value/wording"}


def detection_accuracy(results: List[Dict]) -> Dict:
    """Judgment accuracy over scenarios where the auditor returned a call."""
    calls = [r for r in results if r.get("audit", {}).get("compliant") is not None]
    if not calls:
        return {"accuracy": None, "n": 0, "note": "no auditor calls"}
    ok = sum(1 for r in calls
             if bool(r["audit"]["compliant"]) == (r["expected"] == "compliant"))
    return {"accuracy": ok / len(calls), "n": len(calls),
            "n_total": len(results)}


def retrieval_quality(s: Scenario, systems: Dict[str, list]) -> Dict:
    """Per-config: did top-k retrieval surface the responsible provision?

    ``systems`` = {config_name: top-k lineage_ids (rank order)}.
    """
    target = s.provision_lineage_id
    out = {}
    for cfg, lids in systems.items():
        hit = target in lids
        out[cfg] = {
            "hit": hit,
            "rank": (lids.index(target) + 1) if hit else None,
            "top3": lids[:3],
        }
    return out


def run_scenario_evaluation(
        scenarios: Sequence[Scenario],
        retrievals: Optional[Dict[str, Dict[str, list]]] = None,
        n_scenarios: Optional[int] = None,
        out_dir: Path = E.OUT_GRAPHS) -> List[Dict]:
    """Evaluate audit decision + per-baseline retrieval + correction quality.

    ``retrievals`` (optional) is ``{scenario_id: {baseline_name:
    [top-k lineage_ids]}}`` -- produced by ``baselines.baseline_retrievals``.
    Passing ``None`` still runs the audit + correction checks but leaves
    ``retrieval`` empty (per "No Fabricated Results"; no fabricated retrieval numbers).
    """
    scenarios = list(scenarios)
    if n_scenarios:
        scenarios = scenarios[:n_scenarios]
    retrievals = retrievals or {}
    rows = []
    for s in scenarios:
        audit = audit_scenario(s)
        expected = s.label
        judged = bool(audit.get("compliant")) if audit.get("compliant") is not None else None
        row = {
            "scenario_id": s.scenario_id,
            "construction": s.construction,
            "expected": expected,
            "audit": audit,
            "judgment_correct": (judged is not None
                                 and judged == (expected == "compliant")),
            "retrieval": retrieval_quality(
                s, retrievals.get(s.scenario_id, {})),
        }
        if expected == "non_compliant" and s.deviation:
            row["correction"] = correction_resolved(
                s, audit.get("correction", ""))
        rp = (audit.get("responsible_provision") or "").lower()
        art_no = s.provision_lineage_id.rsplit(":", 1)[-1]
        row["cites_responsible"] = bool(
            re.search(r"art\s*\.?\s*" + re.escape(art_no) + r"\b", rp)
            or s.source_doc.split("_")[0].lower() in rp) or None
        rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scenario_evaluation.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # aggregates
    det = detection_accuracy(rows)
    corr_rows = [r for r in rows if r.get("correction") is not None]
    corr_ok = sum(1 for r in corr_rows if r["correction"]["resolved"])
    cites_vals = [r.get("cites_responsible") for r in rows
                  if isinstance(r.get("cites_responsible"), bool)]
    # per-baseline retrieval summary
    baseline_hits: Dict[str, dict] = {}
    for r in rows:
        for b, q in (r.get("retrieval") or {}).items():
            h = baseline_hits.setdefault(b, {"hit": 0, "n": 0,
                                              "ranks": []})
            h["n"] += 1
            if q["hit"]:
                h["hit"] += 1
                h["ranks"].append(q["rank"])
    for b, h in baseline_hits.items():
        h["hit_rate"] = h["hit"] / h["n"] if h["n"] else None
        h["mean_rank"] = (sum(h["ranks"]) / len(h["ranks"])) \
            if h["ranks"] else None
        del h["ranks"]
    agg = {
        "detection": det,
        "correction_resolved": (corr_ok / len(corr_rows)) if corr_rows else None,
        "n_correction_checked": len(corr_rows),
        "cites_responsible": (sum(cites_vals) / len(cites_vals))
                             if cites_vals else None,
        "baseline_retrieval": baseline_hits,
        "n_rows": len(rows),
        "audit_model": E.EvalConfig().llm_model,
    }
    (out_dir / "scenario_agg.json").write_text(json.dumps(agg, indent=2))
    return rows
