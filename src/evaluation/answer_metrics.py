"""Answer-level metrics for Layer B (per the "Answer-Level Metrics", "Groundedness / Faithfulness" and "RAG Triangulation" requirements).

Because the gold set carries a *target passage* as its reference (not a
human-authored gold answer -- see ``benchmark.py``), each answer metric is
chosen to be meaningful against a passage and is reported with that caveat:

* ``exact_match`` / ``token_f1`` -- against the target passage.  Will be low
  for summary-style answers; that difference is itself a finding.
* ``semantic_similarity`` -- L2-normalised bge-m3 cosine of the generated
  answer vs the target passage (model / function / normalization documented
  in ``semantic.py``).
* ``cites_target`` -- does the answer name the target instrument / article?
  A cheap, defensible groundedness proxy (per the "Groundedness / Faithfulness" requirement).
* ``judge`` -- the LLM-as-judge dict, now spec-aligned (see below):
  a 7-section evaluation packet drives a primary judge that emits both
  legacy 4-axis scores (``relevance / faithfulness / groundedness /
  completeness`` on 1..5, retained for backward compatibility) and the
  spec-required boolean / float fields (``correct``, ``faithful``,
  ``complete``, ``evidence_supported``, ``unsupported_claims``,
  ``graph_reasoning_correct``, ``overall_score``, ``confidence``,
  ``needs_escalation``), plus per-claim support rows (``claims``) with
  a derived ``unsupported_claim_rate``.
  A confidence gate re-routes rows whose primary-judge ``confidence`` is
  below ``judge_confidence_threshold`` -- or whose ``overall_score``
  falls in ``judge_escalate_scores`` -- to ``judge_escalator`` (spec §13
  escalation policy).  The ``judge`` dict therefore records ``final_model``
  = the model that produced the emitted scores and ``escalated`` /
  ``needs_escalation`` booleans.

The judge is never treated as ground truth; it is one axis among
several and is logged with its model id.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import config as E
from .metrics import exact_match, token_f1
from .benchmark import BenchmarkItem
from . import semantic


def semantic_similarity(answer: str, reference: str,
                        model: str = semantic.DEFAULT_MODEL
                        ) -> Optional[float]:
    """L2-normalised bge-m3 cosine of answer vs reference (``None`` if blank)."""
    return semantic.cosine(answer, reference, model)


def cites_target(answer: str, item: BenchmarkItem) -> bool:
    """Heuristic: does the answer reference the target instrument / article?"""
    if not answer:
        return False
    a = answer.lower()
    doc = (item.doc_id or "").lower()
    tokens = [t for t in re.split(r"[_\s]", doc) if len(t) >= 3]
    hit_doc = any(t in a for t in tokens)
    num = re.search(r":article:(\d+)", item.target_lineage_id) or \
          re.search(r":preamble:(\d+)", item.target_lineage_id)
    hit_art = bool(num and re.search(r"article\s*" + num.group(1) + r"\b", a))
    return hit_doc or hit_art


#: The 4 legacy axes (kept for backward compatibility with existing Table B
#: consumers; the spec-aligned boolean fields are added alongside, not in
#: place of, these).
_JUDGE_AXES = ("relevance", "faithfulness", "groundedness", "completeness")

#: spec §12 boolean / score fields (all optional in the judge's JSON).
_SPEC_FIELDS = (
    "correct", "faithful", "complete", "evidence_supported",
    "unsupported_claims", "graph_reasoning_correct",
)


def _rubric_prompt(packet: Dict[str, str]) -> str:
    """Render the 7-section evaluation packet (spec §12).

    ``packet`` keys (all optional; missing = "(none)"):
        ``question, answer, gold_answer, gold_facts, gold_evidence,
        retrieved_evidence, graph_evidence``.
    """
    g = lambda key: (packet.get(key) or "").strip() or "(none)"
    return (
        "You are grading a compliance-LLM answer about EU/UK energy, "
        "digital and market regulation. Be strict and consistent.  Judge "
        "ONLY against the supplied gold facts, gold evidence, graph "
        "evidence, and retrieved evidence provided below -- do not use "
        "outside knowledge.  Claim-level support matters more than a "
        "single overall judgement.  For graph-dependent answers, verify "
        "that the relation direction (AMENDS vs AMENDED_BY) and relation "
        "type cited in the answer are correct.\n\n"
        "=================  EVALUATION PACKET  =================\n\n"
        f"[1] QUERY:\n{g('question')}\n\n"
        f"[2] GOLD ANSWER (source-derived ground-truth reference from the PDF):\n"
        f"{g('gold_answer')}\n\n"
        f"[3] GOLD FACTS (canonical structured facts extracted from the source graph):\n"
        f"{g('gold_facts')}\n\n"
        f"[4] GOLD EVIDENCE (the target chunk from the source instrument):\n"
        f"{g('gold_evidence')}\n\n"
        f"[5] RETRIEVED TEXTUAL EVIDENCE (what the RAG system retrieved for this answer):\n"
        f"{g('retrieved_evidence')}\n\n"
        f"[6] GRAPH EVIDENCE (typed/directional relations of the target instrument):\n"
        f"{g('graph_evidence')}\n\n"
        f"[7] GENERATED RAG ANSWER (the answer being evaluated):\n"
        f"{g('answer')}\n\n"
        "=======================================================\n\n"
        "Score EACH of the four legacy axes on 1..5 (1 worst, 5 best, "
        "never 0):\n"
        "  relevance     - addresses the question, no off-topic filler\n"
        "  faithfulness  - uses only the supplied evidence, no invention\n"
        "  groundedness  - claims cite a correct instrument + article number\n"
        "  completeness  - covers all obligations the evidence raises\n\n"
        "Then emit the spec fields:\n"
        "  correct             (bool) -- the answer is factually correct on the "
        "                          material points\n"
        "  faithful            (bool) -- every claim is grounded in the supplied "
        "                          gold/retrieved evidence (no invention)\n"
        "  complete            (bool) -- covers the material obligations the gold "
        "                          evidence raises\n"
        "  evidence_supported  (bool) -- the key claims are backed by the "
        "                          supplied evidence\n"
        "  unsupported_claims  (bool) -- true if ANY claim is unsupported by the "
        "                          supplied evidence\n"
        "  graph_reasoning_correct (bool or null when no graph evidence applies) -- "
        "                          relation direction and type are correct\n"
        "  overall_score       (int 1..5)\n"
        "  confidence          (float 0.0..1.0; how sure you are of these "
        "                          scores)\n"
        "  needs_escalation    (bool; true if the case is ambiguous / you are "
        "                          not confident / gold-vs-retrieved conflict)\n\n"
        "Then emit per-claim rows (up to 10 claims; use 0 when trivial):\n"
        "  claims : a list of {\"claim\": <short text>, \"supported\": <bool>} "
        "           rows, one per material claim made in the answer.\n\n"
        "Respond with ONLY this JSON, no prose, no code fence:\n"
        "{\n"
        '  "correct": bool,\n'
        '  "faithful": bool,\n'
        '  "complete": bool,\n'
        '  "evidence_supported": bool,\n'
        '  "unsupported_claims": bool,\n'
        '  "graph_reasoning_correct": bool | null,\n'
        '  "overall_score": int (1..5),\n'
        '  "confidence": float (0.0..1.0),\n'
        '  "needs_escalation": bool,\n'
        '  "claims": [ {"claim": str, "supported": bool}, ... ],\n'
        '  "relevance": int (1..5),\n'
        '  "faithfulness": int (1..5),\n'
        '  "groundedness": int (1..5),\n'
        '  "completeness": int (1..5)\n'
        "}"
    )


def _coerce_int(v, lo: int, hi: int) -> Optional[int]:
    """Coerce to int clamped to [lo, hi]; ``None`` when not a real number."""
    if v is None:
        return None
    try:
        iv = int(v)
    except (TypeError, ValueError):
        return None
    return max(lo, min(hi, iv))


def _coerce_bool(v) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "t", "1"}:
            return True
        if s in {"false", "f", "0"}:
            return False
        return None
    return None


def _coerce_confidence(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f < 0.0:
        f = 0.0
    if f > 1.0:
        f = 1.0
    return f


def _coerce_claims(v) -> List[Dict]:
    if not isinstance(v, list):
        return []
    out: List[Dict] = []
    for row in v[:10]:
        if not isinstance(row, dict):
            continue
        c = (row.get("claim") or "").strip()
        if not c:
            continue
        sup = _coerce_bool(row.get("supported"))
        out.append({"claim": c, "supported": (True if sup is None else sup)})
    return out


def _extract_judge_json(raw: str) -> Dict:
    """Parse the judge's JSON, coercing to the spec schema.

    Returns a dict with every spec key present (``None`` when not emitted);
    ``unsupported_claim_rate`` (0.0..1.0 when claims are present, ``None``
    otherwise), ``_claims`` (the coerced claim list for downstream use).
    """
    import common as c
    obj = c.extract_json(raw)
    if not isinstance(obj, dict):
        raise ValueError("judge JSON was not a dict")
    out: Dict = {}
    for f in _SPEC_FIELDS:
        out[f] = _coerce_bool(obj.get(f)) if f != "graph_reasoning_correct" \
            else _coerce_bool(obj.get(f))
    out["overall_score"] = _coerce_int(obj.get("overall_score"), 1, 5)
    out["confidence"] = _coerce_confidence(obj.get("confidence"))
    out["needs_escalation"] = _coerce_bool(obj.get("needs_escalation")) \
        or False
    claims = _coerce_claims(obj.get("claims"))
    out["_claims"] = claims
    n_unsup = sum(1 for c in claims if c.get("supported") is False)
    out["unsupported_claim_rate"] = (n_unsup / len(claims)) if claims else None
    # legacy 4 axes (kept for downstream backward compatibility)
    for ax in _JUDGE_AXES:
        out[ax] = _coerce_int(obj.get(ax), 1, 5)
    return out


def _should_escalate(v: Dict, cfg: E.EvalConfig) -> bool:
    """Spec §13 escalation policy: confidence < threshold, or the overall
    score falling in the configured set, or a model-declared
    ``needs_escalation``.
    """
    conf = v.get("confidence")
    if conf is not None and conf < cfg.judge_confidence_threshold:
        return True
    score = v.get("overall_score")
    if score is not None and score in cfg.judge_escalate_scores:
        return True
    return bool(v.get("needs_escalation"))


def _primary_judge(packet: Dict[str, str], cfg: E.EvalConfig,
                   use_cache: bool) -> Optional[Dict]:
    """Ask ``cfg.judge_model`` to score the packet; return a coerced dict.

    ``None`` when the LLM is unreachable or the output cannot be parsed.
    """
    import common as c
    prompt = _rubric_prompt(packet)
    cache_key = c.stable_hash("judge", cfg.judge_model, cfg.judge_scale,
                              prompt)
    try:
        raw = c.ollama_chat(
            model=cfg.judge_model,
            prompt=prompt,
            temperature=0.0,
            use_cache=use_cache,
            cache_key=cache_key,
        )
    except Exception as exc:  # noqa: BLE001 -- record, never fabricate
        return None
    try:
        return _extract_judge_json(raw)
    except Exception as exc:  # noqa: BLE001
        return None  # caller falls back to a zero-valued "record, don't fabricate" row


def _escalation_judge(packet: Dict[str, str], cfg: E.EvalConfig,
                      use_cache: bool) -> Optional[Dict]:
    """Ask ``cfg.judge_escalator`` (Nemotron) to re-score the same packet."""
    import common as c
    prompt = _rubric_prompt(packet)
    cache_key = c.stable_hash("judge_esc", cfg.judge_escalator,
                              cfg.judge_scale, prompt)
    try:
        raw = c.ollama_chat(
            model=cfg.judge_escalator,
            prompt=prompt,
            temperature=0.0,
            use_cache=use_cache,
            cache_key=cache_key,
        )
    except Exception as exc:  # noqa: BLE001
        raise
    try:
        return _extract_judge_json(raw)
    except Exception:  # noqa: BLE001
        return None


def llm_judge(answer: str,
              question: str,
              evidence: Optional[str] = None,
              *,
              gold_answer: Optional[str] = None,
              gold_facts: Optional[str] = None,
              gold_evidence: Optional[str] = None,
              graph_evidence: Optional[str] = None,
              model: Optional[str] = None,
              escalator_model: Optional[str] = None,
              confidence_threshold: Optional[float] = None,
              use_cache: bool = True) -> Dict:
    """Spec-aligned answer judge.

    The model (default ``cfg.judge_model``; overridden by ``model`` when
    given) scores the 7-section evaluation packet and returns a dict with
    the spec fields plus the legacy 4 axes.  When the confidence gate
    fires (confidence below ``confidence_threshold`` or overall_score in
    the configured set), ``cfg.judge_escalator`` (or ``escalator_model``
    when given) re-scores the same packet and its output is what the
    caller sees -- the ``judge`` dict records ``escalated=True`` and
    ``primary_model`` / ``final_model`` so the audit trail stays intact.

    If either model is unreachable or output cannot be parsed, the dict is
    a zero-valued fallback with ``error`` (per the "No Fabricated Results"
    requirement).
    """
    cfg = E.EvalConfig()
    primary = model or cfg.judge_model
    escalator = escalator_model or cfg.judge_escalator
    if confidence_threshold is None:
        threshold = cfg.judge_confidence_threshold
    else:
        threshold = float(confidence_threshold)

    packet = {
        "question": question,
        "answer": answer,
        "gold_answer": gold_answer,
        "gold_facts": gold_facts,
        "gold_evidence": gold_evidence,
        "retrieved_evidence": evidence,
        "graph_evidence": graph_evidence,
    }

    out = _primary_judge(packet, cfg, use_cache)
    if out is None:
        # model down -- record as empty, not fabricated (per spec)
        out = {f: 0 for f in _JUDGE_AXES}
        out.update({k: None for k in _SPEC_FIELDS})
        out["overall_score"] = None
        out["confidence"] = None
        out["needs_escalation"] = False
        out["_claims"] = []
        out["unsupported_claim_rate"] = None
        out["escalated"] = False
        out["primary_model"] = primary
        out["final_model"] = primary
        out["error"] = "primary judge unavailable"
        return out
    # -- escalation gate (spec §13) --
    escalated = _should_escalate(out, cfg)
    if escalated:
        try:
            esc = _escalation_judge(packet, cfg, use_cache)
        except Exception as exc:  # noqa: BLE001 -- escalator down
            esc = None
            esc_err = f"{type(exc).__name__}: {exc}"
        else:
            esc_err = None
        if esc is not None:
            # adopt the escalator's output as the final judgment
            for f in _SPEC_FIELDS + _JUDGE_AXES:
                out[f] = esc.get(f)
            out["overall_score"] = esc.get("overall_score")
            out["confidence"] = esc.get("confidence")
            out["needs_escalation"] = True
            out["unsupported_claim_rate"] = esc.get("unsupported_claim_rate")
            out["_claims"] = esc.get("_claims", [])
            out["escalated"] = True
            out["final_model"] = escalator
        else:
            # escalator unavailable / unparseable -- keep the primary (partial)
            # result and record the miss so we can audit later.
            out["escalated"] = True
            out["escalation_error"] = esc_err or "escalator unavailable"
            out["final_model"] = primary
            out["needs_escalation"] = True
    else:
        out["escalated"] = False
        out["final_model"] = primary
    # ``_claims`` is intentionally retained -- it is a spec §15 field the
    # report consumes (per-claim support rows + derived unsupported_claim_rate).
    out["primary_model"] = primary
    return out


def _item_from_row(row: Dict) -> BenchmarkItem:
    lid = row.get("target", "")
    doc = lid.split(":", 1)[0] if lid else ""
    return BenchmarkItem(
        query_id=row.get("query_id", ""),
        question=row.get("question", ""),
        doc_id=doc,
        target_lineage_id=lid,
        category=row.get("category", "unknown"),
    )


def score_generation(row: Dict,
                     judge_model: Optional[str] = None,
                     emb_model: str = semantic.DEFAULT_MODEL,
                     run_judge: bool = True) -> Dict:
    """Attach answer metrics to one generation row.

    The judge receives the spec §12 7-section evaluation packet.  Fields:

    * ``gold_answer`` = ``row["reference_answer"]`` (the target-passage text;
      benchmark.py ``reference_basis`` = "target_chunk_text", so the gold
      *is* a source passage, not a paraphrase).
    * ``gold_evidence`` = same as ``gold_answer`` (this gold set has no
      separate facts layer).
    * ``gold_facts``    = ``row["graph_evidence"]`` when present (typed/directional
      relations of the target), else empty.
    * ``retrieved_evidence`` = the prompt context (the retrieved chunks) from
      ``row["user_prompt"]`` (the generation prompt template in ``generation.py``
      is the single source of truth for what was retrieved).
    * ``graph_evidence`` = ``row["graph_evidence"]`` when present.
    """
    ans = row.get("answer", "")
    ref = row.get("reference_answer", "")
    item = _item_from_row(row)
    out = {
        "exact_match": exact_match(ans, ref),
        "token_f1": token_f1(ans, ref),
        "semantic_similarity": semantic_similarity(ans, ref, emb_model),
        "cites_target": int(cites_target(ans, item)),
    }
    if run_judge:
        question = row.get("question", "")
        # user_prompt is the standardised context template -- strip the
        # preamble so the judge sees the actual evidence sections, not the
        # boilerplate instruction.
        user_prompt = row.get("user_prompt", "")
        retrieved = re.sub(
            r"^Relevant regulatory evidence \(top \d+ retrieved chunks\):\s*\n",
            "", user_prompt, count=1)
        retrieved = re.sub(
            r"Question:\s*.*$", "", retrieved,
            flags=re.DOTALL).strip() or user_prompt
        gr_ev = (row.get("graph_evidence") or "").strip()
        gf = gr_ev  # this gold set has no separate facts layer; reuse graph
        out["judge"] = llm_judge(
            answer=ans,
            question=question,
            evidence=retrieved,
            gold_answer=ref,
            gold_facts=gf,
            gold_evidence=ref,
            graph_evidence=gr_ev,
            model=judge_model,
        )
    else:
        out["judge"] = None
    return out


def _unsupported_claim_rate(rows: Sequence[Dict]) -> Optional[float]:
    """Mean ``unsupported_claim_rate`` across rows that have a claim list."""
    vals = [r.get("judge", {}).get("unsupported_claim_rate")
            for r in rows
            if isinstance((r.get("judge") or {}).get("unsupported_claim_rate"),
                          (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _graph_reasoning_accuracy(rows: Sequence[Dict]) -> Optional[float]:
    """Fraction of rows where ``graph_reasoning_correct`` is True (or None /
    N/A rows are excluded).
    """
    applicable = [r["judge"]["graph_reasoning_correct"] for r in rows
                  if isinstance((r.get("judge") or {}).get(
                      "graph_reasoning_correct"), bool)]
    if not applicable:
        return None
    return sum(1 for v in applicable if v) / len(applicable)


def _escalation_rate(rows: Sequence[Dict]) -> float:
    n = len(rows)
    if not n:
        return 0.0
    return sum(1 for r in rows
               if (r.get("judge") or {}).get("escalated")) / n


def run(items: Sequence[BenchmarkItem],
        gen: Dict[str, list],
        judge_model: Optional[str] = None,
        emb_model: str = semantic.DEFAULT_MODEL,
        run_judge: bool = True,
        out_dir: Path = E.OUT_GENERATION) -> Dict[str, list]:
    """Score every (system, item) generation row and persist per-system."""
    scored: Dict[str, list] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for system, rows in gen.items():
        srows = []
        for r in rows:
            rr = dict(r)
            rr.update(score_generation(rr, judge_model=judge_model,
                                       emb_model=emb_model,
                                       run_judge=run_judge))
            srows.append(rr)
        scored[system] = srows
        with open(out_dir / f"answers_{system}.jsonl", "w") as f:
            for rr in srows:
                f.write(json.dumps(rr, ensure_ascii=False, default=str)
                        + "\n")
    return scored


def aggregate(scored: Dict[str, list]) -> List[dict]:
    """Long-form answer-metric table (system / metric / value / n).

    In addition to the legacy axes, exposes the spec §12 derived metrics:
    ``judge__correct``, ``judge__faithful``, ``judge__complete``,
    ``judge__evidence_supported``, ``judge__overall_score``,
    ``judge__confidence``, ``judge__unsupported_claim_rate``,
    ``judge__graph_reasoning_correct``, ``judge__escalation_rate``.
    """
    out = []
    for system, rows in scored.items():
        for m in ("exact_match", "token_f1", "semantic_similarity",
                  "cites_target"):
            vals = [r.get(m) for r in rows
                    if isinstance(r.get(m), (int, float))]
            if vals:
                out.append({"system": system, "metric": m,
                            "value": sum(vals) / len(vals), "n": len(vals)})
        # legacy 4 axes
        for ax in _JUDGE_AXES:
            vals = []
            for r in rows:
                j = r.get("judge") or {}
                if not j.get("error") and isinstance(j.get(ax), (int, float)):
                    vals.append(j[ax])
            if vals:
                out.append({"system": system, "metric": f"judge__{ax}",
                            "value": sum(vals) / len(vals), "n": len(vals)})
        # spec §12 boolean fields (reported as 0/1 means)
        for f in _SPEC_FIELDS:
            vals = []
            for r in rows:
                j = r.get("judge") or {}
                if not j.get("error") and isinstance(j.get(f), bool):
                    vals.append(1 if j[f] else 0)
            if vals:
                out.append({"system": system, "metric": f"judge__{f}",
                            "value": sum(vals) / len(vals), "n": len(vals)})
        # overall_score (1..5), confidence (0..1), unsupported_claim_rate
        for f in ("overall_score", "confidence", "unsupported_claim_rate"):
            vals = []
            for r in rows:
                j = r.get("judge") or {}
                if not j.get("error") and isinstance(j.get(f), (int, float)):
                    vals.append(j[f])
            if vals:
                out.append({"system": system, "metric": f"judge__{f}",
                            "value": sum(vals) / len(vals), "n": len(vals)})
        # graph_reasoning_correct -- fraction of rows with bool True
        gr_acc = _graph_reasoning_accuracy(rows)
        if gr_acc is not None:
            out.append({"system": system,
                        "metric": "judge__graph_reasoning_correct",
                        "value": gr_acc, "n": len(rows)})
        # unsupported_claim_rate -- already per-row; also emit the mean
        ucr = _unsupported_claim_rate(rows)
        if ucr is not None:
            out.append({"system": system,
                        "metric": "judge__unsupported_claim_rate",
                        "value": ucr, "n": len(rows)})
        # escalation rate (informational)
        er = _escalation_rate(rows)
        out.append({"system": system,
                    "metric": "judge__escalation_rate",
                    "value": er, "n": len(rows)})
    out.sort(key=lambda r: (
        {s: i for i, s in enumerate(E.GENERATION_SYSTEMS)
         if s in scored}.get(r["system"], 99), r["metric"]))
    return out
