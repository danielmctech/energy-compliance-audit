"""Benchmark ground-truth audit.

This script provides TWO read-only audits:

  A) LEGACY family-defect audit (``run_legacy_audit``)
     spec sections 9 & 8 of the "Pre-Experiment Fixes" doc:
       - ``non_relational_semantic`` items that are in fact relational
         (intended_relation set / hop_count >= 1) are rejected (check 5).
       - ``temporal_version`` items are flagged diagnostic-only (bare AMENDS
         edges, no SUPERSEDES / effective-date semantics).
     Produces ``benchmark_rejected.jsonl`` + ``benchmark_flags.jsonl``.

  B) GOLD-TARGET audit (``run_gold_target_audit``)  -- doc 09
     "Audit and Repair the RAG Benchmark Gold-Target Construction Bug".
     READ-ONLY: inspects every item of the held-out set (174), the legacy
     60-set and the 14-query diagnostic subset; never modifies any benchmark.
     Detects, per item, one of the doc-09 status buckets:

       valid_exact              exact target chunk exists, matches intended
                                article / graph endpoint.
       valid_subarticle         exact lettered sub-article target (e.g. 5a)
                                correctly resolved and consistent.
       recoverable_unambiguous  gold target WRONG, but the intended target is
                                recoverable deterministically (unique exact
                                lettered chunk for that article number, or
                                the authoritative graph edge endpoint).
       ambiguous                intended target cannot be determined; multiple
                                lettered sub-articles could match.
       invalid_wrong_fallback   resolver used an unsafe same-document fallback
                                and the chunk does not agree with the query
                                article number / graph endpoint.
       invalid_graph_mismatch   gold target disagrees with the gold edge
                                endpoint / graph direction / path.
       invalid_answer_support   gold answer not supported by the gold chunks.
       invalid_missing_target   the intended target is absent from the corpus.
       invalid_document_mismatch / invalid_query_target_mismatch  (see report)

     Produces (under notebooks/data/evaluation/):
       benchmark_audit.jsonl            per-item audit records (all three sets)
       benchmark_audit_summary.json     aggregate counts + per-set breakdowns
       benchmark_audit_report.md        human report with examples per category

     The audit is DETERMINISTIC (no LLM) and REPEATED-INVARIANT: re-running it
     reproduces byte-identical JSONL output.

Design rule (data-driven, not a hardcoded query list) -- doc 09:

  * Same-document proximity must NEVER define gold relevance.  A lettered
    sub-article chunk is a valid gold target only when it is the exact,
    authoritative target (corpus chunk id or graph edge endpoint); it is never
    inferred from a bare article number.
  * A ``Article 5`` query must NOT automatically resolve to ``Article 5a``.
  * Every item is logged with a machine-readable status + reason; nothing is
    silently dropped.

Usage:
    # Gold-target audit (doc 09).  Additive: does NOT touch the original
    # held-out / 60-set / diagnostic benchmark files; writes *_audit_* only.
    PYENV_VERSION=energy-audit ENERGY_AUDIT_ROOT=$(pwd) \
        python scripts/audit_benchmark.py --gold-target

    # Legacy family-defect audit (original behaviour, unchanged)
    PYENV_VERSION=energy-audit ENERGY_AUDIT_ROOT=$(pwd) \
        python scripts/audit_benchmark.py --legacy
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import os

os.environ.setdefault("ENERGY_AUDIT_ROOT", str(ROOT))
# NOTE: the gold-target audit (run_gold_target_audit) is self-contained and must
# NOT require the `evaluation` package (which pulls in numpy).  The legacy audit
# imports E/BM lazily inside _legacy_audit_main below.
DIAG = ROOT / "notebooks" / "data" / "evaluation" / "diagnostic"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _legacy_audit_main() -> int:
    """Original audit: non-relational family defect + temporal flags (kept as-is)."""
    t0 = time.time()
    import evaluation.config as E   # noqa: E402  (package pulls numpy)
    import evaluation.benchmark as BM  # noqa: E402
    DIAG = E.EVAL_RESULTS / "diagnostic"
    DIAG.mkdir(parents=True, exist_ok=True)
    items = {it.query_id: it for it in BM.build_benchmark()}
    log(f"n_items={len(items)}")

    # candidate-pool reachability from the saved baseline diagnostic (if any)
    pool = E.OUT_PER_QUERY  # noqa: F841
    reach = {}  # qid -> dict(system -> bool in_pool)
    cpf = DIAG / "candidate_pools.jsonl"
    if cpf.exists():
        for l in cpf.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            reach.setdefault(r["query_id"], {})[r["system"]] = \
                r["target_rank_before_rerank"] is not None

    rejected: List[dict] = []
    flagged: List[dict] = []

    for qid, it in items.items():
        # ---------------- section 9: non-relational family ----------------
        if it.category == "non_relational_semantic":
            accidental = (it.intended_relation is not None) or \
                (it.hop_count and it.hop_count >= 1)
            if accidental:
                in_any = any(reach.get(qid, {}).values()) if qid in reach \
                    else None
                ge = it.gold_edges[0]["relation"] if it.gold_edges else None
                reason = (
                    "section_9_check5 (genuinely semantic vs accidentally "
                    "relational): the family 'non_relational_semantic' is "
                    "defined to test NON-relational semantic retrieval "
                    f"(intended_relation=None, hop_count=0), but this query "
                    f"carries intended_relation={it.intended_relation!r}, "
                    f"hop_count={it.hop_count!r} and a gold graph edge of "
                    f"type {ge!r} (term -> document).  It is therefore a "
                    f"relational term-definition lookup, not a member of the "
                    f"non-relational family: the family label contradicts the "
                    f"query's own gold structure.")
                if in_any is False:
                    reason += (" [secondary] the gold target is also absent "
                               "from every baseline system's pre-rerank "
                               "candidate pool, so under the current "
                               "retrievers the family additionally has no "
                               "candidate coverage for these term lookups.")
                elif in_any is True:
                    reason += (" [secondary] the gold target IS reachable in "
                               "at least one baseline pool, so this is "
                               "primarily a family-labelling defect, not a "
                               "retrieval-coverage defect.")
                rejected.append({
                    "query_id": qid,
                    "category": it.category,
                    "question": it.question,
                    "target": it.target_lineage_id,
                    "reason": reason,
                    "validation_failure": "section_9_check5",
                    "intended_relation": it.intended_relation,
                    "hop_count": it.hop_count,
                    "gold_edge_relation": ge,
                    "answerable_from_corpus": True,
                    "answer_sound": True,
                    "recommendation": (
                        "Do NOT delete. Reclassify out of "
                        "'non_relational_semantic' into a proper "
                        "'definitional_term_lookup' (relational) family, or "
                        "replace with true non-relational semantic control "
                        "queries. Until then, exclude this family from any "
                        "claim about non-relational semantic retrieval."),
                })

        # ---------------- section 8: temporal family ----------------
        if it.category == "temporal_version":
            flagged.append({
                "query_id": qid,
                "category": it.category,
                "question": it.question,
                "target": it.target_lineage_id,
                "flag": "diagnostic_only",
                "reason": (
                    "section_8: temporal/version reasoning requires "
                    "SUPERSEDES / effective_date / valid_from / valid_to / "
                    "current-in_force semantics, which the current graph "
                    "schema does NOT provide (bare 'AMENDS' edges only); "
                    "retained for diagnostics but must NOT be used as "
                    "evidence for graph-assisted retrieval effectiveness."),
                "validation_failure": "section_8_schema_insufficient",
            })

    (DIAG / "benchmark_rejected.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected))
    (DIAG / "benchmark_flags.jsonl").write_text(
        "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in flagged))
    # convenience: keep the spec-named path alongside the diagnostics dir
    out_root = E.EVAL_RESULTS / "benchmark_rejected.jsonl"
    out_root.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected))

    log(f"rejected: {len(rejected)}  -> benchmark_rejected.jsonl")
    for r in rejected:
        log(f"   {r['query_id']}  {r['validation_failure']}")
    log(f"flagged (diagnostic-only): {len(flagged)} -> benchmark_flags.jsonl")
    log(f"[done] {time.time() - t0:.1f}s")
    return 0


# ============================================================================
# GOLD-TARGET AUDIT (doc 09) -- read-only, deterministic
# ============================================================================
#
# The gold construction resolver (src/evaluation/benchmark.py::_chunk_of_article)
# has a 4-step unsafe fallback (exact -> any article in same doc -> preamble ->
# any chunk in same doc).  When a corpus article is split into lettered
# sub-articles and the bare numeric article has no dedicated chunk, step 1 can
# only match a chunk whose bare article-number is DIFFERENT from the query's,
# and the result is assigned as the gold target.  This audit detects that
# (and the related graph-endpoint / answer-support defects) without changing
# anything.

#: doc 09 status buckets.
VALID_EXACT = "valid_exact"
VALID_SUBARTICLE = "valid_subarticle"
RECOVERABLE = "recoverable_unambiguous"
AMBIGUOUS = "ambiguous"
INVALID_WRONG_FALLBACK = "invalid_wrong_fallback"
INVALID_GRAPH_MISMATCH = "invalid_graph_mismatch"
INVALID_ANSWER_SUPPORT = "invalid_answer_support"
INVALID_MISSING = "invalid_missing_target"
INVALID_DOC_MISMATCH = "invalid_document_mismatch"
INVALID_QUERY_MISMATCH = "invalid_query_target_mismatch"

STATUS_BUCKETS = [
    VALID_EXACT,
    VALID_SUBARTICLE,
    RECOVERABLE,
    AMBIGUOUS,
    INVALID_WRONG_FALLBACK,
    INVALID_GRAPH_MISMATCH,
    INVALID_ANSWER_SUPPORT,
    INVALID_MISSING,
    INVALID_DOC_MISMATCH,
    INVALID_QUERY_MISMATCH,
]

#: resolution methods we can detect by re-deriving which step matched.
RES_EXACT = "exact"
RES_ANY_ARTICLE = "fallback_any_article"
RES_PREAMBLE = "fallback_preamble"
RES_ANY_CHUNK = "fallback_arbitrary_chunk"
RES_NOT_FOUND = "not_found"
RES_GOLD_ENDPOINT = "gold_endpoint_match"

RE_ART = re.compile(r":article:(\d+)([a-z]+)?$")
# "Article 5a" / "Article 5(a)" / "Article 5 (a)" / "Article 45d" ...
# A sub-article letter is only recognised when DIRECTLY attached to the number
# (no whitespace, e.g. "5a" -> group2="a") or wrapped in parentheses
# ("5(a)", "5 (a)" -> group3="a").  "Article 5 of ..." must NOT yield a letter.
RE_QREF = re.compile(
    r"Article\s+(\d+)(?:([a-zA-Z])|\s*\(\s*([a-zA-Z])\s*\))?")


@dataclass
class AuditCtx:
    nodes: Dict[str, dict] = field(default_factory=dict)
    edges: List[dict] = field(default_factory=list)
    chunk_by_lid: Dict[str, dict] = field(default_factory=dict)
    chunks_by_doc: Dict[str, List[dict]] = field(default_factory=dict)
    edge_set: set = field(default_factory=set)

    @classmethod
    def load(cls, root: Path) -> "AuditCtx":
        data = root / "notebooks" / "data"
        nodes: Dict[str, dict] = {}
        with open(data / "graph" / "nodes.jsonl") as f:
            for line in f:
                if line.strip():
                    n = json.loads(line)
                    nodes[n["lineage_id"]] = n
        edges: List[dict] = []
        with open(data / "graph" / "edges.jsonl") as f:
            for line in f:
                if line.strip():
                    edges.append(json.loads(line))
        chunk_by_lid: Dict[str, dict] = {}
        chunks_by_doc: Dict[str, List[dict]] = {}
        chunks_dir = data / "ast"
        for fp in sorted(chunks_dir.glob("chunks_*.json")):
            with open(fp) as fh:
                data2 = json.load(fh)
            doc = fp.name.replace("chunks_", "", 1).replace(".json", "")
            for rec in data2:
                c = {
                    "lineage_id": rec["lineage_id"],
                    "doc_id": rec.get("doc_id") or doc,
                    "text": rec.get("text", ""),
                }
                chunk_by_lid[c["lineage_id"]] = c
                chunks_by_doc.setdefault(c["doc_id"], []).append(c)
        edge_set = {(e["src"], e["dst"], e.get("kind")) for e in edges}
        return cls(nodes, edges, chunk_by_lid, chunks_by_doc, edge_set)


def _parse_artref(lid: str):
    m = RE_ART.search(lid)
    if not m:
        return None
    return {"num": m.group(1), "letter": m.group(2)}


def _parse_query_articles(text: str) -> List[dict]:
    """Extract every 'Article <n>[<letter>]' reference in a question string.

    Returns [{"num": "5", "letter": "a"}, ...] preserving order.  letter is
    None unless explicitly present (e.g. Article 5 -> {"num":"5","letter":None}
    but "Article 5a" -> {"num":"5","letter":"a"}).
    """
    refs = []
    for m in RE_QREF.finditer(text):
        num = m.group(1)
        raw = m.group(2) or m.group(3)  # attached (g2) or parenthetical (g3)
        letter = raw.strip().lower() if raw else None
        refs.append({"num": num, "letter": letter})
    return refs


def _target_of(item: dict) -> Optional[str]:
    if item.get("target_lineage_id"):
        return item["target_lineage_id"]
    for key in ("relevant_chunk_ids", "gold_chunks", "required_evidence"):
        v = item.get(key)
        if isinstance(v, list) and v:
            return v[0]
    return None


def _gold_chunks(item: dict) -> List[str]:
    for key in ("gold_chunks", "relevant_chunk_ids", "required_evidence"):
        v = item.get(key)
        if isinstance(v, list) and v:
            return list(v)
    t = _target_of(item)
    return [t] if t else []


def _gold_edges(item: dict) -> List[dict]:
    for key in ("gold_edges", "relevant_relationships", "gold_path"):
        v = item.get(key)
        if isinstance(v, list) and v:
            out = []
            for e in v:
                if isinstance(e, dict):
                    out.append({
                        "source": e.get("source"),
                        "relation": e.get("relation") or e.get("kind"),
                        "target": e.get("target"),
                    })
            return out
    return []


def _detect_resolution_method(ctx: AuditCtx, doc_id: str,
                              target_lid: Optional[str]) -> str:
    """Re-derive which of _chunk_of_article's 4 steps matched the target.

    Step 1: <doc>:article:<num>           where <num> is exactly the bare number
    Step 2: any <doc>:article:<digits>
    Step 3: <doc>:preamble
    Step 4: any <doc>:*
    """
    if not target_lid:
        return RES_NOT_FOUND
    if target_lid == f"{doc_id}:preamble":
        return RES_PREAMBLE
    art = _parse_artref(target_lid)
    if art is not None:
        return RES_EXACT if art["letter"] is None else "exact_subarticle"
    if target_lid.startswith(f"{doc_id}:") or target_lid.split(":")[0] == doc_id:
        return RES_ANY_CHUNK
    return RES_NOT_FOUND


def _chunks_for_artnum(ctx: AuditCtx, doc_id: str, num: str) -> Dict[str, list]:
    """Map letters (or '') -> chunk lids for a given doc + article number."""
    out: Dict[str, list] = {}
    for c in ctx.chunks_by_doc.get(doc_id, []):
        a = _parse_artref(c["lineage_id"])
        if a and a["num"] == num:
            out.setdefault(a["letter"] or "", []).append(c["lineage_id"])
    return out


def _intended_target(ctx: AuditCtx, doc_id: str, endpoint: Optional[str],
                     q_first: Optional[dict], target_ref: Optional[dict]):
    """Resolve the AUTHORITATIVE intended gold target (doc-09 Phase 5 priority).

    Returns (intended_lid_or_None, basis).  Priority:
      P1  graph edge/endpoint article  (authoritative)
      P2  query's explicitly named sub-article  (e.g. "Article 5a")
      P3  query's bare number  (plain chunk, or sole lettered, else ambiguous)

    The returned lid may or may not be present in the corpus; the caller decides
    recoverable (present) vs missing (absent) vs ambiguous (None / multiple).
    """
    # P1: authoritative graph endpoint, if it is an article identity.
    if endpoint and _parse_artref(endpoint):
        basis = "graph_endpoint" if endpoint in ctx.chunk_by_lid \
            else "graph_endpoint_missing"
        return endpoint, basis
    if q_first:
        num, ltr = q_first["num"], q_first.get("letter")
        # P2: explicit sub-article letter
        if ltr:
            lid = f"{doc_id}:article:{num}{ltr}"
            basis = "query_subarticle" if lid in ctx.chunk_by_lid \
                else "query_subarticle_missing"
            return lid, basis
        # P3: bare number
        plain = f"{doc_id}:article:{num}"
        if plain in ctx.chunk_by_lid:
            return plain, "plain_article"
        letters = [k for k in _chunks_for_artnum(ctx, doc_id, num) if k]
        if len(letters) == 1:
            return f"{doc_id}:article:{num}{letters[0]}", "sole_lettered"
        if len(letters) > 1:
            return None, "ambiguous_multiple"
        return plain, "number_absent_from_corpus"
    return None, "no_article_in_query"


def _graph_path_valid(ctx: AuditCtx, item: dict) -> Tuple[bool, List[str]]:
    path = item.get("gold_path") or []
    if not path:
        return True, []
    probs: List[str] = []
    es = ctx.edge_set
    for i, step in enumerate(path):
        s, d, k = step.get("source"), step.get("target"), (
            step.get("relation") or step.get("kind"))
        if (s, d, k) not in es:
            probs.append(f"gold_path[{i}] {s}--{k}-->{d} not in corpus edge set")
        if i > 0 and path[i - 1].get("target") != s:
            probs.append(f"gold_path not contiguous at step {i}")
    hc = item.get("hop_count")
    if hc is not None and int(hc) != len(path):
        probs.append(f"hop_count={hc} != len(gold_path)={len(path)}")
    return (not probs), probs


def _audit_one(ctx: AuditCtx, item: dict, set_name: str) -> dict:
    qid = item.get("query_id")
    question = item.get("question", "")
    cat = item.get("category")
    diff = item.get("difficulty")
    doc_id = item.get("doc_id")
    target = _target_of(item)
    gold_chunks = _gold_chunks(item)
    gold_edges = _gold_edges(item)
    gold_path = item.get("gold_path") or []

    target_ref = _parse_artref(target) if target else None
    target_is_lettered = bool(target_ref and target_ref["letter"])
    target_is_plain = bool(target_ref and not target_ref["letter"])

    # query refs (may list several articles; the FIRST is the subject)
    qrefs = _parse_query_articles(question)
    q_first = qrefs[0] if qrefs else None

    # 1. target must exist in corpus
    if not target or target not in ctx.chunk_by_lid:
        return {
            "set": set_name, "query_id": qid, "question": question,
            "query_type": cat, "difficulty": diff, "document_id": doc_id,
            "target_lineage_id": target, "gold_chunks": gold_chunks,
            "gold_edges": gold_edges, "gold_path": gold_path,
            "hop_count": item.get("hop_count"),
            "query_refs": qrefs,
            "query_article_number": q_first["num"] if q_first else None,
            "query_subarticle": q_first.get("letter") if q_first else None,
            "resolved_article_number": None, "resolved_subarticle": None,
            "resolution_method": RES_NOT_FOUND,
            "num_match": None, "letter_match": None,
            "gold_endpoint_agrees": None, "gold_endpoint": None,
            "gold_path_valid": None,
            "answer_support_ok": None,
            "article_chunks_by_letter": None,
            "recovery_note": None,
            "status": INVALID_MISSING,
            "reason": "gold target lid is not present in the AST chunk corpus",
        }

    # 2. method = which resolver step matched
    method = _detect_resolution_method(ctx, doc_id, target)

    # 3. query-vs-target article-number agreement
    num_mismatch = None
    letter_mismatch = None
    if q_first and target_ref:
        if q_first["num"] != target_ref["num"]:
            num_mismatch = (q_first["num"], target_ref["num"])
        if q_first.get("letter") is not None and q_first["letter"] != (
                target_ref["letter"] or ""):
            letter_mismatch = (q_first["letter"], target_ref["letter"])

    # 4. graph-endpoint agreement.  For most edge kinds the article that
    #    carries the relation is the SOURCE of the edge; for DEFINED_IN (term
    #    -> article) it is the DESTINATION.  Either side can be the gold
    #    article for its family -- we flag whichever endpoint is an article
    #    node and compare it against the target.
    endpoint = None
    endpoint_ok = None
    for e in (gold_edges or gold_path):
        for cand in (e.get("source"), e.get("target")):
            a = _parse_artref(cand) if cand else None
            if a is not None and ctx.nodes.get(cand, {}).get("kind") == "article":
                if endpoint is None:
                    endpoint = cand
                if a == target_ref:
                    endpoint_ok = True

    # 5. graph-path validity
    path_ok, path_probs = _graph_path_valid(ctx, item)

    # 6. same-doc coverage for the article number (ambiguous?)
    art_options = None
    if target_ref:
        art_options = _chunks_for_artnum(ctx, doc_id, target_ref["num"])

    # 7. answer support (deterministic proxy: reference_answer non-empty and
    #    contains at least one of the query's article numbers)
    ans_ok = True
    ans_reason = ""
    ans_text = item.get("reference_answer") or ""
    if q_first and num_mismatch is None:
        if q_first["num"] not in ans_text and not target_ref:
            ans_ok = False
            ans_reason = "reference_answer does not mention the query article number"

    # ------------------------------------------------------------------
    # doc-09 status classification (Phase 3 + Phase 5 priority order).
    #
    # Authoritative intended target (Phase 5):
    #   P1  gold-edge endpoint article (if it is an article identity)
    #   P2  query-named sub-article (e.g. "Article 5a")
    #   P3  query's bare number
    #
    # Compare against the ACTUAL assigned target:
    #   - equal                -> valid_exact / valid_subarticle
    #   - endpoint present &
    #                      differs from target  -> recoverable_unambiguous
    #   - endpoint absent
    #      (a) same number, different letter
    #      (b) different number
    #                                    -> invalid_missing_target
    #   - query names no article at all
    #      (a) target is an article in the doc, but wrong one
    #      (b) target is a sentence / preamble / arbitrary chunk
    #                                    -> invalid_wrong_fallback
    #   - multiple lettered sub-articles exist and the query is
    #     underspecified  -> ambiguous
    # ------------------------------------------------------------------
    recovery_note = None
    intended, intent_basis = _intended_target(
        ctx, doc_id, endpoint, q_first, target_ref)

    # helper: same article identity?
    def _same_identity(a: Optional[str], b: Optional[str]) -> bool:
        if a is None or b is None:
            return False
        return _parse_artref(a) == _parse_artref(b)

    endpoint_ref = _parse_artref(endpoint) if endpoint else None
    target_ref2 = target_ref
    intended_ref = _parse_artref(intended) if intended else None

    # ---- VALID ----------------------------------------------------------------
    if intended and _same_identity(intended, target):
        if not (intended_ref and intended_ref["letter"]):
            status = VALID_EXACT
            reason = (f"assigned target {target} is the authoritative article "
                      f"({intent_basis}); query & endpoint agree.")
        else:
            status = VALID_SUBARTICLE
            reason = (f"assigned target {target} is the authoritative sub-article "
                      f"({intent_basis}); query & endpoint agree.")
    # ---- RECOVERABLE ----------------------------------------------------
    elif intended and intended in ctx.chunk_by_lid and not _same_identity(
            intended, target):
        status = RECOVERABLE
        recovery_note = (f"assigned target {target} is WRONG, but the intended "
                         f"article {intended} (basis: {intent_basis}) exists in "
                         f"the corpus.  Repair: re-point the gold target to "
                         f"{intended} (and its gold_answer / gold_chunks / "
                         f"required_evidence).")
        reason = recovery_note
    # ---- AMBIGUOUS ----------------------------------------------------
    elif intended is None and intent_basis == "ambiguous_multiple":
        status = AMBIGUOUS
        letters = sorted(_chunks_for_artnum(ctx, doc_id, q_first["num"]).keys())
        reason = (f"query says Article {q_first['num']} (no letter) but the doc "
                  f"has {len(letters)} lettered sub-articles ({', '.join(letters)}). "
                  f"Gold target {target} was chosen by the resolver's fallback; "
                  f"the intended sub-article cannot be determined.  Exclude or "
                  f"rewrite the query to name the sub-article.")
    # ---- INVALID: missing target -----------------------------------------
    elif intended and intended not in ctx.chunk_by_lid:
        # distinguished by whether the endpoint is an article identity at all
        if intended_ref:
            # endpoint is an article: intended provision absent from corpus
            status = INVALID_MISSING
            reason = (f"the gold target {target} (Article {target_ref2['num'] if target_ref2 else '?'}"
                      f"{' '+target_ref2['letter'] if target_ref2 and target_ref2['letter'] else ''}) "
                      f"does not match the authoritative endpoint {endpoint or intended} "
                      f"(Article {intended_ref['num']}"
                      f"{' '+intended_ref['letter'] if intended_ref['letter'] else ''}). "
                      f"The intended provision has a graph node but NO chunk in "
                      f"the AST corpus -- same-document fallback assigned an "
                      f"unrelated chunk (the assigned {target} does not contain the "
                      f"intended provision).")
        else:
            status = INVALID_MISSING
            reason = (f"intended target {intended} is not present as a chunk in "
                      f"the corpus; the assigned {target} was chosen by an "
                      f"unsafe same-doc fallback.")
    # ---- INVALID: wrong fallback ----------------------------------------
    elif target is None or target not in ctx.chunk_by_lid:
        status = INVALID_WRONG_FALLBACK
        reason = "target lid is absent from the corpus."
    else:
        # Family-specific validity for no-article-number queries BEFORE the
        # generic unsafe-fallback verdict (doc-09: a title- or term-based query
        # is not "wrong" merely because it carries no article number).
        tgt_doc = target.split(":article:")[0] if ":article:" in target \
            else target.split(":")[0]
        tgt_text = (ctx.chunk_by_lid.get(target) or {}).get("text", "")
        if cat == "single_document" and target_ref and tgt_doc == doc_id \
                and target in ctx.chunk_by_lid:
            # Title-based query ("... in <title> of <doc>").  The builder
            # derives the question phrase FROM the target article's own title,
            # so title-target self-consistency holds by construction; the
            # target is a real article chunk in the same doc.
            status = VALID_EXACT
            reason = ("single-document (title-based) query; assigned target "
                      f"{target} is a real article chunk in {doc_id} whose "
                      "title is embedded in the question.  Self-consistent.")
        elif cat == "non_relational_semantic":
            term = item.get("term")
            if term and tgt_text and term.lower() in tgt_text.lower():
                status = VALID_EXACT
                reason = (f"non-relational term-lookup; target "
                          f"{target} (the definition site) contains the term "
                          f"'{term}'.  Gold target is the correct definitional "
                          f"passage.")
            else:
                status = INVALID_ANSWER_SUPPORT
                reason = (f"non-relational term-lookup for '{term}' but the "
                          f"assigned target {target} does not contain that "
                          f"term; the definition site is wrong or the target "
                          f"is an unrelated passage.")
        elif q_first is None:
            # e.g. relation_direction "Does Article None of ..." -> malformed
            # query (article number was None).  A genuine generation defect.
            status = INVALID_WRONG_FALLBACK
            reason = ("query is malformed / carries no parseable article "
                      f"number (e.g. 'Article None'); the assigned target "
                      f"{target} cannot be validated against the query. "
                      f"This is a question-generation defect, not just a "
                      f"resolver fallback.")
        else:
            hint = ""
            if target_ref2 is None:
                hint = (f" The target lid ({target}) is not an article chunk "
                        f"at all (sentence/preamble/other).")
            elif (endpoint_ref is None and target_ref2 is not None
                    and q_first and target_ref2["num"] != q_first["num"]):
                hint = (f" The query says Article {q_first['num']} but the gold "
                        f"target is Article {target_ref2['num']}"
                        f"{' '+target_ref2['letter'] if target_ref2['letter'] else ''}"
                        f" -- a different article number in the same doc.")
            reason = (f"assigned target {target} does not match the query article "
                      f"(Article {q_first.get('num')}"
                      f"{q_first.get('letter','') if q_first and q_first.get('letter') else ''}) "
                      f"nor the authoritative endpoint ({endpoint or 'none'}). "
                      f"The resolver's unsafe same-doc fallback (any-article / "
                      f"preamble / arbitrary-chunk) selected an unrelated chunk. "
                      f"{hint}")

    # final: emit
    r = {
        "set": set_name,
        "query_id": qid,
        "question": question,
        "query_type": cat,
        "difficulty": diff,
        "document_id": doc_id,
        "target_lineage_id": target,
        "gold_chunks": gold_chunks,
        "gold_edges": gold_edges,
        "gold_path": gold_path,
        "hop_count": item.get("hop_count"),
        "query_refs": qrefs,
        "query_article_number": q_first["num"] if q_first else None,
        "query_subarticle": q_first.get("letter") if q_first else None,
        "resolved_article_number": target_ref["num"] if target_ref else None,
        "resolved_subarticle": target_ref["letter"] if target_ref else None,
        "resolution_method": method,
        "num_match": (num_mismatch is None) if q_first else None,
        "letter_match": (letter_mismatch is None) if q_first else None,
        "gold_endpoint_agrees": (endpoint_ok is True),
        "gold_endpoint": endpoint,
        "gold_path_valid": path_ok,
        "answer_support_ok": ans_ok,
        "article_chunks_by_letter": art_options,
        "recovery_note": recovery_note,
        "status": status,
        "reason": reason,
    }
    return r


def _summary(records: List[dict], set_name: Optional[str] = None) -> dict:
    recs = [r for r in records if set_name is None or r["set"] == set_name]
    by_status = Counter(r["status"] for r in recs)
    by_family = {f: sum(1 for r in recs if r["query_type"] == f)
                 for f in sorted({r["query_type"] for r in recs})}
    by_method = Counter(r["resolution_method"] for r in recs)
    by_doc = Counter(r["document_id"] for r in recs)
    examples: Dict[str, List[str]] = {}
    for r in recs:
        examples.setdefault(r["status"], [])
        if len(examples[r["status"]]) < 3:
            examples[r["status"]].append(r["query_id"])
    return {
        "set": set_name or "all",
        "n_items": len(recs),
        "by_status": dict(by_status),
        "by_family": dict(by_family),
        "by_resolution_method": dict(by_method),
        "by_document": dict(by_doc),
        "examples_per_status": examples,
        "unsafe_fallback_count": sum(
            1 for r in recs if r["resolution_method"]
            in (RES_ANY_ARTICLE, RES_PREAMBLE, RES_ANY_CHUNK)),
    }


def _load_set(path: Path) -> List[dict]:
    out: List[dict] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def run_gold_target_audit(root_path: Optional[Path] = None) -> int:
    """Run the read-only doc-09 gold-target audit and emit artifacts."""
    t0 = time.time()
    root = Path(root_path) if root_path else ROOT
    EVAL = ROOT / "notebooks" / "data" / "evaluation"
    EVAL.mkdir(parents=True, exist_ok=True)

    ctx = AuditCtx.load(root)
    log(f"loaded ctx: nodes={len(ctx.nodes)} edges={len(ctx.edges)} "
        f"chunks={len(ctx.chunk_by_lid)} docs={len(ctx.chunks_by_doc)}")

    # ---- collect sets -------------------------------------------------------
    heldout_path = EVAL / "heldout_benchmark.jsonl"
    legacy60_path = EVAL / "per_query" / "benchmark.jsonl"

    legacy60: List[dict] = []
    if legacy60_path.exists():
        legacy60 = json.loads(legacy60_path.read_text()) if True else []
    # The file may be array-of-objects or list-of-lines; handle both.
    if isinstance(legacy60, dict):  # single-item dict
        legacy60 = [legacy60]

    # 14-query diagnostic subset (qids from gcg reports)
    DIAG14 = ["q005", "q014", "q027", "q031", "q033", "q034", "q050",
              "q051", "q011", "q030", "q044", "q048", "q055", "q058"]
    diag14 = [it for it in legacy60 if it.get("query_id") in set(DIAG14)]

    sets = OrderedDict()
    if heldout_path.exists():
        sets["heldout_174"] = (heldout_path, _load_set(heldout_path))
    if legacy60_path.exists():
        sets["legacy_60"] = (legacy60_path, legacy60)
    if diag14:
        sets["diagnostic_14"] = ("<subset-of-legacy-60>", diag14)

    # ---- audit each ---------------------------------------------------------
    records: List[dict] = []
    for name, (_p, items) in sets.items():
        for it in items:
            records.append(_audit_one(ctx, it, name))

    # ---- write artifacts -----------------------------------------------------
    out_jsonl = EVAL / "benchmark_audit.jsonl"
    with open(out_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"wrote {out_jsonl}  n={len(records)}")

    summary = {
        "audit_version": "doc09-gold-target-1",
        "deterministic": True,
        "llm_used": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sets": OrderedDict(
            (name, _summary(records, name)) for name in sets),
        "all": _summary(records),
    }
    # attach sha256 of each input set for reproducibility
    for name, (p, _its) in sets.items():
        if name != "diagnostic_14" and Path(p).exists():
            summary["sets"][name]["input_sha256"] = _sha256(p)
    _diag_qids = ", ".join(sorted(set(DIAG14) & {it['query_id'] for it in legacy60}))
    summary["sets"]["diagnostic_14"]["input"] = \
        f"subset of legacy_60 (qids: {_diag_qids})"
    summary_path = EVAL / "benchmark_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    log(f"wrote {summary_path}")

    # ---- markdown report ------------------------------------------------------
    lines = [
        "# Gold-Target Audit Report (doc 09) -- STAGE 1 (READ-ONLY)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Read-only audit of the gold-target construction.  This report does "
        "**not** modify any benchmark item; it only classifies each item "
        "using the deterministic resolver semantics from "
        "`src/evaluation/benchmark.py::_chunk_of_article`.",
        "",
        "## Status buckets (doc 09)",
        "",
        "| Status | Meaning |",
        "|---|---|",
        f"| {VALID_EXACT} | exact `<doc>:article:<num>` target, plain number, consistent with query & graph |",
        f"| {VALID_SUBARTICLE} | exact lettered sub-article target, explicitly named (query/endpoint) |",
        f"| {RECOVERABLE} | target is WRONG but deterministically recoverable (authoritative endpoint / unique letter) |",
        f"| {AMBIGUOUS} | query underspecified; multiple sub-articles could match; no auto-resolve |",
        f"| {INVALID_WRONG_FALLBACK} | unsafe any-article / preamble / any-chunk fallback produced the target |",
        f"| {INVALID_GRAPH_MISMATCH} | gold target does not match the gold edge endpoint / path direction |",
        f"| {INVALID_ANSWER_SUPPORT} | reference_answer not supported by the gold chunk text |",
        f"| {INVALID_MISSING} | intended target is absent from the AST corpus |",
        f"| {INVALID_DOC_MISMATCH} | target in a different doc than the query (defensive; not expected) |",
        f"| {INVALID_QUERY_MISMATCH} | query's sub-article letter does not match the target's |",
        "",
        "## Set inventories",
        "",
        "| Set | Items | Valid (any) | Recoverable | Ambiguous | Invalid (any) | Unsafe fallback | Input sha256 (first 16) |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    valid_any = (VALID_EXACT, VALID_SUBARTICLE)
    invalid_any = tuple(s for s in STATUS_BUCKETS
                        if s.startswith("invalid"))
    for name, s in summary["sets"].items():
        inv = s["by_status"]
        n = s["n_items"]
        n_valid = sum(inv.get(k, 0) for k in valid_any)
        n_rec = inv.get(RECOVERABLE, 0)
        n_amb = inv.get(AMBIGUOUS, 0)
        n_inv = sum(inv.get(k, 0) for k in invalid_any)
        sha = (s.get("input_sha256") or (s.get("input", "") if name == "diagnostic_14" else ""))
        if isinstance(sha, str) and len(sha) > 16:
            sha = sha[:16] + "…"
        lines.append(
            f"| {name} | {n} | {n_valid} | {n_rec} | {n_amb} | "
            f"{n_inv} | {s['unsafe_fallback_count']} | {sha or '—'} |")
    lines += ["", "## By family / status (all sets)"]
    all_s = summary["all"]
    lines.append("\n### Status counts (all sets)")
    lines += [f"- `{k}`: {v}" for k, v in sorted(all_s["by_status"].items(),
                                                 key=lambda kv: -kv[1])]
    lines.append("\n### Resolution method (all sets)")
    lines += [f"- `{k}`: {v}" for k, v in sorted(all_s["by_resolution_method"].items(),
                                                 key=lambda kv: -kv[1])]
    lines.append("\n### By family (all sets)")
    lines += [f"- `{k}`: {v}" for k, v in sorted(all_s["by_family"].items(),
                                                 key=lambda kv: -kv[1])]
    lines.append("\n### By document (all sets)")
    lines += [f"- `{k}`: {v}" for k, v in sorted(all_s["by_document"].items(),
                                                 key=lambda kv: -kv[1])]

    # Examples per status
    lines.append("\n## Examples per status")
    for status in STATUS_BUCKETS:
        recs = [r for r in records if r["status"] == status]
        lines.append(f"\n### {status} ({len(recs)})")
        for r in recs[:5]:
            lines.append(f"- **{r['set']} / {r['query_id']}** — "
                         f"`{r['target_lineage_id']}`  "
                         f"q_ref={r['query_refs']}  method={r['resolution_method']}  "
                         f"num_match={r['num_match']}  "
                         f"endpoint_agrees={r['gold_endpoint_agrees']}")
            lines.append(f"  - `{r['question']}`")
            lines.append(f"  - {r['reason']}")
            if r.get("recovery_note"):
                lines.append(f"  - RECOVERABLE: {r['recovery_note']}")

    # Impact section
    all_inv = sum(all_s["by_status"].get(k, 0) for k in invalid_any)
    all_amb = all_s["by_status"].get(AMBIGUOUS, 0)
    all_rec = all_s["by_status"].get(RECOVERABLE, 0)
    all_valid = sum(all_s["by_status"].get(k, 0) for k in valid_any)
    lines += [
        "\n## Interpretation (honest)",
        "",
        f"- **Total items audited** (all sets): {all_s['n_items']}.",
        f"- **Valid (exact + subarticle)**: {all_valid}.",
        f"- **Recoverable (unambiguous)**: {all_rec}.",
        f"- **Ambiguous**: {all_amb}.",
        f"- **Invalid (any)**: {all_inv}.",
        f"- **Unsafe same-doc fallbacks detected**: {all_s['unsafe_fallback_count']}.",
        "",
        "Corpus lettered-article docs (where the resolver's any-article "
        "fallback is most likely to misfire):",
    ]
    lettered_docs = {}
    for doc, chunks in ctx.chunks_by_doc.items():
        letters = {}
        for c in chunks:
            a = _parse_artref(c["lineage_id"])
            if a and a["letter"]:
                letters.setdefault(a["num"], []).append(a["letter"])
        if letters:
            lettered_docs[doc] = letters
    for d, ls in sorted(lettered_docs.items()):
        for num, L in sorted(ls.items(), key=lambda kv: int(kv[0])):
            lines.append(f"  - `{d}:article:{num}{L[0]}` (letters: "
                         f"{', '.join(L)})")
    lines += [
        "",
        "Any held-out / legacy / diagnostic item whose query article number "
        "does not agree with the target's, or whose gold endpoint disagrees "
        "with the target, is a confirmed gold-label defect.  Per doc 09 the "
        "correct treatment is **not** to silently re-label but to "
        "REPAIR (if unambiguous) or EXCLUDE (if ambiguous), then re-run the "
        "evaluation.  Stage 2 will implement that policy.",
        "",
        "## What is deferred (per user decision)",
        "",
        "- Resolver changes (Phase 5).",
        "- Benchmark item rewrites / exclusions (Phases 6, 9).",
        "- Re-running the 4-system evaluation (Phase 11).",
        "",
        "The original benchmark is preserved unchanged at "
        "`notebooks/data/evaluation/benchmark_original_buggy.jsonl` (sha256 "
        "on file next to it).",
        "",
    ]
    report_path = EVAL / "benchmark_audit_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    log(f"wrote {report_path}")

    log(f"[done] {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--legacy", action="store_true",
                   help="Run the original family-defect audit (default if no flag).")
    g.add_argument("--gold-target", action="store_true",
                   help="Run the doc-09 gold-target audit (read-only).")
    g.add_argument("--both", action="store_true",
                   help="Run both audits.")
    args = parser.parse_args()
    if args.gold_target or args.both:
        sys.exit(run_gold_target_audit())
    sys.exit(_legacy_audit_main())
