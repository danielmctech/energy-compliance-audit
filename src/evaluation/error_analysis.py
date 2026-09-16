"""Error classification and data-leakage checks.

For every (system, query) pair we attach a suggested failure label, kept as
a routing aid with the raw evidence behind it so a human can review.  The
label is never treated as a verified verdict about what the model actually knew.

The leakage check looks at the real training files rather than the benchmark
answers, and flags three ways the gold set could bleed into them:
  1. the gold question shows up verbatim as a training query;
  2. the gold target chunk is one of the training positives (the encoder was
     trained to rank exactly this passage first for a similar question);
  3. the gold reference text is embedded in a training positive's text.

Each probe returns a per-item boolean plus an aggregate risk note.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import config as E
from .benchmark import BenchmarkItem

TAXONOMY = E.ERROR_LABELS


def _rank_of(rows: Dict[str, list], qid: str, target: str) -> Optional[int]:
    for r in rows:
        if r.get("query_id") == qid:
            top = r.get("retrieved_top20", [])
            return (top.index(target) + 1) if target in top else None
    return None


def classify(item: BenchmarkItem,
             retrieval_rows: Dict[str, list],
             generation_rows: Optional[Dict[str, list]] = None,
             k: int = 5) -> Dict[str, str]:
    """Per-system suggested failure label.

      RETRIEVAL_FAILURE                 - target not in stored top-20
      PARTIAL_RETRIEVAL                 - target retrieved but past top-k
      CORRECT / CORRECT_RETRIEVAL_WRONG_ANSWER / GENERATION_FAILURE
                                        - answer-level (when generation rows)
    """
    out: Dict[str, str] = {}
    target = item.target_lineage_id
    for system, rows in retrieval_rows.items():
        rank = _rank_of(rows, item.query_id, target)
        if rank is None:
            out[system] = "RETRIEVAL_FAILURE"
            continue
        if rank > k:
            out[system] = "PARTIAL_RETRIEVAL"
            continue
        # target in top-k: look at the answer (if generation rows available)
        ans = (generation_rows or {}).get(system) or []
        arow = next((a for a in ans if a["query_id"] == item.query_id), None)
        if arow is None:
            # retrieved but no answer row -- context-layer only
            out[system] = "CORRECT"
            continue
        j = arow.get("judge") or {}
        f1 = arow.get("token_f1", 0) or 0
        cites = int(arow.get("cites_target", 0) or 0)
        faith = j.get("faithfulness", 0) or 0
        if (f1 >= 0.5 or cites == 1) and faith >= 3:
            out[system] = "CORRECT"
        elif f1 > 0.1 or cites == 1:
            out[system] = "CORRECT_RETRIEVAL_WRONG_ANSWER"
        else:
            out[system] = "GENERATION_FAILURE"
    return out


def _load_pairs(pairs_path: Path):
    pos, qs, pos_texts = set(), set(), set()
    if not pairs_path.exists():
        return pos, qs, pos_texts
    for line in pairs_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            p = json.loads(line)
        except json.JSONDecodeError:
            continue
        if p.get("positive", {}).get("lineage_id"):
            pos.add(p["positive"]["lineage_id"])
        if p.get("query"):
            qs.add(p["query"])
        if p.get("positive", {}).get("text"):
            pos_texts.add(p["positive"]["text"][:400])
    return pos, qs, pos_texts


def leakage_checks(items: Sequence[BenchmarkItem],
                   pairs_path: Path,
                   ) -> Dict:
    """Leakage probes (per "Avoid Data Leakage"): per-item booleans + aggregate risk note."""
    pos, qs, pos_texts = _load_pairs(pairs_path)
    per_item, n1 = [], 0
    n2 = n3 = 0
    for it in items:
        b1 = it.question in qs
        b2 = it.target_lineage_id in pos
        ref = (it.reference_answer or "")[:400]
        b3 = bool(ref) and any(ref in t for t in pos_texts)
        n1 += int(b1)
        n2 += int(b2)
        n3 += int(b3)
        per_item.append({
            "query_id": it.query_id,
            "question_matches_training_query": b1,
            "target_is_training_positive": b2,
            "reference_text_in_training_positive": b3,
        })
    n = max(len(items), 1)
    return {
        "per_item": per_item,
        "counts": {
            "question_verbatim_in_training": n1,
            "targets_that_are_training_positives": n2,
            "reference_text_in_training_positives": n3,
            "n_queries": len(items),
        },
        "risk_note": (
            f"LoRA dense encoders were trained ON the gold-set targets "
            f"({n2}/{len(items)}). Finetuned-dense rows are therefore "
            "inflated vs a clean baseline; the base-encoder rows are the "
            "unbiased estimate. Report both and flag the leakage rather "
            "than presenting the finetuned number as a clean win."
        ),
    }


def save(classification: Dict, leakage: Dict,
         out_dir: Path = E.OUT_AGGREGATE) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / "error_analysis.json"
    p1.write_text(json.dumps(classification, indent=2))
    p2 = out_dir / "leakage_checks.json"
    p2.write_text(json.dumps(leakage, indent=2))
    return p1
