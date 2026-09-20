"""Error classification.

For every (system, query) pair we attach a suggested failure label, kept as
a routing aid with the raw evidence behind it so a human can review.  The
label is never treated as a verified verdict about what the model actually knew.
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


def save(classification: Dict,
         out_dir: Path = E.OUT_AGGREGATE) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / "error_analysis.json"
    p1.write_text(json.dumps(classification, indent=2))
    return p1
