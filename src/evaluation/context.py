"""Level-2 context triangulation (per "RAG Triangulation": retrieval -> context -> generation).

Sits between "did retrieval find the target" and "did the LLM use it".  For
each system + query we record the exact top-K chunk *context window* (what
goes into the prompt) and whether the target passage made it into that
window.  A system can retrieve a target at rank 30 (passing recall@20) yet
drop it from a 5-chunk prompt (context failure) -- this layer separates those
two, which is exactly the distinction the "Groundedness / Faithfulness" requirement demands (retrieval relevance
vs answer faithfulness).

Context window size is documented and identical for every system
(``context_window`` below) so the comparison is neutral (per the "Methodological Neutrality" requirement).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from . import config as E
from .benchmark import BenchmarkItem

#: How many of the top-ranked chunks are actually surfaced to the LLM.
#: Identical for every system -> a fair "context" slice (per "Methodological Neutrality").
CONTEXT_WINDOW = 5


def context_coverage(per_query_rows: Dict[str, list],
                     item: BenchmarkItem,
                     window: int = CONTEXT_WINDOW,
                     rank_cap: int = 50) -> dict:
    """Does the target make the top-``window`` context of one system?

    ``per_query_rows`` is the per-query record (has ``retrieved_top20`` and
    ``target_rank``).  We only have top-20 stored, so coverage is measured at
    the stored depth; ``window`` defaults to 5 but may be compared at any
    depth <= 20.
    """
    target = item.target_lineage_id
    top = per_query_rows.get("retrieved_top20", [])
    in_window = target in top[:window] if top else False
    rank = per_query_rows.get("target_rank")
    return {
        "in_context_window": bool(in_window),
        "target_rank": rank,
        "window": window,
        "rank_in_top50": rank if rank and rank <= 20 else None,
        "context_text_present": bool(
            target in top[:len(top)] and target in top),
    }


def triage(item: BenchmarkItem,
           systems: Dict[str, dict]) -> Dict[str, str]:
    """Per-system failure class for the context layer (the "context" row of "Error Classification").

    systems = {system: {"target_rank": int|None, ...}}.
      RETRIEVED_TOP_WINDOW  -> target in top-window (context should carry it)
      RETRIEVED_BELOW_WINDOW -> rank beyond window (context-dropped)
      NOT_RETRIEVED         -> not in stored top list
    """
    out = {}
    for s, row in systems.items():
        rank = row.get("target_rank")
        if rank is None:
            out[s] = "NOT_RETRIEVED"
        elif rank <= CONTEXT_WINDOW:
            out[s] = "RETRIEVED_TOP_WINDOW"
        else:
            out[s] = "RETRIEVED_BELOW_WINDOW"
    return out


def save(rows: List[dict], out_dir: Path = E.OUT_AGGREGATE) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "context_coverage.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p
