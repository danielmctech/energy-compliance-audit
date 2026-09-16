"""Re-point the 35 missing-target items to their materialized article chunk.

Root cause: _chunk_of_article fell through to "any article in same doc" for
lettered sub-articles, assigning a wrong chunk as the gold.  The 12 real
article chunks are now materialized (materialize_missing_chunks.py), so for
each item whose exact endpoint is one of them, swap the 5 gold-pointer
fields.

The 4 `mica_2023_1114:article:149` items are left untouched: that article
does not exist (MiCA max = 81).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO   = Path(__file__).resolve().parent.parent
BENCH  = REPO / "notebooks/data/evaluation/heldout_benchmark.jsonl"
AUDIT  = REPO / "notebooks/data/evaluation/benchmark_audit.jsonl"
AST    = REPO / "notebooks/data/ast"


def load_chunk_texts() -> dict:
    out = {}
    for p in sorted(AST.glob("chunks_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for c in d:
            lid = c.get("lineage_id") or c.get("id")
            if lid:
                out[lid] = c.get("text", "")
    return out


def main() -> int:
    audit = [json.loads(l) for l in AUDIT.read_text().splitlines() if l.strip()]
    # only "invalid_missing_target" items whose endpoint is one of the 12
    # materialized articles (NOT the mica:149 ones)
    re_point = {}
    for r in audit:
        if r["set"] == "heldout_174" and r["status"] == "invalid_missing_target":
            ep = r["gold_endpoint"]
            if ep and ":article:" in ep and ep != "mica_2023_1114:article:149":
                re_point[r["query_id"]] = ep
    print(f"re-point candidates: {len(re_point)}")

    CT = load_chunk_texts()
    items = [json.loads(l) for l in BENCH.read_text().splitlines() if l.strip()]
    by_qid = {it["query_id"]: it for it in items}

    missing = []
    for qid, ep in sorted(re_point.items()):
        if ep not in CT:
            missing.append(f"{qid} -> {ep}")
            continue
        if not CT[ep].strip():
            missing.append(f"{qid} -> {ep} (empty)")
    if missing:
        print("ABORT -- endpoint missing:", missing)
        return 1

    backup = BENCH.with_suffix(".jsonl.bak_35repoint")
    if not backup.exists():
        shutil.copy2(BENCH, backup)
        print(f"backup -> {backup}")

    changed = 0
    new_lines = []
    for l in BENCH.read_text().splitlines():
        if not l.strip():
            new_lines.append(l)
            continue
        it = json.loads(l)
        qid = it["query_id"]
        if qid in re_point:
            ep = re_point[qid]
            text = CT[ep]
            it["gold_chunks"] = [ep]
            it["reference_answer"] = text
            it["gold_answer"]     = text
            it["required_evidence"] = [ep]
            it["relevant_chunk_ids"] = [ep]
            changed += 1
        new_lines.append(json.dumps(it, ensure_ascii=False))

    BENCH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"re-pointed {changed} of {len(re_point)} items -> {BENCH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
