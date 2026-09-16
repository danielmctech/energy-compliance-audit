"""Re-point the 59 recoverable benchmark items to their correct endpoint chunk.

Root cause: ``_chunk_of_article`` (benchmark.py:137) has a 4-step fallback.
For lettered sub-articles (e.g. ``article:45d``) the regex ``:\\d+$`` in step 2
never matches, so step 3/4 silently assign a different article's chunk.

Fix: for every ``recoverable_unambiguous`` item (audit JSONL), swap
``gold_chunks``, ``reference_answer``, ``gold_answer``, ``required_evidence``,
``relevant_chunk_ids`` to the authoritative ``gold_endpoint`` (which IS
present in the corpus and same-doc).

Safe / deterministic / no LLM.  Backs up the original file first.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "notebooks/data/evaluation/heldout_benchmark.jsonl"
AUDIT = REPO / "notebooks/data/evaluation/benchmark_audit.jsonl"
CHUNKS_DIR = REPO / "notebooks/data/ast"


def load_chunk_texts() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for f in sorted(CHUNKS_DIR.glob("chunks_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("chunks", data)
        for c in items:
            lid = c.get("lineage_id") or c.get("id")
            if lid:
                out[lid] = c.get("text", "")
    return out


def main() -> int:
    # -- load audit: get the 59 recoverable items + their endpoints ----------
    audit = [json.loads(l) for l in AUDIT.read_text().splitlines() if l.strip()]
    recoverable = {}
    for r in audit:
        if r["set"] == "heldout_174" and r["status"] == "recoverable_unambiguous":
            recoverable[r["query_id"]] = r["gold_endpoint"]
    print(f"recoverable items to re-point: {len(recoverable)}")

    # -- load chunk texts -----------------------------------------------------
    CT = load_chunk_texts()

    # -- verify: endpoint exists, non-empty, same-doc --------------------------
    bench_lines = BENCH.read_text().splitlines()
    items = [json.loads(l) for l in bench_lines if l.strip()]
    item_map = {it["query_id"]: it for it in items}

    errors = []
    for qid, endpoint in sorted(recoverable.items()):
        if endpoint not in CT:
            errors.append(f"{qid}: endpoint {endpoint!r} not in corpus")
            continue
        if not CT[endpoint].strip():
            errors.append(f"{qid}: endpoint {endpoint!r} has empty text")
            continue
        item = item_map.get(qid)
        if item is None:
            errors.append(f"{qid}: not found in benchmark")
            continue
        if item["doc_id"] not in endpoint:
            errors.append(f"{qid}: doc mismatch {item['doc_id']} vs {endpoint}")
            continue

    if errors:
        print("REPAIR ABORTED — verification failures:")
        for e in errors:
            print(f"  {e}")
        return 1

    # -- back up ----------------------------------------------------------------
    backup = BENCH.with_suffix(".jsonl.bak_59repoint")
    if not backup.exists():
        shutil.copy2(BENCH, backup)
        print(f"backup -> {backup}")

    # -- apply the swap ----------------------------------------------------------
    changed = 0
    new_lines = []
    for l in bench_lines:
        if not l.strip():
            new_lines.append(l)
            continue
        item = json.loads(l)
        qid = item["query_id"]
        if qid in recoverable:
            ep = recoverable[qid]
            text = CT[ep]
            # swap the 5 gold-pointer fields
            old_gc = item.get("gold_chunks", [])
            item["gold_chunks"] = [ep]
            item["reference_answer"] = text
            item["gold_answer"] = text
            item["required_evidence"] = [ep]
            item["relevant_chunk_ids"] = [ep]
            # keep reference_basis (already "target_chunk_text")
            # do NOT touch: gold_edges, gold_path, gold_documents,
            #   gold_entities, question, term, doc_id, article_title, etc.
            changed += 1
        new_lines.append(json.dumps(item, ensure_ascii=False))

    BENCH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"re-pointed {changed} of {len(recoverable)} items -> {BENCH}")
    print("fields swapped: gold_chunks, reference_answer, gold_answer,")
    print("               required_evidence, relevant_chunk_ids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
