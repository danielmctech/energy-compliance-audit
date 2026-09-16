"""Verify build_benchmark_heldout() (rebuild path used by run_heldout_systems)
now produces the same gold targets as the hand-repaired JSONL artifact."""
from __future__ import annotations
import json, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"; SCRIPTS = ROOT / "scripts"
for p in (str(SRC), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("ENERGY_AUDIT_ROOT", str(ROOT))

import evaluation.benchmark as BM
root = BM._repo_root()
art_path = root / "notebooks/data/evaluation/heldout_benchmark.jsonl"

# corpus lids
targets = BM._load_chunk_targets(root / "notebooks/data/ast")
corpus = {x["lineage_id"] for x in targets}

art = {}
with open(art_path) as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            art[d["query_id"]] = d
rebmap = {x.query_id: x for x in BM.build_benchmark_heldout()}

print(f"artifact items: {len(art)}   rebuild items: {len(rebmap)}")
missing_in_reb = [q for q in art if q not in rebmap]
if missing_in_reb:
    print(f"!! query_ids in artifact but not in rebuild: {missing_in_reb}")

match = tgt_diff = both_bad = art_bad_reb_ok = 0
divergent = []
for q, a in art.items():
    r = rebmap.get(q)
    if r is None:
        continue
    a_t = a.get("target_lineage_id") or a.get("relevant_chunk_ids", [None])[0]
    a_gc = list(a.get("gold_chunks") or [a_t])
    a_ok = a_t in corpus
    r_ok = r.target_lineage_id in corpus
    if a_ok and r_ok:
        if a_t == r.target_lineage_id and a_gc == list(r.gold_chunks or [r.target_lineage_id]):
            match += 1
        else:
            tgt_diff += 1
            divergent.append((q, a_t, r.target_lineage_id))
    elif (not a_ok) and r_ok:
        art_bad_reb_ok += 1
        divergent.append((q, a_t, r.target_lineage_id))
    elif (not r_ok) and a_ok:
        tgt_diff += 1
        divergent.append((q, a_t, r.target_lineage_id))
    else:
        both_bad += 1
        divergent.append((q, a_t, r.target_lineage_id))

print(f"match={match} tgt_diff={tgt_diff} "
      f"artifact_bad_rebuild_ok={art_bad_reb_ok} both_bad={both_bad}")
print("--- divergent (artifact_target -> rebuild_target) ---")
for q, a_t, r_t in divergent:
    print(f"  {q}: {a_t}  ->  {r_t}")

nv = [(q, a.get("target_lineage_id"), a.get("gold_chunks")) for q, a in art.items()
      if a.get("target_lineage_id") not in corpus]
print(f"\nartifact items whose target is NOT in corpus ({len(nv)}):")
for q, t, g in nv:
    print(f"  {q}: {t}  gold={g}")
