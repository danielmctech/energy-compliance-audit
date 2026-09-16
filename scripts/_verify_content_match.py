"""CORRECT content-based comparison.
artifact target  = relevant_chunk_ids[0]   (what load() uses as target_lineage_id)
rebuild target   = item.target_lineage_id  (dataclass field, always set)
Match by question text (stable across the hNNN shuffle).
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "scripts"):
    if str(ROOT/p) not in sys.path: sys.path.insert(0, str(ROOT/p))
os.environ.setdefault("ENERGY_AUDIT_ROOT", str(ROOT))
import evaluation.benchmark as BM
root = BM._repo_root()
art_path = root/"notebooks/data/evaluation/heldout_benchmark.jsonl"
corpus = {x["lineage_id"] for x in BM._load_chunk_targets(root/"notebooks/data/ast")}

art = {}
with open(art_path) as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            art[d["question"].strip()] = d
reb = BM.build_benchmark_heldout()
# dedup rebuild by question (report count)
reb_q = {}
dup = 0
for r in reb:
    key = r.question.strip()
    if key in reb_q: dup += 1
    reb_q[key] = r
print(f"artifact items: {len(art)}  rebuild items: {len(reb)}  rebuild dup-questions: {dup}")

st=dt=ao=ro=0; rows=[]
for q,a in art.items():
    at = (a.get("relevant_chunk_ids") or [None])[0]
    if q in reb_q:
        rt = reb_q[q].target_lineage_id
        if at==rt: st+=1
        else:
            dt+=1; rows.append((q[:50],at,rt,at in corpus,rt in corpus))
    else: ao+=1
for q in reb_q:
    if q not in art: ro+=1
print(f"both_same_target={st}  both_diff_target={dt}  artifact_only={ao}  rebuild_only={ro}")
print("--- SAME question, DIFFERENT target (art[OK] -> reb[OK]) ---")
for q,at,rt,oka,okr in rows:
    print(f"  {q}\n    art={at}[{'OK' if oka else 'MISS'}]  reb={rt}[{'OK' if okr else 'MISS'}]")
print("\n--- artifact_only ---")
for q,a in art.items():
    if q not in reb_q:
        print(f"  {a['query_id']}: tgt={(a.get('relevant_chunk_ids') or [None])[0]} | {q[:60]}")
print("\n--- rebuild_only ---")
for q,r in reb_q.items():
    if q not in art:
        print(f"  tgt={r.target_lineage_id} | {q[:60]}")
