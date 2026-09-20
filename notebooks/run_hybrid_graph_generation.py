"""Driver: generate + score answers for JUST hybrid_graph using the 60 existing
retrieval rows on disk, then append to the aggregate tables.

Runs the neutral pipeline exactly like notebook 08, but scoped to hybrid_graph
so the other four systems' rows stay byte-identical (no re-generation,
no re-judging of existing rows -- only new prompts hit Ollama).
"""
import os, sys, time, json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation import config as E, benchmark as BM
import evaluation.retrieval_run  as RR
import evaluation.generation    as GEN
import evaluation.answer_metrics as AM
import evaluation.tables        as TB
import evaluation.error_analysis as EA
from evaluation.semantic import DEFAULT_MODEL as EMB

t0 = time.time()
print(f"[setup] repo: {ROOT}")
items = BM.build_benchmark()
print(f"[bench] n_items={len(items)}")

# --- Load the 60 hybrid_graph per-query rows off disk (built during the re-run) ---
pq_file = E.OUT_PER_QUERY / "retrieval_hybrid_graph.jsonl"
retrieval_hg = [json.loads(l) for l in pq_file.read_text().splitlines() if l.strip()]
print(f"[ret]   hybrid_graph rows loaded: {len(retrieval_hg)}")

# --- Step 1: Generate answers (60 LLM calls) ---
print("\n[step 1/4] GEN.run_generation(hybrid_graph) ...")
t1 = time.time()
gen = GEN.run_generation(items, {"hybrid_graph": retrieval_hg}, systems=["hybrid_graph"])
n_hg = len(gen.get("hybrid_graph", []))
print(f"        done in {time.time()-t1:.1f}s  rows={n_hg}")

# --- Step 2: Score those 60 rows (60 LLM judge calls + local bge-m3) ---
print("\n[step 2/4] AM.run(hybrid_graph) ...")
t2 = time.time()
scored = AM.run(items, gen, run_judge=True)
n_scored = len(scored.get("hybrid_graph", []))
print(f"        done in {time.time()-t2:.1f}s  scored={n_scored}")

# --- Step 3: Build a 5-system answer aggregate by merging existing rows w/ new ---
# Existing 4 systems live in answers_{sparse,dense,hybrid,neo4j}.jsonl, pre-aggregated
# in notebooks/data/evaluation/aggregate/answer_aggregate.csv (long form).
print("\n[step 3/4] Re-aggregate Table B across GENERATION_SYSTEMS ...")
# For each existing system, load the scored rows from disk and aggregate.
all_scored = {}
for s in E.GENERATION_SYSTEMS:
    if s == "hybrid_graph":
        all_scored[s] = scored[s]
        continue
    f = E.OUT_GENERATION / f"answers_{s}.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    all_scored[s] = rows
    print(f"        {s:<12} {len(rows):>3} rows from disk")
print(f"        {'hybrid_graph':<12} {len(all_scored['hybrid_graph']):>3} rows freshly scored")

ans_agg = AM.aggregate(all_scored)
E.OUT_AGGREGATE.mkdir(parents=True, exist_ok=True)
with open(E.OUT_AGGREGATE / "answer_aggregate.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["system", "metric", "value", "n"])
    w.writeheader()
    for r in ans_agg:
        w.writerow(r)
print(f"        wrote answer_aggregate.csv ({len(ans_agg)} rows)")

# Re-render Table B via the official table_b()
tB = TB.table_b(ans_agg)
print("\n[step 3/4] new table_B.csv rows:")
for r in tB:
    print("         ", json.dumps(r))

# --- Step 4: Re-render full tables.md (needs retrieval agg) ---
print("\n[step 4/4] Re-render tables ...")
agg = TB.load_aggregate()           # retrieval agg from disk (8 systems, unmodified)
tA = TB.table_a(agg)
tComp = TB.comparison(agg)
written = TB.write({"A": tA, "B": tB, "comparison": tComp})
for name, p in written.items():
    print(f"        wrote {p}")

# --- Error analysis (all 8 retrieval systems + 5 gen systems) ---
# Load all 8 systems' per-query rows to feed EA.classify; only the 5 gen
# systems have answer rows (sparse/dense/hybrid/hybrid_graph/neo4j).
retrieval_all = {}
for sname in E.EXPERIMENTS:
    f = E.OUT_PER_QUERY / f"retrieval_{sname}.jsonl"
    if f.exists():
        retrieval_all[sname] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
# Load the existing 4 systems' generation rows so EA.classify can classify
# them with the answer layer, not just the retrieval layer.
gen_all = {"hybrid_graph": all_scored["hybrid_graph"]}
for sname in ("sparse", "dense", "hybrid", "neo4j"):
    f = E.OUT_GENERATION / f"answers_{sname}.jsonl"
    if f.exists():
        gen_all[sname] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
classification = {item.query_id: EA.classify(item, retrieval_all, gen_all, k=5)
                  for item in items}
EA.save(classification, {})
from collections import Counter
labels = Counter(v for d in classification.values() for v in d.values())
print("\nerror-analysis label distribution:")
for k in sorted(labels):
    print(f"        {k:<35} {labels[k]}")

dt = time.time() - t0
print(f"\n[done] total {dt:.1f}s")
