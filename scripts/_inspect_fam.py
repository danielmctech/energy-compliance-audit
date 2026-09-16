import sys, json, re
sys.path.insert(0, "src")
import evaluation.benchmark as BM
items = BM.build_benchmark()
print(f"{'qid':<6}{'category':<28}{'term':<28}{'relation':<16}{'hop':<5}")
for it in items:
    if it.category == "non_relational_semantic":
        m = re.search(r"'(.+?)'", it.question)
        term = m.group(1) if m else it.question[:28]
        print(f"{it.query_id:<6}{it.category:<28}{term:<28}"
              f"{str(it.intended_relation or ''):<16}{it.hop_count:<5}")
print("--- TEMPORAL ---")
for it in items:
    if it.category == "temporal_version":
        print(f"  {it.query_id}: {it.question[:75]}  target={it.target_lineage_id}")
