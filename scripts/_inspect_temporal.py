import sys, json
sys.path.insert(0, "src")
import evaluation.benchmark as BM
for it in BM.build_benchmark():
    if it.category == "temporal_version":
        rels = sorted({e.get("relation") for e in it.gold_edges})
        has_temporal = any(k.lower() in (json.dumps(it.gold_edges).lower()+ (it.metadata or {}).get("version","").lower())
                           for k in ("supersedes","effective_date","valid_from","valid_to","current","in_force"))
        print(f"  {it.query_id}  gold_edge_relations={rels}  "
              f"mentions_temporal_semantics={has_temporal}")
