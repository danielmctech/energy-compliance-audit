import sys, json, re
sys.path.insert(0, "src")
import evaluation.benchmark as BM
items = BM.build_benchmark()
for it in items:
    if it.category == "non_relational_semantic":
        m = re.search(r"'(.+?)'", it.question)
        term = m.group(1) if m else "?"
        print("="*80)
        print(f"qid={it.query_id}  term='{term}'")
        print(f"  intended_relation={it.intended_relation!r}  "
              f"intended_direction={it.intended_direction!r}  hop={it.hop_count}")
        print(f"  gold_edges={it.gold_edges}")
        print(f"  gold_path={it.gold_path}")
        print(f"  gold_entities={it.gold_entities}")
        print(f"  gold_chunks={it.gold_chunks}")
        print(f"  required_evidence={it.required_evidence}")
        print(f"  difficulty={it.difficulty!r}  term_field={it.term!r}")
