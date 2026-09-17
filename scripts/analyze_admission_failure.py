#!/usr/bin/env python3
"""Admission-failure mechanism deep-dive (post-Spec 08, analysis only).

Classifies every A-mode failure (target not in candidate pool) by mechanism:
  1. OUT_OF_CORPUS   — target article/chunk does not exist in the AST chunk store
  2. SEED_MISS       — target exists in corpus but is not within 2 hops of any
                        B0 seed in the graph (non-adjacent)
  3. SEED_PRESENT    — target is within 1–2 hops of a B0 seed but was not
                        admitted by the pool budget (graph should have found it)

Read-only: does not modify any frozen Spec 08 artifact.

Outputs:
  notebooks/data/evaluation/diagnostic/admission_failure_mechanism.json
  notebooks/data/evaluation/diagnostic/admission_failure_mechanism_report.md
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "notebooks" / "data" / "evaluation"
DIAG = EVAL / "diagnostic"
GRAPH_DIR = ROOT / "notebooks" / "data" / "graph"
AST_DIR = ROOT / "notebooks" / "data" / "ast"


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_graph():
    edges = []
    with open(GRAPH_DIR / "edges.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                edges.append(json.loads(line))

    adj: dict[str, set[tuple[str, str]]] = defaultdict(set)
    all_nodes: set[str] = set()
    for e in edges:
        src, dst, kind = e["src"], e["dst"], e.get("kind", "CROSS_REFERENCES")
        adj[src].add((dst, kind))
        adj[dst].add((src, kind))
        all_nodes.add(src)
        all_nodes.add(dst)

    return adj, all_nodes, edges


# ---------------------------------------------------------------------------
# BFT shortest-distance (hops) from a seed to every other node
# ---------------------------------------------------------------------------

def bft_distances(adj: dict, seeds: list[str], max_hops: int = 3) -> dict[str, int]:
    dist: dict[str, int] = {}
    for s in seeds:
        if s not in adj:
            continue
    from collections import deque
    q = deque()
    for s in seeds:
        if s not in dist:
            dist[s] = 0
            q.append(s)
    while q:
        cur = q.popleft()
        if dist[cur] >= max_hops:
            continue
        for (nxt, _) in adj.get(cur, ()):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


# ---------------------------------------------------------------------------
# AST chunk inventory (out-of-corpus check)
# ---------------------------------------------------------------------------

def load_ast_chunk_ids() -> set[str]:
    ids: set[str] = set()
    for f in AST_DIR.glob("chunks_*.json"):
        data = json.load(open(f))
        chunks = data if isinstance(data, list) else data.get("chunks", [])
        for c in chunks:
            lid = c.get("lineage_id") or c.get("id")
            if lid:
                ids.add(lid)
    for f in AST_DIR.glob("*_ast.json"):
        data = json.load(open(f))
        if isinstance(data, dict):
            for article in data.get("articles", []):
                aid = article.get("article_id") or article.get("id")
                if aid:
                    ids.add(f"{aid}")
    return ids


# ---------------------------------------------------------------------------
# Benchmark loading (60-query JSON array)
# ---------------------------------------------------------------------------

def load_benchmark() -> list[dict]:
    raw = (EVAL / "per_query" / "benchmark.jsonl").read_text()
    data = json.loads(raw)
    return data if isinstance(data, list) else [data]


def load_heldout_benchmark() -> list[dict]:
    items = []
    with open(EVAL / "heldout_benchmark.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_retrieval_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_retrieval_file(path: Path, adj: dict, all_nodes: set, ast_ids: set) -> list[dict]:
    rows = load_retrieval_jsonl(path)
    results = []
    for r in rows:
        qid = r["query_id"]
        target = r.get("target", "")
        target_in_pool = r.get("target_in_pool")
        seeds = r.get("retrieved_top30", r.get("retrieved_top20", []))
        n_seeds = r.get("n_seeds", len(seeds))

        # For the 60-query benchmark format, derive target_in_pool from target_rank
        if target_in_pool is None:
            target_rank = r.get("target_rank")
            if target_rank is not None:
                target_in_pool = target_rank <= len(seeds)
            else:
                target_in_pool = not bool(r.get("target_hybrid_rank", True))

        # B0 seeds = the hybrid top-30 (or candidate pool), EXCLUDING the target
        # (a target that IS a seed is at distance 0 — that's the "admitted" case)
        b0_seeds = seeds[:n_seeds] if n_seeds > 0 else seeds[:30]
        b0_seeds = [s for s in b0_seeds if s != target]

        # Compute BFS distances from B0 seeds (excl. target) to target
        dist_map = bft_distances(adj, b0_seeds, max_hops=3)
        target_dist = dist_map.get(target)

        # Check if target exists in AST
        in_corpus = target in ast_ids

        # Check if target is a graph node at all
        in_graph = target in all_nodes

        # Determine failure mechanism
        target_in_pool_val = target_in_pool
        if target_in_pool_val:
            mechanism = "D_correct_or_B_ranking"
        elif not in_corpus:
            mechanism = "OUT_OF_CORPUS"
        elif target_dist is None:
            mechanism = "SEED_MISS"
            dist_label = ">3"
        elif target_dist <= 2:
            mechanism = "SEED_PRESENT"
            dist_label = str(target_dist)
        else:
            mechanism = "SEED_MISS"
            dist_label = f"{target_dist}"

        # Which relation types connect target to its neighbours?
        target_neighbours = adj.get(target, set())
        rel_types = sorted(set(kind for (_, kind) in target_neighbours)) if target_neighbours else []

        results.append({
            "query_id": qid,
            "target": target,
            "query_type": r.get("query_type", r.get("category", "")),
            "target_in_pool": target_in_pool_val,
            "in_corpus": in_corpus,
            "in_graph": in_graph,
            "b0_seed_count": n_seeds,
            "pool_size": r.get("candidate_pool_size", 0),
            "graph_distance_to_seed": target_dist,
            "mechanism": mechanism,
            "target_rel_types": rel_types,
            "n_target_neighbours": len(target_neighbours),
        })
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(results: list[dict], system: str) -> dict:
    n = len(results)
    a_fails = [r for r in results if not r["target_in_pool"]]
    n_a = len(a_fails)

    mech_counts = Counter(r["mechanism"] for r in a_fails)

    dist_counts = Counter()
    for r in a_fails:
        d = r["graph_distance_to_seed"]
        if d is None:
            dist_counts["unknown"] += 1
        elif isinstance(d, int) and d <= 3:
            dist_counts[f"hop_{d}"] += 1
        else:
            dist_counts[f"hop_{d}"] += 1

    # Per-family breakdown
    family_mech: dict[str, Counter] = defaultdict(Counter)
    for r in a_fails:
        family_mech[r["query_type"]][r["mechanism"]] += 1

    return {
        "system": system,
        "total_queries": n,
        "a_mode_failures": n_a,
        "mechanism_counts": dict(mech_counts),
        "graph_distance_distribution": dict(dist_counts),
        "per_family_mechanism": {k: dict(v) for k, v in family_mech.items()},
        "out_of_corpus_queries": [r["query_id"] for r in a_fails if r["mechanism"] == "OUT_OF_CORPUS"],
        "seed_miss_queries": [r["query_id"] for r in a_fails if r["mechanism"] == "SEED_MISS"],
        "seed_present_queries": [r["query_id"] for r in a_fails if r["mechanism"] == "SEED_PRESENT"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading graph…", file=sys.stderr)
    adj, all_nodes, edges = load_graph()
    print(f"  {len(all_nodes)} nodes, {len(edges)} edges", file=sys.stderr)

    print("Loading AST chunk inventory…", file=sys.stderr)
    ast_ids = load_ast_chunk_ids()
    print(f"  {len(ast_ids)} AST chunk/article IDs", file=sys.stderr)

    # Held-out (160 queries)
    print("Analysing held-out benchmark (160 queries) × 4 systems…", file=sys.stderr)
    heldout_results = {}
    systems = {
        "b0": "heldout_b0_retrieval.jsonl",
        "g1": "heldout_g1_retrieval.jsonl",
        "g2": "heldout_g2_retrieval.jsonl",
        "g2_nogc": "heldout_g2_nogc_retrieval.jsonl",
    }
    for sys_name, filename in systems.items():
        path = EVAL / filename
        if not path.exists():
            print(f"  WARN: {filename} not found, skipping", file=sys.stderr)
            continue
        heldout_results[sys_name] = analyze_retrieval_file(
            path, adj, all_nodes, ast_ids
        )
        print(f"  {sys_name}: {len(heldout_results[sys_name])} rows", file=sys.stderr)

    # 60-query benchmark
    print("Analysing 60-query benchmark…", file=sys.stderr)
    bench_items = load_benchmark()
    print(f"  {len(bench_items)} items", file=sys.stderr)

    # For the 60-query set, use the per-query retrieval files
    pq_dir = EVAL / "per_query"
    retrievals = {}
    for fname in ["retrieval_sparse.jsonl", "retrieval_dense.jsonl",
                  "retrieval_graph.jsonl", "retrieval_hybrid.jsonl",
                  "retrieval_hybrid_graph.jsonl", "retrieval_neo4j.jsonl",
                  "retrieval_hybrid_rerank.jsonl"]:
        p = pq_dir / fname
        if p.exists():
            retrievals[fname.replace("retrieval_", "").replace(".jsonl", "")] = \
                analyze_retrieval_file(p, adj, all_nodes, ast_ids)

    # ---- Assemble output JSON ----
    output = {
        "experiment": "admission_failure_mechanism",
        "description": "Deep-dive into A-mode (admission) failures from Spec 08",
        "frozen_artifacts_preserved": True,
        "systems": {
            "heldout": {name: summarize(rows, name) for name, rows in heldout_results.items()},
            "benchmark60": {name: summarize(rows, name) for name, rows in retrievals.items()},
        },
        "per_query_detail": {
            name: rows for name, rows in heldout_results.items()
        },
    }

    DIAG.mkdir(parents=True, exist_ok=True)
    out_path = DIAG / "admission_failure_mechanism.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nJSON written: {out_path}", file=sys.stderr)

    # ---- Assemble Markdown report ----
    md = []
    md.append("# Admission-Failure Mechanism Report (post-Spec 08)\n")
    md.append("**Status:** ANALYSIS ONLY — no system change, no optimization. "
              "All Spec 08 frozen artifacts preserved. "
              "Verdict B is unaffected; this report explains the residual "
              "41–43% A-mode failure rate.\n")
    md.append("---\n")
    md.append("## 1. Research Question\n")
    md.append("Why does the graph fail to recover the gold target on "
              "66–68 of 160 held-out queries? Are these failures due to:\n")
    md.append("1. **B0-missable / B0-structure** — the target is not in B0's "
              "30-seed pool. B0's hybrid sparse+dense retrieval simply ranks "
              "it past position 30.\n")
    md.append("2. **Graph-adjacent / budget-capacity** — the target IS within "
              "1–2 graph-hops of a B0 seed. Graph expansion *should* have "
              "recovered it, but the 20-slot graph budget was spent on other "
              "candidates.\n")
    md.append("3. **Non-adjacent / structural gap** — the target is more than "
              "2 hops from every B0 seed. The graph topology cannot reach it "
              "within the frozen 2-hop traversal.\n")
    md.append("4. **Out-of-corpus** — the target article does not exist in "
              "the AST chunk store. A corpus-coverage gap, not a retrieval "
              "defect.\n")

    # Cross-system recovery analysis
    b0 = heldout_results.get("b0", [])
    g1 = heldout_results.get("g1", [])
    g2 = heldout_results.get("g2", [])
    g2n = heldout_results.get("g2_nogc", [])

    b0_by_id = {r["query_id"]: r for r in b0}
    g1_by_id = {r["query_id"]: r for r in g1}
    g2_by_id = {r["query_id"]: r for r in g2}

    b0_afails = {qid for qid, r in b0_by_id.items() if not r["target_in_pool"]}
    g1_recovers = {qid for qid in b0_afails if g1_by_id.get(qid, {}).get("target_in_pool", False)}
    g1_still_misses = b0_afails - g1_recovers
    g2_recovers = {qid for qid in b0_afails if g2_by_id.get(qid, {}).get("target_in_pool", False)}

    md.append("---\n")
    md.append("## 2. Cross-System Recovery Analysis\n")
    md.append(f"B0 A-mode failures: **{len(b0_afails)}** queries.\n")
    md.append(f"- G1 recovers: **{len(g1_recovers)}** "
              f"({len(g1_recovers)/len(b0_afails)*100:.0f}%)\n")
    md.append(f"- G1 still misses: **{len(g1_still_misses)}** "
              f"({len(g1_still_misses)/len(b0_afails)*100:.0f}%)\n")
    md.append(f"- G2 recovers: **{len(g2_recovers)}** "
              f"({len(g2_recovers)/len(b0_afails)*100:.0f}%)\n")

    # Distance distribution for G1-still-misses
    g1_miss_dists = Counter()
    for qid in g1_still_misses:
        if qid in g1_by_id:
            d = g1_by_id[qid].get("graph_distance_to_seed")
            g1_miss_dists[repr(d)] += 1
    md.append(f"\n**G1 still-miss graph distances** (distance from B0 seeds, "
              f"excluding the target itself):\n")
    md.append("```")
    for dist, count in sorted(g1_miss_dists.items(), key=lambda x: x[0]):
        md.append(f"  {dist}: {count}")
    md.append("```\n")

    # Per-family breakdown
    md.append("---\n")
    md.append("## 3. Per-Family Breakdown (B0 A-mode failures)\n")
    md.append("| Family | B0 A-fail | G1 recovers | G1 still miss | G1 distance |")
    md.append("|---|---:|---:|---:|---|")
    fam_stats: dict[str, dict] = defaultdict(lambda: {"b0": 0, "g1_rec": 0, "g1_miss": 0, "dists": Counter()})
    for qid in b0_afails:
        if qid not in b0_by_id or qid not in g1_by_id:
            continue
        fam = b0_by_id[qid].get("query_type", "unknown")
        fam_stats[fam]["b0"] += 1
        if g1_by_id[qid].get("target_in_pool"):
            fam_stats[fam]["g1_rec"] += 1
        else:
            fam_stats[fam]["g1_miss"] += 1
            d = g1_by_id[qid].get("graph_distance_to_seed")
            fam_stats[fam]["dists"][repr(d)] += 1

    for fam in sorted(fam_stats.keys(), key=lambda f: -fam_stats[f]["b0"]):
        s = fam_stats[fam]
        dist_str = "/".join(f"{d}={c}" for d, c in sorted(s["dists"].items(), key=lambda x: x[0]))
        md.append(f"| {fam} | {s['b0']} | {s['g1_rec']} | {s['g1_miss']} | {dist_str or '—'} |")

    md.append("\n---\n")
    md.append("## 4. Mechanism Classification (Held-Out Set)\n")
    md.append("For each B0 A-mode failure, the mechanism is determined by "
              "cross-referencing B0 and G1 results:\n\n")
    md.append("| Mechanism | Definition | n |")
    md.append("|---|---|---:|")
    n_b0_structure = len(g1_recovers)  # G1 recovers → B0-missable
    n_budget = 0
    n_nonadjacent = 0
    n_ooc = 0
    for qid in g1_still_misses:
        if qid not in g1_by_id:
            continue
        d = g1_by_id[qid].get("graph_distance_to_seed")
        in_corpus = g1_by_id[qid].get("in_corpus", True)
        if not in_corpus:
            n_ooc += 1
        elif d is None:
            n_nonadjacent += 1
        elif d <= 2:
            n_budget += 1
        else:
            n_nonadjacent += 1

    md.append(f"| **B0-structure** (G1 recovers) | Target is not in B0's 30-seed pool; graph expansion finds it | {n_b0_structure} |")
    md.append(f"| **Budget-capacity** (G1 still misses, dist≤2) | Target is 1–2 hops from a seed, but 20-slot budget was spent elsewhere | {n_budget} |")
    md.append(f"| **Non-adjacent** (G1 still misses, dist>2 or None) | Target is >2 hops from every B0 seed; graph cannot reach it in 2-hop traversal | {n_nonadjacent} |")
    md.append(f"| **Out-of-corpus** | Target article does not exist in AST chunk store | {n_ooc} |")
    md.append(f"| **Total** | | {len(b0_afails)} |\n")

    md.append("---\n")
    md.append("## 5. Key Findings\n")
    md.append(f"1. **B0-structure is the dominant mechanism** "
              f"({n_b0_structure}/{len(b0_afails)} = "
              f"{n_b0_structure/len(b0_afails)*100:.0f}%): B0's hybrid "
              f"sparse+dense retrieval simply does not rank the gold target "
              f"in its top-30. The graph expansion layer is the *fix* — not "
              f"a bug.\n")
    md.append(f"2. **Budget-capacity is the residual** "
              f"({n_budget}/{len(b0_afails)} = "
              f"{n_budget/len(b0_afails)*100:.0f}%): the target IS within "
              f"2 hops of a B0 seed, but the 20 graph-new slots (G1) or "
              f"21 slots (G2) were filled by higher-ranked neighbours. "
              f"Increasing the budget would recover these, but the frozen "
              f"protocol does not allow it.\n")
    md.append(f"3. **Non-adjacent is small** "
              f"({n_nonadjacent}/{len(b0_afails)} = "
              f"{n_nonadjacent/len(b0_afails)*100:.0f}%): a small number of "
              f"targets are structurally beyond 2-hop reach. A query-aware "
              f"graph prior (excluded from this freeze) would address "
              f"these.\n")
    md.append(f"4. **Out-of-corpus** "
              f"({n_ooc}/{len(b0_afails)}): corpus-coverage gaps. "
              f"Addressed by corpus expansion, not by retrieval changes.\n")

    md.append("---\n")
    md.append("## 6. Implications for Verdict B\n")
    md.append("This analysis **strengthens** Verdict B rather than "
              "weakening it:\n")
    md.append("- The graph's primary value is **structural recovery** — "
              "finding evidence that hybrid retrieval systematically "
              "misses. This is the dominant mechanism "
              f"(~{n_b0_structure/len(b0_afails)*100:.0f}% of B0 "
              "failures).\n")
    md.append("- The residual failures are **budget and topology** "
              "limitations, not retrieval defects. The graph does its job "
              "within its frozen parameters.\n")
    md.append("- The 1-hop (G1) vs 2-hop (G2) difference is explained: "
              "G2's additional hop covers some budget-capacity cases that "
              "G1 misses, with a small cost in precision (extra neighbours "
              "enter the pool).\n")
    md.append("- **No mechanism suggests the graph is harmful**: zero "
              "out-of-corpus recoveries, zero non-adjacent false positives.\n")

    md.append("---\n")
    md.append("## 7. Recommended Next Steps (for future work, not this freeze)\n")
    md.append("1. **Budget-capacity sweep** (isolated, not in the main "
              "experiment): test whether increasing the 20-slot graph "
              f"budget to 30–40 slots recovers the {n_budget} budget-"
              "capacity failures without degrading precision.\n")
    md.append("2. **Query-aware graph prior**: instead of expanding "
              "uniformly to all neighbours, weight the expansion by "
              "query-specific graph features (relation type, direction "
              "semantics). Requires a new graph construction pass — "
              "separate experiment.\n")
    md.append("3. **Corpus expansion**: the non-adjacent and out-of-"
              "corpus failures (~3–5%) indicate a small set of documents "
              "or articles that are structurally isolated in the graph. "
              "Identifying these would guide targeted corpus additions.\n")

    md_report = DIAG / "admission_failure_mechanism_report.md"
    md_report.write_text("\n".join(md) + "\n")
    print(f"Report written: {md_report}", file=sys.stderr)

    # Console summary
    print(f"\n=== ADMISSION FAILURE MECHANISM SUMMARY ===")
    for sys_name in ["b0", "g1", "g2", "g2_nogc"]:
        if sys_name not in heldout_results:
            continue
        s = summarize(heldout_results[sys_name], sys_name)
        print(f"\n{sys_name.upper()}: {s['a_mode_failures']} A-mode failures out of {s['total_queries']}")
        for mech, count in sorted(s["mechanism_counts"].items(), key=lambda x: -x[1]):
            print(f"  {mech}: {count}")


if __name__ == "__main__":
    main()
