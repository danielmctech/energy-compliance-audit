"""One-shot / re-runnable Neo4j ingestion (P1).

Usage:
    python3 scripts/ingest_neo4j.py             # full run
    python3 scripts/ingest_neo4j.py --dry-run   # load + search_text audit only
    python3 scripts/ingest_neo4j.py --verify    # post-run sanity checks

Idempotent: safe to re-run.  ``emb`` is only filled where it is currently
null, so a second run is cheap (embeddings are also disk-cached).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# make `src` importable regardless of CWD
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import graphrag_n4j as n4j          # noqa: E402
import neo4j_config as cfg         # noqa: E402


def _dry_run() -> int:
    nodes = n4j.load_nodes()
    edges = n4j.load_edges()
    print(f"nodes: {len(nodes):>4d}    edges: {len(edges):>4d}")
    by_kind = {}
    for n in nodes:
        by_kind.setdefault(n["kind"], []).append(n)
    for k in sorted(by_kind):
        print(f"  {k:18s} {len(by_kind[k]):>4d}")
    print("\nsearch_text_for() sample per kind:")
    for n in nodes[: 3 * len(by_kind)]:
        print(f"  [{n['kind']:16s}] {n['lineage_id'][:44]:44s} -> {n4j.search_text_for(n)}")
    print(f"\nEMBEDDABLE_KINDS = {n4j.EMBEDDABLE_KINDS}")
    kind_count = {}
    for n in nodes:
        if n["kind"] in n4j.EMBEDDABLE_KINDS:
            kind_count[n["kind"]] = kind_count.get(n["kind"], 0) + 1
    print(f"em-bdable counts: {kind_count}  (total={sum(kind_count.values())})")
    # sanity: unique lineage ids
    lids = [n["lineage_id"] for n in nodes]
    print("unique lineage_ids:", len(set(lids)), "/", len(lids))
    # sanity: no dangling src/dst
    lid_set = set(lids)
    dang = sum(1 for e in edges if e["src"] not in lid_set or e["dst"] not in lid_set)
    print("dangling edges  :", dang)
    return 0


def _run(dry_run: bool, verify: bool) -> int:
    if dry_run:
        return _dry_run()

    driver = cfg.make_driver()
    db = cfg.neo4j_settings().database
    try:
        with driver.session(database=db) as s:
            s.run("MATCH (n) DETACH DELETE n")
            s.run("MATCH ()-[r]->() DELETE r")
        print("cleared any stale graph (idempotent MERGE would handle this, "
              "but we start clean to make first-run deterministic).")
        summary = n4j.ingest(driver, database=db)
    finally:
        driver.close()
    print(json.dumps(summary, indent=2, default=str))

    if verify:
        return _verify(driver_uri=cfg.neo4j_settings().uri)
    return 0


def _verify(*, driver_uri: str) -> int:
    from neo4j import GraphDatabase

    n = cfg.neo4j_settings()
    d = GraphDatabase.driver(n.uri, auth=(n.username, n.password))
    d.verify_connectivity()
    ok = True

    def check(name, actual, expected):
        nonlocal ok
        good = actual == expected
        ok = ok and good
        print(f"  [{'OK ' if good else 'BAD'}] {name}: {actual} (expected {expected})")

    # ground-truth embeddable count straight from the JSONL
    nodes = n4j.load_nodes()
    expect_emb = sum(1 for x in nodes if x["kind"] in n4j.EMBEDDABLE_KINDS)

    try:
        with d.session(database=n.database) as s:
            counts = n4j.counts(d)
            print("\n== counts ==")
            for k, v in counts.items():
                print(f"  {k:14s} : {v}")
            print("\n== assertions ==")
            check("nodes   ", counts["nodes"],    2049)
            check("edges   ", counts["edges"],    4139)
            check("terms   ", counts["terms"],    839)
            check("articles", counts["articles"], 1021)
            check("embedded", counts["embedded"], expect_emb)

            vec_index = [i for i in counts.get("indexes", {}).items()
                         if i[1].startswith("VECTOR")]
            print(f"  [info] vector indexes: {vec_index}")
            if not vec_index:
                ok = False

            # spot-check: a real article -> searchable text + 1024-d emb
            rec = s.run(
                "MATCH (n:Article) "
                "WHERE n.lineage_id = 'data_act_2023_2854:article:11' "
                "RETURN n.search_text AS st, "
                "       CASE WHEN n.emb IS NULL THEN -1 ELSE size(n.emb) END AS dim"
            ).single()
            print(f"  [spot] article 11 search_text = {rec['st']!r}, "
                  f"emb dim = {rec['dim']}")
            if rec["dim"] != 1024:
                ok = False
    finally:
        d.close()
    print("\nVERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="load + audit only, no Neo4j write")
    ap.add_argument("--verify", action="store_true",
                    help="run post-ingest sanity checks")
    args = ap.parse_args()
    return _run(dry_run=args.dry_run, verify=args.verify)


if __name__ == "__main__":
    raise SystemExit(main())
