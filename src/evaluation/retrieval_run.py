"""Layer A -- run every retrieval config over the shared benchmark.

Per "Evaluation Architecture" (modular pipeline), "Retrieval Metrics at Multiple K Values" (multi-K table),
"Dense Semantic Search Evaluation" / "Sparse Search Evaluation" (per-query +
aggregate), "Graph + Hybrid Evaluation" (hybrid + graph) and "Evaluation Configuration Matrix" (config matrix).

Every system is evaluated over the SAME 60 queries with the SAME relevance
judgments and the SAME k-set (the "Methodological Neutrality" requirement).  A system
is just a (mode, dense_model) pair from ``EXPERIMENTS``:

    sparse / dense(base) / hybrid /
    graph / neo4j_graph / hybrid_graph

``hybrid`` and ``hybrid_graph`` are now structurally distinct retrieval modes:
``hybrid`` fuses sparse+dense, fixes the top-k, and only *annotates* the
survivors with a graph-boost (no re-ordering, no new nodes).  ``hybrid_graph``
feeds the 1-2 hop graph neighbours as a third RRF list, so graph-sourced chunks
can enter the top-k and shift the ordering -- the dedicated "hybrid + graph"
system named by the "Graph + Hybrid Evaluation" requirement.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import config as E
from .benchmark import BenchmarkItem
from . import metrics

BASE_DENSE = "all-MiniLM-L6-v2"


def _dense_specs_for(variant: Optional[str]) -> tuple:
    """(dense_models, dense_specs) for a config's dense encoder variant."""
    if variant is None:
        return None, None
    if variant == "base":
        return ["base"], {"base": BASE_DENSE}
    raise ValueError(f"unknown dense variant {variant!r}")


class RetrievalEngine:
    """Lazy per-variant Retriever instances (each variant loads its own
    dense model); sparse/graph modes are shared corpus so we keep one
    retriever per variant for isolation."""

    def __init__(self):
        self._by_variant: Dict[str, object] = {}
        self._by_mode: Dict[str, object] = {}

    def for_system(self, system: str):
        spec = E.EXPERIMENTS[system]
        mode = spec["retrieval"]
        variant = spec["dense_model"]
        if variant in self._by_variant:
            return self._by_variant[variant]
        if mode != "dense":
            # sparse/hybrid/hybrid_graph/graph/neo4j share one base retriever
            if None not in self._by_mode:
                from retrieval import Retriever
                self._by_mode[None] = Retriever(
                    dense_models=["base"],
                    dense_specs={"base": BASE_DENSE})
            return self._by_mode[None]
        from retrieval import Retriever
        models, specs = _dense_specs_for(variant)
        r = Retriever(dense_models=models, dense_specs=specs)
        self._by_variant[variant] = r
        return r

    def retrieve(self, system: str, query: str, k: int) -> List[dict]:
        r = self.for_system(system)
        spec = E.EXPERIMENTS[system]
        mode = spec["retrieval"]
        # The graph-aware re-ranker systems honour the EvalConfig feature
        # flags (gr_enable, hops, snippets, ...); ``hybrid_rerank`` (A0)
        # deliberately does not, so its results stay the pre-v2 numbers.
        gr_cfg, expand, pool_n = (None, False, None)
        if mode in E.GRAPH_RERANK_MODES:
            cfg = E.EvalConfig()
            gr_cfg, expand = cfg.context_config(mode)
            # spec §2: honour the configurable candidate depth so the
            # re-ranker pool is ``gr_candidate_k`` chunks (not the historical
            # ``max(k, 5)``).  A0 ``hybrid_rerank`` never reaches here, so its
            # results stay the pre-v2 numbers.
            pool_n = cfg.gr_candidate_k
        out = []
        for rank, x in enumerate(
                r.retrieve(query, k=k, mode=mode,
                           graph_cfg=gr_cfg, with_expand=expand,
                           pool_n=pool_n),
                start=1):
            rec = {
                "rank": rank,
                "lineage_id": x.lineage_id,
                "doc_id": x.doc_id,
                "score": x.score,
                "methods": list(x.source_methods),
                "graph_edge_type": x.graph_edge_type,
                "model_variant": x.model_variant,
            }
            if x.extra:
                rec["extra"] = {k_: v for k_, v in x.extra.items()}
            out.append(rec)
        return out

    def retrieve_chunks(self, system: str, query: str, k: int) -> List[dict]:
        """Top-k with full chunk ``text`` (for context construction), in the
        SAME rank order as :meth:`retrieve`.

        Mirrors :meth:`retrieve` for the graph-aware re-ranker modes: when the
        mode is in ``GRAPH_RERANK_MODES`` the call passes ``graph_cfg``,
        ``with_expand`` and ``pool_n`` (spec §2 / ``gr_candidate_k``) so the
        *context* the answer generator sees is built from the same deeper,
        re-ranked pool -- not from the historical ``max(k, 5)`` seed. Without
        this, the doc's ``#3 -> #7 -> #8`` answer-level causal chain would be
        scored on a context that ignores the very re-ranking under test.
        """
        r = self.for_system(system)
        mode = E.EXPERIMENTS[system]["retrieval"]
        gr_cfg, expand, pool_n = (None, False, None)
        if mode in E.GRAPH_RERANK_MODES:
            cfg = E.EvalConfig()
            gr_cfg, expand = cfg.context_config(mode)
            pool_n = cfg.gr_candidate_k
        return [
            {"lineage_id": x.lineage_id, "doc_id": x.doc_id,
             "score": x.score, "text": x.text}
            for x in r.retrieve(query, k=k, mode=mode,
                                graph_cfg=gr_cfg, with_expand=expand,
                                pool_n=pool_n)
        ]


def per_query_metrics(system: str, query: str, item: BenchmarkItem,
                      engine: RetrievalEngine,
                      k_set: Sequence[int] = E.EvalConfig().k_values
                      ) -> dict:
    """Retrieve once at the largest k, truncate to each k (no re-retrieval).

    Truncating a single top-K list keeps rankings comparable across k and
    avoids k-dependent re-normalisation from the fusion layer.
    """
    k_max = max(k_set)
    retrieved = engine.retrieve(system, query, k_max)
    lids = [x["lineage_id"] for x in retrieved]
    m: Dict[str, dict] = {}
    target = item.target_lineage_id
    for k in k_set:
        rl = lids[:k]
        m[str(k)] = {
            "system": system,
            "query_id": item.query_id,
            "k": k,
            "recall": metrics.recall_at_k(rl, [target], k),
            "precision": metrics.precision_at_k(rl, [target], k),
            "hit": metrics.hit_rate_at_k(rl, [target], k),
            "mrr": metrics.reciprocal_rank(rl, [target]),
            "ndcg": metrics.ndcg_at_k(rl, [target], k),
        }
    first_rank = lids.index(target) + 1 if target in lids else None
    return {
        "query_id": item.query_id,
        "question": item.question,
        "system": system,
        "category": item.category,
        "target": target,
        "retrieved_top20": lids[:20],
        "scores": [round(x["score"], 6) for x in retrieved[:20]],
        "methods": [x["methods"] for x in retrieved[:20]],
        "target_rank": first_rank,
        "metrics": m,
    }


def run_retrieval(items: Sequence[BenchmarkItem],
                  systems: Optional[Sequence[str]] = None,
                  engine: Optional[RetrievalEngine] = None,
                  out_dir: Path = E.OUT_PER_QUERY,
                  ) -> Dict[str, list]:
    """Run all systems over all items; persist per-query + aggregate.

    Returns ``{system: [per-query dicts]}``.  Gracefully skips a system on
    hard error (e.g. Neo4j down for neo4j_graph) instead of aborting the
    whole run, and records the skip (per "No Fabricated Results")."""
    systems = list(systems) if systems else list(E.EXPERIMENTS)
    engine = engine or RetrievalEngine()
    cfg = E.EvalConfig()
    results: Dict[str, list] = {}
    errors: Dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for system in systems:
        rows = []
        failed = False
        for qno, item in enumerate(items, start=1):
            t0 = time.time()
            try:
                row = per_query_metrics(system, item.question, item,
                                        engine, cfg.k_values)
            except Exception as exc:  # noqa: BLE001 - record, don't abort
                if not failed:
                    errors[system] = f"{type(exc).__name__}: {exc}"
                    failed = True
                    break
                raise
            row["retrieval_ms"] = round((time.time() - t0) * 1000, 1)
            rows.append(row)
        results[system] = rows
        if failed:
            continue
        _save_system(system, rows, out_dir)
    if errors:
        (out_dir / "skipped_systems.json").write_text(
            json.dumps(errors, indent=2))
    agg = aggregate(results, items)
    _save_aggregate(agg, out_dir)
    return results


def _save_system(system: str, rows: list, out_dir: Path):
    p = out_dir / f"retrieval_{system}.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def aggregate(results: Dict[str, list],
              items: Sequence[BenchmarkItem]) -> List[dict]:
    """Long-form table system/metric/k/value + counts (per "Retrieval Metrics at Multiple K Values" / "Machine-Readable Results")."""
    cfg = E.EvalConfig()
    rows = []
    for system, data in results.items():
        for k in cfg.k_values:
            vals = {m: [] for m in
                    ("recall", "precision", "hit", "mrr", "ndcg")}
            for r in data:
                mk = r["metrics"].get(str(k))
                if not mk:
                    continue
                for m in vals:
                    if mk.get(m) is not None:
                        vals[m].append(mk[m])
            for m, vs in vals.items():
                if vs:
                    rows.append({"system": system, "metric": m, "k": k,
                                 "value": sum(vs) / len(vs),
                                 "n": len(vs)})
    rows.sort(key=lambda r: (
        {s: i for i, s in enumerate(E.EXPERIMENTS)}[r["system"]],
        r["metric"], r["k"]))
    return rows


def _save_aggregate(agg: List[dict], out_dir: Path):
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    cp = out_dir.parent / "aggregate" / "retrieval_aggregate.csv"
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "metric", "k",
                                          "value", "n"])
        w.writeheader()
        w.writerows(agg)
    jp = out_dir.parent / "aggregate" / "retrieval_aggregate.json"
    jp.write_text(json.dumps(agg, indent=2))


def load_aggregate(path: Path = E.OUT_AGGREGATE / "retrieval_aggregate.csv"
                   ) -> List[dict]:
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["k"] = int(r["k"])
        r["value"] = float(r["value"])
        r["n"] = int(r["n"])
    return rows
