"""End-to-end RAG answer generation -- ONE neutral pipeline for all systems.

Per "End-to-End RAG Evaluation" (run retrieval -> context -> generate),
"Methodological Neutrality" (same generation config across systems -- comparison
is on *retrieval*, not prompt engineering), and "Answer-Level Metrics"
(document model + generation settings).

Neutrality design (important): every system -- dense / sparse / hybrid /
graph / neo4j / finetuned -- is asked the SAME question with the SAME prompt
template, the SAME context window size, and the SAME LLM settings.  The only
thing that differs between runs is *which chunks each system retrieved*.
This is what lets the thesis attribute answer-quality differences to
retrieval rather than to prompt luck (per "Groundedness / Faithfulness": separate retrieval, context,
generation).

The LLM is reached through ``common.ollama_chat`` which caches by prompt
hash, so a re-run is deterministic and free (per the "Reproducibility" requirement)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import config as E
from .benchmark import BenchmarkItem

SYSTEM_PROMPT = (
    "You are a precise compliance auditor answering questions about "
    "EU/UK energy, digital and market regulation. "
    "Ground every claim in the supplied evidence. "
    "If the evidence is insufficient to answer, say so in one sentence "
    "instead of guessing. Cite the instrument and article number "
    "for each obligation you state."
)

#: The context slice.  Identical for every system (per "Methodological Neutrality").
CONTEXT_WINDOW = 5

_CONTEXT_TEMPLATE = (
    "Relevant regulatory evidence (top {n} retrieved chunks):\n\n"
    "{evidence}\n\n"
    "Question: {question}\n\n"
    "Answer using only the evidence above. Be concise and cite "
    "instrument + article number."
)


def _format_evidence(chunks: Sequence[Dict], window: int = CONTEXT_WINDOW) -> str:
    parts = []
    for i, c in enumerate(list(chunks)[:window], start=1):
        head = f"[{i}] "
        if c.get("doc_id"):
            head += f"{c['doc_id']}  "
        if c.get("lineage_id"):
            head += f"({c['lineage_id']})\n"
        parts.append(head + (c.get("text") or "").strip())
    return "\n\n".join(parts)


def format_graph_evidence(gold_edges: Sequence[dict]) -> str:
    """Render the item's gold typed/directional relations as judge-facing text.

    Each gold edge is a ``{source, relation, target}`` dict from
    ``benchmark.py``.  Returned as a compact bullet list; empty string when
    the item carries no gold edges (e.g. the single_document family).
    """
    if not gold_edges:
        return ""
    lines = []
    for e in gold_edges:
        lines.append(
            f"- {e.get('source', '?')}  --[{e.get('relation', '?')}-->  "
            f"{e.get('target', '?')}")
    return ("\n".join(lines) if lines else "")


def build_prompt(question: str, chunks: Sequence[Dict],
                 window: int = CONTEXT_WINDOW) -> str:
    """Render the standardised user prompt from retrieved chunks."""
    return _CONTEXT_TEMPLATE.format(
        n=min(window, max(len(chunks), 1)),
        evidence=_format_evidence(chunks, window) or "(no evidence retrieved)",
        question=question,
    )


def generate(question: str, chunks: Sequence[Dict],
             cfg: Optional[E.EvalConfig] = None,
             use_cache: bool = True) -> Dict:
    """One LLM answer for a (question, retrieved-chunks) pair.

    Deterministic given the same inputs (cached via common.ollama_chat).
    Returns {answer, system_prompt, user_prompt, n_chunks, model, ...}.
    """
    import common as c

    cfg = cfg or E.EvalConfig()
    user = build_prompt(question, chunks)
    prompt_sha = c.stable_hash(cfg.llm_model, cfg.llm_temperature,
                               cfg.llm_max_tokens, SYSTEM_PROMPT, user)
    try:
        answer = c.ollama_chat(
            model=cfg.llm_model,
            prompt=user,
            system=SYSTEM_PROMPT,
            temperature=cfg.llm_temperature,
            num_predict=cfg.llm_max_tokens,
            use_cache=use_cache,
            cache_key=f"gen_{prompt_sha}",
        )
    except Exception as exc:  # noqa: BLE001 -- record as empty, report gap
        answer = ""
        _record_error(exc)
    return {
        "answer": (answer or "").strip(),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user,
        "n_chunks": len(chunks),
        "model": cfg.llm_model,
        "temperature": cfg.llm_temperature,
        "max_tokens": cfg.llm_max_tokens,
        "window": CONTEXT_WINDOW,
    }


def run_generation(items: Sequence[BenchmarkItem],
                   results: Dict[str, list],
                   systems: Optional[Sequence[str]] = None,
                   engine=None,
                   out_dir: Path = E.OUT_GENERATION
                   ) -> Dict[str, list]:
    """Generate answers for every (item, system) that has retrieval rows.

    ``results`` is the per-query retrieval map from retrieval_run.run_retrieval.
    """
    from .retrieval_run import RetrievalEngine

    systems = list(systems) if systems else [s for s in E.GENERATION_SYSTEMS
                                             if s in results]
    engine = engine or RetrievalEngine()
    cfg = E.EvalConfig()
    out: Dict[str, list] = {}
    for system in systems:
        rows = []
        for item in items:
            rq = next((r for r in results.get(system, [])
                       if r["query_id"] == item.query_id), None)
            if rq is None:
                continue
            chunks = engine.retrieve_chunks(system, item.question,
                                            CONTEXT_WINDOW)
            gen = generate(item.question, chunks, cfg)
            gen.update({
                "query_id": item.query_id,
                "system": system,
                "category": item.category,
                "target": item.target_lineage_id,
                "reference_answer": item.reference_answer,
                "reference_basis": "target_chunk_text",
                "retrieval_target_rank": rq.get("target_rank"),
                "target_in_context": any(
                    c["lineage_id"] == item.target_lineage_id
                    for c in chunks),
                "graph_evidence": format_graph_evidence(item.gold_edges),
                "intended_relation": item.intended_relation,
                "intended_direction": item.intended_direction,
            })
            rows.append(gen)
        out[system] = rows
        _save(system, rows, out_dir)
    return out


def _save(system, rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"generation_{system}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _record_error(exc):
    try:
        from . import config as E2
        E2.OUT_GENERATION.mkdir(parents=True, exist_ok=True)
        with open(E2.OUT_GENERATION / "generation_errors.jsonl", "a") as f:
            f.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
    except Exception:
        pass
