"""GraphRAG query API for the Neo4j-backed graph layer (P4).

Required interface:

* ``from src.graphrag_n4j import GraphRAG;  GraphRAG().query(q)``
* ``result.answer``, ``result.sources``, ``result.chunks``,
  ``result.entities``, ``result.relationships``, ``result.cypher``,
  ``result.retrieval_scores``.
* ``rag.query(q, debug=True)`` must surface the full retrieval trace.
* Prompts live in a dedicated module (not notebooks) -- see
  ``graphrag_n4j.context``.

Implementation:

* reuse the already-built P2 ``Neo4jGraphRetriever`` for all retrieval /
  Cypher / debug plumbing -- zero duplication of traversal logic;
* add ``chunks`` (full-article text + metadata) via a single Cypher call
  on the returned lineage ids;
* add a deterministic :func:`build_context` over those chunks;
* wrap the ``neo4j_config.make_llm()`` Ollama LLM;
* expose a :class:`QueryResult` matching every attribute in the required interface;
* fall back to a deterministic "insufficient context" answer when the
  chunk set is empty or below a floor -- matches the "Say when the context
  is insufficient" prompt rule without burning a token.

This file does NOT subclass ``neo4j_graphrag.retrievers.Retriever``.
That official wrapper is still available via
``neo4j_graphrag.generation.GraphRAG`` for callers who want the raw
``RetrieverResult`` contract; ``graphrag_n4j.GraphRAG`` is the
public entry point and owns the provenance + debug shape.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from neo4j_config import make_driver, make_llm, neo4j_settings
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from neo4j_config import make_driver, make_llm, neo4j_settings

try:
    from .context import (
        SYSTEM_PROMPT,
        build_context,
        format_question,
    )
    from .retriever import (
        EDGE_KINDS,
        Neo4jGraphRetriever,
        TARGET_LABELS,
    )
except ImportError:  # running as top-level script
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from context import SYSTEM_PROMPT, build_context, format_question
    from retriever import EDGE_KINDS, Neo4jGraphRetriever, TARGET_LABELS


# ---------------------------------------------------------------------------
# result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """The fields the query API must return: ``answer``, ``sources``,
    ``chunks``, ``entities``, ``relationships``, ``cypher``,
    ``retrieval_scores``, plus the full retrieval-debug trace."""

    # --- core -------------------------------------------------------------
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)             # doc_ids
    chunks: List[Dict[str, Any]] = field(default_factory=list)   # per-lineage dict
    entities: List[str] = field(default_factory=list)            # lineage_ids that are Term/Entity hubs
    relationships: List[str] = field(default_factory=list)       # human-readable edge strings
    cypher: str = ""
    retrieval_scores: Dict[str, float] = field(default_factory=dict)
    context: Mapping[str, str] = field(default_factory=dict)     # 4 sections
    context_text: str = ""                                        # what the LLM would see
    prompt: str = ""                                              # full prompt sent (system+context+question), already rendered
    elapsed_ms: int = 0

    # --- convenience aliases --------------------------------------------
    @property
    def retrieved_entities(self) -> List[str]:
        return self.entities

    @property
    def retrieved_relationships(self) -> List[str]:
        return self.relationships

    # --- debug mode -------------------------------------------------------
    def debug(self) -> Dict[str, Any]:
        """Render the full retrieval trace for :meth:`GraphRAG.query( ..., debug=True )`.

        Shape: question -> seeds -> traversal -> ranked -> context -> LLM ->
        answer."""
        return {
            "question": self.question,
            "context": dict(self.context),
            "prompt": self.prompt,
            "cypher": self.cypher,
            "seeds": self._seeds,                       # type: ignore[attr-defined]
            "ranked": self._ranked,                     # type: ignore[attr-defined]
            "scores": self.retrieval_scores,
            "elapsed_ms": self.elapsed_ms,
            "answer": self.answer,
        }

    def __repr__(self) -> str:
        return (
            "QueryResult(question={!r}, {} chunks, {} docs, {}ms)".format(
                self.question, len(self.chunks), len(self.sources), self.elapsed_ms
            )
        )


# ---------------------------------------------------------------------------
# main API
# ---------------------------------------------------------------------------

class GraphRAG:
    """Entry point to the local GraphRAG layer over the Neo4j-backed
    graph, `from src.rag import GraphRAG`."""

    def __init__(
        self,
        *,
        llm: Optional[Any] = None,
        llm_model: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_max_tokens: int = 2048,  # qwen3.8 needs headroom: thinking tokens + final answer
        max_hops: Optional[int] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        max_results: Optional[int] = None,
        edge_kinds: Optional[Sequence[str]] = None,
        min_chunks_for_answer: int = 1,
        prompt_temperature: Optional[float] = None,
    ) -> None:
        self.llm_model = llm_model or "qwen3.8:27b"
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.min_chunks_for_answer = min_chunks_for_answer
        self.prompt_temperature = prompt_temperature

        # build the underlying P2 retriever (configurable parameters)
        self.retriever = Neo4jGraphRetriever(
            max_hops=max_hops,
            top_k=top_k,
            min_score=min_score,
            max_results=max_results,
            edge_kinds=tuple(edge_kinds or EDGE_KINDS),
        )

        # build the LLM (lazy: only on query, so importing this module costs
        # nothing)
        self._llm = llm
        mp: Dict[str, Any] = {"max_tokens": llm_max_tokens, "temperature": 0.1}
        if llm_temperature is not None:
            mp["temperature"] = float(llm_temperature)
        self._mp = mp
        self._corpus_cache: Optional[Dict[str, Any]] = None

    def query(
        self,
        question: str,
        *,
        k: int = 10,
        debug: bool = False,
        message_history: Optional[Sequence[Any]] = None,
        examples: str = "",
        timeout_seconds: int = 60,
    ) -> QueryResult:
        """Run the full GraphRAG pipeline.

        1. retrieval  -- ``Neo4jGraphRetriever.search`` (vector seeds +
           bounded Cypher traversal),
        2. context    -- :func:`build_context`,
        3. generation -- the Ollama LLM via ``neo4j_config.make_llm``.

        ``debug=True`` does not change the return type -- it just fills in
        all the debug fields (seeds, ranked, cypher, prompt) on the same
        :class:`QueryResult`.  Use ``result.debug()`` to render.
        """
        t0 = time.time()
        question = (question or "").strip()

        # ---- 1. retrieval ---------------------------------------------------
        ranked = self.retriever.search(question, k=k)
        dbg = self.retriever.last_debug or {}
        dbg_dict = dbg.dict() if hasattr(dbg, "dict") else dict(dbg)  # type: ignore[call-arg]

        # ---- 2. context -----------------------------------------------------
        chunk_lookup = self._fetch_chunk_details(ranked)
        # merge in the full article text from the 03b chunk corpus (graph nodes
        # carry only title/number for :Article -- the body lives in the corpus)
        chunk_lookup = self._merge_corpus_text(chunk_lookup)
        # pull Term/Entity hub metadata for the "Relevant Entities" section
        entity_lids = [
            s for s in (dbg_dict.get("seed_ids") or [])
            if ":term:" in s or ":entity:" in s
        ]
        entity_lookup = self._fetch_entity_details(entity_lids)
        context = build_context(
            ranked,
            chunk_lookup,
            entities=entity_lids,
            entity_lookup=entity_lookup,
        )
        context_text = "\n\n".join(
            context[k] for k in ("documents", "chunks", "entities", "relationships", "provenance")
        )

        # build the final prompt (system + context + question)
        # the official neo4j_graphrag.GraphRAG path uses RagTemplate + llm
        # invoke with system_instruction; we do the equivalent directly so
        # we control the exact text sent to the LLM.
        system = SYSTEM_PROMPT
        user = format_question(context=context_text, question=question)
        if examples:
            user = f"Examples:\n{examples}\n{user}"

        # relationship views (result.relationships) -- the hop >= 1
        # edge strings.
        rel_strings: List[str] = []
        for lid, hop, kinds in ranked:
            if hop == 0:
                continue
            kinds_str = ";".join(kinds) or "PATH"
            rel_strings.append(f"{lid}  {kinds_str}@{hop}hop")

        chunk_rows = [
            {k: v for k, v in chunk_lookup.get(lid, {}).items()}
            for lid, _h, _k in ranked if lid in chunk_lookup
        ]

        # ---- 3. generation --------------------------------------------------
        # Fall back to a deterministic short answer when context is too thin
        # (saves an LLM call; mirrors the "say when context is insufficient" rule).
        if not ranked or len(chunk_rows) < max(1, self.min_chunks_for_answer):
            answer = "The supplied context is insufficient to answer this question. " + (
                "No retrieval hits were produced for the graph search."
                if not ranked else
                f"Only {len(chunk_rows)} chunk(s) were retrieved, below the "
                f"minimum of {self.min_chunks_for_answer}."
            )
            # mark clearly so callers can detect the insufficient-context fallback
            answer = "[INSUFFICIENT CONTEXT] " + answer
            elapsed = int((time.time() - t0) * 1000)
            return self._result(
                question=question,
                answer=answer,
                dbg=dbg_dict,
                ranked=ranked,
                chunk_lookup=chunk_lookup,
                context=context,
                context_text=context_text,
                system=system,
                user=user,
                elapsed_ms=elapsed,
                entity_lids=entity_lids,
                rel_strings=rel_strings,
                chunk_rows=chunk_rows,
                timeout_seconds=timeout_seconds,
                llm_was_called=False,
            )

        if self._llm is None:
            from neo4j_config import graphrag_settings, make_llm as _mk_llm

            settings = graphrag_settings()
            settings.llm_model = self.llm_model
            self._llm = _mk_llm(settings, model_params=self._mp)
            llm = self._llm
        else:
            llm = self._llm
        # OpenAI-compatible invoke via the official neo4j_graphrag LLM adapter
        res = llm.invoke(
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        answer = (res.content or "").strip() if hasattr(res, "content") else str(res)
        elapsed = int((time.time() - t0) * 1000)

        return self._result(
            question=question,
            answer=answer,
            dbg=dbg_dict,
            ranked=ranked,
            chunk_lookup=chunk_lookup,
            context=context,
            context_text=context_text,
            system=system,
            user=user,
            elapsed_ms=elapsed,
            entity_lids=entity_lids,
            rel_strings=rel_strings,
            chunk_rows=chunk_rows,
            timeout_seconds=timeout_seconds,
            llm_was_called=True,
        )

    # -----------------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------------

    def _fetch_chunk_details(
        self, ranked: Sequence[Tuple[str, int, Sequence[str]]],
    ) -> Dict[str, Dict[str, Any]]:
        """One Cypher call to pull text/metadata for every ranked lineage.

        Only Article / Preamble nodes carry the chunk text we want for the
        prompt -- Term / Entity hubs are not returned.
        """
        if not ranked:
            return {}
        lids = [lid for lid, _h, _k in ranked if lid]
        if not lids:
            return {}
        driver = make_driver()
        try:
            with driver.session() as s:
                rows = list(s.run(
                    "UNWIND $ids AS id "
                    "MATCH (n) WHERE n.lineage_id = id "
                    "  AND (n:Article OR n:Preamble) "
                    "RETURN n.lineage_id AS lid, "
                    "       n.kind AS node_type, "
                    "       n.title AS title, "
                    "       n.term AS term, "
                    "       n.number AS number, "
                    "       n.search_text AS search_text, "
                    "       n.doc_id AS doc_id, "
                    "       n.text AS text, "
                    "       n.lineage_id AS lineage_id",
                    {"ids": lids},
                ))
        finally:
            driver.close()
        out: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            d = dict(r)
            # strip long fields we don't need (keep text for the prompt, cap
            # later in context builder)
            out[d.get("lid")] = d
        return out

    def _fetch_entity_details(self, lids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """One Cypher call for :Term / :Entity hubs (no chunk text; metadata only).

        Used :func:`GraphRAG.query` to populate ``result.entities`` with just
        enough detail (search_text) for the LLM to reference a named concept.
        """
        if not lids:
            return {}
        driver = make_driver()
        try:
            with driver.session() as s:
                rows = list(s.run(
                    "UNWIND $ids AS id "
                    "MATCH (n) WHERE n.lineage_id = id "
                    "  AND (n:Term OR n:Entity) "
                    "RETURN n.lineage_id AS lid, "
                    "       n.kind AS node_type, "
                    "       n.term AS term, "
                    "       n.search_text AS search_text, "
                    "       n.doc_id AS doc_id",
                    {"ids": list(lids)},
                ))
        finally:
            driver.close()
        return {r["lid"]: dict(r) for r in rows if r.get("lid")}

    def _merge_corpus_text(
        self, chunk_lookup: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Fill ``text`` (and ``node_type``) from the 03b chunk corpus.

        Graph nodes of label :Article / :Preamble carry only ``title`` /
        ``number`` / ``search_text``; the full regulatory body lives in the
        chunk corpus (the same one that ``sparse`` / ``dense`` / ``hybrid``
        use).  This keeps P4 output consistent with the rest of the retrieval
        layer.  Falls back quietly if the corpus is not present (e.g. running
        before the 03b notebook).
        """
        if self._corpus_cache is None:
            try:
                from retrieval._corpus import load_corpus

                self._corpus_cache = load_corpus()
            except Exception:
                self._corpus_cache = {"chunks": []}
        chunks = self._corpus_cache.get("chunks", [])
        text_by_lid: Dict[str, str] = {
            ch.lineage_id: ch.text for ch in chunks if ch.lineage_id
        }
        for lid, rec in chunk_lookup.items():
            if not rec.get("text") and lid in text_by_lid:
                rec["text"] = text_by_lid[lid]
        return chunk_lookup

    def _result(self, **kwargs: Any) -> QueryResult:
        q: str = kwargs.pop("question")
        ans: str = kwargs.pop("answer")
        dbg: Any = kwargs.pop("dbg", {})
        ranked: List[Tuple[str, int, Sequence[str]]] = kwargs.pop("ranked", [])
        chunk_lookup: Dict[str, Dict[str, Any]] = kwargs.pop("chunk_lookup", {})
        context: Mapping[str, str] = kwargs.pop("context", {})
        context_text: str = kwargs.pop("context_text", "")
        system: str = kwargs.pop("system", "")
        user: str = kwargs.pop("user", "")
        elapsed: int = kwargs.pop("elapsed_ms", 0)
        entity_lids: List[str] = kwargs.pop("entity_lids", [])
        rel_strings: List[str] = kwargs.pop("rel_strings", [])
        chunk_rows: List[Dict[str, Any]] = kwargs.pop("chunk_rows", [])

        # build the documented result fields
        doc_ids = []
        for c in chunk_rows:
            d = c.get("doc_id")
            if d and d not in doc_ids:
                doc_ids.append(d)
        scores = dict((dbg or {}).get("score_of_seed") or {})

        r = QueryResult(
            question=q,
            answer=ans,
            sources=doc_ids,
            chunks=chunk_rows,
            entities=entity_lids,
            relationships=rel_strings,
            cypher=(dbg or {}).get("cypher", ""),
            retrieval_scores=scores,
            context=context,
            context_text=context_text,
            prompt=system + "\n\n" + user,
            elapsed_ms=elapsed,
        )
        # stash the full debug payload (used by .debug() only)
        r._seeds = (dbg or {}).get("seeds", {})          # type: ignore[attr-defined]
        r._ranked = [(lid, hop, list(kinds)) for lid, hop, kinds in ranked]  # type: ignore[attr-defined]
        return r


def default() -> GraphRAG:
    """Build the shared :class:`GraphRAG` instance from ``.env``."""
    return GraphRAG()


__all__ = ["GraphRAG", "QueryResult", "default"]
