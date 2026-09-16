"""Listwise LLM re-ranker (the §6.1 "Reranking" component).

Re-orders a *fixed candidate pool* by relevance to a query. It is strictly a
permutation: the re-ranker never admits a new candidate and never drops one.
That is what makes it directly comparable to ``hybrid`` in the benchmark --
Recall/Precision/Hit are set-based and therefore identical to ``hybrid`` by
construction, and only the rank-sensitive axes (MRR, nDCG) can move. If the
LLM cannot be reached or its output does not parse to a valid permutation, the
re-ranker falls back to the original order (a no-op) and records that in the
returned metadata, rather than producing a partial / truncated list
("No Fabricated Results" / "record, don't abort").

The heavy lifting (Ollama call + prompt-hash cache) is delegated to
``common.ollama_chat``, so a rerun over the same (query, pool) is
deterministic and free -- consistent with every other LLM use in this repo.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

try:  # allow use as a top-level module (sys.path points at src/)
    import common as c
except ImportError:  # allow `import reranking` from within src/
    from .. import common as c  # type: ignore


SYSTEM_PROMPT = (
    "You are a precision re-ranker for EU energy, digital and market "
    "regulation. You are given a query and a numbered list of candidate "
    "passages. Judge each candidate on how directly and specifically it "
    "answers the query, and rank them from MOST to LEAST relevant. Do not "
    "invent passages or drop any candidate: use every number exactly once."
)


class Reranker:
    """Listwise LLM re-ranker.

    ``rerank(query, candidates)`` takes the candidate pool (already retrieved)
    and returns the reordered candidate ids. ``candidates`` items must carry at
    least ``lineage_id`` and ``text`` (``doc_id`` is used for context only).
    """

    #: hard cap on how many candidates we ever send to the model -- bounds the
    #: prompt/num_ctx and keeps a single re-rank cheap.
    MAX_CANDIDATES = 20
    #: per-candidate text length cap in characters (bounds prompt size).
    PER_CAND_CHARS = 500
    #: Spec §6 "graph context should be evidence, not relevance".  When any
    #: candidate carries a ``graph_context`` string, these guardrail sentences
    #: are appended to the ranking instructions before the JSON output
    #: requirement.  When NO candidate has context (the A0 ``hybrid_rerank``
    #: baseline) the prompt is byte-identical to the historical v1 prompt --
    #: the guardrails only activate in the graph-aware modes where they belong.
    GRAPH_GUARDRAILS = (
        "The candidate text is the primary evidence of relevance; "
        "Graph evidence lines are supporting context only. "
        "Graph connectivity does not imply relevance. "
        "A candidate's relationship type and direction (AMENDS vs AMENDED_BY) "
        "matter -- weigh them accordingly. "
        "Direct textual evidence in the candidate should generally be "
        "preferred over weak graph proximity, and irrelevant graph "
        "neighbours must not raise a candidate's rank."
    )

    def __init__(self, model: Optional[str] = None,
                 temperature: float = 0.0,
                 max_candidates: int = MAX_CANDIDATES,
                 per_candidate_chars: int = PER_CAND_CHARS):
        self.model = model or c.env_models()["reasoner_default"]
        self.temperature = temperature
        self.max_candidates = max_candidates
        self.per_candidate_chars = per_candidate_chars

    def rerank(self, query: str, candidates: Sequence[Dict],
               max_candidates: Optional[int] = None) -> Dict:
        """Re-rank ``candidates`` by relevance to ``query``.

        ``max_candidates`` is a per-call override of :attr:`MAX_CANDIDATES` (the
        number of candidates actually sent to the model).  ``None`` (the
        default, used by the A0 ``hybrid_rerank`` baseline) keeps the
        historical cap of 20 -- the baseline prompt/pool and cache key are
        therefore byte-identical to the v1 re-ranker.  Graph modes pass their
        pool depth (e.g. ``gr_candidate_k=30``) so a deeper hybrid pool is not
        silently truncated before the model sees it.
        Returns a dict:
          ``order`` : list of lineage_id in the new order (a permutation of the
                      input candidates, same set)
          ``status``: "llm" (model produced the order) | "fallback" (model
                      unavailable / output invalid -> original order kept)
          ``model`` : the model id (or None when no LLM call was made)
          ``pool``  : number of candidates actually given to the model
        """
        ids = [c["lineage_id"] for c in candidates]
        n = len(ids)
        if n <= 1:
            return {"order": ids, "status": "fallback",
                    "model": None, "pool": n}

        cap = self.max_candidates if max_candidates is None else max_candidates
        pool_n = min(n, cap)
        # if the pool exceeds the model cap, re-rank it and keep the overflow
        # at the tail (original relative order) -- still a valid permutation.
        head = list(candidates[:pool_n])
        tail_ids = ids[pool_n:]

        prompt = self._prompt(query, head, pool_n)
        # Cache-key invariant: a call with per-candidate graph context must
        # NOT be served a baseline's cached ordering, since the prompts differ
        # (the evidence lines change the model's judgement).  When NO candidate
        # carries context we keep the historical key (model, temp, query, ids,
        # n) so the ``hybrid_rerank`` (A0) baseline remains cache-hit identical
        # to its pre-v2 numbers; when context is present we hash the full
        # prompt content so each distinct evidence set gets its own entry.
        any_ctx = any((c.get("graph_context") or "").strip() for c in head)
        if any_ctx:
            ck = c.stable_hash("gr", self.model, self.temperature, prompt)
        else:
            parts = [self.model, self.temperature, query, ids, n]
            # An explicit max_candidates changes the prompted candidate set,
            # so it must not share a cached ordering with a call that used the
            # default cap for the same (query, ids, n).  The A0 baseline
            # (max_candidates=None) keeps its historical key unchanged.
            if max_candidates is not None:
                parts += ["mc", max_candidates]
            ck = c.stable_hash(*parts)
        try:
            raw = c.ollama_chat(
                model=self.model,
                prompt=prompt,
                system=SYSTEM_PROMPT,
                temperature=self.temperature,
                num_ctx=32768,
                num_predict=4096,
                use_cache=True,
                cache_key=ck,
            )
            parsed = c.extract_json(raw)
        except Exception:  # noqa: BLE001 -- model down / unreachable
            return {"order": ids, "status": "fallback",
                    "model": self.model, "pool": pool_n}

        ordered = self._to_order(parsed, head)
        if ordered is None:
            return {"order": ids, "status": "fallback",
                    "model": self.model, "pool": pool_n}
        return {"order": ordered + tail_ids, "status": "llm",
                "model": self.model, "pool": pool_n}

    # -- internals ----------------------------------------------------------
    def _prompt(self, query: str, head: Sequence[Dict],
                pool_n: int) -> str:
        lines = [f"Query: {query}", "", f"Candidates ({pool_n}):"]
        # v2 extension: per-candidate ``graph_context`` (a few typed/directional
        # graph relations rendered by ``reranking.graph_context``).  When no
        # candidate carries one, the prompt is byte-identical to the v1
        # re-ranker so the ``hybrid_rerank`` baseline is preserved bit-for-bit.
        for i, c in enumerate(head, start=1):
            text = (c.get("text") or "").strip()
            if len(text) > self.per_candidate_chars:
                text = text[: self.per_candidate_chars] + " …"
            doc = c.get("doc_id") or ""
            block = f"[{i}] {doc}\n{text}".strip()
            gc = (c.get("graph_context") or "").strip()
            if gc:
                block = f"{block}\n  Graph evidence:\n" + "\n".join(
                    "  - " + l for l in gc.splitlines())
            lines.append(block)
        # Spec §6: guardrails activate only when any candidate actually
        # carries graph evidence -- this keeps the A0 baseline
        # (``hybrid_rerank``) prompt byte-identical to the historical v1 text
        # while giving graph modes an explicit "context is supporting, not
        # dominant" instruction.
        any_ctx = any((c.get("graph_context") or "").strip() for c in head)
        if any_ctx:
            lines += ["", self.GRAPH_GUARDRAILS]
        lines += [
            "",
            "Rank the candidates from most to least relevant to the query.",
            "Respond with ONLY a JSON array of the candidate numbers in that "
            "order, e.g. [3, 1, 4, 2]. Use every number exactly once.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _to_order(parsed, head: Sequence[Dict]) -> Optional[List[str]]:
        """Turn model output into an ordered id list that is a valid
        permutation of ``head``, else return None (caller falls back).

        Accepts: [numbers...] (any ints that map to distinct candidates) where
        numbers are the 1-based candidate indices in the prompted order.
        """
        if not isinstance(parsed, (list, tuple)):
            return None
        seen_set: set = set()
        seen_list: List[int] = []
        for item in parsed:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= len(head) and idx not in seen_set:
                seen_set.add(idx)
                seen_list.append(idx)
        # require the model covered *every* candidate before trusting it -- a
        # partial ordering would silently drop candidates (not a permutation).
        if len(seen_list) != len(head):
            return None
        return [head[i - 1]["lineage_id"] for i in seen_list]
