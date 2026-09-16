"""Graph-aware context for the listwise re-ranker (v2, §6.1 extension).

Given a retrieved candidate (a chunk whose graph identity is an article /
preamble / document node) and the user query, this module renders a
compact, typed, direction-aware description of the candidate's local graph
neighbourhood.  That text is then injected (per candidate) into the
re-ranker prompt as *evidence* -- cross-references the candidate relies on,
the instruments that amend or supersede it, the definitions it hosts --
so the LLM can weigh relational relevance, not just lexical overlap.

Design invariants (see task spec §"Context builder"):
  * **read-only**: never mutates the graph; pure function over ``GraphLike``
  * **graceful**: any internal error -> return ``""`` (empty context); the
    caller then falls back to the base re-ranker prompt byte-for-byte
  * **budgeted**: bounded by ``max_edges`` / ``max_paths`` and an overall
    character cap (``token_limit`` * 4, chars-per-token heuristic), so one
    candidate can never blow out the re-rank prompt
  * **direction-aware**: outgoing edges use the stored kind's label
    (CITES / AMENDS / SUPERSEDES / ...), incoming edges use the ``*_BY``
    form (CITED_BY / AMENDED_BY / SUPERSEDED_BY / ...); this is the whole
    ``*_BY`` story -- a runtime view over ``adj_in``, no new edges
  * **cycle-safe**: the 2-hop pass excludes seeds and 1-hop neighbours,
    so a 2-cycle A<->B renders each direction exactly once
  * **deduplicated**: one line per (kind, direction, neighbour); multiple
    edges to the same neighbour from the same kind fold into one line
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import common as c
except ImportError:  # allow `import reranking.graph_context` from within src/
    from .. import common as c  # type: ignore


# ---------------------------------------------------------------------------
# direction-aware relation labels
# ---------------------------------------------------------------------------
# OUT_LABEL: outgoing edges (src == candidate), as stored in the graph.
# IN_LABEL : incoming edges  (dst == candidate), presented to the LLM as
#            ``<candidate> <IN_LABEL> <neighbour>``.  Where no natural
#            passive exists we mirror the verb (DEFINES / CONTAINS).
OUT_LABEL: Dict[str, str] = {
    "CROSS_REFERENCES": "CITES",
    "AMENDS":           "AMENDS",
    "DEFINED_IN":       "DEFINES",       # term -> article; presented via the term side
    "APPLIES_TO":       "APPLIES_TO",
    "PART_OF":          "PART_OF",
    "IMPLEMENTS":       "IMPLEMENTS",
    "SUPERSEDES":       "SUPERSEDES",
}
IN_LABEL: Dict[str, str] = {
    "CROSS_REFERENCES": "CITED_BY",
    "AMENDS":           "AMENDED_BY",
    "DEFINED_IN":       "DEFINES",       # article DEFINES term: article <-DEFINED_IN- term
    "APPLIES_TO":       "APPLIED_TO",
    "PART_OF":          "CONTAINS",      # document CONTAINS article
    "IMPLEMENTS":       "IMPLEMENTED_BY",
    "SUPERSEDES":       "SUPERSEDED_BY",
}
# kinds we do NOT surface in the context (e.g. pure containment noise):
SKIP_KINDS = frozenset()


@dataclass(frozen=True)
class GraphContextConfig:
    """Budgets + toggles for :func:`build_graph_context`.

    Values are deliberately small: the context is *evidence* for a 20-
    candidate re-rank prompt, not a full subgraph dump.
    """
    include_outgoing: bool = True        # candidate --OUT--> neighbours
    include_incoming: bool = True        # neighbours --IN-> candidate
    include_paths: bool = True          # 2-hop candidate -> mid -> end
    include_snippets: bool = True       # append edge evidence snippet (<=120 chars)
    include_definitions: bool = True    # DEFINED_IN / APPLIES_TO lines
    query_aware: bool = True            # order neighbours by query-token overlap
    max_edges: int = 8                  # 1-hop lines total (out+in combined)
    max_paths: int = 3                  # 2-hop lines
    token_limit: int = 500              # hard cap ~chars (chars = token_limit * 4)
    # restrict which kinds appear (empty = all known kinds)
    kinds: Optional[Tuple[str, ...]] = None
    # restrict directions: any subset of {"out", "in", "path"}; default all
    directions: Optional[Tuple[str, ...]] = None

    def char_cap(self) -> int:
        return self.token_limit * 4


# kind filter helper -------------------------------------------------------
def _allowed(kind: str, cfg: GraphContextConfig) -> bool:
    if kind in SKIP_KINDS:
        return False
    if cfg.kinds is not None and kind not in cfg.kinds:
        return False
    if not cfg.include_definitions and kind in ("DEFINED_IN", "APPLIES_TO"):
        return False
    return True


def _direction_ok(cfg: GraphContextConfig, d: str) -> bool:
    return cfg.directions is None or d in cfg.directions


# node labelling -------------------------------------------------------------
def node_label(nodes: dict, lid: str, max_chars: int = 80) -> str:
    """Human-readable label for a graph node id.

    Prefers the stored ``title`` / ``term`` / ``label``; falls back to a
    structural slug (``<doc> §<n>`` / ``<term>`` / ``entity:<label>``).
    """
    n = nodes.get(lid) if nodes else None
    if isinstance(n, dict):
        if n.get("kind") == "document":
            # full titles are noisy ("REGULATION (EU) 2016/679 OF THE EUROPEAN
            # PARLIAMENT...") -- compress to "<KIND> (<yyyy/nnnn>)" when a
            # number is present, else the short doc_id slug.
            title = (n.get("title") or "").strip()
            m = re.search(r"\b\d{4}/\d{3,5}", title)
            k = re.match(r"(REGULATION|DIRECTIVE|DECISION)", title, re.I)
            if m and k:
                return f"{k.group(1).capitalize()} ({m.group(0)})"
            doc = n.get("doc_id")
            if doc:
                return doc
        for key in ("title", "term", "label"):
            v = n.get(key)
            if isinstance(v, str) and v.strip():
                s = v.strip()
                if len(s) > max_chars:
                    s = s[: max_chars - 1] + "…"
                return s
        doc = n.get("doc_id")
        if doc:
            return doc
    # structural: <doc>:article:<num> / <doc>:preamble / ext:article:<num>
    if lid.startswith("ext:"):
        m = re.match(r"ext:article:(.+)", lid)
        if m:
            return f"Art. {m.group(1)} (external)"
        m = re.match(r"ext:instrument:(.+)", lid)
        if m:
            return f"Regulation {m.group(1).replace('-', '/')} (external)"
    m = re.match(r"([^:]+):article:([0-9]+[a-z]?)", lid)
    if m:
        return f"{m.group(1)} Art. {m.group(2)}"
    if lid.endswith(":preamble"):
        return f"{lid.split(':', 1)[0]} Preamble"
    return lid.replace(":", " / ")[:max_chars]


# query-aware scoring ---------------------------------------------------------
_WORD = re.compile(r"[a-z0-9]+")

def _tokens(s: str) -> frozenset:
    return frozenset(t for t in _WORD.findall(s.lower()) if len(t) >= 3)

def _qscore(query: str, text: str) -> int:
    q = _tokens(query)
    if not q:
        return 0
    return len(q & _tokens(text))


# edge-line rendering ---------------------------------------------------------
def _snippet_of(edges: Optional[Sequence[dict]], src: str, dst: str,
                kind: str, max_chars: int = 120) -> str:
    if not edges:
        return ""
    best = ""
    for e in edges:
        if e.get("src") == src and e.get("dst") == dst and e.get("kind") == kind:
            s = (e.get("snippet") or "").strip()
            if len(s) > len(best):
                best = s
    if not best:
        return ""
    best = re.sub(r"\s+", " ", best)
    if len(best) > max_chars:
        best = best[: max_chars - 1] + "…"
    return best


def _line(kind: str, direction: str, neighbour: str, label: str,
          snippet: str) -> str:
    base = f"{OUT_LABEL.get(kind, kind)} {label}" \
        if direction == "out" else f"{IN_LABEL.get(kind, kind)} {label}"
    if snippet:
        base = f"{base} — \"{snippet}\""
    return base


# main entry ------------------------------------------------------------------
def context_lines(graph, candidate_lid: str, query: str = "",
                  cfg: Optional[GraphContextConfig] = None) -> List[str]:
    """The raw relation lines for ``candidate_lid`` (no header).

    Never raises -- returns ``[]`` on any failure (the caller then
    degrades to the base prompt).
    """
    try:
        return _build_lines(graph, candidate_lid, query or "",
                            cfg or GraphContextConfig())
    except Exception:  # noqa: BLE001 -- graceful degradation is a spec invariant
        return []


def build_graph_context(graph, candidate_lid: str, query: str = "",
                        cfg: Optional[GraphContextConfig] = None) -> str:
    """Render the typed/directional local graph context for ``candidate_lid``.

    Parameters
    ----------
    graph : GraphLike (or any object exposing ``.nodes``, ``.adj``,
            ``.adj_in``; optional ``.edges`` list of raw edge dicts for
            snippets)
    candidate_lid : lineage id of the retrieved node
    query : user query (used for query-aware neighbour ordering)
    cfg : budget/toggle overrides; defaults are conservative

    Returns a multi-line string (possibly empty).  Never raises.
    """
    lines = context_lines(graph, candidate_lid, query, cfg)
    if not lines:
        return ""
    return f"Graph context ({len(lines)} relations)\n" + "\n".join(lines)


def _build_lines(graph, seed: str, query: str,
                 cfg: GraphContextConfig) -> List[str]:
    nodes = getattr(graph, "nodes", None) or {}
    adj = getattr(graph, "adj", None) or {}
    adj_in = getattr(graph, "adj_in", None) or {}
    edges = getattr(graph, "edges", None)  # may be None -> no snippets
    if seed not in nodes:
        return []

    cap = cfg.char_cap()
    lines: List[str] = []
    used: set = set()   # (kind, direction, neighbour) already rendered

    def add(kind: str, direction: str, neighbour: str, label: str,
            snippet: str) -> bool:
        key = (kind, direction, neighbour)
        if neighbour == seed or not _allowed(kind, cfg) or not _direction_ok(cfg, direction):
            return False
        if key in used:
            return False
        line = _line(kind, direction, neighbour, label, snippet)
        if cap and sum(len(l) for l in lines) + len(line) + 2 > cap:
            return False
        lines.append(line)
        used.add(key)
        return True

    # ---- 1-hop outgoing ----------------------------------------------------
    if cfg.include_outgoing:
        cands = []
        for dst, kind in adj.get(seed, []):
            if dst == seed:
                continue
            label = node_label(nodes, dst)
            snip = _snippet_of(edges, seed, dst, kind) if cfg.include_snippets else ""
            score = _qscore(query, label + " " + snip) if cfg.query_aware else 0
            cands.append((-score, dst, kind, label, snip))
        cands.sort(key=lambda r: (r[0], r[1]))
        for _score, dst, kind, label, snip in cands:
            if not add(kind, "out", dst, label, snip):
                continue
            if len(lines) >= cfg.max_edges:
                break

    # ---- 1-hop incoming ----------------------------------------------------
    if cfg.include_incoming:
        cands = []
        for src, kind in adj_in.get(seed, []):
            if src == seed:
                continue
            label = node_label(nodes, src)
            snip = _snippet_of(edges, src, seed, kind) if cfg.include_snippets else ""
            score = _qscore(query, label + " " + snip) if cfg.query_aware else 0
            cands.append((-score, src, kind, label, snip))
        cands.sort(key=lambda r: (r[0], r[1]))
        for _score, src, kind, label, snip in cands:
            if not add(kind, "in", src, label, snip):
                continue
            if len(lines) >= cfg.max_edges:
                break

    # ---- 2-hop paths (candidate -> mid -> end) -----------------------------
    # The *end* must be a *new* node (not the seed, not a 1-hop neighbour,
    # not an earlier path end); the *mid* is the intermediate hop and is the
    # typical 1-hop neighbour, so we do NOT exclude it here.  Cycles terminate
    # because a node can only appear once as a path end.
    if cfg.include_paths:
        ends = {seed}                     # nodes already used as a path end
        # 1-hop neighbours already rendered (out/in) -- used as a dedup set
        hop1 = {x[2] for x in used if isinstance(x, tuple)}
        cands = []
        for mid, _kind in adj.get(seed, []):
            if mid == seed:
                continue
            for end, kind in adj.get(mid, []):
                if end in ends or end in hop1:
                    continue
                label = f"{node_label(nodes, mid)} → {node_label(nodes, end)}"
                snip = _snippet_of(edges, mid, end, kind) if cfg.include_snippets else ""
                score = _qscore(query, label + " " + snip) if cfg.query_aware else 0
                cands.append((-score, mid, end, label, snip))
        cands.sort(key=lambda r: (r[0], r[1]))
        n_via = sum(1 for l in lines if l.startswith("VIA "))
        for _nscore, mid, end, label, snip in cands:
            if n_via >= cfg.max_paths:
                break
            line = f"VIA {label}" + (f" — \"{snip}\"" if snip else "")
            if cap and sum(len(l) for l in lines) + len(line) + 2 > cap:
                break
            if end in ends:
                continue
            lines.append(line)
            ends.add(end)
            n_via += 1

    return lines
