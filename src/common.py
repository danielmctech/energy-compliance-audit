"""
Shared helpers used by every pipeline notebook.

Covers the bits that all the notebooks need but shouldn't each carry:
figuring out where the repo is, the canonical corpus paths, finding the
PDFs and processed markdown, an Ollama chat wrapper with retry and a small
per-model response cache, and pulling JSON out of model output.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None


# ---------------------------------------------------------------------------
# repo root resolution
# ---------------------------------------------------------------------------

def repo_root(start: Optional[Path] = None) -> Path:
    """
    find the energy-audit repo root from anywhere inside it (or from CWD).

    strategy: start at `start` (default CWD), walk up; the root is the first
    directory containing both `notebooks/` and `data/`. override with
    ENERGY_AUDIT_ROOT.
    """
    env = os.getenv("ENERGY_AUDIT_ROOT")
    if env:
        p = Path(env).resolve()
        if p.exists():
            return p

    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "notebooks").is_dir() and (candidate / "data").is_dir():
            return candidate
    # last resort: parent of this file's directory (src/ -> repo root)
    return Path(__file__).resolve().parent.parent


ROOT = repo_root()


def rel_to_root(p: Path) -> str:
    """path as a repo-relative string when possible (portable artifacts)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def resolve_repo_path(p) -> Path:
    """resolve a possibly relative or stale absolute path against the repo."""
    p = Path(p)
    if p.is_absolute() and p.exists():
        return p
    return ROOT / p


# ---------------------------------------------------------------------------
# canonical paths
# ---------------------------------------------------------------------------

RAW_BASE = ROOT / "data" / "raw" / "eu"
PROCESSED_BASE = ROOT / "data" / "processed" / "eu"

NOTEBOOKS_DATA = ROOT / "notebooks" / "data"
GROUND_TRUTH_PATH = NOTEBOOKS_DATA / "ground_truth"
EVALUATIONS_PATH = NOTEBOOKS_DATA / "evaluations"
CACHE_PATH = NOTEBOOKS_DATA / "llm_cache"
METADATA_PATH = ROOT / "outputs" / "corpus_metadata.json"

# document folders under {lang}/  (network_codes is a sibling, not nested)
DOC_TYPE_FOLDERS = ["regulations", "directives", "guidance", "network_codes"]

# fallback: legacy layout without the language level (kept for safety)
LEGACY_RAW_BASE = ROOT / "data" / "raw" / "eu"


def ensure_dirs() -> None:
    for p in (
        GROUND_TRUTH_PATH,
        EVALUATIONS_PATH,
        CACHE_PATH,
    ):
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# corpus discovery
# ---------------------------------------------------------------------------

def discover_language_dirs() -> List[Path]:
    """
    return language dirs under data/raw/eu (e.g. [en]). if the tree uses the
    legacy layout (regulations/ directly under eu/), return [] and discovery
    falls back to the legacy base.
    """
    if not RAW_BASE.exists():
        return []
    langs = []
    for child in sorted(RAW_BASE.iterdir()):
        if child.is_dir() and any((child / t).is_dir() for t in DOC_TYPE_FOLDERS):
            langs.append(child)
    return langs


def raw_base_for(doc_type: str) -> Path:
    """resolve the raw directory for a doc type, lang-aware."""
    langs = discover_language_dirs()
    if langs:
        # prefer the primary language (first found / en)
        primary = next((l for l in langs if l.name == "en"), langs[0])
        candidate = primary / doc_type
        if candidate.is_dir():
            return candidate
        # network codes may live under regulations/network_codes in legacy trees
        legacy = LEGACY_RAW_BASE / doc_type
        if legacy.is_dir():
            return legacy
    return LEGACY_RAW_BASE / doc_type


def discover_pdf(doc_type: str = None) -> List[Path]:
    """all raw pdfs in the corpus (single language level)."""
    if doc_type:
        base = raw_base_for(doc_type)
        if base.is_dir():
            return sorted(
                [p for p in base.glob("*.pdf") if p.suffix.lower() == ".pdf"]
            )
        return []

    out: List[Path] = []
    langs = discover_language_dirs()
    if langs:
        primary = next((l for l in langs if l.name == "en"), langs[0])
        for t in DOC_TYPE_FOLDERS:
            d = primary / t
            if d.is_dir():
                out.extend(sorted(p for p in d.glob("*.pdf") if p.suffix.lower() == ".pdf"))
    else:
        for t in DOC_TYPE_FOLDERS:
            d = LEGACY_RAW_BASE / t
            if d.is_dir():
                out.extend(sorted(p for p in d.glob("*.pdf") if p.suffix.lower() == ".pdf"))
            legacy = LEGACY_RAW_BASE / "regulations" / t
            if legacy.is_dir() and t == "network_codes":
                out.extend(sorted(p for p in legacy.glob("*.pdf") if p.suffix.lower() == ".pdf"))
    # dedupe, stable order
    seen, unique = set(), []
    for p in out:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            unique.append(p)
    return unique


def doc_type_for(path: Path) -> str:
    """infer document type from folder structure."""
    try:
        parts = [p.lower() for p in path.resolve().parts]
    except Exception:
        return "unknown"
    if "network_codes" in parts:
        return "network_codes"
    for t in ("regulations", "directives", "guidance"):
        if t in parts:
            return t
    return "unknown"


def discover_markdown() -> List[Path]:
    """all processed markdown files (the 02/03a input)."""
    out: List[Path] = []
    if PROCESSED_BASE.exists():
        out = sorted(p for p in PROCESSED_BASE.rglob("*.md") if p.suffix == ".md")
    return out


def md_for_pdf(pdf_path: Path) -> Path:
    """expected markdown output location for a given raw pdf."""
    dtype = doc_type_for(pdf_path)
    return PROCESSED_BASE / dtype / (pdf_path.stem + ".md")


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def env_models() -> Dict[str, str]:
    return {
        "reasoner_default": os.getenv("ENERGY_AUDIT_REASONER", "qwen3.8:27b"),
        "reasoner_baseline": os.getenv("ENERGY_AUDIT_REASONER_BASELINE", "deepseek-r1:32b"),
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    }


# ---------------------------------------------------------------------------
# ollama chat (retry, budgets, cache)
# ---------------------------------------------------------------------------

def ollama_chat(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.0,
    num_ctx: int = 32768,
    num_predict: int = 16384,
    max_retries: int = 3,
    extra_options: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
    cache_key: Optional[str] = None,
) -> str:
    """
    chat with an ollama model. returns the raw message content.
    caches by (model, cache_key or prompt-hash) under CACHE_PATH.
    """
    if ollama is None:
        raise RuntimeError("ollama package not installed")

    ck = cache_key or hashlib.sha1(prompt.encode("utf-8")).hexdigest()
    cache_file = CACHE_PATH / f"{model.replace(':', '_')}_{ck}.json"

    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return cached["content"]
        except Exception:
            pass  # corrupt cache entry -> recompute

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    options: Dict[str, Any] = {
        "temperature": temperature,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
    }
    if extra_options:
        options.update(extra_options)

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = ollama.chat(model=model, messages=messages, options=options)
            content = resp["message"]["content"]
            if use_cache:
                ensure_dirs()
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "model": model,
                            "cache_key": ck,
                            "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest(),
                            "content": content,
                            "stats": resp.get("stats", {}),
                        },
                        f,
                        ensure_ascii=False,
                    )
            return content
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 5 * (attempt + 1)
            time.sleep(wait)
    raise RuntimeError(f"ollama.chat failed for {model} after {max_retries} tries: {last_err}")


# ---------------------------------------------------------------------------
# json extraction from llm output (robust against fences/preamble)
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """
    pull a json object/array out of llm text.
    handles: clean json, single fenced block, preamble+json, reasoning+json.
    """
    if text is None:
        raise ValueError("empty llm response")
    t = text.strip()

    candidates: List[str] = []
    blocks = _JSON_BLOCK.findall(t)
    candidates.extend(b.strip() for b in blocks)
    candidates.append(t)

    # also try the outermost balanced braces/brackets scan
    for opener, closer in (("{", "}"), ("[", "]")):
        i = t.find(opener)
        if i != -1:
            depth = 0
            for j in range(i, len(t)):
                if t[j] == opener:
                    depth += 1
                elif t[j] == closer:
                    depth -= 1
                    if depth == 0:
                        candidates.append(t[i : j + 1])
                        break

    last_err: Optional[Exception] = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, (dict, list)):
                return obj
        except json.JSONDecodeError as e:
            last_err = e
    raise ValueError(f"could not parse json from llm output: {last_err}")


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()
