# -*- coding: utf-8 -*-
"""Unit tests for src/common.py -- repo-root resolution, path helpers,
env model config, and robust JSON extraction (all offline, no LLM)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_repo_root_from_repo():
    from common import ROOT, repo_root

    assert (ROOT / "notebooks").is_dir()
    assert (ROOT / "notebooks" / "data").is_dir()
    # resolves from anywhere inside the tree
    sub = ROOT / "src"
    assert repo_root(start=sub) == ROOT
    assert repo_root(start=ROOT) == ROOT


def test_rel_to_root_roundtrip():
    from common import ROOT, rel_to_root, resolve_repo_path

    rel = rel_to_root(ROOT / "src" / "common.py")
    assert rel == "src/common.py"
    # stale/relative path resolves back to the same absolute path
    assert resolve_repo_path(rel) == ROOT / "src" / "common.py"


def test_env_models_keys_and_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ("ENERGY_AUDIT_REASONER", "ENERGY_AUDIT_REASONER_BASELINE",
                "OLLAMA_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    from common import env_models

    m = env_models()
    assert set(m) == {"reasoner_default", "reasoner_baseline", "base_url"}
    assert m["base_url"].endswith("11434")


def _extract(text):
    from common import extract_json
    return extract_json(text)


def test_extract_json_clean():
    assert _extract('{"a": 1}') == {"a": 1}
    assert _extract("[1, 2, 3]") == [1, 2, 3]


def test_extract_json_fenced():
    assert _extract('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract('```\n[1, 2]\n```') == [1, 2]


def test_extract_json_preamble_and_reasoning():
    txt = ("some preamble text\n"
           '{"answer": "yes", "refs": ["Article 1"]}\n'
           "trailing commentary")
    assert _extract(txt) == {"answer": "yes", "refs": ["Article 1"]}


def test_extract_json_rejects_garbage():
    with pytest.raises(ValueError):
        _extract("no json at all")
    with pytest.raises(ValueError):
        _extract(None)


def test_stable_hash_deterministic():
    from common import stable_hash

    h1 = stable_hash("a", "b", 1)
    h2 = stable_hash("a", "b", 1)
    h3 = stable_hash("a", "b", 2)
    assert h1 == h2
    assert len(h1) == 40  # sha1 hex
    assert h1 != h3
