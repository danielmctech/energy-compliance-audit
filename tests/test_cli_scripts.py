# -*- coding: utf-8 -*-
"""Smoke tests for the P1/P2 CLI scripts (offline paths only)."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, *args], cwd=ROOT, capture_output=True, text=True, timeout=300
    )


def test_ingest_dry_run_offline():
    """`ingest_neo4j.py --dry-run` loads the corpus and audits it without
    touching Neo4j or Ollama."""
    r = _run("scripts/ingest_neo4j.py", "--dry-run")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "nodes:" in r.stdout and "edges:" in r.stdout
    assert "unique lineage_ids:" in r.stdout
    assert "dangling edges" in r.stdout


def test_ingest_dry_run_unique_ids_and_no_dangling():
    r = _run("scripts/ingest_neo4j.py", "--dry-run")
    assert r.returncode == 0
    # dry-run invariants: every lineage_id unique, zero dangling edges
    for line in r.stdout.splitlines():
        if "unique lineage_ids:" in line:
            uniq, total = line.split("unique lineage_ids:")[1].split("/")
            assert int(uniq.strip()) == int(total.strip())
        if "dangling edges" in line:
            assert int(line.split(":", 1)[1].strip()) == 0


def test_graphrag_n4j_package_imports_cleanly():
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "import graphrag_n4j as n4j;"
        "print(sorted(n4j.KNOWN_EDGE_KINDS));"
        "print(n4j.EMBEDDABLE_KINDS)"
    )
    r = subprocess.run([PY, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    out = r.stdout.strip().splitlines()
    assert ast.literal_eval(out[0]) == [
        "AMENDS", "APPLIES_TO", "CROSS_REFERENCES", "DEFINED_IN"
    ]
