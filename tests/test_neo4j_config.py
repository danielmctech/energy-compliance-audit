# -*- coding: utf-8 -*-
"""Unit tests for src/neo4j_config.py -- env-driven settings and client
factories. Client construction is lazy/offline (no socket opened)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_neo4j_settings_env_override(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://db:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "audit")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("NEO4J_DATABASE", "graphrag")
    import neo4j_config as NC
    import importlib
    importlib.reload(NC)

    s = NC.neo4j_settings()
    assert s.uri == "bolt://db:7687"
    assert s.username == "audit"
    assert s.is_configured is True
    assert s.auth() == ("audit", "secret")
    assert s.database == "graphrag"


def test_neo4j_settings_defaults(monkeypatch):
    import neo4j_config as NC
    import importlib
    importlib.reload(NC)
    # note: neo4j_config auto-loads .env on import/reload -> clear vars AFTER
    for var in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"):
        monkeypatch.delenv(var, raising=False)

    s = NC.neo4j_settings()
    assert s.uri == "bolt://localhost:7687"
    assert s.username == "neo4j"
    assert s.is_configured is False
    assert s.database == "neo4j"


def test_graphrag_settings_defaults(monkeypatch):
    for var in ("OLLAMA_OPENAI_BASE_URL", "OLLAMA_BASE_URL",
                "GRAPHRAAG_LLM_MODEL", "GRAPHRAAG_EMBEDDING_MODEL"):
        monkeypatch.delenv(var, raising=False)
    import neo4j_config as NC
    import importlib
    importlib.reload(NC)

    g = NC.graphrag_settings()
    assert g.ollama_openai_base_url.endswith("/v1")
    assert g.embedding_model == "bge-m3"
    assert g.embedding_dim == 1024
    assert g.api_key  # auth header present for Ollama /v1
    assert g.llm_model


def test_settings_are_dataclasses():
    import dataclasses
    import neo4j_config as NC

    assert dataclasses.is_dataclass(NC.Neo4jSettings)
    assert dataclasses.is_dataclass(NC.GraphRAGSettings)


def test_env_summary_contains_key_fields():
    import neo4j_config as NC

    summary = NC.env_summary()
    assert "NEO4J_URI" in summary or "uri" in summary


def test_make_llm_and_embeddings_construct(monkeypatch):
    import neo4j_config as NC

    llm = NC.make_llm()
    assert llm is not None

    emb = NC.make_embeddings()
    assert emb is not None


def test_make_driver_requires_password(monkeypatch):
    import neo4j_config as NC
    # clear AFTER import/reload so .env's auto-loader can't refill it
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    with pytest.raises(RuntimeError):
        NC.make_driver()


def test_env_summary_never_leaks_password(monkeypatch):
    import neo4j_config as NC

    monkeypatch.setenv("NEO4J_PASSWORD", "top-secret-value")
    summary = NC.env_summary()
    assert summary["NEO4J_PASSWORD"] == "***set***"
    assert "top-secret-value" not in str(summary)
