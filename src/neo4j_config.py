"""Neo4j + GraphRAG configuration.

Reads NEO4J_* / GRAPHRAAG_* env vars (from `.env` or the environment)
and builds the `neo4j-graphrag` LLM / embedding clients.

Ollama is reached through its OpenAI-compatible endpoint
(`http://localhost:11434/v1`). We therefore use graphrag's OpenAI
client classes pointed at that base URL (the `[ollama]` extra would
downgrade `ollama` which the rest of the repo pins to 0.6.2).
Ollama's /v1 endpoint expects an Authorization header; any non-empty
key satisfies it, hence `NEO4J_GRAPHRAAG_API_KEY` (default "ollama").
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


def _load_dotenv_best_effort() -> None:
    """Load `.env` from repo root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    try:
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        load_dotenv(root / ".env")
    except Exception:
        pass


_load_dotenv_best_effort()


@dataclass
class Neo4jSettings:
    uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687")
    )
    username: str = field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    database: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))

    @property
    def is_configured(self) -> bool:
        return bool(self.password)

    def auth(self) -> tuple:
        return (self.username, self.password)


@dataclass
class GraphRAGSettings:
    ollama_openai_base_url: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_OPENAI_BASE_URL",
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1",
        )
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("NEO4J_GRAPHRAAG_API_KEY", "ollama")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv(
            "GRAPHRAAG_LLM_MODEL",
            os.getenv("ENERGY_AUDIT_REASONER", "qwen3.8:27b"),
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("GRAPHRAAG_EMBEDDING_MODEL", "bge-m3")
    )
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("GRAPHRAAG_EMBEDDING_DIM", "1024"))
    )


def neo4j_settings() -> Neo4jSettings:
    return Neo4jSettings()


def graphrag_settings() -> GraphRAGSettings:
    return GraphRAGSettings()


def make_llm(settings: Optional[GraphRAGSettings] = None, **kwargs):
    """graphrag OpenAI LLM client pointed at Ollama's /v1 endpoint."""
    from neo4j_graphrag.llm import OpenAILLM

    s = settings or graphrag_settings()
    params = kwargs.pop("model_params", {"temperature": 0.1})
    return OpenAILLM(
        model_name=s.llm_model,
        base_url=s.ollama_openai_base_url,
        api_key=s.api_key,
        model_params=params,
        **kwargs,
    )


def make_embeddings(settings: Optional[GraphRAGSettings] = None, **kwargs):
    """graphrag OpenAI embeddings client pointed at Ollama's /v1 endpoint."""
    from neo4j_graphrag.embeddings import OpenAIEmbeddings

    s = settings or graphrag_settings()
    return OpenAIEmbeddings(
        model=s.embedding_model,
        base_url=s.ollama_openai_base_url,
        api_key=s.api_key,
        **kwargs,
    )


def make_driver(settings: Optional[Neo4jSettings] = None):
    from neo4j import GraphDatabase

    s = settings or neo4j_settings()
    if not s.is_configured:
        raise RuntimeError(
            "NEO4J_PASSWORD is not set. Add it to .env "
            "(see .env.example) or export it in the shell."
        )
    return GraphDatabase.driver(s.uri, auth=s.auth())


def env_summary() -> Dict[str, str]:
    """Redacted config overview (never include the password)."""
    n, g = neo4j_settings(), graphrag_settings()
    return {
        "NEO4J_URI": n.uri,
        "NEO4J_USERNAME": n.username,
        "NEO4J_PASSWORD": "***set***" if n.password else "<missing>",
        "NEO4J_DATABASE": n.database,
        "OLLAMA_OPENAI_BASE_URL": g.ollama_openai_base_url,
        "GRAPHRAAG_LLM_MODEL": g.llm_model,
        "GRAPHRAAG_EMBEDDING_MODEL": g.embedding_model,
        "GRAPHRAAG_EMBEDDING_DIM": str(g.embedding_dim),
    }
