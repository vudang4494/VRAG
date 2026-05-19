"""Test configuration and shared fixtures."""

import asyncio
import os

import pytest

os.environ.setdefault("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        os.environ.setdefault("OLLAMA_MODEL", "qwen3.5:9b")
os.environ.setdefault("OLLAMA_EMBED_MODEL", "bge-m3")
os.environ.setdefault("OLLAMA_EMBED_URL", "http://host.docker.internal:11434")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_COLLECTION", "enterprise_kb")
os.environ.setdefault("NEO4J_URL", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault(
    "POSTGRES_URL", "postgresql://raguser:46b0a29fb12290c116e2cf8995334e07c1c1@localhost:5432/ragdb"
)
os.environ.setdefault("LANGFUSE_HOST", "http://localhost:3000")
os.environ.setdefault("MAX_CONCURRENT_REQUESTS", "8")
os.environ.setdefault("REQUEST_TIMEOUT_S", "120")
os.environ.setdefault("SEMANTIC_CACHE_TTL_S", "3600")
os.environ.setdefault("ENABLE_SEMANTIC_CACHE", "false")
os.environ.setdefault("RETRIEVAL_TOP_K", "8")
os.environ.setdefault("RETRIEVAL_VECTOR_TOP_K", "20")
os.environ.setdefault("RETRIEVAL_GRAPH_TOP_K", "20")
os.environ.setdefault("RRF_K", "60")


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
