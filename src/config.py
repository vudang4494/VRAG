"""Configuration — all settings read from environment variables (Mac Mini M4 Optimized)."""
import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "qwen3.5:4b"

    # ── Embedding ───────────────────────────────────────────────────────────
    ollama_embed_model: str = "bge-m3"
    ollama_embed_url: str = "http://host.docker.internal:11434"
    embed_dimension: int = 1024
    # M4 optimized: batch 32 (larger than default 16 for throughput)
    embed_batch_size: int = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
    # M4 optimized: only 3 concurrent Ollama embedding calls (M4 can handle 3 streams)
    embed_concurrent_limit: int = int(os.environ.get("EMBED_CONCURRENT_LIMIT", "3"))

    # ── Vector DB ────────────────────────────────────────────────────────────
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "enterprise_kb"

    # ── Knowledge Graph ───────────────────────────────────────────────────────
    neo4j_url: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # ── Cache ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://redis:6379/0"
    # M4 optimized: 2h cache (longer TTL, less memory pressure)
    semantic_cache_ttl_s: int = int(os.environ.get("SEMANTIC_CACHE_TTL", "7200"))
    enable_semantic_cache: bool = os.environ.get("ENABLE_SEMANTIC_CACHE", "true").lower() != "false"

    # ── App ─────────────────────────────────────────────────────────────────
    app_env: str = "production"
    log_level: str = "INFO"
    # M4 optimized: 6 concurrent (M4 efficiency cores handle I/O well)
    max_concurrent_requests: int = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "6"))
    request_timeout_s: int = int(os.environ.get("REQUEST_TIMEOUT_S", "120"))

    # ── Retrieval tuning ────────────────────────────────────────────────────
    retrieval_top_k: int = 8
    retrieval_vector_top_k: int = 20
    retrieval_graph_top_k: int = 15  # reduced from 20 to save memory
    rrf_k: int = 60

    # ── Multi-tenancy ────────────────────────────────────────────────────────
    multi_tenant_enabled: bool = True
    api_internal_key: str = ""
    enforce_api_key: bool = False

    # ── Reranking ────────────────────────────────────────────────────────────
    enable_reranker: bool = True
    reranker_type: str = "semantic"  # semantic is fast, uses cosine similarity
    reranker_top_k: int = 10

    # ── Dashboard ───────────────────────────────────────────────────────────
    dashboard_port: int = 7860


@lru_cache
def get_settings() -> Settings:
    return Settings()
