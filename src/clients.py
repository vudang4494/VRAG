"""Global client holders — initialized once at app startup (Mac Mini M4 Optimized)."""

import asyncio
import hashlib
from typing import TYPE_CHECKING, Any

import httpx
import redis.asyncio as redis
from loguru import logger
from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

if TYPE_CHECKING:
    pass


class SemanticCache:
    """
    Redis-backed semantic query cache using embedding similarity.

    Improvements for Mac Mini:
    - SHA-256 first 32 dims (same as before, fast hash)
    - TTL configurable via env (default 7200s = 2h, up from 1h)
    - Lazy eviction — Redis allkeys-lru handles overflow automatically
    """

    def __init__(self, redis_client: redis.Redis, ttl: int = 7200):
        self.redis = redis_client
        self.ttl = ttl

    @staticmethod
    def _cache_key(embedding: list[float], top_k: int) -> str:
        # Truncate to 32 dims for fast hashing (embedding space is continuous anyway)
        vec_hash = hashlib.sha256(
            ",".join(f"{v:.4f}" for v in embedding[:32]).encode()
        ).hexdigest()[:16]
        return f"rag:cache:{vec_hash}:{top_k}"

    async def get(self, embedding: list[float], top_k: int) -> list[dict] | None:
        key = self._cache_key(embedding, top_k)
        try:
            data = await self.redis.get(key)
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
            return None
        if not data:
            return None
        try:
            import json

            return json.loads(data)
        except Exception as e:
            logger.debug(f"Cache get: JSON decode failed: {e}")
            return None

    async def set(self, embedding: list[float], top_k: int, results: list[dict]) -> None:
        key = self._cache_key(embedding, top_k)
        try:
            import json

            await self.redis.setex(key, self.ttl, json.dumps(results))
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")

    async def clear(self) -> None:
        try:
            keys = []
            async for key in self.redis.scan_iter(match="rag:cache:*"):
                keys.append(key)
            if keys:
                await self.redis.delete(*keys)
        except Exception as e:
            logger.debug(f"Cache clear failed: {e}")


class Clients:
    """Holds all external-service client instances."""

    llm: AsyncOpenAI  # semantic LLM — generation, chat, validation
    entity_extractor: Any  # separate entity LLM/NER — GLiNER / OpenAI / etc.
    qdrant: AsyncQdrantClient
    neo4j: Any
    redis: redis.Redis
    cache: SemanticCache
    http: httpx.AsyncClient
    concurrent_semaphore: asyncio.Semaphore


_clients: Clients | None = None


def get_clients() -> Clients:
    global _clients
    if _clients is None:
        _clients = Clients()
    return _clients


async def init_clients() -> Clients:
    """Initialize all clients. Called from lifespan in main.py."""
    from src.config import get_settings

    settings = get_settings()

    clients = get_clients()

    # LLM — Ollama (Metal GPU on Mac Mini M4)
    # Connection pool tuned for async workloads
    clients.llm = AsyncOpenAI(
        base_url=f"{settings.ollama_base_url}/v1",
        api_key="ollama",
        timeout=settings.request_timeout_s,
        max_retries=2,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=16,  # reduced from 20 (M4 optimized)
                max_keepalive_connections=8,  # reduced from 10
            ),
            timeout=httpx.Timeout(settings.request_timeout_s, connect=10.0),
        ),
    )

    # Monkey-patch chat.completions.create to inject Ollama-specific options:
    #   • keep_alive=-1 → model stays in VRAM (avoid 5-min expiry cold reload)
    #   • think=False   → disable Qwen3-style thinking mode (content was empty
    #                     because all tokens consumed by hidden thinking).
    _original_create = clients.llm.chat.completions.create

    async def _create_with_ollama_opts(**kwargs):
        extra_body = kwargs.pop("extra_body", None) or {}
        extra_body.setdefault("keep_alive", -1)
        extra_body.setdefault("think", False)
        return await _original_create(extra_body=extra_body, **kwargs)

    clients.llm.chat.completions.create = _create_with_ollama_opts  # type: ignore[method-assign]

    # Qdrant — async client, minimal timeout
    clients.qdrant = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=30.0,
    )

    # Neo4j — connection pool tuned for small deployment
    clients.neo4j = AsyncGraphDatabase.driver(
        settings.neo4j_url,
        auth=(settings.neo4j_user, settings.neo4j_password),
        max_connection_pool_size=10,  # reduced from 20 (M4 has limited RAM)
        connection_timeout=15.0,
        encrypted=False,
    )

    # Redis — semantic cache
    clients.redis = redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=False,
        max_connections=10,  # reduced from 20
    )
    clients.cache = SemanticCache(
        clients.redis,
        ttl=settings.semantic_cache_ttl_s,
    )

    # Shared HTTP client for embedding calls
    clients.http = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=16,
            max_keepalive_connections=8,
        ),
        timeout=httpx.Timeout(120.0, connect=5.0),
    )

    # Concurrency limiter — M4 optimized
    clients.concurrent_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    # Entity extractor (separate from semantic LLM)
    # Pre-load at startup so memory is committed early and predictable.
    # If we lazy-load inside an ingest request, model load can OOM under
    # concurrent memory pressure from qdrant/neo4j/embedding clients.
    #
    # HOWEVER: eager-loading GLiNER (168M params ≈ 700MB RAM) consumes a chunk
    # of the 6GB container limit on every startup even when no ingest happens.
    # With M4 24GB host RAM, 6GB container + GLiNER + httpx buffers is tight.
    # FIXED: lazy-load — model is loaded on first extract() call, not at startup.
    # The first ingest call pays the load cost; subsequent calls reuse the cached model.
    clients.entity_extractor = None
    try:
        from src.services.entity_extractor import create_entity_extractor

        clients.entity_extractor = create_entity_extractor(
            provider=settings.entity_extractor_provider,
            model=settings.entity_extractor_model,
            threshold=settings.entity_extractor_threshold,
            llm_for_relations=clients.llm if settings.entity_relations_enabled else None,
            relation_model=settings.ollama_model,
            extract_relations=settings.entity_relations_enabled,
        )
        # Lazy-load: only load model into memory when first needed.
        # Don't call ner._load() here — let FastAPI handle requests immediately
        # without waiting for GLiNER to load first.
        # Model loads on first extract() call inside ingest_document_v2().
        logger.info("Entity extractor initialized (lazy-load mode — model loads on first use)")
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"Entity extractor init failed: {e}")
        clients.entity_extractor = None

    return clients
