"""Global client holders — initialized once at app startup (Mac Mini M4 Optimized)."""
import asyncio
import hashlib
from typing import TYPE_CHECKING, Any

import httpx
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from neo4j import AsyncGraphDatabase
import redis.asyncio as redis

if TYPE_CHECKING:
    from src.config import Settings


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
            if data:
                import json
                return json.loads(data)
        except Exception:
            pass
        return None

    async def set(self, embedding: list[float], top_k: int, results: list[dict]) -> None:
        key = self._cache_key(embedding, top_k)
        try:
            import json
            await self.redis.setex(key, self.ttl, json.dumps(results))
        except Exception:
            pass

    async def clear(self) -> None:
        try:
            keys = []
            async for key in self.redis.scan_iter(match="rag:cache:*"):
                keys.append(key)
            if keys:
                await self.redis.delete(*keys)
        except Exception:
            pass


class Clients:
    """Holds all external-service client instances."""

    llm: AsyncOpenAI                # semantic LLM — generation, chat, validation
    entity_extractor: Any           # separate entity LLM/NER — GLiNER / OpenAI / etc.
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
                max_connections=16,      # reduced from 20 (M4 optimized)
                max_keepalive_connections=8,  # reduced from 10
            ),
            timeout=httpx.Timeout(settings.request_timeout_s, connect=10.0),
        ),
    )

    # Monkey-patch chat.completions.create to inject keep_alive=-1 globally.
    # Ollama's default 5-minute model expiry causes ~50s cold-load latency
    # mid-pipeline. With keep_alive=-1 on every call, model stays in VRAM
    # for the lifetime of the API process.
    _original_create = clients.llm.chat.completions.create

    async def _create_keep_alive(**kwargs):
        extra_body = kwargs.pop("extra_body", None) or {}
        extra_body.setdefault("keep_alive", -1)
        return await _original_create(extra_body=extra_body, **kwargs)

    clients.llm.chat.completions.create = _create_keep_alive  # type: ignore[method-assign]

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
        # Force eager load of underlying model (GLiNER) to commit memory up front.
        try:
            ner = getattr(clients.entity_extractor, "ner", None)
            if ner is not None and hasattr(ner, "_load"):
                import asyncio as _aio
                await _aio.to_thread(ner._load)
        except Exception as _e:
            import logging as _lg
            _lg.getLogger(__name__).warning(f"Entity model eager-load skipped: {_e}")
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Entity extractor init failed: {e}")
        clients.entity_extractor = None

    return clients
