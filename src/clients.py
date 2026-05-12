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

    llm: AsyncOpenAI
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

    return clients
