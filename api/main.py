"""VRAG API — Hybrid GraphRAG (Apple Silicon optimized).

Architecture:
  - Vector search: Qdrant + BGE-M3 embeddings (1024-dim, 5 named views + graph_aware)
  - Graph: Neo4j knowledge graph (entities, relations, communities, temporal)
  - Entity extraction: GLiNER (zero-shot, 168M params)
  - LLM: Ollama (Qwen3.5-9B, Metal-accelerated, thinking-mode bypassed)
  - Cache: Redis semantic cache (embedding-keyed, TTL configurable)

Endpoints live under /api/v3 (REST API contract version — see api/routes/):
  chat / chat/stream / chat/react / ingest/upload /
  gaea/refine / hefr/populate / hefr/retrieve / rerank/l2r/test /
  cross_doc/build / community/build / health / health/deep

Root-level endpoints kept: /health, /metrics (Prometheus), /api/metrics (JSON).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from loguru import logger

from src.clients import get_clients, init_clients
from src.config import get_settings
from src.models import HealthResponse, ServiceCheck


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Starting VRAG API...")
    logger.info(f"  Ollama:    {settings.ollama_base_url} / {settings.ollama_model}")
    logger.info(f"  Embedding: {settings.ollama_embed_url} / {settings.ollama_embed_model}")
    logger.info(f"  Qdrant:    {settings.qdrant_url}")
    logger.info(f"  Neo4j:     {settings.neo4j_url}")

    clients = await init_clients()
    logger.info("VRAG API ready")
    yield

    logger.info("Shutting down VRAG API...")
    if clients.http:
        await clients.http.aclose()
    if clients.llm and clients.llm._client:
        await clients.llm._client.aclose()
    if clients.qdrant:
        await clients.qdrant.close()
    if clients.neo4j:
        await clients.neo4j.close()
    if clients.redis:
        await clients.redis.aclose()


app = FastAPI(
    title="VRAG API",
    version="3.0.0",
    description="VRAG — Hybrid GraphRAG: multi-view vectors + GAEA + ReAct + L2R + HEFR",
    lifespan=lifespan,
)

from api.routes import router as v3_router  # noqa: E402

app.include_router(v3_router, prefix="/api/v3", tags=["vrag"])

from src.metrics import MetricsMiddleware, get_metrics  # noqa: E402

app.add_middleware(MetricsMiddleware)


@app.get("/metrics")
async def metrics():
    """Prometheus-format metrics endpoint."""
    return Response(
        content=get_metrics().prometheus_output(),
        media_type="text/plain",
    )


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "version": "3.0.0"})


@app.get("/health/deep", response_model=HealthResponse)
async def health_deep():
    clients = get_clients()
    checks: dict[str, ServiceCheck] = {}

    try:
        models = await clients.llm.models.list()
        checks["ollama"] = ServiceCheck(status="ok", models=[m.id for m in models.data])
    except Exception as e:
        checks["ollama"] = ServiceCheck(status="fail", detail=str(e)[:100])

    try:
        cols = await clients.qdrant.get_collections()
        checks["qdrant"] = ServiceCheck(status="ok", collections=len(cols.collections))
    except Exception as e:
        checks["qdrant"] = ServiceCheck(status="fail", detail=str(e)[:100])

    try:
        async with clients.neo4j.session() as s:
            result = await s.run("RETURN 1 AS x")
            await result.single()
        checks["neo4j"] = ServiceCheck(status="ok")
    except Exception as e:
        checks["neo4j"] = ServiceCheck(status="fail", detail=str(e)[:100])

    try:
        await clients.redis.ping()
        checks["redis"] = ServiceCheck(status="ok")
    except Exception as e:
        checks["redis"] = ServiceCheck(status="fail", detail=str(e)[:100])

    overall = "ok" if all(c.status == "ok" for c in checks.values()) else "degraded"
    return HealthResponse(status=overall, checks=checks)


@app.get("/api/metrics")
async def json_metrics():
    """JSON-format metrics endpoint."""
    clients = get_clients()
    try:
        cached = await clients.redis.dbsize()
    except Exception:
        cached = 0
    return {
        "total_requests": 0,
        "total_errors": 0,
        "cache_entries": cached,
        "cache_enabled": get_settings().enable_semantic_cache,
    }
