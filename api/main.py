"""VRAG API — Hybrid GraphRAG (Apple Silicon optimized).

Architecture:
  - Vector search: Qdrant + BGE-M3 embeddings (1024-dim, 6 named views:
    dense, paraphrase, question, summary, keywords, graph_aware — see vector.py)
  - Graph: Neo4j knowledge graph (entities, relations, communities, temporal)
  - Entity extraction: GLiNER (zero-shot, 168M params)
  - LLM: Ollama (default gemma4:e4b, Metal-accelerated, thinking-mode bypassed via
    ollama_chat(think=False) — the model does advertise a thinking capability and
    costs ~5.5x the tokens if it is left on; see src/config.py for the measurements)
  - Cache: Redis semantic cache (embedding-keyed, TTL configurable)

Startup prints the effective config and verifies every configured model tag exists
in Ollama (src/services/config_report.py) — do not trust this docstring over that
banner. Docs describe intent; the banner describes the process you are looking at.

Endpoints live under /api (REST contract — see api/routes/):
  chat / chat/stream / chat/react / ingest/upload /
  gaea/refine / hefr/populate / hefr/retrieve / rerank/l2r/test /
  cross_doc/build / community/build / health / health/deep

Root-level endpoints kept: /health, /metrics (Prometheus), /api/metrics (JSON).
"""

import time
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

    # Say out loud what this process actually resolved to, and whether the models it
    # intends to call exist. Every config layer (config.py / .env / compose / shell)
    # may legitimately override the one below it; none of them may do it silently.
    # Best-effort: a self-report must never be the thing that takes the API down.
    try:
        from src.services.config_report import log_startup_report

        await log_startup_report(clients, settings)
    except Exception as e:
        logger.warning(f"config self-report failed (non-fatal): {e}")

    # Self-heal the Qdrant collection: create it with the canonical schema if it
    # doesn't exist yet, so ingestion never fails with a 404 on a fresh volume.
    try:
        from src.services.vector import ensure_collection

        created = await ensure_collection(
            clients.qdrant, settings.qdrant_collection, settings.embed_dimension
        )
        logger.info(
            f"  Qdrant collection '{settings.qdrant_collection}': "
            f"{'created' if created else 'present'}"
        )
    except Exception as e:
        logger.warning(f"ensure_collection failed (ingest may 404 until init-qdrant.sh): {e}")

    # Self-heal the Neo4j schema (constraints + indexes). Without these, MERGE does
    # full label scans and graph retrieval is unindexed — same bootstrap gap as Qdrant.
    try:
        from src.services.kg import ensure_schema

        await ensure_schema(clients.neo4j)
    except Exception as e:
        logger.warning(f"ensure_schema failed (run `make init-neo4j` manually): {e}")

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
    version="1.0.0",
    description="VRAG — Hybrid GraphRAG: multi-view vectors + GAEA + ReAct + L2R + HEFR",
    lifespan=lifespan,
)

from api.routes import router as api_router  # noqa: E402

app.include_router(api_router, prefix="/api", tags=["rag"])

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
    return JSONResponse({"status": "ok", "version": "1.0.0"})


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
    """JSON-format metrics endpoint.

    total_requests/total_errors used to be the literals 0 and 0, so this endpoint
    reported an idle server while /metrics — fed by the same MetricsMiddleware, one
    screen up — reported hundreds of requests and real errors. Anything trusting the
    JSON view saw a system that had never been touched. Read the counters.
    """
    clients = get_clients()
    try:
        cached = await clients.redis.dbsize()
    except Exception:
        cached = 0

    m = get_metrics()
    return {
        "total_requests": sum(m._requests_total.values()),
        "total_errors": sum(m._requests_errors.values()),
        "active_requests": m._active_requests,
        "uptime_seconds": round(time.time() - m._start_time, 1),
        "cache_hits": m._cache_hits,
        "cache_misses": m._cache_misses,
        "chunks_indexed": m._chunks_indexed,
        "entities_extracted": m._entities_extracted,
        "cache_entries": cached,
        "cache_enabled": get_settings().enable_semantic_cache,
    }
