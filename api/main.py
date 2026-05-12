"""Enterprise RAG API — FastAPI application (Mac Mini M4 Optimized).

Architecture:
  - Vector search: Qdrant + BGE-M3 embeddings (1024 dim, scalar quantization)
  - Graph search: Neo4j entity extraction + relationships
  - Fusion: RRF (Reciprocal Rank Fusion, k=60)
  - LLM: Ollama (Qwen3.5-4B Q4_K_M GGUF, Metal-accelerated)
  - Cache: Redis semantic cache (embedding-keyed, TTL 2h)
  - No Langfuse: saves ~200MB RAM

RAM Budget (Mac Mini M4 24GB):
  Ollama (Metal):     ~6-8GB (Qwen3.5-4B Q4_K_M + BGE-M3 loaded)
  Neo4j:               ~1GB
  Qdrant:             ~0.5GB
  Redis:              ~0.2GB
  rag-api container:  ~0.8GB
  rag-dashboard:      ~0.5GB
  System overhead:    ~5GB
  --------------------------------
  Total:              ~14-16GB (leaves headroom)
"""
import json
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse, Response
from loguru import logger

from src.config import get_settings
from src.clients import get_clients, init_clients
from src.models import (
    ChatCompletionRequest,
    ChatMessage,
    HealthResponse,
    ServiceCheck,
    IngestResponse,
    ModelList,
)
from src.services.retrieval import hybrid_retrieve


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    logger.info("Starting RAG API (Mac Mini M4 Optimized)...")
    logger.info(f"  Ollama:     {settings.ollama_base_url} / {settings.ollama_model}")
    logger.info(f"  Embedding:  {settings.ollama_embed_url} / {settings.ollama_embed_model}")
    logger.info(f"  Qdrant:     {settings.qdrant_url}")
    logger.info(f"  Neo4j:      {settings.neo4j_url}")
    logger.info(f"  Cache:      {settings.enable_semantic_cache} (TTL={settings.semantic_cache_ttl_s}s)")

    clients = await init_clients()
    logger.info("RAG API ready")
    yield

    logger.info("Shutting down RAG API...")
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


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise RAG API",
    version="2.0.0",
    description="Hybrid GraphRAG: Vector (Qdrant + BGE-M3) + Graph (Neo4j) + RRF Fusion",
    lifespan=lifespan,
)

# Multi-tenant API (tenants, sources, documents, RAG, stats)
from api.multi_tenant import router as multi_tenant_router
app.include_router(multi_tenant_router, prefix="/api/v2")

# Prometheus metrics
from src.metrics import MetricsMiddleware, get_metrics
app.add_middleware(MetricsMiddleware)


@app.get("/metrics")
async def metrics():
    """Prometheus-format metrics endpoint."""
    return Response(
        content=get_metrics().prometheus_output(),
        media_type="text/plain",
    )


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "version": "2.0.0"})


@app.get("/health/deep", response_model=HealthResponse)
async def health_deep():
    clients = get_clients()
    checks: dict[str, ServiceCheck] = {}

    # Ollama
    try:
        models = await clients.llm.models.list()
        checks["ollama"] = ServiceCheck(
            status="ok",
            models=[m.id for m in models.data],
        )
    except Exception as e:
        checks["ollama"] = ServiceCheck(status="fail", detail=str(e)[:100])

    # Qdrant
    try:
        cols = await clients.qdrant.get_collections()
        checks["qdrant"] = ServiceCheck(
            status="ok",
            collections=len(cols.collections),
        )
    except Exception as e:
        checks["qdrant"] = ServiceCheck(status="fail", detail=str(e)[:100])

    # Neo4j
    try:
        async with clients.neo4j.session() as s:
            result = await s.run("RETURN 1 AS x")
            await result.single()
        checks["neo4j"] = ServiceCheck(status="ok")
    except Exception as e:
        checks["neo4j"] = ServiceCheck(status="fail", detail=str(e)[:100])

    # Redis
    try:
        await clients.redis.ping()
        checks["redis"] = ServiceCheck(status="ok")
    except Exception as e:
        checks["redis"] = ServiceCheck(status="fail", detail=str(e)[:100])

    overall = "ok" if all(c.status == "ok" for c in checks.values()) else "degraded"
    return HealthResponse(status=overall, checks=checks)


# ── Chat Completions ──────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible RAG-augmented chat endpoint.

    Pipeline:
      1. Extract last user message
      2. Semantic cache lookup (Redis, keyed by query embedding)
      3. Cache miss → hybrid retrieval (vector + graph, concurrent)
      4. RRF fusion
      5. Generate response with context
      6. Cache result
    """
    settings = get_settings()
    clients = get_clients()

    async with clients.concurrent_semaphore:
        last_msg = next(
            (m for m in reversed(req.messages) if m.role == "user"),
            None,
        )
        if not last_msg:
            raise HTTPException(400, "At least one user message is required")

        query = last_msg.content.strip()

        # ── Step 1: Embed query (for cache key) ───────────────────────────
        embed_start = time.monotonic()
        try:
            from src.services.embedding import embed_single
            query_vec = await embed_single(
                clients.http,
                settings.ollama_embed_url,
                settings.ollama_embed_model,
                query,
                timeout=60.0,
            )
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            query_vec = None

        # ── Step 2: Semantic cache check ──────────────────────────────────
        cached = None
        if query_vec and settings.enable_semantic_cache:
            cached = await clients.cache.get(query_vec, settings.retrieval_top_k)

        if cached:
            logger.info(f"Cache HIT for query (len={len(query)})")
            docs = cached
        else:
            # ── Step 3: Hybrid retrieval ──────────────────────────────────
            docs = await hybrid_retrieve(
                query,
                clients,
                top_k=settings.retrieval_top_k,
                vector_top_k=settings.retrieval_vector_top_k,
                graph_top_k=settings.retrieval_graph_top_k,
            )

            # ── Step 4: Cache the result ─────────────────────────────────
            if query_vec and settings.enable_semantic_cache and docs:
                await clients.cache.set(
                    query_vec, settings.retrieval_top_k, docs
                )

        # ── Step 5: Build context ────────────────────────────────────────
        context = _format_context(docs)

        rag_system = (
            "Bạn là trợ lý AI nội bộ của doanh nghiệp. "
            "Trả lời dựa trên context được cung cấp. "
            "Nếu không có thông tin, nói rõ thay vì bịa đặt.\n\n"
            f"Context:\n{context}"
        )

        messages = [
            {"role": "system", "content": rag_system},
            *[{"role": m.role, "content": m.content} for m in req.messages],
        ]

        # ── Step 6: Generate ─────────────────────────────────────────────
        if req.stream:
            return StreamingResponse(
                _stream_chat(messages, req, settings),
                media_type="text/event-stream",
            )
        else:
            response = await clients.llm.chat.completions.create(
                model=settings.ollama_model,
                messages=messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
            return response


async def _stream_chat(
    messages: list[dict],
    req: ChatCompletionRequest,
    settings,
) -> AsyncIterator[str]:
    clients = get_clients()
    try:
        stream = await clients.llm.chat.completions.create(
            model=settings.ollama_model,
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


def _format_context(docs: list[dict]) -> str:
    if not docs:
        return "(Khong co context phu hop — tra loi dua tren kien thuc cua ban)"
    blocks = []
    for i, d in enumerate(docs, 1):
        source = d.get("source", "unknown")
        modes = d.get("retrieval_modes", [])
        tag = f"[{' + '.join(modes)}]" if modes else ""
        blocks.append(f"[Doc {i} — nguon: {source}] {tag}\n{d.get('text', '')}\n")
    return "\n".join(blocks)


# ── Ingestion ────────────────────────────────────────────────────────────────

@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(file: UploadFile = File(...)):
    """Upload and index a document. Pipeline: parse → chunk → embed + KG → upsert."""
    settings = get_settings()

    if file.size and file.size > 200 * 1024 * 1024:
        raise HTTPException(413, "File exceeds 200MB limit")

    clients = get_clients()
    content = await file.read()

    if len(content) < 100:
        raise HTTPException(422, "File too small or empty")

    result = await ingest_document(
        file_content=content,
        filename=file.filename or "unknown",
        clients=clients,
    )
    return JSONResponse(result)


# ── Models ──────────────────────────────────────────────────────────────────

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    clients = get_clients()
    try:
        models = await clients.llm.models.list()
        return ModelList(data=[{"id": m.id} for m in models.data])
    except Exception as e:
        raise HTTPException(503, f"Failed to list models: {e}")


# ── JSON Metrics ───────────────────────────────────────────────────────────────

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


# ── Cache Management ────────────────────────────────────────────────────────

@app.post("/cache/clear")
async def clear_cache():
    clients = get_clients()
    await clients.cache.clear()
    return {"status": "ok", "message": "Semantic cache cleared"}


# Import here to avoid circular reference
from src.services.ingestion import ingest_document
