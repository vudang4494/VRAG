"""Multi-tenant API — FastAPI with auth, tenants, sources, documents."""
import asyncio
import hashlib
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Form, Header, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

from src.config import get_settings
from src.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    DocumentListResponse,
    IngestJobResponse,
    RetrievalFilters,
    RetrievalResult,
    RetrievalResponse,
    SourceCreate,
    SourceUpdate,
    SystemStats,
    Tenant,
    TenantCreate,
    TenantStats,
)


# =============================================================================
# Auth middleware
# =============================================================================

API_KEY_DB: dict[str, dict[str, Any]] = {}


async def verify_api_key(x_api_key: Annotated[str | None, Header()] = None) -> dict[str, Any]:
    """Validate API key and return tenant info. Requires API key in X-API-Key header."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
    if key_hash not in API_KEY_DB:
        raise HTTPException(status_code=401, detail="Invalid API key")
    key_info = API_KEY_DB[key_hash]
    if not key_info["is_active"]:
        raise HTTPException(status_code=401, detail="API key deactivated")
    if key_info.get("expires_at") and key_info["expires_at"] < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="API key expired")
    return key_info


async def verify_ingest_scope(ctx: Annotated[dict, Depends(verify_api_key)]) -> dict:
    if "ingest" not in ctx.get("scopes", []):
        raise HTTPException(status_code=403, detail="Insufficient scope: ingest required")
    return ctx


# =============================================================================
# Multi-tenant routers
# =============================================================================

router = APIRouter()


# --- Tenants ---

TENANT_DB: dict[str, dict[str, Any]] = {}


@router.get("/tenants", tags=["tenants"])
async def list_tenants():
    return [
        {"id": tid, "name": t["name"], "status": t["status"]}
        for tid, t in TENANT_DB.items()
    ]


@router.post("/tenants", tags=["tenants"])
async def create_tenant(body: TenantCreate) -> Tenant:
    tid = str(uuid.uuid4())
    tenant = {
        "id": tid,
        "slug": body.slug,
        "name": body.name,
        "description": body.description,
        "owner_email": body.owner_email,
        "plan": body.plan,
        "status": "active",
        "default_chunk_strategy": body.default_chunk_strategy.value,
        "default_chunk_size": body.default_chunk_size,
        "default_chunk_overlap": body.default_chunk_overlap,
        "retrieval_top_k": body.retrieval_top_k,
        "vector_weight": body.vector_weight,
        "graph_weight": body.graph_weight,
        "enable_semantic_cache": body.enable_semantic_cache,
        "semantic_cache_ttl_s": body.semantic_cache_ttl_s,
        "max_concurrent_requests": body.max_concurrent_requests,
        "custom_llm_model": body.custom_llm_model,
        "custom_embed_model": body.custom_embed_model,
        "metadata": body.metadata_,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "source_count": 0,
        "document_count": 0,
        "chunk_count": 0,
    }
    TENANT_DB[tid] = tenant
    api_key = uuid.uuid4().hex
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    API_KEY_DB[key_hash] = {
        "id": str(uuid.uuid4()),
        "tenant_id": tid,
        "name": f"{body.name} default key",
        "scopes": ["chat", "ingest", "read"],
        "rate_limit_per_minute": 60,
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
        "expires_at": None,
        "last_used_at": None,
    }
    return {**tenant, "api_key": api_key}


@router.get("/tenants/{tenant_id}", tags=["tenants"])
async def get_tenant(tenant_id: str):
    if tenant_id not in TENANT_DB:
        raise HTTPException(404, "Tenant not found")
    return TENANT_DB[tenant_id]


@router.delete("/tenants/{tenant_id}", tags=["tenants"])
async def delete_tenant(tenant_id: str):
    if tenant_id in TENANT_DB:
        del TENANT_DB[tenant_id]
    return {"status": "ok"}


# --- Sources ---

SOURCE_DB: dict[str, dict[str, Any]] = {}


@router.post("/sources", tags=["sources"])
async def create_source(body: SourceCreate, ctx: Annotated[dict, Depends(verify_ingest_scope)]):
    sid = str(uuid.uuid4())
    source = {
        "id": sid,
        "tenant_id": body.tenant_id,
        "name": body.name,
        "source_type": body.source_type.value,
        "description": body.description,
        "status": body.status.value,
        "access_level": body.access_level.value,
        "tags": body.tags,
        "schedule_cron": body.schedule_cron,
        "is_recurring": body.is_recurring,
        "crawl_depth": body.crawl_depth,
        "filters": body.filters,
        "custom_config": body.custom_config,
        "last_sync_at": None,
        "last_sync_status": None,
        "document_count": 0,
        "error_message": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    SOURCE_DB[sid] = source
    return source


@router.get("/sources", tags=["sources"])
async def list_sources(
    ctx: Annotated[dict, Depends(verify_api_key)],
    tenant_id: str = Query(...),
):
    return [s for s in SOURCE_DB.values() if s["tenant_id"] == tenant_id]


@router.patch("/sources/{source_id}", tags=["sources"])
async def update_source(source_id: str, body: SourceUpdate):
    if source_id not in SOURCE_DB:
        raise HTTPException(404, "Source not found")
    src = SOURCE_DB[source_id]
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(src, field, value)
    src["updated_at"] = datetime.now(timezone.utc)
    return src


@router.post("/sources/{source_id}/sync", tags=["sources"])
async def sync_source(source_id: str, ctx: Annotated[dict, Depends(verify_ingest_scope)]):
    if source_id not in SOURCE_DB:
        raise HTTPException(404, "Source not found")
    source = SOURCE_DB[source_id]

    from plugins.registry import registry
    from plugins.base import PluginConfig

    plugin = registry.create_source_plugin(
        name=source["source_type"],
        config=PluginConfig(
            raw=source["custom_config"],
            tenant_id=source["tenant_id"],
            source_id=source_id,
        ),
    )
    await plugin.initialize()
    try:
        result = await plugin.sync()
        source["last_sync_at"] = datetime.now(timezone.utc)
        source["last_sync_status"] = "success"
        source["document_count"] = len(result.documents)
        return {
            "status": "ok",
            "documents_crawled": len(result.documents),
            "errors": result.errors,
            "duration_s": result.duration_seconds,
        }
    finally:
        await plugin.close()


# --- Documents ---

@router.post("/documents", tags=["documents"])
async def ingest_document(
    file: UploadFile,
    tenant_id: Annotated[str, Form()],
    source_id: Annotated[str, Form()],
    title: Annotated[str, Form()],
    ctx: Annotated[dict, Depends(verify_ingest_scope)],
    access_level: Annotated[str, Form()] = "internal",
    tags: Annotated[str, Form()] = "",
    department: Annotated[str, Form()] = "",
    author: Annotated[str, Form()] = "",
):
    content = await file.read()
    if len(content) < 50:
        raise HTTPException(422, "File too small or empty")
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 50MB)")

    from plugins.registry import registry
    from plugins.base import PluginConfig, ParsedDocument

    plugin = registry.create_source_plugin(
        name="file",
        config=PluginConfig(
            raw={},
            tenant_id=tenant_id,
            source_id=source_id,
        ),
    )

    doc = await plugin.fetch(
        url_or_path=file.filename,
        content=content,
        filename=file.filename,
        metadata={
            "title": title,
            "access_level": access_level,
            "tags": tags.split(",") if tags else [],
            "department": department,
            "author": author,
        },
    )

    from src.storage.document_store import DocumentStore
    store = DocumentStore()
    result = await store.ingest_document(
        tenant_id=tenant_id,
        source_id=source_id,
        doc=doc,
    )

    return {
        "status": "success",
        "document_id": result["document_id"],
        "chunks_indexed": result["chunk_count"],
        "entities_extracted": result["entity_count"],
        "relationships_extracted": result["relationship_count"],
    }


@router.get("/documents", tags=["documents"])
async def list_documents(
    ctx: Annotated[dict, Depends(verify_api_key)],
    tenant_id: str = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    source_id: str | None = None,
    status: str | None = None,
):
    return DocumentListResponse(
        documents=[],
        total=0,
        page=page,
        page_size=page_size,
        has_more=False,
    )


@router.delete("/documents/{document_id}", tags=["documents"])
async def delete_document(
    document_id: str,
    ctx: Annotated[dict, Depends(verify_ingest_scope)],
    tenant_id: str = Query(...),
):
    from src.storage.document_store import DocumentStore
    store = DocumentStore()
    await store.delete_document(tenant_id=tenant_id, doc_id=document_id)
    return {"status": "ok"}


# --- RAG / Chat ---

@router.post("/v1/chat/completions", tags=["chat"])
async def chat_completions(
    body: ChatCompletionRequest,
    ctx: Annotated[dict, Depends(verify_api_key)],
):
    from src.services.retrieval import hybrid_retrieve
    from src.clients import get_clients
    from src.reranking.reranker import SemanticReranker, NoOpReranker

    clients = get_clients()
    tenant_id = ctx["tenant_id"]
    start = time.monotonic()

    filters = body.filters.model_dump() if body.filters else None
    top_k = 8

    try:
        results = await hybrid_retrieve(
            query=body.messages[-1].content,
            clients=clients,
            top_k=top_k,
        )
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        results = []

    reranker = SemanticReranker(
        embed_url=get_settings().ollama_embed_url,
        embed_model=get_settings().ollama_embed_model,
    )
    reranked = await reranker.rerank(
        query=body.messages[-1].content,
        candidates=[
            {
                "text": r.get("text", ""),
                "score": r.get("rrf_score", 0.0),
                "chunk_id": r.get("chunk_id", ""),
                "source": r.get("source", ""),
            }
            for r in results
        ],
        top_k=top_k,
    )

    context = "\n\n".join(
        f"[Source: {r['source']}] {r['text'][:300]}..." for r in reranked
    )

    system_prompt = (
        "Bạn là trợ lý AI chuyên nghiệp. Trả lời dựa trên ngữ cảnh được cung cấp. "
        f"Nếu không có thông tin, hãy nói rõ.\n\nNgữ cảnh:\n{context}"
    )

    if body.system_prompt_override:
        system_prompt = body.system_prompt_override

    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m.role, "content": m.content} for m in body.messages
    ]

    from openai import AsyncOpenAI
    llm = AsyncOpenAI(
        api_key="ollama",
        base_url=get_settings().ollama_base_url,
        http_client=clients.http,
    )

    try:
        response = await llm.chat.completions.create(
            model=body.model,
            messages=messages,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            stream=body.stream,
            stop=body.stop,
        )
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(500, f"LLM error: {e}")

    retrieval_time_ms = (time.monotonic() - start) * 1000

    if body.stream:
        async def stream_generator():
            async for chunk in response:
                yield f"data: {chunk.model_dump_json()}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"X-Retrieval-Time-Ms": str(retrieval_time_ms)},
        )

    msg = response.choices[0].message
    content = msg.content or msg.reasoning or ""

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(datetime.now(timezone.utc).timestamp()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning": msg.reasoning if hasattr(msg, "reasoning") else None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": response.usage.model_dump() if hasattr(response, "usage") and response.usage else {},
        "sources": [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                text=r["text"][:300],
                score=r["rerank_score"],
                retrieval_modes=["reranked"],
                source=r.get("source", ""),
                document_id=r.get("doc_id", ""),
                document_title=r.get("title", ""),
            )
            for r in reranked
        ] if body.include_sources else None,
        "retrieval_time_ms": retrieval_time_ms,
    }


# --- Stats ---

@router.get("/plugins", tags=["system"])
async def list_plugins():
    from plugins.registry import registry
    return registry.list_source_plugins()


@router.get("/stats/tenant/{tenant_id}", tags=["stats"])
async def get_tenant_stats(tenant_id: str):
    return TenantStats(
        tenant_id=tenant_id,
        total_sources=0,
        active_sources=0,
        total_documents=0,
        documents_by_status={},
        total_chunks=0,
        total_entities=0,
        total_relationships=0,
        total_api_calls=0,
        cache_hit_rate=0.0,
        avg_retrieval_time_ms=0.0,
        avg_generation_time_ms=0.0,
        storage_used_mb=0.0,
    )


@router.get("/stats/system", tags=["stats"])
async def get_system_stats():
    return SystemStats(
        total_tenants=len(TENANT_DB),
        active_tenants=sum(1 for t in TENANT_DB.values() if t["status"] == "active"),
        total_documents=0,
        total_chunks=0,
        total_entities=0,
        total_api_calls_today=0,
        system_health="ok",
        uptime_seconds=0.0,
    )
