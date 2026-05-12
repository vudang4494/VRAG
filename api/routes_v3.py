"""V3 API — Quality-first GraphRAG endpoints (Pipeline V2).

Mounted at /api/v3. New endpoints:
  POST /api/v3/chat                — full quality-first chat
  POST /api/v3/ingest/upload        — ingest_document_v2
  POST /api/v3/community/build      — trigger Leiden + summary build for tenant
  GET  /api/v3/health               — liveness for v3 pipeline
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import json
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger

from src.clients import get_clients
from src.config import get_settings
from src.services.community import build_communities_for_tenant
from src.services.ingestion_v2 import ingest_document_v2
from src.services.query_understanding import understand_query
from src.services.rerank_stages import rerank_full_pipeline
from src.services.retrieval_v2 import multi_path_retrieve
from src.services.validation import validate_answer


router = APIRouter()


# ─── Prompt templates ──────────────────────────────────────────────────────────

_OUTLINE_PROMPT = """Bạn là trợ lý AI nội bộ. Đọc các đoạn văn bản tham khảo và tạo
dàn ý ngắn (3-5 gạch đầu dòng) cho câu trả lời. KHÔNG trả lời chi tiết, chỉ dàn ý.

Câu hỏi: {query}

Các đoạn tham khảo:
{context}

Dàn ý:"""


_DRAFT_PROMPT = """Bạn là trợ lý AI nội bộ. Trả lời câu hỏi sau bằng tiếng Việt
NGẮN GỌN (2-5 câu), dựa HOÀN TOÀN trên các đoạn văn bản tham khảo bên dưới.

QUY TẮC NGHIÊM NGẶT (PHẢI TUÂN THỦ):
1. MỖI câu trong câu trả lời PHẢI kết thúc bằng [chunk_id] của đoạn nguồn.
   Ví dụ: "BGE-M3 hỗ trợ 100 ngôn ngữ [doc_abc::para::5]."
2. CHỈ dùng tên thực thể (entity) XUẤT HIỆN trong các đoạn tham khảo.
   KHÔNG đưa thêm entity từ kiến thức training của bạn.
3. Nếu thực sự không có thông tin trong đoạn tham khảo, chỉ trả lời:
   "Tôi không có đủ thông tin chắc chắn dựa trên tài liệu hiện có."
4. KHÔNG suy luận, KHÔNG bổ sung, KHÔNG diễn giải ngoài context.

Câu hỏi: {query}

Dàn ý gợi ý (tham khảo):
{outline}

Các đoạn tham khảo (chunk_id ở đầu mỗi đoạn):
{context}

Câu trả lời (mỗi câu KÈM [chunk_id] ở cuối):"""


_JUDGE_PROMPT = """Trong 3 bản trả lời sau, bản nào CHÍNH XÁC NHẤT (dựa trên context),
RÕ RÀNG NHẤT, và có TRÍCH DẪN ĐẦY ĐỦ nhất? Trả lời CHỈ số 1, 2, hoặc 3.

Câu hỏi: {query}

Bản 1:
{d1}

Bản 2:
{d2}

Bản 3:
{d3}

Số bản tốt nhất:"""


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _format_context(candidates: list[dict]) -> str:
    """Format reranked candidates into a prompt-ready context block.

    Includes:
      - Top entities extracted from the query (if any) — helps LLM stay focused
      - Per-chunk: chunk_id, source, retrieval_path, matched entities (if entity-pivot)
      - Chunk text
    """
    if not candidates:
        return ""

    # Pull query entities (set by retrieval_v2 on each candidate)
    query_entities: list[str] = []
    for c in candidates:
        qe = c.get("_query_entities")
        if qe:
            query_entities = qe
            break

    header_parts = []
    if query_entities:
        header_parts.append(
            "Các thực thể chính trong câu hỏi (được trích từ knowledge graph): "
            + ", ".join(query_entities[:8])
        )
    header = ("\n".join(header_parts) + "\n\n") if header_parts else ""

    lines = []
    for i, c in enumerate(candidates, 1):
        cid = c.get("chunk_id", f"cand_{i}")
        src = c.get("source", "unknown")
        text = (c.get("text") or "")[:1200]
        # Surface entity-pivot signal if this chunk came via that path
        matched = c.get("matched_entities") or []
        path = c.get("retrieval_path", "")
        annotations = []
        if "entity_pivot" in path or matched:
            annotations.append(f"matched_entities: {', '.join(matched[:6])}")
        ann = f" ({'; '.join(annotations)})" if annotations else ""
        lines.append(f"[{cid}] (source: {src}{ann})\n{text}")

    return header + "\n\n---\n\n".join(lines)


async def _llm_complete(llm, model: str, prompt: str, max_tokens: int = 800, temperature: float = 0.3) -> str:
    """Non-streaming completion. Delegates to ollama_helper for consistency."""
    from src.services.ollama_helper import ollama_chat
    return await ollama_chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ─── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/health", tags=["v3"])
async def health_v3():
    settings = get_settings()
    return {
        "status": "ok",
        "pipeline_v2_enabled": settings.pipeline_v2_enabled,
        "consistency_num_views": settings.consistency_num_views,
        "rerank_stage3_enabled": settings.rerank_stage3_enabled,
        "validation_enabled": settings.validation_enabled,
        "community_enabled": settings.community_enabled,
    }


@router.get("/health/deep", tags=["v3"])
async def health_v3_deep():
    """
    Detailed health of V2 pipeline components.
    Reports availability of cross-encoder, igraph (Leiden), and dependent libs.
    """
    settings = get_settings()
    clients = get_clients()

    def _check(name, import_path):
        try:
            __import__(import_path)
            return {"name": name, "ok": True}
        except Exception as e:
            return {"name": name, "ok": False, "error": str(e)[:200]}

    deps = [
        _check("sentence-transformers", "sentence_transformers"),
        _check("python-docx", "docx"),
        _check("openpyxl", "openpyxl"),
        _check("python-igraph", "igraph"),
        _check("leidenalg", "leidenalg"),
        _check("networkx", "networkx"),
        _check("docling", "docling"),
        _check("pypdf", "pypdf"),
    ]

    # Quick component pings
    qdrant_ok = True
    qdrant_collections: list[str] = []
    try:
        cols = await clients.qdrant.get_collections()
        qdrant_collections = [c.name for c in cols.collections]
    except Exception:
        qdrant_ok = False

    neo4j_ok = True
    neo4j_node_count = 0
    try:
        async with clients.neo4j.session() as s:
            r = await s.run("MATCH (n) RETURN count(n) AS c LIMIT 1")
            row = await r.single()
            neo4j_node_count = int(row["c"]) if row else 0
    except Exception:
        neo4j_ok = False

    ollama_ok = True
    ollama_models: list[str] = []
    try:
        resp = await clients.http.get(f"{settings.ollama_base_url}/api/tags", timeout=10.0)
        ollama_models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        ollama_ok = False

    try:
        from src.metrics import get_metrics
        v2_metrics = get_metrics().get_v2_metrics()
    except Exception:
        v2_metrics = {}

    return {
        "status": "ok" if all([qdrant_ok, neo4j_ok, ollama_ok]) else "degraded",
        "pipeline_v2_enabled": settings.pipeline_v2_enabled,
        "components": {
            "qdrant": {"ok": qdrant_ok, "collections": qdrant_collections},
            "neo4j": {"ok": neo4j_ok, "node_count": neo4j_node_count},
            "ollama": {"ok": ollama_ok, "models": ollama_models, "url": settings.ollama_base_url},
        },
        "dependencies": deps,
        "config_summary": {
            "consistency_num_views": settings.consistency_num_views,
            "entity_vote_passes": settings.entity_vote_passes,
            "query_reformulations": settings.query_reformulations,
            "rerank_stage1_enabled": settings.rerank_stage1_enabled,
            "rerank_stage3_enabled": settings.rerank_stage3_enabled,
            "generation_drafts": settings.generation_drafts,
            "validation_enabled": settings.validation_enabled,
            "community_enabled": settings.community_enabled,
        },
        "metrics_v2": v2_metrics,
    }


@router.post("/ingest/upload", tags=["v3"])
async def ingest_upload_v3(
    file: UploadFile = File(...),
    filename: str | None = Form(default=None),
    tenant_id: str = Form(default="default"),
    access_level: str = Form(default="INTERNAL"),
    department: str | None = Form(default=None),
    author: str | None = Form(default=None),
):
    """Ingest a single document through Pipeline V2."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) > 200 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (>200MB)")

    fname = filename or file.filename or "upload"
    clients = get_clients()
    result = await ingest_document_v2(
        content=content,
        filename=fname,
        clients=clients,
        tenant_id=tenant_id,
        access_level=access_level,
        department=department,
        author=author,
    )
    return result


@router.post("/chat", tags=["v3"])
async def chat_v3(body: dict[str, Any]):
    """
    Quality-first chat. Body:
      {
        "query": "...",
        "tenant_id": "default",
        "access_levels": ["PUBLIC", "INTERNAL"],
        "format_filter": null | ["pdf", "xlsx"],
        "include_sources": true,
        "max_retries": 1
      }
    """
    settings = get_settings()
    clients = get_clients()
    started_total = time.monotonic()
    latency: dict[str, float] = {}

    query = (body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    tenant_id = body.get("tenant_id") or "default"
    access_levels = body.get("access_levels")
    format_filter = body.get("format_filter")
    include_sources = body.get("include_sources", True)
    max_retries = int(body.get("max_retries", 1))

    # 1. Query understanding
    t0 = time.monotonic()
    understanding = await understand_query(
        query, clients.llm, model=settings.ollama_model, timeout=settings.query_understanding_timeout_s,
    )
    latency["query_understanding_ms"] = (time.monotonic() - t0) * 1000

    candidates: list[dict] = []
    top_reranked: list[dict] = []
    answer = ""
    validation: dict[str, Any] = {}
    refused = False
    refusal_reason = None

    for attempt in range(max_retries + 1):
        # 2. Multi-path retrieval
        t0 = time.monotonic()
        top_k_per_path = settings.retrieval_v2_path_top_k * (1 + attempt)  # retry với broader scope
        candidates = await multi_path_retrieve(
            understanding,
            clients,
            tenant_id=tenant_id,
            format_filter=format_filter,
            access_levels=access_levels,
            top_k_per_path=top_k_per_path,
            final_top_k=settings.rerank_stage1_top_k * (1 + attempt),
        )
        latency[f"retrieval_attempt{attempt}_ms"] = (time.monotonic() - t0) * 1000

        if not candidates:
            refused = True
            refusal_reason = "no_candidates"
            answer = settings.refusal_message_vi
            break

        # 3. 3-stage rerank
        t0 = time.monotonic()
        top_reranked = await rerank_full_pipeline(
            query=understanding.get("rewrite") or query,
            candidates=candidates,
            http=clients.http,
            embed_url=settings.ollama_embed_url,
            llm=clients.llm,
            embed_model=settings.ollama_embed_model,
            llm_model=settings.ollama_model,
            stage1_top_k=settings.rerank_stage2_top_k,
            stage2_top_k=settings.rerank_stage3_top_k,
            stage3_top_k=settings.final_top_k,
            enable_stage1=settings.rerank_stage1_enabled,
            enable_stage3=settings.rerank_stage3_enabled,
        )
        latency[f"rerank_attempt{attempt}_ms"] = (time.monotonic() - t0) * 1000

        if not top_reranked:
            refused = True
            refusal_reason = "rerank_empty"
            answer = settings.refusal_message_vi
            break

        # 4. Context assembly
        context = _format_context(top_reranked)

        # 5. Generation: outline → 3 drafts // → judge
        t0 = time.monotonic()
        if settings.generation_outline_enabled:
            outline = await _llm_complete(
                clients.llm, settings.ollama_model,
                _OUTLINE_PROMPT.format(query=query, context=context),
                max_tokens=300, temperature=0.2,
            )
        else:
            outline = ""

        drafts = await asyncio.gather(*[
            _llm_complete(
                clients.llm, settings.ollama_model,
                _DRAFT_PROMPT.format(query=query, outline=outline or "(no outline)", context=context),
                max_tokens=settings.generation_max_tokens,
                temperature=0.2 + i * 0.15,
            )
            for i in range(settings.generation_drafts)
        ])
        drafts = [d for d in drafts if d]
        if not drafts:
            refused = True
            refusal_reason = "no_drafts"
            answer = settings.refusal_message_vi
            break

        # Judge
        if settings.generation_judge_enabled and len(drafts) > 1:
            verdict = await _llm_complete(
                clients.llm, settings.ollama_model,
                _JUDGE_PROMPT.format(
                    query=query,
                    d1=drafts[0],
                    d2=drafts[1] if len(drafts) > 1 else "(không có)",
                    d3=drafts[2] if len(drafts) > 2 else "(không có)",
                ),
                max_tokens=10, temperature=0.1,
            )
            import re
            match = re.search(r"\d", verdict)
            best_idx = int(match.group(0)) - 1 if match else 0
            best_idx = max(0, min(best_idx, len(drafts) - 1))
            answer = drafts[best_idx]
        else:
            answer = drafts[0]
        latency[f"generation_attempt{attempt}_ms"] = (time.monotonic() - t0) * 1000

        # 6. Validation gates
        if settings.validation_enabled:
            t0 = time.monotonic()
            validation = await validate_answer(
                answer=answer,
                context=context,
                llm=clients.llm,
                neo4j_driver=clients.neo4j,
                tenant_id=tenant_id,
                model=settings.ollama_model,
                min_grounded_ratio=settings.validation_min_grounded_ratio,
                max_invalid_entities=settings.validation_max_invalid_entities,
                min_citation_ratio=settings.validation_min_citation_ratio,
            )
            latency[f"validation_attempt{attempt}_ms"] = (time.monotonic() - t0) * 1000
            if validation.get("passed"):
                break
            else:
                logger.info(f"Validation failed (attempt {attempt}): {validation.get('failure_reason')}")
                if attempt >= max_retries:
                    if settings.validation_retry_on_fail:
                        refused = True
                        refusal_reason = validation.get("failure_reason") or "validation_failed"
                        answer = settings.refusal_message_vi
                    break
                # else: loop with broader retrieval
        else:
            validation = {"passed": True, "grounded_ratio": 1.0, "confidence": 1.0, "failure_reason": None}
            break

    total_ms = (time.monotonic() - started_total) * 1000
    latency["total_ms"] = total_ms

    try:
        from src.metrics import get_metrics
        get_metrics().record_v2_chat(
            refused=refused,
            validation_passed=validation.get("passed", True),
            grounded_ratio=validation.get("grounded_ratio", 0.0),
            stage_latencies_ms=latency,
        )
    except Exception:
        pass

    sources_out = []
    if include_sources:
        for c in top_reranked[:settings.final_top_k]:
            sources_out.append({
                "chunk_id": c.get("chunk_id"),
                "text": (c.get("text") or "")[:500],
                "source": c.get("source"),
                "format": c.get("format"),
                "chunk_level": c.get("chunk_level"),
                "final_score": c.get("final_score"),
                "consistency_score": c.get("consistency_score"),
                "judge_reason": c.get("judge_reason"),
            })

    return {
        "id": f"chat_v3_{uuid.uuid4().hex[:12]}",
        "created": int(time.time()),
        "model": settings.ollama_model,
        "answer": answer,
        "refused": refused,
        "refusal_reason": refusal_reason,
        "intent": understanding.get("intent"),
        "confidence": validation.get("confidence", 0.0),
        "validation": {
            "passed": validation.get("passed", True),
            "grounded_ratio": validation.get("grounded_ratio", 1.0),
            "invalid_entities": validation.get("invalid_entities", []),
            "citation_ratio": validation.get("citation_ratio", 1.0),
            "failure_reason": validation.get("failure_reason"),
        },
        "sources": sources_out,
        "latency_breakdown_ms": latency,
    }


@router.post("/chat/stream", tags=["v3"])
async def chat_v3_stream(body: dict[str, Any]):
    """
    Streaming chat via Server-Sent Events (SSE).

    Perceived latency: ~2-3s (first token) vs 18-30s end-to-end.
    Same retrieval/rerank as /api/v3/chat but streams generation tokens.

    Event types emitted (each line: `data: <json>\\n\\n`):
      - meta:       {intent, sources, retrieval_timings}
      - token:      {text}
      - validation: {passed, grounded_ratio, citation_ratio, ...}
      - done:       {refused, refusal_reason, total_ms}
      - error:      {error}

    Body same shape as /api/v3/chat.
    """
    settings = get_settings()
    clients = get_clients()
    started_total = time.monotonic()

    query = (body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    tenant_id = body.get("tenant_id") or "default"
    access_levels = body.get("access_levels")
    format_filter = body.get("format_filter")
    include_sources = body.get("include_sources", True)

    async def _event_stream():
        nonlocal query, tenant_id, access_levels, format_filter
        latency: dict[str, float] = {}

        def sse(payload: dict) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        try:
            # 1. Query understanding
            t0 = time.monotonic()
            understanding = await understand_query(
                query, clients.llm, model=settings.ollama_model,
                timeout=settings.query_understanding_timeout_s,
            )
            latency["query_understanding_ms"] = (time.monotonic() - t0) * 1000

            # 2. Retrieval
            t0 = time.monotonic()
            candidates = await multi_path_retrieve(
                understanding, clients,
                tenant_id=tenant_id,
                format_filter=format_filter,
                access_levels=access_levels,
                top_k_per_path=settings.retrieval_v2_path_top_k,
                final_top_k=settings.rerank_stage1_top_k,
            )
            latency["retrieval_ms"] = (time.monotonic() - t0) * 1000

            if not candidates:
                yield sse({"type": "done", "refused": True, "refusal_reason": "no_candidates",
                          "total_ms": (time.monotonic() - started_total) * 1000})
                return

            # 3. Rerank
            t0 = time.monotonic()
            top_reranked = await rerank_full_pipeline(
                query=understanding.get("rewrite") or query,
                candidates=candidates,
                http=clients.http,
                embed_url=settings.ollama_embed_url,
                llm=clients.llm,
                embed_model=settings.ollama_embed_model,
                llm_model=settings.ollama_model,
                stage1_top_k=settings.rerank_stage2_top_k,
                stage2_top_k=settings.rerank_stage3_top_k,
                stage3_top_k=settings.final_top_k,
                enable_stage1=settings.rerank_stage1_enabled,
                enable_stage3=settings.rerank_stage3_enabled,
            )
            latency["rerank_ms"] = (time.monotonic() - t0) * 1000

            if not top_reranked:
                yield sse({"type": "done", "refused": True, "refusal_reason": "rerank_empty",
                          "total_ms": (time.monotonic() - started_total) * 1000})
                return

            # 4. Send META event — UI gets sources before tokens arrive
            sources_out = [
                {
                    "chunk_id": c.get("chunk_id"),
                    "text": (c.get("text") or "")[:300],
                    "source": c.get("source"),
                    "format": c.get("format"),
                    "chunk_level": c.get("chunk_level"),
                    "final_score": c.get("final_score"),
                }
                for c in top_reranked[:settings.final_top_k]
            ] if include_sources else []
            yield sse({
                "type": "meta",
                "intent": understanding.get("intent"),
                "sources": sources_out,
                "retrieval_timings": latency,
            })

            # 5. Stream generation
            context = _format_context(top_reranked)
            prompt = _DRAFT_PROMPT.format(
                query=query,
                outline="(streaming mode, no separate outline)",
                context=context,
            )

            from src.services.ollama_helper import ollama_chat_stream
            t0 = time.monotonic()
            full_answer = ""
            async for chunk in ollama_chat_stream(
                messages=[{"role": "user", "content": prompt}],
                model=settings.ollama_model,
                temperature=0.2,
                max_tokens=settings.generation_max_tokens,
            ):
                tok = chunk.get("token") or ""
                if tok:
                    full_answer += tok
                    yield sse({"type": "token", "text": tok})
                if chunk.get("done"):
                    break
            latency["generation_ms"] = (time.monotonic() - t0) * 1000

            # 6. Validation on full answer (post-stream)
            validation = {"passed": True, "grounded_ratio": 1.0, "citation_ratio": 1.0}
            if settings.validation_enabled and full_answer.strip():
                t0 = time.monotonic()
                validation = await validate_answer(
                    answer=full_answer,
                    context=context,
                    llm=clients.llm,
                    neo4j_driver=clients.neo4j,
                    tenant_id=tenant_id,
                    model=settings.ollama_model,
                    min_grounded_ratio=settings.validation_min_grounded_ratio,
                    max_invalid_entities=settings.validation_max_invalid_entities,
                    min_citation_ratio=settings.validation_min_citation_ratio,
                )
                latency["validation_ms"] = (time.monotonic() - t0) * 1000

            yield sse({
                "type": "validation",
                "passed": validation.get("passed", True),
                "grounded_ratio": validation.get("grounded_ratio", 1.0),
                "citation_ratio": validation.get("citation_ratio", 1.0),
                "invalid_entities": validation.get("invalid_entities", []),
                "failure_reason": validation.get("failure_reason"),
            })

            # 7. Done event with full stats
            total_ms = (time.monotonic() - started_total) * 1000
            yield sse({
                "type": "done",
                "refused": not validation.get("passed", True),
                "refusal_reason": validation.get("failure_reason") if not validation.get("passed", True) else None,
                "total_ms": total_ms,
                "latency_breakdown_ms": latency,
                "answer_length": len(full_answer),
            })

            # Record metrics
            try:
                from src.metrics import get_metrics
                get_metrics().record_v2_chat(
                    refused=not validation.get("passed", True),
                    validation_passed=validation.get("passed", True),
                    grounded_ratio=validation.get("grounded_ratio", 0.0),
                    stage_latencies_ms=latency,
                )
            except Exception:
                pass

        except Exception as e:
            logger.exception("Streaming chat failed")
            yield sse({"type": "error", "error": str(e)[:500]})

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
        },
    )


@router.post("/chat/react", tags=["v3"])
async def chat_v3_react(body: dict[str, Any]):
    """
    Multi-step ReAct chat — explicit Thought→Action→Observation reasoning.

    Each loop step: LLM picks next action (search_entity, expand_relation,
    retrieve_chunks, graph_aware_search, rerank, FINISH). Max 4 steps then
    synthesize final answer from accumulated chunks.

    Returns:
      {
        "answer": "...",
        "trace": [ { step, thought, action, args, observation_summary } ],
        "steps_used": N,
        "chunks_examined": K,
        "discovered_entities": [...],
        "sources": [...],
        "latency_ms": {total, synthesize}
      }

    Body: {"query": "...", "tenant_id": "default", "max_steps": 4}
    """
    from src.services.react_loop import react_chat as react_chat_fn

    settings = get_settings()
    clients = get_clients()
    query = (body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    tenant_id = body.get("tenant_id") or "default"
    max_steps = int(body.get("max_steps", 4))

    result = await react_chat_fn(query, clients, settings, tenant_id, max_steps)
    return result


@router.post("/gaea/refine", tags=["v3"])
async def gaea_refine(body: dict[str, Any]):
    """
    GAEA — Graph-Augmented Embedding Aggregation.

    Refines all chunk embeddings for a tenant using entity-neighborhood
    attention. Adds `graph_aware` named vector to Qdrant collection.

    Body: {
      "tenant_id": "eval",
      "alpha": 0.35,            // blend factor (0-1)
      "neighbor_cap": 20,        // max co-mention chunks per chunk
      "batch_size": 50
    }

    Run AFTER ingest + cross_doc build. Idempotent (re-run updates the vector).
    """
    from src.services.graph_embeddings import batch_refine_tenant

    settings = get_settings()
    clients = get_clients()
    tenant_id = body.get("tenant_id") or "default"

    started = time.monotonic()
    result = await batch_refine_tenant(
        clients.neo4j,
        clients.qdrant,
        settings.qdrant_collection,
        tenant_id=tenant_id,
        alpha=float(body.get("alpha", 0.35)),
        neighbor_cap=int(body.get("neighbor_cap", 20)),
        batch_size=int(body.get("batch_size", 50)),
    )
    result["duration_seconds"] = time.monotonic() - started
    return result


@router.post("/cross_doc/build", tags=["v3"])
async def cross_doc_build(body: dict[str, Any]):
    """
    Build cross-document relationships:
      - (:Document)-[:SHARES_ENTITIES]->(:Document)
      - (:Chunk)-[:SIMILAR_TO {cross_doc: true}]->(:Chunk)
      - (:Document)-[:SIMILAR_DOC]->(:Document)
    """
    from src.services.cross_doc import build_cross_doc_graph

    settings = get_settings()
    clients = get_clients()
    tenant_id = body.get("tenant_id") or "default"

    started = time.monotonic()
    result = await build_cross_doc_graph(
        clients.neo4j,
        clients.qdrant,
        settings.qdrant_collection,
        tenant_id=tenant_id,
        sample_chunks=int(body.get("sample_chunks", 500)),
        min_chunk_score=float(body.get("min_chunk_score", 0.75)),
        min_shared_entities=int(body.get("min_shared_entities", 3)),
        min_entity_jaccard=float(body.get("min_entity_jaccard", 0.10)),
        min_chunk_edges_for_doc=int(body.get("min_chunk_edges_for_doc", 5)),
        min_doc_avg_score=float(body.get("min_doc_avg_score", 0.78)),
    )
    result["duration_seconds"] = time.monotonic() - started
    return result


@router.post("/community/build", tags=["v3"])
async def community_build(body: dict[str, Any]):
    """
    Trigger Leiden clustering + LLM summary build for a tenant.
    Body: {"tenant_id": "default", "levels": 1, "resolution": 1.0, "min_size": 3}
    """
    settings = get_settings()
    clients = get_clients()

    tenant_id = body.get("tenant_id") or "default"
    levels = int(body.get("levels", settings.community_levels))
    resolution = float(body.get("resolution", settings.community_resolution))
    min_size = int(body.get("min_size", settings.community_min_size))
    vote_passes = int(body.get("vote_passes", settings.community_summary_vote_passes))

    started = time.monotonic()
    stats = await build_communities_for_tenant(
        clients.neo4j,
        clients.llm,
        tenant_id=tenant_id,
        levels=levels,
        resolution=resolution,
        min_size=min_size,
        vote_passes=vote_passes,
        llm_model=settings.ollama_model,
    )
    stats["duration_seconds"] = time.monotonic() - started
    return stats
