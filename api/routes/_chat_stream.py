"""Streaming chat endpoint — /chat/stream (SSE)."""

import json
import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from loguru import logger

from api.routes._prompts import DRAFT_PROMPT
from api.routes._utils import format_context

router = APIRouter()


@router.post("/chat/stream", tags=["v3"])
async def chat_stream(body: dict[str, Any]):
    """
    Streaming chat via Server-Sent Events (SSE).

    Perceived latency: ~2-3s (first token) vs 18-30s end-to-end.
    Same retrieval/rerank as /chat but streams generation tokens.

    Event types emitted (each line: `data: <json>\n\n`):
      - meta:       {intent, sources, retrieval_timings}
      - token:      {text}
      - validation: {passed, grounded_ratio, citation_ratio, ...}
      - done:       {refused, refusal_reason, total_ms}
      - error:      {error}

    Body same shape as /chat.
    """
    settings_get = __import__("src.config", fromlist=["get_settings"]).get_settings
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    settings = settings_get()
    clients = clients_get()
    started_total = time.monotonic()

    query = (body.get("query") or "").strip()
    if not query:
        return StreamingResponse(
            iter([json.dumps({"type": "error", "error": "Missing 'query'"})]),
            media_type="application/x-ndjson",
        )
    tenant_id = body.get("tenant_id") or "default"
    access_levels = body.get("access_levels")
    format_filter = body.get("format_filter")
    include_sources = body.get("include_sources", True)

    async def _event_stream():
        latency: dict[str, float] = {}

        def sse(payload: dict) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        try:
            # 1. Query understanding
            from src.services.query_understanding import understand_query

            t0 = time.monotonic()
            understanding = await understand_query(
                query,
                clients.llm,
                model=settings.ollama_model,
                timeout=settings.query_understanding_timeout_s,
                query_type="factual",  # streaming uses standard pipeline; query_type irrelevant
            )
            latency["query_understanding_ms"] = (time.monotonic() - t0) * 1000

            # 2. Retrieval
            from src.services.rerank_l2r import rerank_l2r
            from src.services.rerank import rerank_full_pipeline
            from src.services.retrieval import multi_path_retrieve

            t0 = time.monotonic()
            candidates = await multi_path_retrieve(
                understanding,
                clients,
                tenant_id=tenant_id,
                format_filter=format_filter,
                access_levels=access_levels,
                top_k_per_path=settings.retrieval_path_top_k,
                final_top_k=settings.rerank_stage1_top_k,
            )
            latency["retrieval_ms"] = (time.monotonic() - t0) * 1000

            if not candidates:
                yield sse(
                    {
                        "type": "done",
                        "refused": True,
                        "refusal_reason": "no_candidates",
                        "total_ms": (time.monotonic() - started_total) * 1000,
                    }
                )
                return

            # 3. Rerank → L2R final
            t0 = time.monotonic()
            qe: list[str] = []
            for c in candidates:
                qe = c.get("_query_entities") or []
                if qe:
                    break

            stage2_results = await rerank_full_pipeline(
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
                enable_stage3=False,
            )
            top_reranked = await rerank_l2r(
                query=understanding.get("rewrite") or query,
                candidates=stage2_results,
                query_entities=qe,
                top_k=settings.final_top_k,
            )
            latency["rerank_ms"] = (time.monotonic() - t0) * 1000

            if not top_reranked:
                yield sse(
                    {
                        "type": "done",
                        "refused": True,
                        "refusal_reason": "rerank_empty",
                        "total_ms": (time.monotonic() - started_total) * 1000,
                    }
                )
                return

            # 4. Send META event — UI gets sources before tokens arrive
            sources_out = (
                [
                    {
                        "chunk_id": c.get("chunk_id"),
                        "text": (c.get("text") or "")[:300],
                        "source": c.get("source"),
                        "format": c.get("format"),
                        "chunk_level": c.get("chunk_level"),
                        "final_score": c.get("final_score"),
                    }
                    for c in top_reranked[: settings.final_top_k]
                ]
                if include_sources
                else []
            )
            yield sse(
                {
                    "type": "meta",
                    "intent": understanding.get("intent"),
                    "sources": sources_out,
                    "retrieval_timings": latency,
                }
            )

            # 5. Stream generation
            context = format_context(top_reranked)

            # VRAG Tier 3c: optional context compression before LLM gen
            if settings.context_compression_enabled and context.strip():
                from src.services.context_compress import compress_context

                context, _stats = await compress_context(
                    context, rate=settings.context_compression_rate
                )

            prompt = DRAFT_PROMPT.format(
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
                from src.services.validation import validate_answer

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

            yield sse(
                {
                    "type": "validation",
                    "passed": validation.get("passed", True),
                    "grounded_ratio": validation.get("grounded_ratio", 1.0),
                    "citation_ratio": validation.get("citation_ratio", 1.0),
                    "invalid_entities": validation.get("invalid_entities", []),
                    "failure_reason": validation.get("failure_reason"),
                }
            )

            # 7. Done event with full stats
            total_ms = (time.monotonic() - started_total) * 1000
            yield sse(
                {
                    "type": "done",
                    "refused": not validation.get("passed", True),
                    "refusal_reason": validation.get("failure_reason")
                    if not validation.get("passed", True)
                    else None,
                    "total_ms": total_ms,
                    "latency_breakdown_ms": latency,
                    "answer_length": len(full_answer),
                }
            )

            # Record metrics
            try:
                metrics_get = __import__("src.metrics", fromlist=["get_metrics"]).get_metrics
                get_metrics = metrics_get()
                get_metrics.record_chat(
                    refused=not validation.get("passed", True),
                    validation_passed=validation.get("passed", True),
                    grounded_ratio=validation.get("grounded_ratio", 0.0),
                    stage_latencies_ms=latency,
                )
            except Exception as e:
                logger.debug(f"Metrics recording failed: {e}")

        except Exception as e:
            logger.exception("Streaming chat failed")
            yield sse({"type": "error", "error": str(e)[:500]})

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
