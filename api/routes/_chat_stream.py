"""Streaming chat endpoint — /chat/stream (SSE)."""

import asyncio
import json
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from api.routes._prompts import DRAFT_PROMPT
from api.routes._utils import format_context
from src.clients import get_clients
from src.config import get_settings

router = APIRouter()


@router.post("/chat/stream", tags=["chat"])
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
    settings = get_settings()
    clients = get_clients()
    started_total = time.monotonic()

    query = (body.get("query") or "").strip()
    if not query:
        # Match /chat: reject a missing query with a real 400 before the stream opens,
        # instead of a 200 whose body was not even valid SSE.
        raise HTTPException(status_code=400, detail="Missing 'query'")
    tenant_id = body.get("tenant_id") or "default"
    access_levels = body.get("access_levels")
    format_filter = body.get("format_filter")
    include_sources = body.get("include_sources", True)
    session_id = body.get("session_id")
    disable_intent = bool(body.get("disable_intent", False))
    disable_history_cache = bool(body.get("disable_history_cache", False))

    async def _event_stream():
        latency: dict[str, float] = {}

        def sse(payload: dict) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        def _stream_text_as_tokens(text: str, chunk_size: int = 24) -> list[str]:
            # Break a finished string into pseudo-tokens so cache hits feel
            # like the live stream. UI render path stays identical.
            out: list[str] = []
            for i in range(0, len(text), chunk_size):
                out.append(text[i : i + chunk_size])
            return out

        try:
            # ─── Pre-RAG layer 1: intent classifier (greeting/OOD short-circuit)
            intent_info: dict[str, Any] = {"intent": "question", "source": "skipped"}
            if not disable_intent:
                from src.services.intent_classifier import (
                    GREETING_RESPONSE_VI,
                    OOD_RESPONSE_VI,
                    classify_intent,
                )

                t0 = time.monotonic()
                intent_info = await classify_intent(query, model=settings.light_llm)
                latency["intent_classification_ms"] = (time.monotonic() - t0) * 1000

                if intent_info["intent"] in ("greeting", "ood"):
                    text = (
                        GREETING_RESPONSE_VI
                        if intent_info["intent"] == "greeting"
                        else OOD_RESPONSE_VI
                    )
                    yield sse(
                        {
                            "type": "meta",
                            "intent": intent_info["intent"],
                            "intent_source": intent_info.get("source"),
                            "sources": [],
                            "retrieval_timings": latency,
                            "shortcut": "intent",
                        }
                    )
                    for tok in _stream_text_as_tokens(text):
                        yield sse({"type": "token", "text": tok})
                    yield sse(
                        {
                            "type": "done",
                            "refused": intent_info["intent"] == "ood",
                            "refusal_reason": (
                                "out_of_domain_intent" if intent_info["intent"] == "ood" else None
                            ),
                            "total_ms": (time.monotonic() - started_total) * 1000,
                            "latency_breakdown_ms": latency,
                            "answer_length": len(text),
                        }
                    )
                    return

            # ─── Pre-RAG layer 2: chat-history semantic cache
            if not disable_history_cache:
                from src.services.chat_history import lookup as _history_lookup

                t0 = time.monotonic()
                try:
                    hit = await _history_lookup(
                        clients.qdrant,
                        clients.http,
                        settings.ollama_embed_url,
                        tenant_id=tenant_id,
                        query=query,
                        session_id=session_id,
                    )
                except Exception as e:
                    logger.debug(f"history cache stream lookup skipped: {e!r}")
                    hit = None
                latency["history_cache_lookup_ms"] = (time.monotonic() - t0) * 1000

                if hit:
                    yield sse(
                        {
                            "type": "meta",
                            "intent": intent_info.get("intent"),
                            "intent_source": intent_info.get("source"),
                            "sources": hit.get("sources") or [],
                            "history_cache": {
                                "hit": True,
                                "score": hit["score"],
                                "matched_variant": hit.get("matched_variant"),
                            },
                            "retrieval_timings": latency,
                            "shortcut": "history_cache",
                        }
                    )
                    cached_answer = hit.get("answer", "")
                    for tok in _stream_text_as_tokens(cached_answer):
                        yield sse({"type": "token", "text": tok})
                    yield sse(
                        {
                            "type": "done",
                            "refused": False,
                            "refusal_reason": None,
                            "total_ms": (time.monotonic() - started_total) * 1000,
                            "latency_breakdown_ms": latency,
                            "answer_length": len(cached_answer),
                        }
                    )
                    return

            # ─── Global-query short-circuit: LazyGraphRAG map-reduce (mirror /chat).
            # Gated: flag OFF ⇒ no router call, stream behavior byte-identical.
            if settings.global_query_enabled:
                from src.services.query_router import classify_query

                t0 = time.monotonic()
                query_type = await asyncio.to_thread(classify_query, query)
                latency["router_ms"] = (time.monotonic() - t0) * 1000
                if query_type == "global":
                    from src.services.global_query import global_map_reduce

                    t0 = time.monotonic()
                    gq = await global_map_reduce(
                        query,
                        tenant_id,
                        clients,
                        settings,
                        max_communities=settings.global_query_max_communities,
                    )
                    latency["global_map_reduce_ms"] = (time.monotonic() - t0) * 1000
                    gq_answer = gq.get("answer", "")
                    yield sse(
                        {
                            "type": "meta",
                            "intent": "global",
                            "sources": (gq.get("sources") or []) if include_sources else [],
                            "retrieval_timings": latency,
                            "shortcut": "global",
                            "communities_used": gq.get("communities_used", 0),
                            "communities_total": gq.get("communities_total", 0),
                        }
                    )
                    for tok in _stream_text_as_tokens(gq_answer):
                        yield sse({"type": "token", "text": tok})
                    if not disable_history_cache and gq_answer and not gq.get("no_data"):
                        try:
                            from src.services.chat_history import store as _history_store

                            await _history_store(
                                clients.qdrant,
                                clients.http,
                                settings.ollama_embed_url,
                                tenant_id=tenant_id,
                                query=query,
                                answer=gq_answer,
                                sources=gq.get("sources") or [],
                                session_id=session_id,
                            )
                        except Exception as e:
                            logger.debug(f"history cache store (global stream) skipped: {e!r}")
                    yield sse(
                        {
                            "type": "done",
                            "refused": gq.get("communities_used", 0) == 0,
                            "refusal_reason": ("no_community_data" if gq.get("no_data") else None),
                            "total_ms": (time.monotonic() - started_total) * 1000,
                            "latency_breakdown_ms": latency,
                            "answer_length": len(gq_answer),
                        }
                    )
                    return

            # 1. Query understanding
            from src.services.query_understanding import understand_query

            t0 = time.monotonic()
            understanding = await understand_query(
                query,
                clients.llm,
                model=settings.ollama_model,
                timeout=settings.query_understanding_timeout_s,
            )
            latency["query_understanding_ms"] = (time.monotonic() - t0) * 1000

            # 2. Retrieval
            from src.services.rerank import rerank_full_pipeline
            from src.services.rerank_l2r import rerank_l2r
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
                enable_stage3=settings.rerank_stage3_enabled,
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

            # 4c. Sufficient-Context Gate
            gate_refused = False
            gate_reason = ""
            if settings.sufficient_context_gate_enabled:
                from src.services.sufficient_context import check_sufficient_context

                t0_gate = time.monotonic()
                gate_result = await check_sufficient_context(
                    query=query, context=context, llm_model=settings.light_llm
                )
                latency["sufficient_context_gate_ms"] = (time.monotonic() - t0_gate) * 1000

                if not gate_result["is_sufficient"]:
                    gate_refused = True
                    gate_reason = gate_result["reason"]
                    logger.info(f"[Gate] Stream refused: {gate_reason}")

            if gate_refused:
                # Stream the refusal message and stop
                toks = settings.refusal_message_vi.split()
                for i, tok in enumerate(toks):
                    yield sse({"type": "token", "text": tok + (" " if i < len(toks) - 1 else "")})
                yield sse(
                    {
                        "type": "validation",
                        "passed": False,
                        "failure_reason": "insufficient_context",
                        "grounded_ratio": 0.0,
                        "citation_ratio": 0.0,
                        "invalid_entities": [],
                    }
                )
                yield sse({"type": "done", "refused": True, "latency_breakdown_ms": latency})
                return

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
                temperature=0.0,
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
                    passages=[c.get("text") or "" for c in top_reranked if c.get("text")],
                    http=clients.http,
                    embed_url=settings.ollama_embed_url,
                    embed_model=settings.ollama_embed_model,
                    use_cosine_grounding=settings.validation_cosine_grounding_enabled,
                    grounding_sim_hi=settings.validation_grounding_sim_hi,
                    grounding_sim_lo=settings.validation_grounding_sim_lo,
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

            # Persist answer to chat history so subsequent (paraphrased) queries
            # can hit the cache. Skips refusals via store()'s internal guard.
            if not disable_history_cache and full_answer and validation.get("passed", True):
                try:
                    from src.services.chat_history import store as _history_store

                    await _history_store(
                        clients.qdrant,
                        clients.http,
                        settings.ollama_embed_url,
                        tenant_id=tenant_id,
                        query=query,
                        answer=full_answer,
                        sources=sources_out,
                        session_id=session_id,
                    )
                except Exception as e:
                    logger.debug(f"history cache store (stream) skipped: {e!r}")

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

            # Record metrics — record_pipeline_chat (matches /chat; record_chat never existed
            # on PrometheusMetrics, so streaming metrics were silently swallowed every request).
            try:
                get_metrics = __import__("src.metrics", fromlist=["get_metrics"]).get_metrics
                get_metrics().record_pipeline_chat(
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
