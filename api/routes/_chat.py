"""Main chat endpoint — /chat (non-streaming, quality-first)."""

import asyncio
import re
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger

from api.routes._prompts import DRAFT_PROMPT, JUDGE_PROMPT, OUTLINE_PROMPT, REFINE_PROMPT
from api.routes._utils import build_sources_out, format_context, llm_complete
from src.clients import get_clients
from src.config import get_settings

router = APIRouter()


@router.post("/chat", tags=["chat"])
async def chat(body: dict[str, Any]):
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
    max_react_steps = int(body.get("max_react_steps", 4))
    force_react = bool(body.get("force_react", False))
    session_id = body.get("session_id")
    # Benchmark mode: skip expensive quality gates for faster evaluation
    disable_validation = bool(body.get("disable_validation", False))
    disable_intent = bool(body.get("disable_intent", False))
    disable_history_cache = bool(body.get("disable_history_cache", False))

    # ─── Pre-RAG layer 1: Intent classification ───────────────────────────────
    # Classifies into question/follow_up/greeting/ood. Greeting/OOD short-circuit
    # the pipeline entirely (no embedding, no retrieval, no LLM gen).
    intent_info: dict[str, Any] = {"intent": "question", "confidence": 0.5, "source": "skipped"}
    if not disable_intent:
        t0 = time.monotonic()
        from src.services.intent_classifier import (
            GREETING_RESPONSE_VI,
            OOD_RESPONSE_VI,
            classify_intent,
        )

        intent_info = await classify_intent(query, model=settings.light_llm)
        latency["intent_classification_ms"] = (time.monotonic() - t0) * 1000

        if intent_info["intent"] == "greeting":
            return {
                "id": f"chat_{uuid.uuid4().hex[:12]}",
                "created": int(time.time()),
                "model": settings.light_llm,
                "answer": GREETING_RESPONSE_VI,
                "refused": False,
                "refusal_reason": None,
                "intent": "greeting",
                "intent_confidence": intent_info["confidence"],
                "intent_source": intent_info.get("source"),
                "routing": {
                    "query_type": "greeting",
                    "react_used": False,
                    "reason": "greeting_shortcut",
                },
                "sources": [],
                "latency_breakdown_ms": latency,
            }
        if intent_info["intent"] == "ood":
            return {
                "id": f"chat_{uuid.uuid4().hex[:12]}",
                "created": int(time.time()),
                "model": settings.light_llm,
                "answer": OOD_RESPONSE_VI,
                "refused": True,
                "refusal_reason": "out_of_domain_intent",
                "intent": "ood",
                "intent_confidence": intent_info["confidence"],
                "intent_source": intent_info.get("source"),
                "routing": {"query_type": "ood", "react_used": False, "reason": "ood_shortcut"},
                "sources": [],
                "latency_breakdown_ms": latency,
            }

    # ─── Pre-RAG layer 2: Chat-history semantic cache ─────────────────────────
    # EmbeddingGemma-backed cosine search against past (tenant, session) chats.
    # Cache hit at score >= HIT_THRESHOLD (0.80) returns cached answer without running RAG.
    if not disable_history_cache:
        t0 = time.monotonic()
        try:
            from src.services.chat_history import lookup as _history_lookup

            hit = await _history_lookup(
                clients.qdrant,
                clients.http,
                settings.ollama_embed_url,
                tenant_id=tenant_id,
                query=query,
                session_id=session_id,
            )
            latency["history_cache_lookup_ms"] = (time.monotonic() - t0) * 1000
            if hit:
                return {
                    "id": f"chat_{uuid.uuid4().hex[:12]}",
                    "created": int(time.time()),
                    "model": settings.heavy_llm,
                    "answer": hit["answer"],
                    "refused": False,
                    "refusal_reason": None,
                    "intent": intent_info["intent"],
                    "intent_confidence": intent_info["confidence"],
                    "history_cache": {
                        "hit": True,
                        "score": hit["score"],
                        "original_query": hit["original_query"],
                        "embed_model": hit["embed_model"],
                    },
                    "routing": {
                        "query_type": intent_info["intent"],
                        "react_used": False,
                        "reason": "history_cache_hit",
                    },
                    "sources": hit.get("sources", []),
                    "latency_breakdown_ms": latency,
                }
        except Exception as e:
            logger.debug(f"history cache lookup skipped: {e!r}")

    # Smart routing: classify query type
    from src.services.query_router import classify_query, describe_routing, should_use_react

    # classify_query does a blocking bge-m3 embed; run it off the event loop so a slow
    # embed on one request does not stall every other concurrent request.
    query_type = await asyncio.to_thread(classify_query, query)
    use_react = should_use_react(query_type, query=query) or force_react
    routing_reason = describe_routing(query_type, use_react)
    logger.info(f"[router] '{query[:60]}' → type={query_type}, react={use_react}")

    # Global-query: LazyGraphRAG map-reduce over communities (bypass top-k retrieval)
    if query_type == "global" and settings.global_query_enabled:
        from src.services.global_query import global_map_reduce

        gq_result = await global_map_reduce(
            query,
            tenant_id,
            clients,
            settings,
            max_communities=settings.global_query_max_communities,
        )
        latency["total_ms"] = (time.monotonic() - started_total) * 1000
        gq_answer = gq_result.get("answer", "")

        if not disable_history_cache and gq_answer and not gq_result.get("no_data"):
            try:
                from src.services.chat_history import store as _history_store

                await _history_store(
                    clients.qdrant,
                    clients.http,
                    settings.ollama_embed_url,
                    tenant_id=tenant_id,
                    query=query,
                    answer=gq_answer,
                    sources=gq_result.get("sources", []),
                    session_id=session_id,
                )
            except Exception as e:
                logger.debug(f"history cache store (global) skipped: {e!r}")

        return {
            "id": f"chat_{uuid.uuid4().hex[:12]}",
            "created": int(time.time()),
            "model": settings.heavy_llm,
            "answer": gq_answer,
            "refused": gq_result.get("communities_used", 0) == 0,
            "refusal_reason": "no_community_data" if gq_result.get("no_data") else None,
            "intent": intent_info.get("intent") or query_type,
            "intent_confidence": intent_info.get("confidence"),
            "intent_source": intent_info.get("source"),
            "history_cache": {"hit": False},
            "routing": {
                "query_type": query_type,
                "react_used": False,
                "reason": "global_map_reduce",
                "communities_total": gq_result.get("communities_total"),
                "communities_used": gq_result.get("communities_used"),
            },
            "trace": [],
            "sources": gq_result.get("sources", []),
            "latency_breakdown_ms": latency,
        }

    # ReAct: delegate to react endpoint
    if use_react:
        react_chat_fn = __import__("src.services.react_loop", fromlist=["react_chat"]).react_chat
        react_result = await react_chat_fn(
            query,
            clients,
            settings,
            tenant_id,
            max_react_steps,
            query_type=query_type,
        )
        total_ms = (time.monotonic() - started_total) * 1000
        latency["total_ms"] = total_ms

        react_answer = react_result.get("answer", "")
        react_sources = react_result.get("sources", [])

        # Persist ReAct answer to chat history too — same cache as the standard path.
        if not disable_history_cache and react_answer:
            try:
                from src.services.chat_history import store as _history_store

                await _history_store(
                    clients.qdrant,
                    clients.http,
                    settings.ollama_embed_url,
                    tenant_id=tenant_id,
                    query=query,
                    answer=react_answer,
                    sources=react_sources,
                    session_id=session_id,
                )
            except Exception as e:
                logger.debug(f"history cache store (react) skipped: {e!r}")

        return {
            "id": f"chat_{uuid.uuid4().hex[:12]}",
            "created": int(time.time()),
            "model": settings.heavy_llm,
            "answer": react_answer,
            "refused": False,
            "refusal_reason": None,
            "intent": intent_info.get("intent") or query_type,
            "intent_confidence": intent_info.get("confidence"),
            "intent_source": intent_info.get("source"),
            "history_cache": {"hit": False},
            "routing": {
                "query_type": query_type,
                "react_used": True,
                "reason": routing_reason,
                "steps_used": react_result.get("steps_used"),
                "chunks_examined": react_result.get("chunks_examined"),
            },
            "trace": react_result.get("trace", []),
            "sources": react_sources,
            "latency_breakdown_ms": latency,
        }

    # Standard pipeline: understanding → retrieval → rerank → generation → validation
    from src.services.query_understanding import understand_query
    from src.services.rerank import rerank_full_pipeline
    from src.services.rerank_l2r import rerank_l2r
    from src.services.retrieval import multi_path_retrieve
    from src.services.validation import validate_answer

    # 1. Query understanding (2 LLM calls: rewrite + keywords)
    t0 = time.monotonic()
    understanding = await understand_query(
        query,
        clients.llm,
        model=settings.ollama_model,
        timeout=settings.query_understanding_timeout_s,
        intent=query_type,  # reuse router's classify_query result — don't embed twice
    )
    latency["query_understanding_ms"] = (time.monotonic() - t0) * 1000

    # 1b. Query entities feed entity_pivot retrieval, entity-aware rerank, and the
    # generation prompt. understand_query ALREADY ran GLiNER on this query (via
    # extract_entities_fast) and returned them, so reuse that result instead of loading
    # a SECOND 168M GLiNER copy (clients.entity_extractor) and re-running inference on
    # the identical string. The old comment claimed this ran "in parallel"; it did not —
    # it awaited sequentially after understand_query, paying the model cost twice.
    query_entities: list[str] = [e for e in (understanding.get("entities") or []) if len(e) >= 2]
    if query_entities:
        logger.info(
            f"[query_entities] {len(query_entities)} from understand_query: {query_entities[:8]}"
        )

    candidates: list[dict] = []
    top_reranked: list[dict] = []
    answer = ""
    validation: dict[str, Any] = {}
    refused = False
    refusal_reason = None

    for attempt in range(max_retries + 1):
        # 2. Multi-path retrieval
        t0 = time.monotonic()
        top_k_per_path = settings.retrieval_path_top_k * (1 + attempt)
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

        # 3. 3-stage rerank → L2R final (with entity-aware scoring)
        t0 = time.monotonic()
        # If entity extraction already ran (Step 1b), use those entities.
        # Otherwise fall back to entities from retrieval (entity_pivot path).
        qe: list[str] = query_entities or []
        if not qe:
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
            early_exit_threshold=settings.rerank_early_exit_threshold,
        )
        top_reranked = await rerank_l2r(
            query=understanding.get("rewrite") or query,
            candidates=stage2_results,
            query_entities=qe,
            top_k=settings.final_top_k,
        )
        latency[f"rerank_attempt{attempt}_ms"] = (time.monotonic() - t0) * 1000

        if not top_reranked:
            refused = True
            refusal_reason = "rerank_empty"
            answer = settings.refusal_message_vi
            break

        # 4. Context assembly
        context = format_context(top_reranked)

        # 4b. VRAG Tier 3c: LLMLingua-2 context compression (opt-in via env).
        # Compresses once; the compressed string is reused across outline/draft/refine/validate.
        # Conditional: skip when context is already short (≤5000 chars) — model
        # already nuốt được nguyên context, compression chỉ thêm latency ~2-3s.
        _ctx_compress_threshold = 5000
        if (
            settings.context_compression_enabled
            and context.strip()
            and len(context) > _ctx_compress_threshold
        ):
            from src.services.context_compress import compress_context

            t0_compress = time.monotonic()
            context, compress_stats = await compress_context(
                context, rate=settings.context_compression_rate
            )
            latency[f"context_compression_attempt{attempt}_ms"] = (
                time.monotonic() - t0_compress
            ) * 1000
            if compress_stats.get("compressed"):
                logger.info(
                    f"  context_compression: {compress_stats.get('original_tokens', 0)} → "
                    f"{compress_stats.get('compressed_tokens', 0)} tokens "
                    f"(ratio={compress_stats.get('ratio', 'n/a')})"
                )
        elif settings.context_compression_enabled and context.strip():
            # Short context — skip compression, just record 0ms for telemetry.
            latency[f"context_compression_attempt{attempt}_ms"] = 0.0
            logger.debug(
                f"  context_compression: skipped (len={len(context)} ≤ {_ctx_compress_threshold})"
            )

        # 4c. Sufficient-Context Gate (Tier 3.5: Fast anti-hallucination/OOD check)
        if not disable_validation and settings.sufficient_context_gate_enabled:
            from src.services.sufficient_context import check_sufficient_context

            t0_gate = time.monotonic()
            gate_result = await check_sufficient_context(
                query=query, context=context, llm_model=settings.light_llm
            )
            latency[f"sufficient_context_gate_attempt{attempt}_ms"] = (
                time.monotonic() - t0_gate
            ) * 1000
            if not gate_result["is_sufficient"]:
                logger.info(
                    f"[Gate] Insufficient context for '{query[:40]}...'. Reason: {gate_result['reason']}"
                )
                refused = True
                refusal_reason = "insufficient_context"
                answer = settings.refusal_message_vi
                break

        # 5. Generation: outline → drafts → judge
        t0 = time.monotonic()
        outline = ""
        if settings.generation_outline_enabled:
            outline = await llm_complete(
                settings.ollama_model,
                OUTLINE_PROMPT.format(query=query, context=context),
                max_tokens=300,
                temperature=0.0,
            )

        # Tier 1 fix #1: temperature 0.0 — anti-hallucination, stick to context.
        # Previously 0.2+i*0.15 — invited LLM invention.
        drafts = await asyncio.gather(
            *[
                llm_complete(
                    settings.ollama_model,
                    DRAFT_PROMPT.format(
                        query=query, outline=outline or "(no outline)", context=context
                    ),
                    max_tokens=settings.generation_max_tokens,
                    temperature=0.0 if i == 0 else 0.1 + (i - 1) * 0.1,
                )
                for i in range(settings.generation_drafts)
            ]
        )
        drafts = [d for d in drafts if d]
        if not drafts:
            refused = True
            refusal_reason = "no_drafts"
            answer = settings.refusal_message_vi
            break

        if settings.generation_judge_enabled and len(drafts) > 1:
            verdict = await llm_complete(
                settings.ollama_model,
                JUDGE_PROMPT.format(
                    query=query,
                    d1=drafts[0],
                    d2=drafts[1] if len(drafts) > 1 else "(không có)",
                    d3=drafts[2] if len(drafts) > 2 else "(không có)",
                ),
                max_tokens=10,
                temperature=0.1,
            )
            match = re.search(r"\d", verdict)
            best_idx = int(match.group(0)) - 1 if match else 0
            best_idx = max(0, min(best_idx, len(drafts) - 1))
            answer = drafts[best_idx]
        else:
            answer = drafts[0]

        # 5b. Refinement
        if settings.generation_refine_enabled and answer.strip():
            t0_refine = time.monotonic()
            refined = await llm_complete(
                settings.ollama_model,
                REFINE_PROMPT.format(query=query, context=context, draft=answer),
                max_tokens=settings.generation_max_tokens,
                temperature=0.0,  # Tier 1 fix: anti-hallucination
            )
            latency[f"refinement_attempt{attempt}_ms"] = (time.monotonic() - t0_refine) * 1000
            if refined.strip():
                answer = refined

        latency[f"generation_attempt{attempt}_ms"] = (time.monotonic() - t0) * 1000

        # 6. Validation gates — skip in benchmark mode
        if not disable_validation and settings.validation_enabled:
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
                passages=[c.get("text") or "" for c in top_reranked if c.get("text")],
                http=clients.http,
                embed_url=settings.ollama_embed_url,
                embed_model=settings.ollama_embed_model,
                use_cosine_grounding=settings.validation_cosine_grounding_enabled,
                grounding_sim_hi=settings.validation_grounding_sim_hi,
                grounding_sim_lo=settings.validation_grounding_sim_lo,
            )
            latency[f"validation_attempt{attempt}_ms"] = (time.monotonic() - t0) * 1000

            if validation.get("passed"):
                break

            failure_reason = validation.get("failure_reason") or "unknown"
            logger.info(f"Validation failed (attempt {attempt}): {failure_reason}")

            # Validation failed → the loop's next attempt broadens retrieval (top_k
            # expands via the top_k_per_path formula) and regenerates.
            #
            # A prior "corrective regeneration" step lived here: on attempt 0 it called
            # correct_and_regenerate (a full LLM call), set answer=corrected, then
            # `continue`d — but the next attempt re-retrieved and regenerated, overwriting
            # `corrected` before it was ever re-validated or returned. The corrective
            # answer was therefore always discarded, making that LLM call pure waste.
            # Removed. Real in-place self-correction would need to re-validate the
            # corrected answer WITHOUT re-retrieving; that's a future enhancement.
        else:
            validation = {
                "passed": True,
                "grounded_ratio": 1.0,
                "confidence": 1.0,
                "failure_reason": None,
            }
            break

    total_ms = (time.monotonic() - started_total) * 1000
    latency["total_ms"] = total_ms

    try:
        get_metrics = __import__("src.metrics", fromlist=["get_metrics"]).get_metrics
        get_metrics().record_pipeline_chat(
            refused=refused,
            validation_passed=validation.get("passed", True),
            grounded_ratio=validation.get("grounded_ratio", 0.0),
            stage_latencies_ms=latency,
        )
    except Exception as e:
        logger.debug(f"Metrics recording failed: {e}")
        pass

    sources_out = build_sources_out(top_reranked, include_sources, settings.final_top_k)

    # ─── Post-RAG: persist (query, answer) into chat history for future cache hits.
    # Only stores when the validation gate accepted the answer — refusals and
    # ungrounded answers are skipped (see chat_history.store() guard).
    if not disable_history_cache and not refused and answer and validation.get("passed", True):
        try:
            from src.services.chat_history import store as _history_store

            await _history_store(
                clients.qdrant,
                clients.http,
                settings.ollama_embed_url,
                tenant_id=tenant_id,
                query=query,
                answer=answer,
                citations=validation.get("citations") or [],
                sources=sources_out,
                session_id=session_id,
            )
        except Exception as e:
            logger.debug(f"history cache store skipped: {e!r}")

    return {
        "id": f"chat_{uuid.uuid4().hex[:12]}",
        "created": int(time.time()),
        "model": settings.ollama_model,
        "answer": answer,
        "refused": refused,
        "refusal_reason": refusal_reason,
        "intent": intent_info.get("intent") or understanding.get("intent"),
        "intent_confidence": intent_info.get("confidence"),
        "intent_source": intent_info.get("source"),
        "history_cache": {"hit": False},
        "confidence": validation.get("confidence", 0.0),
        "routing": {
            "query_type": query_type,
            "react_used": False,
            "reason": routing_reason,
        },
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
