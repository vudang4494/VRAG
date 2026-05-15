"""Main chat endpoint — /chat (non-streaming, quality-first)."""

import re
import time
import uuid
from typing import Any

from fastapi import HTTPException
from loguru import logger

from api.routes._prompts import DRAFT_PROMPT, JUDGE_PROMPT, OUTLINE_PROMPT, REFINE_PROMPT
from api.routes._utils import build_sources_out, format_context, llm_complete

router = __import__("fastapi").APIRouter()


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
    settings_get = __import__("src.config", fromlist=["get_settings"]).get_settings
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    settings = settings_get()
    clients = clients_get()
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

    # Smart routing: classify query type
    from src.services.query_router import classify_query, describe_routing, should_use_react

    query_type = classify_query(query)
    use_react = should_use_react(query_type) or force_react
    routing_reason = describe_routing(query_type, use_react)
    logger.info(f"[router] '{query[:60]}' → type={query_type}, react={use_react}")

    # ReAct: delegate to react endpoint
    if use_react:
        react_chat_fn = __import__("src.services.react_loop", fromlist=["react_chat"]).react_chat
        react_result = await react_chat_fn(
            query,
            clients,
            settings,
            tenant_id,
            max_react_steps,
        )
        total_ms = (time.monotonic() - started_total) * 1000
        latency["total_ms"] = total_ms
        return {
            "id": f"chat_v3_{uuid.uuid4().hex[:12]}",
            "created": int(time.time()),
            "model": settings.ollama_model,
            "answer": react_result.get("answer", ""),
            "refused": False,
            "refusal_reason": None,
            "intent": query_type,
            "routing": {
                "query_type": query_type,
                "react_used": True,
                "reason": routing_reason,
                "steps_used": react_result.get("steps_used"),
                "chunks_examined": react_result.get("chunks_examined"),
            },
            "trace": react_result.get("history", []),
            "sources": react_result.get("sources", []),
            "latency_breakdown_ms": latency,
        }

    # Standard pipeline: understanding → retrieval → rerank → generation → validation
    from src.services.ood_detector import detect_ood_mixed
    from src.services.query_understanding import understand_query
    from src.services.rerank_l2r import rerank_l2r
    from src.services.rerank_stages import rerank_full_pipeline
    from src.services.retrieval_v2 import multi_path_retrieve
    from src.services.validation import validate_answer

    # 1. Query understanding
    t0 = time.monotonic()
    understanding = await understand_query(
        query,
        clients.llm,
        model=settings.ollama_model,
        timeout=settings.query_understanding_timeout_s,
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
        top_k_per_path = settings.retrieval_v2_path_top_k * (1 + attempt)
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

        # 2b. OOD detection — refuse BEFORE spending time on rerank + generation
        if settings.ood_detection_enabled:
            t0_ood = time.monotonic()
            ood_result = detect_ood_mixed(candidates, understanding.get("original") or query)
            latency["ood_detection_ms"] = (time.monotonic() - t0_ood) * 1000
            if ood_result["is_ood"]:
                refused = True
                refusal_reason = "out_of_domain"
                answer = settings.refusal_message_vi
                logger.info(
                    f"[OOD] refusing '{query[:50]}' — top_score={ood_result['top_score']}, "
                    f"kw_overlap={ood_result['keyword_overlap_ratio']}, "
                    f"reason={ood_result['reason']}"
                )
                break
            logger.debug(
                f"[OOD] in-domain — top_score={ood_result['top_score']}, "
                f"kw_overlap={ood_result['keyword_overlap_ratio']}"
            )

        # 3. 3-stage rerank → L2R final
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
        latency[f"rerank_attempt{attempt}_ms"] = (time.monotonic() - t0) * 1000

        if not top_reranked:
            refused = True
            refusal_reason = "rerank_empty"
            answer = settings.refusal_message_vi
            break

        # 4. Context assembly
        context = format_context(top_reranked)

        # 5. Generation: outline → drafts → judge
        t0 = time.monotonic()
        import asyncio

        outline = ""
        if settings.generation_outline_enabled:
            outline = await llm_complete(
                settings.ollama_model,
                OUTLINE_PROMPT.format(query=query, context=context),
                max_tokens=300,
                temperature=0.2,
            )

        drafts = await asyncio.gather(
            *[
                llm_complete(
                    settings.ollama_model,
                    DRAFT_PROMPT.format(
                        query=query, outline=outline or "(no outline)", context=context
                    ),
                    max_tokens=settings.generation_max_tokens,
                    temperature=0.2 + i * 0.15,
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
                temperature=0.2,
            )
            latency[f"refinement_attempt{attempt}_ms"] = (time.monotonic() - t0_refine) * 1000
            if refined.strip():
                answer = refined

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
                logger.info(
                    f"Validation failed (attempt {attempt}): {validation.get('failure_reason')}"
                )
                if attempt >= max_retries:
                    if settings.validation_retry_on_fail:
                        refused = True
                        refusal_reason = validation.get("failure_reason") or "validation_failed"
                        answer = settings.refusal_message_vi
                    break
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
        get_metrics().record_v2_chat(
            refused=refused,
            validation_passed=validation.get("passed", True),
            grounded_ratio=validation.get("grounded_ratio", 0.0),
            stage_latencies_ms=latency,
        )
    except Exception:
        pass

    return {
        "id": f"chat_v3_{uuid.uuid4().hex[:12]}",
        "created": int(time.time()),
        "model": settings.ollama_model,
        "answer": answer,
        "refused": refused,
        "refusal_reason": refusal_reason,
        "intent": understanding.get("intent"),
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
        "sources": build_sources_out(top_reranked, include_sources, settings.final_top_k),
        "latency_breakdown_ms": latency,
    }
