"""ReAct chat endpoint — /chat/react (multi-step reasoning)."""

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException

from src.clients import get_clients
from src.config import get_settings
from src.services.query_router import classify_query, should_use_react

router = APIRouter()


@router.post("/chat/react", tags=["react"])
async def chat_react(body: dict[str, Any]):
    """
    Multi-step ReAct chat — explicit Thought→Action→Observation reasoning.

    NOTE: This endpoint now enforces query-type routing. Factual queries are
    automatically rerouted to the standard pipeline. ReAct is only used for
    multi-hop, summarization, and analytical queries.

    Each loop step: LLM picks next action (search_entity, expand_relation,
    retrieve_chunks, graph_aware_search, rerank, FINISH). Max 6 steps (raised
    from 4) then synthesize final answer from accumulated chunks.

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

    Body: {"query": "...", "tenant_id": "default", "max_steps": 6}
    """
    from api.routes._chat import chat

    settings = get_settings()
    clients = get_clients()
    query = (body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    tenant_id = body.get("tenant_id") or "default"
    max_steps = int(body.get("max_steps", 6))
    force_react = bool(body.get("force_react", False))

    # Routing guard: factual queries should NOT go through ReAct.
    # classify_query does a blocking bge-m3 embed — run it off the event loop.
    query_type = await asyncio.to_thread(classify_query, query)
    use_react = should_use_react(query_type, query=query) or force_react

    if not use_react:
        # Reroute to standard pipeline instead of running ReAct on a simple query
        return await chat(
            {
                "query": query,
                "tenant_id": tenant_id,
                "access_levels": body.get("access_levels"),
                "format_filter": body.get("format_filter"),
                "include_sources": body.get("include_sources", True),
                "max_retries": int(body.get("max_retries", 1)),
                "force_react": False,
            }
        )

    react_chat_fn = __import__("src.services.react_loop", fromlist=["react_chat"]).react_chat
    result = await react_chat_fn(query, clients, settings, tenant_id, max_steps)
    return result
