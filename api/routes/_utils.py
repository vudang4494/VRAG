"""Shared helpers used across chat and streaming endpoints."""

import time
from typing import Any

from src.services.ollama_helper import ollama_chat


def format_context(candidates: list[dict]) -> str:
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


async def llm_complete(
    model: str,
    prompt: str,
    max_tokens: int = 800,
    temperature: float = 0.3,
) -> str:
    """Non-streaming completion. Delegates to ollama_helper for consistency."""
    return await ollama_chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def run_retrieval_and_rerank(
    query: str,
    tenant_id: str,
    settings,
    clients,
    format_filter: Any = None,
    access_levels: Any = None,
    max_retries: int = 1,
) -> tuple[list[dict], list[dict], dict]:
    """Shared retrieval + reranking logic used by both /chat and /chat/stream.

    Returns (candidates, top_reranked, latency_dict).
    """
    from src.services.query_understanding import understand_query
    from src.services.rerank_l2r import rerank_l2r
    from src.services.rerank_stages import rerank_full_pipeline
    from src.services.retrieval_v2 import multi_path_retrieve

    latency: dict[str, float] = {}

    # Query understanding
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

    for attempt in range(max_retries + 1):
        # Multi-path retrieval
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
            return [], [], latency

        # Extract query entities from entity-pivot retrieval
        qe: list[str] = []
        for c in candidates:
            qe = c.get("_query_entities") or []
            if qe:
                break

        # 3-stage rerank → L2R final
        t0 = time.monotonic()
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

        if top_reranked:
            return candidates, top_reranked, latency
        # else: loop with broader scope

    return candidates, top_reranked, latency


def extract_query_entities(candidates: list[dict]) -> list[str]:
    """Pull query entities from entity-pivot candidates."""
    for c in candidates:
        qe = c.get("_query_entities")
        if qe:
            return qe
    return []


def build_sources_out(
    top_reranked: list[dict],
    include_sources: bool,
    final_top_k: int,
) -> list[dict]:
    """Build sources list from reranked candidates."""
    if not include_sources:
        return []
    return [
        {
            "chunk_id": c.get("chunk_id"),
            "text": (c.get("text") or "")[:500],
            "source": c.get("source"),
            "format": c.get("format"),
            "chunk_level": c.get("chunk_level"),
            "final_score": c.get("final_score"),
            "consistency_score": c.get("consistency_score"),
            "judge_reason": c.get("judge_reason"),
        }
        for c in top_reranked[:final_top_k]
    ]
