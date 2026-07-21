"""Shared helpers used across chat and streaming endpoints."""

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

    # Pull query entities (set by retrieval on each candidate)
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
        meta = c.get("metadata") or {}
        src = c.get("filename") or meta.get("filename") or c.get("source", "unknown")
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
            # GraphRAG paths — essential for benchmarking
            "retrieval_path": c.get("retrieval_path"),
            "matched_entities": c.get("matched_entities") or [],
            "entity_match_count": c.get("entity_match_count"),
        }
        for c in top_reranked[:final_top_k]
    ]
