"""Global-query LazyGraphRAG — query-time map-reduce over communities.

Thematic/corpus-wide questions bypass top-k retrieval. Fetch community
membership (built cluster-only, no eager LLM summary), MAP each community's
member chunks against the query with the LIGHT model, drop the NO_INFO ones,
then REDUCE the survivors into a global answer with the HEAVY model.

Gated by GLOBAL_QUERY_ENABLED (routing) + community data; fails loud (no_data)
when the tenant has no communities. Prompts live here (not api/routes/_prompts.py)
so the service layer never imports from the API layer.
"""

from __future__ import annotations

import asyncio
import logging

from src.services.community import fetch_chunks_for_entities
from src.services.ollama_helper import ollama_chat

logger = logging.getLogger(__name__)

_NO_INFO = "NO_INFO"

GLOBAL_MAP_PROMPT = """Bạn đang phân tích MỘT cụm chủ đề trong kho tài liệu. Chỉ dựa trên các đoạn dưới đây,
rút ra những ý LIÊN QUAN tới câu hỏi (3-4 gạch đầu dòng ngắn gọn, tiếng Việt).
Nếu cụm này KHÔNG chứa thông tin liên quan tới câu hỏi, trả lời CHÍNH XÁC một từ: NO_INFO

Câu hỏi: {query}

Các đoạn trong cụm [{cluster_id}]:
{context}

Ý liên quan (hoặc NO_INFO):"""

GLOBAL_REDUCE_PROMPT = """Bạn là chuyên gia tổng hợp. Từ các phát hiện theo TỪNG cụm chủ đề dưới đây, viết một
câu trả lời TỔNG THỂ, mạch lạc bằng tiếng Việt cho câu hỏi — bao quát các chủ đề chính, nêu điểm chung và
khác biệt giữa các cụm. Mỗi luận điểm ghi cụm nguồn dạng [cluster_id]. KHÔNG bịa thông tin ngoài các phát hiện.

Câu hỏi: {query}

Phát hiện theo cụm:
{cluster_findings}

Câu trả lời tổng thể:"""


async def _fetch_communities(
    neo4j_driver, tenant_id: str, limit: int
) -> list[tuple[str, list[str]]]:
    """Community id + member entity names (works for lazy communities with no summary)."""
    async with neo4j_driver.session() as s:
        res = await s.run(
            """
            MATCH (com:Community {tenant_id: $tid})<-[:IN_COMMUNITY]-(e:Entity)
            WITH com, collect(e.name) AS members
            RETURN com.id AS id, members AS members
            ORDER BY size(members) DESC
            LIMIT $limit
            """,
            tid=tenant_id,
            limit=limit,
        )
        return [(r["id"], r["members"]) for r in await res.data()]


async def _map_community(
    neo4j_driver,
    cid: str,
    members: list[str],
    query: str,
    tenant_id: str,
    model: str,
    chunks_per_community: int,
) -> tuple[str, str, list[str]] | None:
    """MAP: summarize one community against the query. None if no relevant info."""
    chunks = await fetch_chunks_for_entities(
        neo4j_driver, members, tenant_id, limit=chunks_per_community
    )
    if not chunks:
        return None
    context = "\n\n".join((c.get("text") or "")[:600] for c in chunks if c.get("text"))
    if not context.strip():
        return None
    out = await ollama_chat(
        messages=[
            {
                "role": "user",
                "content": GLOBAL_MAP_PROMPT.format(query=query, cluster_id=cid, context=context),
            }
        ],
        model=model,
        temperature=0.2,
        max_tokens=400,
    )
    out = (out or "").strip()
    if not out or _NO_INFO in out.upper():
        return None
    chunk_ids = [c["chunk_id"] for c in chunks if c.get("chunk_id")]
    return (cid, out, chunk_ids)


async def global_map_reduce(
    query: str,
    tenant_id: str,
    clients,
    settings,
    max_communities: int = 20,
    chunks_per_community: int = 6,
) -> dict:
    """Query-time map-reduce over communities.

    Returns {answer, communities_total, communities_used, sources, no_data}.
    """
    driver = clients.neo4j
    comms = await _fetch_communities(driver, tenant_id, max_communities)
    if not comms:
        logger.warning("global_map_reduce: no communities for tenant=%s", tenant_id)
        return {
            "answer": (
                f'Chưa có dữ liệu community cho tenant "{tenant_id}". Chạy build trước: '
                f'POST /api/community/build {{"tenant_id":"{tenant_id}","lazy":true}}.'
            ),
            "communities_total": 0,
            "communities_used": 0,
            "sources": [],
            "no_data": True,
        }

    mapped = await asyncio.gather(
        *[
            _map_community(
                driver, cid, members, query, tenant_id, settings.light_llm, chunks_per_community
            )
            for cid, members in comms
        ]
    )
    findings = [m for m in mapped if m]
    if not findings:
        return {
            "answer": settings.refusal_message_vi,
            "communities_total": len(comms),
            "communities_used": 0,
            "sources": [],
            "no_data": False,
        }

    cluster_findings = "\n\n".join(f"[{cid}]\n{text}" for cid, text, _ in findings)
    answer = await ollama_chat(
        messages=[
            {
                "role": "user",
                "content": GLOBAL_REDUCE_PROMPT.format(
                    query=query, cluster_findings=cluster_findings
                ),
            }
        ],
        model=settings.heavy_llm,
        temperature=0.3,
        max_tokens=getattr(settings, "generation_max_tokens", 1024),
    )
    sources = [
        {"community_id": cid, "chunk_id": ch} for cid, _, chunk_ids in findings for ch in chunk_ids
    ]
    return {
        "answer": (answer or "").strip(),
        "communities_total": len(comms),
        "communities_used": len(findings),
        "sources": sources,
        "no_data": False,
    }
