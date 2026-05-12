"""Retrieval service — hybrid Vector + Graph retrieval with RRF fusion."""
import asyncio
from typing import Any

from loguru import logger

from src.services.embedding import embed_single
from src.services.vector import vector_search
from src.services.kg import graph_retrieve


def rrf_fuse(
    vector_results: list[dict],
    graph_results: list[dict],
    k: int = 60,
    top_k: int = 10,
    vector_weight: float = 1.0,
    graph_weight: float = 1.0,
) -> list[dict]:
    """
    Reciprocal Rank Fusion to merge ranked lists from vector and graph retrieval.

    RRF_score = Σ (weight / (k + rank)) across all retrieval systems.
    Items appearing in both lists benefit from being ranked well in either system.
    """
    fused: dict[str, dict] = {}

    def add_results(results: list[dict], weight: float = 1.0):
        for rank, item in enumerate(results, 1):
            key = item.get("chunk_id", "")[:64]
            if not key:
                continue

            if key not in fused:
                fused[key] = {
                    "chunk_id": key,
                    "text": item.get("text", ""),
                    "source": item.get("source", "unknown"),
                    "metadata": item.get("metadata", {}),
                    "matched_entities": item.get("matched_entities", []),
                    "vector_score": 0.0,
                    "graph_score": 0.0,
                    "rrf_score": 0.0,
                }

            fused[key]["rrf_score"] += weight / (k + rank)
            if item.get("retrieval_mode") == "vector":
                fused[key]["vector_score"] = max(
                    fused[key]["vector_score"], item.get("score", 0.0)
                )
            elif item.get("retrieval_mode") == "graph":
                fused[key]["graph_score"] = max(
                    fused[key]["graph_score"], item.get("graph_score", 0.0)
                )

    add_results(vector_results, weight=vector_weight)
    add_results(graph_results, weight=graph_weight)

    sorted_results = sorted(
        fused.values(), key=lambda x: x["rrf_score"], reverse=True
    )

    for item in sorted_results[:top_k]:
        modes = []
        if item["vector_score"] > 0:
            modes.append("vector")
        if item["graph_score"] > 0:
            modes.append("graph")
        item["retrieval_modes"] = modes

    return sorted_results[:top_k]


async def hybrid_retrieve(
    query: str,
    clients: Any,
    top_k: int = 8,
    vector_top_k: int = 20,
    graph_top_k: int = 20,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline — optimized for concurrent execution.

    1. Embed query (BGE-M3)
    2. Concurrent vector search (Qdrant) + graph search (Neo4j)
    3. RRF fusion
    4. Return top-K fused results
    """
    from src.config import get_settings
    settings = get_settings()

    # Embed query
    try:
        query_vec = await embed_single(
            clients.http,
            settings.ollama_embed_url,
            settings.ollama_embed_model,
            query,
            timeout=60.0,
        )
    except Exception as e:
        logger.warning(f"Query embedding failed: {e}")
        return []

    # Concurrent vector + graph search
    vector_task = vector_search(
        clients.qdrant,
        settings.qdrant_collection,
        query_vec,
        limit=vector_top_k,
    )
    graph_task = graph_retrieve(
        clients.neo4j,
        query_vec,
        clients.http,
        settings.ollama_embed_url,
        settings.ollama_embed_model,
        top_k=graph_top_k,
    )

    vector_results, graph_results = await asyncio.gather(
        vector_task, graph_task, return_exceptions=True
    )

    if isinstance(vector_results, Exception):
        logger.warning(f"Vector search error: {vector_results}")
        vector_results = []
    if isinstance(graph_results, Exception):
        logger.warning(f"Graph search error: {graph_results}")
        graph_results = []

    if not vector_results and not graph_results:
        return []

    if not graph_results:
        return [{**r, "retrieval_modes": ["vector"]} for r in vector_results[:top_k]]
    if not vector_results:
        return [{**r, "retrieval_modes": ["graph"]} for r in graph_results[:top_k]]

    return rrf_fuse(
        vector_results,
        graph_results,
        k=settings.rrf_k,
        top_k=top_k,
        vector_weight=1.0,
        graph_weight=1.0,
    )
