"""Vector service — Qdrant storage and ANN search."""
import hashlib

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from loguru import logger


def _to_int_id(chunk_id: str) -> int:
    """Convert string chunk_id to deterministic integer for Qdrant."""
    return int(hashlib.md5(chunk_id.encode()).hexdigest()[:12], 16) % (2**63)


async def upsert_points(
    client: AsyncQdrantClient,
    collection: str,
    points: list[dict],
    dimension: int = 1024,
) -> int:
    """Upsert points to Qdrant collection with named 'dense' vector."""
    qdrant_points = [
        PointStruct(
            id=_to_int_id(p["id"]),
            vector={"dense": p["vector"]},
            payload={**p.get("payload", {}), "chunk_id": p["id"]},
        )
        for p in points
    ]

    await client.upsert(collection_name=collection, points=qdrant_points)
    return len(qdrant_points)


async def vector_search(
    client: AsyncQdrantClient,
    collection: str,
    query_vector: list[float],
    limit: int = 10,
) -> list[dict]:
    """Dense vector search in Qdrant using named 'dense' vector."""
    try:
        response = await client.query_points(
            collection_name=collection,
            query=query_vector,
            using="dense",
            limit=limit,
            with_payload=True,
        )
        results = response.points
    except Exception as e:
        logger.warning(f"Qdrant search failed: {e}")
        return []

    scored = []
    for r in results:
        payload = r.payload or {}
        scored.append({
            "chunk_id": payload.get("chunk_id", str(r.id)),
            "text": payload.get("text", ""),
            "source": payload.get("source", "unknown"),
            "metadata": payload,
            "score": float(r.score),
            "retrieval_mode": "vector",
        })
    return scored
