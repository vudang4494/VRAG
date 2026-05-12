"""Phase 4 — HEFR: Hierarchical Entity-First Retrieval.

Invert classic chunk-first retrieval: query → entities → chunks.
At scale (1M+ chunks), entity layer (50K) is much faster ANN target.

Architecture:
   Query → Domain classifier (filter)
         → Entity NER (extract query entities)
         → Entity ANN (Qdrant entities_<tenant>) → top 20 entities
         → Cypher: entities → chunks via CONTAINS_ENTITY
         → Chunk rerank with multi-feature L2R
         → Top-K answer

For our 10-paper test bed, HEFR doesn't show as much speedup (chunks already
small). But framework + entity ANN collection enables scaling.

Entity embeddings already computed by Phase 1 GAEA's
`aggregate_entity_embedding`. This module:
  1. Pushes entity embeddings to separate Qdrant collection
  2. Provides hefr_retrieve() orchestrator
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger


ENTITY_COLLECTION_TEMPLATE = "entities_{tenant}"


async def ensure_entity_collection(qdrant_client, tenant_id: str, dim: int = 1024):
    """Create the per-tenant entity collection if missing."""
    from qdrant_client import models as qm
    col_name = ENTITY_COLLECTION_TEMPLATE.format(tenant=tenant_id)
    try:
        await qdrant_client.get_collection(col_name)
        return col_name
    except Exception:
        pass  # not found, create

    await qdrant_client.create_collection(
        collection_name=col_name,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )
    # Payload indexes for fast filter
    for field in ("name", "type"):
        try:
            await qdrant_client.create_payload_index(
                collection_name=col_name, field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass
    logger.info(f"HEFR: created entity collection {col_name}")
    return col_name


async def populate_entity_collection(
    neo4j_driver,
    qdrant_client,
    chunk_collection: str,
    tenant_id: str,
    batch_size: int = 100,
) -> dict[str, Any]:
    """Compute aggregate embedding for each entity in tenant, upsert to
    entities_<tenant> Qdrant collection."""
    from src.services.graph_embeddings import aggregate_entity_embedding
    from qdrant_client import models as qm
    import hashlib

    col_name = await ensure_entity_collection(qdrant_client, tenant_id)

    # List all entities
    async with neo4j_driver.session() as s:
        r = await s.run(
            "MATCH (e:Entity) WHERE e.tenant_id = $tid RETURN e.name AS name, e.type AS type LIMIT 10000",
            tid=tenant_id,
        )
        entities = await r.data()
    logger.info(f"HEFR populate: {len(entities)} entities for tenant {tenant_id}")

    points = []
    skipped = 0
    sem = asyncio.Semaphore(8)

    async def _one(ent):
        nonlocal skipped
        async with sem:
            emb = await aggregate_entity_embedding(
                ent["name"], neo4j_driver, qdrant_client, chunk_collection, tenant_id,
            )
            if emb is None:
                skipped += 1
                return None
            ent_id = int(hashlib.sha256(ent["name"].encode()).hexdigest()[:15], 16)
            return qm.PointStruct(
                id=ent_id,
                vector=emb.tolist(),
                payload={"name": ent["name"], "type": ent.get("type", "OTHER"), "tenant_id": tenant_id},
            )

    results = await asyncio.gather(*[_one(e) for e in entities])
    points = [p for p in results if p is not None]

    if points:
        # Batch upsert
        for i in range(0, len(points), batch_size):
            await qdrant_client.upsert(collection_name=col_name, points=points[i:i+batch_size])

    return {"collection": col_name, "entities_upserted": len(points), "skipped": skipped, "total": len(entities)}


async def hefr_retrieve(
    query_embedding: list[float],
    query_entity_names: list[str],
    clients: Any,
    settings: Any,
    tenant_id: str,
    top_entities: int = 20,
    top_chunks: int = 50,
) -> tuple[list[dict], list[dict]]:
    """Entity-first retrieval. Returns (chunks, top_entities_found)."""
    col_name = ENTITY_COLLECTION_TEMPLATE.format(tenant=tenant_id)
    started = time.monotonic()

    # 1. Entity ANN search
    try:
        ent_resp = await clients.qdrant.query_points(
            collection_name=col_name,
            query=query_embedding,
            limit=top_entities,
            with_payload=True,
        )
        ent_hits = [
            {"name": p.payload.get("name"), "type": p.payload.get("type"), "score": float(p.score)}
            for p in ent_resp.points
        ]
    except Exception as e:
        logger.warning(f"HEFR entity ANN failed (collection may be missing): {e}")
        ent_hits = []

    # If query entities provided by NER, boost their exact matches
    boosted_entity_names = [e["name"] for e in ent_hits]
    for qen in query_entity_names:
        if qen not in boosted_entity_names:
            boosted_entity_names.insert(0, qen)
    boosted_entity_names = boosted_entity_names[:top_entities]

    if not boosted_entity_names:
        return [], []

    # 2. Cypher: entities → chunks
    cypher = """
    UNWIND $names AS qname
    MATCH (e:Entity {tenant_id: $tid})
    WHERE toLower(e.name) = toLower(qname) OR toLower(e.name) CONTAINS toLower(qname)
    MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e)
    WHERE c.tenant_id = $tid
    WITH c, count(DISTINCT e) AS matches, collect(DISTINCT e.name) AS matched_names
    ORDER BY matches DESC
    LIMIT $limit
    RETURN c.id AS chunk_id, c.text AS text, c.source AS source,
           c.format AS format, c.chunk_level AS chunk_level,
           coalesce(c.consistency_score, 0.7) AS consistency,
           matches, matched_names
    """
    async with clients.neo4j.session() as s:
        r = await s.run(cypher, names=boosted_entity_names, tid=tenant_id, limit=top_chunks)
        rows = await r.data()

    chunks = [
        {
            "chunk_id": row["chunk_id"],
            "text": row["text"] or "",
            "source": row["source"] or "unknown",
            "format": row.get("format"),
            "chunk_level": row.get("chunk_level"),
            "consistency_score": float(row["consistency"]),
            "score": float(row["matches"]) / max(len(boosted_entity_names), 1),
            "matched_entities": row["matched_names"],
            "retrieval_path": "hefr:entity_first",
        }
        for row in rows
    ]

    elapsed = time.monotonic() - started
    logger.info(f"HEFR: {len(ent_hits)} entities → {len(chunks)} chunks in {elapsed*1000:.0f}ms")
    return chunks, ent_hits
