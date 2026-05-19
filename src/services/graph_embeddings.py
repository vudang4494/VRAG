"""GAEA — Graph-Augmented Embedding Aggregation.

Phase 1 novel contribution: parameter-free graph attention to refine chunk
embeddings using entity neighborhood. No training required.

Why this beats existing GraphRAG papers:
  - Microsoft GraphRAG, LightRAG, KG²RAG, E²GraphRAG all use independent chunk
    embeddings. Cross-chunk relationships exist in graph but not in vectors.
  - GAEA injects neighborhood context into the vector itself, offline.
  - Query-time retrieval via cosine on refined embeddings inherently catches
    cross-chunk semantic links that vanilla retrieval misses.

Algorithm (math):
  For each chunk c:
    E_c = entities mentioned in c (from CONTAINS_ENTITY in Neo4j)
    N_c = up to K chunks sharing ≥1 entity with c (co-mention neighborhood)

    KV = [entity_aggregate_emb for e in E_c] ++ [chunk_emb for c' in N_c]
    scores = (KV @ c.emb) / sqrt(d)
    weights = softmax(scores)
    context = weights @ KV                # neighborhood-weighted aggregate

    c.emb_refined = (1-α) · c.emb + α · context
    c.emb_refined = c.emb_refined / ||·||   # unit-norm for cosine

α tunable (default 0.35). Higher α = more neighborhood influence (may blur);
lower α = closer to original (less context).

Storage: refined embedding upserted to Qdrant as 6th named vector "graph_aware"
in the existing enterprise_kb collection.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import numpy as np
from loguru import logger

# Default tuning parameters
DEFAULT_ALPHA = 0.35
DEFAULT_NEIGHBOR_CAP = 20
DEFAULT_BATCH_SIZE = 50


def _softmax(scores: np.ndarray) -> np.ndarray:
    s = scores - scores.max()
    exp = np.exp(s)
    return exp / exp.sum()


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def graph_attention_refine(
    chunk_emb: np.ndarray,
    neighbor_embeddings: list[np.ndarray],
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:
    """Core math: attend over KV pool, blend with chunk emb, normalize.

    No training, no params — pure dot-product attention + linear interpolation.
    """
    if not neighbor_embeddings:
        return _l2_normalize(chunk_emb)

    kv = np.array(neighbor_embeddings)  # (M, d)
    d = chunk_emb.shape[0]
    scores = (kv @ chunk_emb) / math.sqrt(d)
    weights = _softmax(scores)
    context = weights @ kv  # (d,)
    refined = (1.0 - alpha) * chunk_emb + alpha * context
    return _l2_normalize(refined)


# ── Aggregate entity embedding from chunk mentions ─────────────────────────────


async def aggregate_entity_embedding(
    entity_name: str,
    neo4j_driver,
    qdrant_client,
    collection: str,
    tenant_id: str | None = None,
    cap_mentions: int = 50,
) -> np.ndarray | None:
    """Build a single embedding representing an entity by averaging its mention
    chunk embeddings, weighted by chunk consistency_score.

    Returns None if no mentions found.
    """
    where_t = "AND c.tenant_id = $tid" if tenant_id else ""
    cypher = f"""
    MATCH (e:Entity)
    WHERE toLower(e.name) = toLower($name) {where_t}
    MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e)
    WHERE c.id IS NOT NULL {where_t}
    RETURN c.id AS chunk_id, coalesce(c.consistency_score, 0.7) AS weight
    LIMIT $cap
    """
    params: dict[str, Any] = {"name": entity_name, "cap": cap_mentions}
    if tenant_id:
        params["tid"] = tenant_id

    try:
        async with neo4j_driver.session() as s:
            result = await s.run(cypher, **params)
            rows = await result.data()
    except Exception as e:
        logger.debug(f"Entity emb fetch ({entity_name}) failed: {e}")
        return None

    if not rows:
        return None

    # Fetch embeddings from Qdrant
    from src.services.vector import to_int_id

    point_ids = [to_int_id(r["chunk_id"]) for r in rows]
    weights = np.array([float(r["weight"]) for r in rows])

    try:
        points = await qdrant_client.retrieve(
            collection_name=collection,
            ids=point_ids,
            with_vectors=["dense"],
            with_payload=False,
        )
    except Exception as e:
        logger.debug(f"Qdrant fetch for entity {entity_name} failed: {e}")
        return None

    embs = []
    used_weights = []
    for p, w in zip(points, weights, strict=False):
        vec = (p.vector or {}).get("dense") if isinstance(p.vector, dict) else None
        if vec:
            embs.append(np.array(vec))
            used_weights.append(w)

    if not embs:
        return None

    embs_mat = np.array(embs)
    w = np.array(used_weights)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    aggregate = (w[:, None] * embs_mat).sum(axis=0)
    return _l2_normalize(aggregate)


# ── Neighbor chunks for a given chunk ─────────────────────────────────────────


async def fetch_chunk_neighborhood(
    chunk_id: str,
    neo4j_driver,
    qdrant_client,
    collection: str,
    tenant_id: str | None = None,
    cap_neighbors: int = DEFAULT_NEIGHBOR_CAP,
) -> tuple[list[str], list[np.ndarray]]:
    """Get up to N chunks that share ≥1 entity with this chunk, AND their entities.

    Returns (entity_names, neighbor_chunk_ids) — both lists for KV pool building.
    """
    where_t = "AND c2.tenant_id = $tid" if tenant_id else ""
    cypher = f"""
    MATCH (c:Chunk {{id: $cid}})-[:CONTAINS_ENTITY]->(e:Entity)<-[:CONTAINS_ENTITY]-(c2:Chunk)
    WHERE c2.id <> $cid {where_t}
    WITH c2, count(DISTINCT e) AS shared_count
    ORDER BY shared_count DESC
    LIMIT $cap
    RETURN c2.id AS neighbor_id
    """
    params: dict[str, Any] = {"cid": chunk_id, "cap": cap_neighbors}
    if tenant_id:
        params["tid"] = tenant_id

    try:
        async with neo4j_driver.session() as s:
            result = await s.run(cypher, **params)
            neighbor_ids = [r["neighbor_id"] for r in await result.data()]

            # Get entity names for this chunk
            ent_result = await s.run(
                "MATCH (c:Chunk {id: $cid})-[:CONTAINS_ENTITY]->(e:Entity) RETURN e.name AS name LIMIT 30",
                cid=chunk_id,
            )
            entity_names = [r["name"] for r in await ent_result.data()]
    except Exception as e:
        logger.debug(f"Neighborhood fetch ({chunk_id}) failed: {e}")
        return [], []

    return entity_names, neighbor_ids


async def fetch_chunk_embeddings(
    chunk_ids: list[str],
    qdrant_client,
    collection: str,
) -> dict[str, np.ndarray]:
    """Bulk fetch dense embeddings for chunk_ids."""
    if not chunk_ids:
        return {}
    from src.services.vector import to_int_id

    id_map = {to_int_id(cid): cid for cid in chunk_ids}
    try:
        points = await qdrant_client.retrieve(
            collection_name=collection,
            ids=list(id_map.keys()),
            with_vectors=["dense"],
            with_payload=False,
        )
    except Exception as e:
        logger.debug(f"Bulk chunk emb fetch failed: {e}")
        return {}

    out: dict[str, np.ndarray] = {}
    for p in points:
        vec = (p.vector or {}).get("dense") if isinstance(p.vector, dict) else None
        if vec:
            orig_id = id_map.get(p.id)
            if orig_id:
                out[orig_id] = np.array(vec)
    return out


# ── Main refine pipeline ───────────────────────────────────────────────────────


async def refine_chunk_gaea(
    chunk_id: str,
    chunk_dense_emb: np.ndarray,
    neo4j_driver,
    qdrant_client,
    collection: str,
    entity_emb_cache: dict[str, np.ndarray],
    tenant_id: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    neighbor_cap: int = DEFAULT_NEIGHBOR_CAP,
) -> np.ndarray:
    """Refine ONE chunk's embedding via entity-neighborhood graph attention.

    entity_emb_cache: pre-computed entity aggregate embeddings (entity_name → vec).
                      Pass empty dict if no entities (will fall back to chunk-only).
    """
    entity_names, neighbor_ids = await fetch_chunk_neighborhood(
        chunk_id,
        neo4j_driver,
        qdrant_client,
        collection,
        tenant_id,
        neighbor_cap,
    )

    # Build KV pool: entity aggregates + neighbor chunk embeddings
    kv_pool: list[np.ndarray] = []
    for ent_name in entity_names:
        e_emb = entity_emb_cache.get(ent_name)
        if e_emb is not None:
            kv_pool.append(e_emb)

    if neighbor_ids:
        neighbor_embs = await fetch_chunk_embeddings(neighbor_ids, qdrant_client, collection)
        kv_pool.extend(neighbor_embs.values())

    return graph_attention_refine(chunk_dense_emb, kv_pool, alpha=alpha)


async def build_entity_embedding_cache(
    neo4j_driver,
    qdrant_client,
    collection: str,
    tenant_id: str | None = None,
    max_entities: int = 5000,
) -> dict[str, np.ndarray]:
    """Pre-compute aggregate embedding for every entity in tenant. Cached for batch refine."""
    where_t = "WHERE e.tenant_id = $tid" if tenant_id else ""
    cypher = f"MATCH (e:Entity) {where_t} RETURN e.name AS name LIMIT $cap"
    params: dict[str, Any] = {"cap": max_entities}
    if tenant_id:
        params["tid"] = tenant_id

    async with neo4j_driver.session() as s:
        result = await s.run(cypher, **params)
        entity_names = [r["name"] for r in await result.data()]

    logger.info(f"GAEA: building entity embedding cache for {len(entity_names)} entities")

    cache: dict[str, np.ndarray] = {}
    sem = asyncio.Semaphore(8)

    async def _one(name: str):
        async with sem:
            emb = await aggregate_entity_embedding(
                name,
                neo4j_driver,
                qdrant_client,
                collection,
                tenant_id,
            )
            if emb is not None:
                cache[name] = emb

    await asyncio.gather(*[_one(n) for n in entity_names])
    logger.info(f"GAEA: entity cache built — {len(cache)}/{len(entity_names)} have embeddings")
    return cache


async def batch_refine_tenant(
    neo4j_driver,
    qdrant_client,
    collection: str,
    tenant_id: str,
    alpha: float = DEFAULT_ALPHA,
    neighbor_cap: int = DEFAULT_NEIGHBOR_CAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Run GAEA refinement on all chunks for a tenant.

    Updates Qdrant points with `graph_aware` named vector. Existing other
    named vectors preserved.
    """
    from qdrant_client import models as qm

    from src.services.vector import to_int_id

    # Step 1: build entity cache
    entity_cache = await build_entity_embedding_cache(
        neo4j_driver,
        qdrant_client,
        collection,
        tenant_id,
    )

    # Step 2: list all chunks for tenant
    cypher = """
    MATCH (c:Chunk)
    WHERE c.tenant_id = $tid
    RETURN c.id AS chunk_id
    """
    async with neo4j_driver.session() as s:
        result = await s.run(cypher, tid=tenant_id)
        chunk_ids = [r["chunk_id"] for r in await result.data()]
    logger.info(f"GAEA: refining {len(chunk_ids)} chunks for tenant {tenant_id}")

    refined_count = 0
    error_count = 0

    # Step 3: batch process
    for batch_start in range(0, len(chunk_ids), batch_size):
        batch = chunk_ids[batch_start : batch_start + batch_size]

        # Bulk fetch original dense embeddings for this batch
        dense_embs = await fetch_chunk_embeddings(batch, qdrant_client, collection)

        # Refine each chunk
        updated_points = []
        for cid in batch:
            orig_emb = dense_embs.get(cid)
            if orig_emb is None:
                error_count += 1
                continue
            try:
                refined = await refine_chunk_gaea(
                    cid,
                    orig_emb,
                    neo4j_driver,
                    qdrant_client,
                    collection,
                    entity_cache,
                    tenant_id,
                    alpha,
                    neighbor_cap,
                )
                point_id = to_int_id(cid)
                updated_points.append(
                    qm.PointVectors(
                        id=point_id,
                        vector={"graph_aware": refined.tolist()},
                    )
                )
                refined_count += 1
            except Exception as e:
                logger.debug(f"Refine failed for {cid}: {e}")
                error_count += 1

        # Update Qdrant — add graph_aware vector to existing points
        if updated_points:
            try:
                await qdrant_client.update_vectors(
                    collection_name=collection,
                    points=updated_points,
                )
            except Exception as e:
                logger.warning(f"Qdrant update_vectors batch {batch_start} failed: {e}")
                error_count += len(updated_points)
                refined_count -= len(updated_points)

        logger.info(
            f"GAEA: batch {batch_start // batch_size + 1} done "
            f"({refined_count}/{len(chunk_ids)} refined, {error_count} errors)"
        )

    return {
        "tenant_id": tenant_id,
        "chunks_total": len(chunk_ids),
        "chunks_refined": refined_count,
        "errors": error_count,
        "entity_cache_size": len(entity_cache),
        "alpha": alpha,
        "neighbor_cap": neighbor_cap,
    }
