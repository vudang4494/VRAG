"""VRAG Tier 3: Entity vector centroids for cross-document entity-aware retrieval.

Implements entity-level cosine similarity with supernova / hub-entity defenses:
- L1 (TF-IDF weighting): rare entities boosted, hub entities downweighted
- L3 (MMR): diversity-aware top-K selection
- L5 (Sub-graph Hard Limit): entity search bounded by current top-N chunks scope

Lazy computation: entity centroid = mean of dense embeddings of chunks containing
that entity. No re-ingest needed. Cached in-process for the lifetime of the API.

Used by `_entity_cosine_path` in `retrieval.py` and (optional) ReAct actions.

Trade-offs:
- First query touching a new entity is slow (~200-500ms to fetch chunks + compute)
- Subsequent queries: <5ms cache lookup
- Memory: ~4KB per entity (1024-d float32 vector). 10k entities = 40MB.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import numpy as np
from loguru import logger

# In-process caches. Keys are `f"{tenant}:{entity_name_lower}"`.
_ENTITY_VEC_CACHE: dict[str, np.ndarray] = {}
_ENTITY_VEC_CACHE_TTL: dict[str, float] = {}  # access_time per key
_ENTITY_DOC_COUNT_CACHE: dict[str, int] = {}
_TOTAL_DOCS_BY_TENANT: dict[str, int] = {}
_CACHE_LOCKS: dict[str, asyncio.Lock] = {}

# Cache limits — prevent unbounded growth on the Python heap.
_MAX_ENTITY_CACHE_ENTRIES = 5000
_CACHE_TTL_SECONDS = 1800  # 30 min — stale entries evicted on access


def _key(tenant_id: str, name: str) -> str:
    return f"{tenant_id}:{name.lower().strip()}"


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0 else v / n


# ─── Neo4j queries ────────────────────────────────────────────────────────────


async def _fetch_chunks_for_entity(
    neo4j_driver, entity_name: str, tenant_id: str, top_n: int = 30
) -> list[str]:
    """Return top-N chunk IDs containing this entity (by alias OR normalized name).

    Normalized match strips whitespace + underscore so PDF-extract artifacts
    ("A STUTE RAG", "Plan_RAG") match query entity ("AstuteRAG", "PlanRAG").
    """
    import re as _re

    name_norm = _re.sub(r"[\s_]+", "", entity_name).lower()
    cypher = """
    MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
    WHERE c.tenant_id = $tid
      AND (
        toLower(e.name) = toLower($name)
        OR toLower(replace(replace(e.name, ' ', ''), '_', '')) = $name_norm
        OR EXISTS {
            MATCH (a:Entity)-[:ALIAS_OF]->(e)
            WHERE toLower(a.name) = toLower($name)
        }
      )
    RETURN c.id AS chunk_id
    LIMIT $top_n
    """
    try:
        async with neo4j_driver.session() as s:
            result = await s.run(
                cypher,
                name=entity_name,
                name_norm=name_norm,
                tid=tenant_id,
                top_n=top_n,
            )
            return [r["chunk_id"] for r in await result.data()]
    except Exception as e:
        logger.debug(f"_fetch_chunks_for_entity({entity_name!r}) failed: {e}")
        return []


async def _fetch_dense_vectors(
    qdrant_client, collection: str, chunk_ids: list[str]
) -> list[np.ndarray]:
    """Get `dense` named vectors from Qdrant for the given chunk_ids."""
    from src.services.vector import to_int_id

    if not chunk_ids:
        return []
    point_ids = [to_int_id(cid) for cid in chunk_ids]
    try:
        points = await qdrant_client.retrieve(
            collection_name=collection,
            ids=point_ids,
            with_vectors=["dense"],
        )
        vecs: list[np.ndarray] = []
        for p in points:
            v = p.vector.get("dense") if isinstance(p.vector, dict) else None
            if v:
                vecs.append(np.asarray(v, dtype=np.float32))
        return vecs
    except Exception as e:
        logger.debug(f"_fetch_dense_vectors failed: {e}")
        return []


# ─── Entity vector — lazy centroid ────────────────────────────────────────────


def _evict_stale() -> None:
    """Evict oldest entries when cache exceeds limit or entries are stale."""
    import time as _time

    now = _time.time()
    stale = [k for k, ts in _ENTITY_VEC_CACHE_TTL.items() if now - ts > _CACHE_TTL_SECONDS]
    for k in stale:
        _ENTITY_VEC_CACHE.pop(k, None)
        _ENTITY_VEC_CACHE_TTL.pop(k, None)
    while len(_ENTITY_VEC_CACHE) > _MAX_ENTITY_CACHE_ENTRIES:
        oldest_key = min(_ENTITY_VEC_CACHE_TTL, key=_ENTITY_VEC_CACHE_TTL.get)
        _ENTITY_VEC_CACHE.pop(oldest_key, None)
        _ENTITY_VEC_CACHE_TTL.pop(oldest_key, None)


async def get_entity_vector(
    entity_name: str,
    tenant_id: str,
    neo4j_driver,
    qdrant_client,
    collection: str,
) -> np.ndarray | None:
    """Lazy entity centroid. Caches per (tenant, entity_name) with LRU+TTL eviction."""
    if not entity_name or not entity_name.strip():
        return None
    key = _key(tenant_id, entity_name)

    import time as _time

    now = _time.time()

    if key in _ENTITY_VEC_CACHE:
        _ENTITY_VEC_CACHE_TTL[key] = now  # refresh TTL on access
        return _ENTITY_VEC_CACHE[key]

    lock = _CACHE_LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        if key in _ENTITY_VEC_CACHE:
            _ENTITY_VEC_CACHE_TTL[key] = now
            return _ENTITY_VEC_CACHE[key]
        chunk_ids = await _fetch_chunks_for_entity(neo4j_driver, entity_name, tenant_id, top_n=20)
        if not chunk_ids:
            return None
        vecs = await _fetch_dense_vectors(qdrant_client, collection, chunk_ids)
        if not vecs:
            return None
        centroid = _normalize(np.mean(np.stack(vecs), axis=0))
        _evict_stale()
        _ENTITY_VEC_CACHE[key] = centroid
        _ENTITY_VEC_CACHE_TTL[key] = now
        return centroid


# ─── TF-IDF weighting (L1 supernova guard) ────────────────────────────────────


async def get_entity_doc_count(entity_name: str, tenant_id: str, neo4j_driver) -> int:
    """Number of distinct documents containing this entity."""
    key = _key(tenant_id, entity_name)
    if key in _ENTITY_DOC_COUNT_CACHE:
        return _ENTITY_DOC_COUNT_CACHE[key]
    cypher = """
    MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
    WHERE c.tenant_id = $tid AND toLower(e.name) = toLower($name)
    RETURN COUNT(DISTINCT c.doc_id) AS n
    """
    try:
        async with neo4j_driver.session() as s:
            result = await s.run(cypher, name=entity_name, tid=tenant_id)
            rows = await result.data()
            n = int(rows[0]["n"]) if rows else 0
    except Exception as e:
        logger.debug(f"doc_count failed for {entity_name!r}: {e}")
        n = 0
    _ENTITY_DOC_COUNT_CACHE[key] = n
    return n


async def get_total_docs(tenant_id: str, neo4j_driver) -> int:
    """Total docs in tenant — for IDF normalization."""
    if tenant_id in _TOTAL_DOCS_BY_TENANT:
        return _TOTAL_DOCS_BY_TENANT[tenant_id]
    cypher = "MATCH (c:Chunk {tenant_id: $tid}) RETURN COUNT(DISTINCT c.doc_id) AS n"
    try:
        async with neo4j_driver.session() as s:
            result = await s.run(cypher, tid=tenant_id)
            rows = await result.data()
            n = int(rows[0]["n"]) if rows else 1
    except Exception as e:
        logger.debug(f"get_total_docs failed: {e}")
        n = 1
    _TOTAL_DOCS_BY_TENANT[tenant_id] = max(n, 1)
    return _TOTAL_DOCS_BY_TENANT[tenant_id]


def tf_idf_weight(doc_count: int, total_docs: int) -> float:
    """L1: rare entities boosted, hub entities downweighted.
    Standard IDF: log((N+1) / (df+1)). Capped to [0.1, 5.0] to avoid extremes.
    """
    raw = math.log((total_docs + 1) / (doc_count + 1))
    return max(0.1, min(5.0, raw))


# ─── L5: scope entities by current top chunks ─────────────────────────────────


async def get_entities_in_chunks(
    chunk_ids: list[str], tenant_id: str, neo4j_driver, limit: int = 200
) -> list[str]:
    """Distinct entity names referenced by these chunks."""
    if not chunk_ids:
        return []
    cypher = """
    MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
    WHERE c.id IN $chunk_ids AND c.tenant_id = $tid
    RETURN DISTINCT e.name AS name
    LIMIT $limit
    """
    try:
        async with neo4j_driver.session() as s:
            result = await s.run(cypher, chunk_ids=chunk_ids, tid=tenant_id, limit=limit)
            return [r["name"] for r in await result.data() if r.get("name")]
    except Exception as e:
        logger.debug(f"get_entities_in_chunks failed: {e}")
        return []


# ─── L3: MMR diversity selection ──────────────────────────────────────────────


def mmr_select(
    query_vec: np.ndarray,
    candidates: list[tuple[str, np.ndarray, float]],
    k: int = 20,
    lambda_: float = 0.6,
) -> list[tuple[str, float]]:
    """Maximal Marginal Relevance: pick diverse top-K instead of redundant near-neighbors.
    candidates = list of (name, vec, weighted_cosine_score).
    Returns selected list of (name, final_mmr_score).
    """
    if not candidates:
        return []
    pool = list(candidates)
    selected: list[tuple[str, np.ndarray, float]] = []
    while pool and len(selected) < k:
        if not selected:
            best = max(pool, key=lambda c: c[2])
        else:

            def _mmr(c):
                rel = c[2]
                redundancy = max(float(np.dot(c[1], s[1])) for s in selected)
                return lambda_ * rel - (1 - lambda_) * redundancy

            best = max(pool, key=_mmr)
        selected.append(best)
        pool.remove(best)
    return [(s[0], s[2]) for s in selected]


# ─── Pull chunks that contain selected entities ───────────────────────────────


async def chunks_from_entities(
    entity_names: list[str],
    tenant_id: str,
    neo4j_driver,
    chunk_ids_scope: list[str] | None = None,
    top_k: int = 30,
) -> list[dict]:
    """Return chunks containing any of the given entities, ranked by match count."""
    if not entity_names:
        return []
    import re as _re

    lower_names = [n.lower() for n in entity_names]
    norm_names = [_re.sub(r"[\s_]+", "", n).lower() for n in entity_names]
    cypher_parts = [
        "MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)",
        "WHERE c.tenant_id = $tid",
        "  AND (",
        "    toLower(e.name) IN $names",
        "    OR toLower(replace(replace(e.name, ' ', ''), '_', '')) IN $names_norm",
        "  )",
    ]
    if chunk_ids_scope:
        cypher_parts.append("  AND c.id IN $scope")
    cypher_parts.append("""
    WITH c, count(DISTINCT e) AS matched, collect(DISTINCT e.name) AS ents
    ORDER BY matched DESC
    LIMIT $limit
    RETURN c.id AS chunk_id, c.text AS text, c.source AS source,
           c.format AS format, c.chunk_level AS chunk_level,
           matched, ents
    """)
    cypher = "\n".join(cypher_parts)
    params: dict[str, Any] = {
        "names": lower_names,
        "names_norm": norm_names,
        "tid": tenant_id,
        "limit": top_k,
    }
    if chunk_ids_scope:
        params["scope"] = chunk_ids_scope
    try:
        async with neo4j_driver.session() as s:
            result = await s.run(cypher, **params)
            rows = await result.data()
    except Exception as e:
        logger.debug(f"chunks_from_entities failed: {e}")
        return []
    return [
        {
            "chunk_id": r["chunk_id"],
            "text": r["text"] or "",
            "source": r["source"] or "",
            "format": r["format"] or "",
            "chunk_level": r["chunk_level"] or "",
            "matched_entity_count": r["matched"],
            "matched_entities": r["ents"],
            "score": min(1.0, r["matched"] / max(len(entity_names), 1)),
            "retrieval_path": "entity_cosine",
        }
        for r in rows
    ]


# ─── Main entry: entity-cosine retrieval ──────────────────────────────────────


async def entity_cosine_retrieve(
    query_vec: np.ndarray,
    chunk_ids_scope: list[str],
    tenant_id: str,
    neo4j_driver,
    qdrant_client,
    collection: str,
    top_k_entities: int = 20,
    top_k_chunks: int = 30,
    lambda_mmr: float = 0.6,
) -> tuple[list[dict], list[tuple[str, float]]]:
    """End-to-end entity-cosine retrieval with L1+L3+L5 anti-supernova guards.

    Returns: (chunks, top_entities_with_scores).
    """
    # L5: entity scope from current top-N chunks (bounds the cosine search)
    entity_scope = await get_entities_in_chunks(chunk_ids_scope, tenant_id, neo4j_driver, limit=200)
    if not entity_scope:
        return [], []
    logger.info(
        f"  entity_cosine: scope = {len(entity_scope)} entities (from {len(chunk_ids_scope)} chunks)"
    )

    total_docs = await get_total_docs(tenant_id, neo4j_driver)

    # Compute weighted cosine for each candidate entity (parallel)
    async def _score(name: str) -> tuple[str, np.ndarray, float] | None:
        vec = await get_entity_vector(name, tenant_id, neo4j_driver, qdrant_client, collection)
        if vec is None:
            return None
        cosine = float(np.dot(query_vec, vec))
        # L1: TF-IDF weight (rare = boost, hub = penalize)
        doc_count = await get_entity_doc_count(name, tenant_id, neo4j_driver)
        weight = tf_idf_weight(doc_count, total_docs)
        return (name, vec, cosine * weight)

    results = await asyncio.gather(*[_score(n) for n in entity_scope])
    scored = [r for r in results if r is not None]
    if not scored:
        return [], []

    # L3: MMR diverse top-K
    selected = mmr_select(query_vec, scored, k=top_k_entities, lambda_=lambda_mmr)
    if not selected:
        return [], []
    logger.info(f"  entity_cosine: top-{len(selected)} entities (MMR λ={lambda_mmr})")

    # Pull chunks containing selected entities, scoped to chunk_ids_scope for safety
    entity_names = [s[0] for s in selected]
    chunks = await chunks_from_entities(
        entity_names, tenant_id, neo4j_driver, chunk_ids_scope, top_k=top_k_chunks
    )
    return chunks, selected


def clear_caches() -> None:
    """Clear in-process caches. Call from admin endpoint after ingest."""
    _ENTITY_VEC_CACHE.clear()
    _ENTITY_DOC_COUNT_CACHE.clear()
    _TOTAL_DOCS_BY_TENANT.clear()
    _CACHE_LOCKS.clear()


# ─── Primary entity-gate (replaces lossy doc-gate) ────────────────────────────


async def _seed_chunks_via_dense(
    qdrant_client: Any,
    collection: str,
    query_vec: np.ndarray,
    tenant_id: str,
    limit: int,
) -> list[str]:
    """Quick dense vector seed to bootstrap entity candidate discovery.

    Returns a list of chunk_ids ordered by descending cosine score.
    """
    from qdrant_client import models as qm

    flt = qm.Filter(must=[qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))])
    try:
        resp = await qdrant_client.query_points(
            collection_name=collection,
            query=query_vec.tolist(),
            using="dense",
            limit=limit,
            query_filter=flt,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.warning(f"entity_gate seed dense search failed: {e!r}")
        return []
    return [(p.payload or {}).get("chunk_id", str(p.id)) for p in resp.points if p.payload]


async def entity_cosine_primary(
    query_vec: np.ndarray,
    tenant_id: str,
    neo4j_driver,
    qdrant_client,
    collection: str,
    top_k_entities: int = 50,
    seed_chunks: int = 200,
    score_floor: float = 0.20,
    lambda_mmr: float = 0.6,
    top_k_output_chunks: int = 80,
) -> tuple[list[dict], list[str], list[tuple[str, float]]]:
    """Primary entity-gate retrieval. Returns (path_chunks, scope_chunk_ids, entities).

    Pipeline:
      1. Dense seed: top-`seed_chunks` via raw dense vector search.
      2. Discover candidate entities from seed chunks (≤200 distinct names).
      3. Score each candidate: cosine(query, entity_centroid) × TF-IDF(entity).
      4. MMR diversify → top-`top_k_entities`.
      5. Pull ALL chunks containing those entities, NO clamp (cross-doc scope).
      6. Return RRF-ready path candidates + flat chunk_id list for downstream.

    Floor: if no entity scores above `score_floor`, returns ([], [], []) so
    caller treats as OOD signal and refuses without paying further cost.
    """
    if query_vec is None or query_vec.size == 0:
        return [], [], []

    qnorm = float(np.linalg.norm(query_vec))
    if qnorm > 0:
        query_vec = query_vec / qnorm

    seed_chunk_ids = await _seed_chunks_via_dense(
        qdrant_client, collection, query_vec, tenant_id, limit=seed_chunks
    )
    if not seed_chunk_ids:
        logger.info("entity_gate: dense seed empty — returning [] (OOD)")
        return [], [], []

    candidate_entities = await get_entities_in_chunks(
        seed_chunk_ids, tenant_id, neo4j_driver, limit=200
    )
    if not candidate_entities:
        logger.info("entity_gate: no entities in seed chunks — returning []")
        return [], [], []

    logger.info(
        f"entity_gate: seed={len(seed_chunk_ids)} chunks → "
        f"{len(candidate_entities)} candidate entities"
    )

    total_docs = await get_total_docs(tenant_id, neo4j_driver)

    async def _score(name: str) -> tuple[str, np.ndarray, float] | None:
        vec = await get_entity_vector(name, tenant_id, neo4j_driver, qdrant_client, collection)
        if vec is None:
            return None
        cosine = float(np.dot(query_vec, vec))
        doc_count = await get_entity_doc_count(name, tenant_id, neo4j_driver)
        weight = tf_idf_weight(doc_count, total_docs)
        return (name, vec, cosine * weight)

    scored_raw = await asyncio.gather(*[_score(n) for n in candidate_entities])
    scored = [r for r in scored_raw if r is not None]
    if not scored:
        return [], [], []

    best_score = max(s[2] for s in scored)
    if best_score < score_floor:
        logger.info(f"entity_gate: best entity score {best_score:.3f} < floor {score_floor} → []")
        return [], [], []

    selected = mmr_select(query_vec, scored, k=top_k_entities, lambda_=lambda_mmr)
    if not selected:
        return [], [], []
    logger.info(
        f"entity_gate: top-{len(selected)} entities (MMR λ={lambda_mmr}), "
        f"best_raw_score={best_score:.3f}"
    )

    entity_names = [s[0] for s in selected]
    # No chunk_ids_scope → broader cross-doc fetch.
    chunks = await chunks_from_entities(
        entity_names, tenant_id, neo4j_driver, chunk_ids_scope=None, top_k=top_k_output_chunks
    )
    if not chunks:
        return [], [], selected

    chunk_id_list = [c["chunk_id"] for c in chunks if c.get("chunk_id")]
    # Tag with retrieval_path for RRF + add score field for hard-limit ordering.
    for c in chunks:
        c["retrieval_path"] = "entity_gate"
    return chunks, chunk_id_list, selected
