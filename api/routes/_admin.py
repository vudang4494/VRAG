"""Admin endpoints — /gaea/refine, /hefr/populate, /hefr/retrieve,
/cross_doc/build, /community/build, /rerank/l2r/test."""

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger  # noqa: F401 — reserved for future per-endpoint logging

router = APIRouter()


# ── GAEA ──────────────────────────────────────────────────────────────────────


@router.post("/gaea/refine", tags=["v3"])
async def gaea_refine(body: dict[str, Any]):
    """Refine all chunk embeddings for a tenant using GAEA.

    Args:
        tenant_id: tenant to refine
        alpha: graph weight (default 0.35)
        neighbor_cap: max neighbors per chunk (default 20)
        batch_size: Qdrant batch size (default 50)
    """
    from src.services.graph_embeddings import batch_refine_tenant
    settings_get = __import__("src.config", fromlist=["get_settings"]).get_settings
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    settings = settings_get()
    clients = clients_get()
    started = time.monotonic()
    tenant = body.get("tenant_id", "default")
    alpha = float(body.get("alpha", 0.35))
    neighbor_cap = int(body.get("neighbor_cap", 20))
    batch_size = int(body.get("batch_size", 50))
    result = await batch_refine_tenant(
        clients.neo4j,
        clients.qdrant,
        settings.qdrant_collection,
        tenant,
        alpha=alpha,
        neighbor_cap=neighbor_cap,
        batch_size=batch_size,
    )
    result["duration_seconds"] = time.monotonic() - started
    return result


# ── HEFR ──────────────────────────────────────────────────────────────────────


@router.post("/hefr/populate", tags=["v3"])
async def hefr_populate(body: dict[str, Any]):
    """Populate HEFR entity collection from Neo4j entities.

    Args:
        tenant_id: tenant to populate
        batch_size: batch size (default 100)
    """
    from src.services.hefr import populate_entity_collection
    settings_get = __import__("src.config", fromlist=["get_settings"]).get_settings
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    settings = settings_get()
    clients = clients_get()
    started = time.monotonic()
    tenant = body.get("tenant_id", "default")
    batch_size = int(body.get("batch_size", 100))
    result = await populate_entity_collection(
        clients.neo4j,
        clients.qdrant,
        settings.qdrant_collection,
        tenant,
        batch_size=batch_size,
    )
    result["duration_seconds"] = time.monotonic() - started
    return result


@router.post("/hefr/retrieve", tags=["v3"])
async def hefr_retrieve(body: dict[str, Any]):
    """HEFR retrieval: query entity collection + pivot to chunks."""
    from src.services.hefr import hefr_retrieve
    settings_get = __import__("src.config", fromlist=["get_settings"]).get_settings
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    settings = settings_get()
    clients = clients_get()
    tenant = body.get("tenant_id", "default")
    query = body.get("query", "")
    top_k = int(body.get("top_k", 20))
    result = await hefr_retrieve(
        clients.qdrant,
        clients.neo4j,
        query,
        settings.qdrant_collection,
        tenant,
        top_k=top_k,
    )
    return {"results": result, "query": query, "tenant_id": tenant}


# ── Cross-Doc ─────────────────────────────────────────────────────────────────


@router.post("/cross_doc/build", tags=["v3"])
async def cross_doc_build(body: dict[str, Any]):
    """Build cross-document links for a tenant.

    Runs SHARES_ENTITIES, cross-doc SIMILAR_TO, and SIMILAR_DOC aggregation.

    Args:
        tenant_id: tenant to process
        min_shared: min shared entities for SHARES_ENTITIES (default 2)
        min_jaccard: min Jaccard for SHARES_ENTITIES (default 0.05)
        candidates_per_chunk: candidates per chunk for SIMILAR_TO (default 5)
        min_score: min cosine for SIMILAR_TO (default 0.75)
        sample_chunks: max chunks to sample (default 2000)
        min_chunk_edges: min edges for SIMILAR_DOC (default 5)
        min_avg_score: min avg score for SIMILAR_DOC (default 0.78)
    """
    from src.services.cross_doc import (
        link_documents_by_entities,
        link_chunks_cross_doc,
        aggregate_document_similarity,
    )
    settings_get = __import__("src.config", fromlist=["get_settings"]).get_settings
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    settings = settings_get()
    clients = clients_get()
    started = time.monotonic()
    tenant = body.get("tenant_id", "default")

    t0 = time.monotonic()
    r1 = await link_documents_by_entities(
        clients.neo4j, tenant,
        min_shared=int(body.get("min_shared", 2)),
        min_jaccard=float(body.get("min_jaccard", 0.05)),
    )
    t1 = time.monotonic()

    r2 = await link_chunks_cross_doc(
        clients.neo4j, clients.qdrant, settings.qdrant_collection, tenant,
        candidates_per_chunk=int(body.get("candidates_per_chunk", 5)),
        min_score=float(body.get("min_score", 0.75)),
        sample_chunks=int(body.get("sample_chunks", 2000)),
    )
    t2 = time.monotonic()

    r3 = await aggregate_document_similarity(
        clients.neo4j, tenant,
        min_chunk_edges=int(body.get("min_chunk_edges", 5)),
        min_avg_score=float(body.get("min_avg_score", 0.78)),
    )
    t3 = time.monotonic()

    return {
        "shares_entities": r1,
        "similar_to": r2,
        "similar_doc": r3,
        "timing_seconds": {
            "shares_entities": round(t1 - t0, 1),
            "similar_to": round(t2 - t1, 1),
            "similar_doc": round(t3 - t2, 1),
            "total": round(t3 - t0, 1),
        },
        "duration_seconds": time.monotonic() - started,
    }


# ── Community ──────────────────────────────────────────────────────────────────


@router.post("/community/build", tags=["v3"])
async def community_build(body: dict[str, Any]):
    """Build community detection and summaries for a tenant.

    Args:
        tenant_id: tenant to process
        levels: number of hierarchy levels (default 3)
        resolution: Leiden resolution (default 1.0)
        min_size: min community size (default 3)
        vote_passes: LLM voting passes (default 2)
    """
    from src.services.community import build_communities_for_tenant
    settings_get = __import__("src.config", fromlist=["get_settings"]).get_settings
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    settings = settings_get()
    clients = clients_get()
    started = time.monotonic()
    tenant = body.get("tenant_id", "default")
    result = await build_communities_for_tenant(
        clients.neo4j,
        clients.llm,
        tenant,
        levels=int(body.get("levels", 3)),
        resolution=float(body.get("resolution", 1.0)),
        min_size=int(body.get("min_size", 3)),
        vote_passes=int(body.get("vote_passes", 2)),
        llm_model=settings.ollama_model,
    )
    result["duration_seconds"] = time.monotonic() - started
    return result


# ── Rerank L2R test ───────────────────────────────────────────────────────────


@router.post("/rerank/l2r/test", tags=["v3"])
async def rerank_l2r_test(body: dict[str, Any]):
    from src.services.query_understanding import understand_query
    from src.services.rerank_l2r import rerank_l2r
    from src.services.retrieval import multi_path_retrieve

    settings_get = __import__("src.config", fromlist=["get_settings"]).get_settings
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    settings = settings_get()
    clients = clients_get()
    query = body.get("query", "")
    tenant_id = body.get("tenant_id", "default")
    top_k = int(body.get("top_k", 5))

    understanding = await understand_query(
        query,
        clients.llm,
        model=settings.ollama_model,
        timeout=settings.query_understanding_timeout_s,
        query_type="factual",
    )
    candidates = await multi_path_retrieve(
        understanding,
        clients,
        tenant_id=tenant_id,
        top_k=top_k * 3,
    )
    reranked = await rerank_l2r(
        query,
        candidates,
        clients,
        tenant_id=tenant_id,
        top_k=top_k,
    )
    return {
        "query": query,
        "understanding": understanding,
        "candidates": [
            {
                "id": c.get("id"),
                "text": (c.get("text") or "")[:200],
                "source": c.get("source"),
                "final_score": round(c.get("final_score", 0), 4),
                "l2r_score": round(c.get("l2r_score", 0), 4),
                "stage2_score": round(c.get("stage2_score", 0), 4),
                "features": {k: round(v, 3) for k, v in (c.get("l2r_features") or {}).items()},
            }
            for c in reranked
        ],
    }


# ── Entity-Doc Similarity ──────────────────────────────────────────────────────


@router.post("/entity_doc_similarity/build", tags=["v3"])
async def entity_doc_similarity_build(body: dict[str, Any]):
    """
    Build Entity-Entity cosine similarity for Document similarity.

    Pipeline:
      1. Entity-Entity ANN via Qdrant on entities_<tenant> collection
      2. Document similarity via soft Jaccard on entity similarities
      3. Write SIMILAR_ENTITIES edges to Neo4j

    Run AFTER: GAEA refine + HEFR populate + entity extraction.

    Body: {
      "tenant_id": "rag51",
      "top_similar_entities": 50,
      "min_entity_cosine": 0.80,
      "min_doc_similarity": 0.10
    }
    """
    from src.services.entity_doc_similarity import build_entity_doc_similarity

    settings_get = __import__("src.config", fromlist=["get_settings"]).get_settings
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    settings = settings_get()
    clients = clients_get()
    tenant_id = body.get("tenant_id", "default")

    # Default entity collection name from HEFR
    entity_collection = f"entities_{tenant_id}"

    started = time.monotonic()
    result = await build_entity_doc_similarity(
        clients.neo4j,
        clients.qdrant,
        entity_collection=entity_collection,
        tenant_id=tenant_id,
        top_similar_entities=int(body.get("top_similar_entities", 50)),
        min_entity_cosine=float(body.get("min_entity_cosine", 0.80)),
        min_doc_similarity=float(body.get("min_doc_similarity", 0.10)),
    )
    result["duration_seconds"] = time.monotonic() - started
    return result


# ── Repair Graph ───────────────────────────────────────────────────────────────


@router.post("/repair/build", tags=["v3"])
async def repair_build(body: dict[str, Any]):
    """
    Run full repair pipeline on existing chunks:
      1. Extract entities via GLiNER (parallel, all chunks)
      2. Canonicalize entities (Levenshtein 3-tier)
      3. Write Entity nodes + CONTAINS_ENTITY edges
      4. Build SHARES_ENTITIES
      5. Build cross-doc SIMILAR_TO
      6. Aggregate SIMILAR_DOC
      7. GAEA refinement (graph_aware vector)
      8. Community detection + summaries
      9. HEFR entity collection
     10. Entity-Entity cosine Document similarity

    Body: {
      "tenant_id": "rag51",
      "batch_size": 50,
      "gliner_threshold": 0.5,
      "entity_limit": 0
    }
    """
    started = time.monotonic()
    result = await _run_repair_pipeline(body)
    result["duration_seconds"] = time.monotonic() - started
    return result


async def _run_repair_pipeline(body: dict[str, Any]) -> dict[str, Any]:
    import asyncio
    import os
    os.environ["SKIP_ENTITY_EXTRACTOR"] = "0"

    from src.services.cross_doc import (
        aggregate_document_similarity,
        link_chunks_cross_doc,
        link_documents_by_entities,
    )
    from src.services.entity_doc_similarity import build_entity_doc_similarity
    from src.services.entity_extractor import create_entity_extractor
    from src.services.graph_embeddings import batch_refine_tenant
    from src.services.hefr import populate_entity_collection
    from src.services.community import build_communities_for_tenant
    from src.services.kg import canonicalize_entities, upsert_chunk_and_entities

    settings_get = __import__("src.config", fromlist=["get_settings"]).get_settings
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    settings = settings_get()
    clients = clients_get()

    tenant = body.get("tenant_id", "rag51")
    batch_size = int(body.get("batch_size", 50))
    gliner_threshold = float(body.get("gliner_threshold", 0.5))
    entity_limit = int(body.get("entity_limit", 0))

    # Init extractor
    extractor = create_entity_extractor(
        provider="gliner",
        model="urchade/gliner_multi-v2.1",
        threshold=gliner_threshold,
        llm_for_relations=None,
        relation_model=None,
        extract_relations=False,
    )

    # Step 1: Fetch chunks
    cypher = "MATCH (c:Chunk) WHERE c.tenant_id = $tid"
    if entity_limit > 0:
        cypher += f" LIMIT {entity_limit}"
    cypher += " RETURN c.id AS chunk_id, c.text AS text, c.source AS source"
    async with clients.neo4j.session() as s:
        r = await s.run(cypher, tid=tenant)
        rows = await r.data()
    chunks = [(row["chunk_id"], row["text"], row["source"]) for row in rows]
    logger.info(f"Repair step1: fetched {len(chunks)} chunks")
    if not chunks:
        return {"error": f"No chunks for tenant {tenant}"}

    # Step 2: Parallel entity extraction
    all_entities: list[tuple] = []
    lock = asyncio.Lock()

    async def _extract_one(cid: str, text: str):
        if not text or len(text.strip()) < 20:
            return
        try:
            ents, _ = await extractor.extract(text)
            if ents:
                records = [(cid, text, {
                    "name": e.name, "type": e.type,
                    "description": e.description, "confidence": e.confidence,
                }) for e in ents]
                async with lock:
                    all_entities.extend(records)
        except Exception as ex:
            logger.debug(f"Extraction failed for {cid}: {ex}")

    sem = asyncio.Semaphore(8)
    await asyncio.gather(*[_run_with_sem(sem, _extract_one(c[0], c[1])) for c in chunks])
    logger.info(f"Repair step2: {len(all_entities)} entity mentions extracted")
    if not all_entities:
        return {"error": "No entities extracted — check GLiNER model"}

    # Step 3: Canonicalize
    chunk_source: dict[str, str] = {c[0]: c[2] for c in chunks}
    unique_map: dict[str, dict] = {}
    for _, _, ent in all_entities:
        key = ent["name"].lower().strip()
        if key not in unique_map:
            unique_map[key] = ent
    unique_list = list(unique_map.values())
    canonicalized = await canonicalize_entities(clients.neo4j, unique_list, tenant)
    canonical_map: dict[str, str] = {}
    for orig, canon in zip(unique_list, canonicalized, strict=False):
        if orig["name"] != canon.get("canonical_name"):
            canonical_map[orig["name"]] = canon.get("canonical_name", orig["name"])
    logger.info(f"Repair step3: canonicalized {len(canonical_map)} aliases")

    # Step 4: Write entities to Neo4j
    chunk_entities: dict[str, list[dict]] = {}
    for chunk_id, text, ent in all_entities:
        canon_name = canonical_map.get(ent["name"], ent["name"])
        chunk_entities.setdefault(chunk_id, []).append({**ent, "name": canon_name})
    written = 0
    for chunk_id, entities in chunk_entities.items():
        try:
            await upsert_chunk_and_entities(
                clients.neo4j, chunk_id=chunk_id,
                text=next((c[1] for c in chunks if c[0] == chunk_id), ""),
                source=chunk_source.get(chunk_id, "unknown"),
                metadata={"tenant_id": tenant, "format": "pdf"},
                entities=entities, relationships=[],
            )
            written += 1
        except Exception as ex:
            logger.debug(f"Upsert failed {chunk_id}: {ex}")
    logger.info(f"Repair step4: wrote {written} chunks with entities")

    # Step 5: SHARES_ENTITIES
    shares = await link_documents_by_entities(
        clients.neo4j, tenant, min_shared=2, min_jaccard=0.05,
    )
    logger.info(f"Repair step5: {shares}")

    # Step 6: cross-doc SIMILAR_TO
    cross_doc = await link_chunks_cross_doc(
        clients.neo4j, clients.qdrant, settings.qdrant_collection, tenant,
        candidates_per_chunk=5, min_score=0.70,
        sample_chunks=min(len(chunks), 2000),
    )
    logger.info(f"Repair step6: {cross_doc}")

    # Step 7: SIMILAR_DOC
    sim_doc = await aggregate_document_similarity(
        clients.neo4j, tenant, min_chunk_edges=3, min_avg_score=0.70,
    )
    logger.info(f"Repair step7: {sim_doc}")

    # Step 8: GAEA
    gaea = await batch_refine_tenant(
        clients.neo4j, clients.qdrant, settings.qdrant_collection, tenant,
        alpha=0.35, neighbor_cap=20, batch_size=50,
    )
    logger.info(f"Repair step8: {gaea}")

    # Step 9: Community detection
    community = await build_communities_for_tenant(
        clients.neo4j, clients.llm, tenant,
        levels=3, resolution=1.0, min_size=3, vote_passes=2,
        llm_model=settings.ollama_model,
    )
    logger.info(f"Repair step9: {community}")

    # Step 10: HEFR
    hefr = await populate_entity_collection(
        clients.neo4j, clients.qdrant, settings.qdrant_collection, tenant,
        batch_size=100,
    )
    logger.info(f"Repair step10: {hefr}")

    # Step 11: Entity-Doc similarity
    entity_doc = await build_entity_doc_similarity(
        clients.neo4j, clients.qdrant,
        entity_collection=f"entities_{tenant}",
        tenant_id=tenant,
        top_similar_entities=50, min_entity_cosine=0.80, min_doc_similarity=0.10,
    )
    logger.info(f"Repair step11: {entity_doc}")

    return {
        "chunks": len(chunks),
        "entities_extracted": len(all_entities),
        "chunks_with_entities": written,
        "shares_entities": shares,
        "cross_doc": cross_doc,
        "similar_doc": sim_doc,
        "gaea": gaea,
        "community": community,
        "hefr": hefr,
        "entity_doc_similarity": entity_doc,
    }


async def _run_with_sem(sem, coro):
    async with sem:
        return await coro
