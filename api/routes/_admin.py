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
    """
    GAEA — Graph-Augmented Embedding Aggregation.

    Refines all chunk embeddings for a tenant using entity-neighborhood
    attention. Adds `graph_aware` named vector to Qdrant collection.

    Body: {
      "tenant_id": "eval",
      "alpha": 0.35,            // blend factor (0-1)
      "neighbor_cap": 20,        // max co-mention chunks per chunk
      "batch_size": 50
    }

    Run AFTER ingest + cross_doc build. Idempotent (re-run updates the vector).
    """
    from src.services.graph_embeddings import batch_refine_tenant

    settings_get = __import__("src.config", fromlist=["get_settings"]).get_settings
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    settings = settings_get()
    clients = clients_get()
    tenant_id = body.get("tenant_id") or "default"

    started = time.monotonic()
    result = await batch_refine_tenant(
        clients.neo4j,
        clients.qdrant,
        settings.qdrant_collection,
        tenant_id=tenant_id,
        alpha=float(body.get("alpha", 0.35)),
        neighbor_cap=int(body.get("neighbor_cap", 20)),
        batch_size=int(body.get("batch_size", 50)),
    )
    result["duration_seconds"] = time.monotonic() - started
    return result


# ── HEFR ──────────────────────────────────────────────────────────────────────


@router.post("/hefr/populate", tags=["v3"])
async def hefr_populate(body: dict[str, Any]):
    """Phase 4: populate per-tenant entity Qdrant collection with aggregate embeddings.
    Run after ingest + GAEA. Body: {tenant_id, batch_size}."""
    from src.services.hefr_retrieval import populate_entity_collection

    settings_get = __import__("src.config", fromlist=["get_settings"]).get_settings
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    settings = settings_get()
    clients = clients_get()
    tenant_id = body.get("tenant_id", "default")
    started = time.monotonic()
    result = await populate_entity_collection(
        clients.neo4j,
        clients.qdrant,
        chunk_collection=settings.qdrant_collection,
        tenant_id=tenant_id,
        batch_size=int(body.get("batch_size", 100)),
    )
    result["duration_seconds"] = time.monotonic() - started
    return result


@router.post("/hefr/retrieve", tags=["v3"])
async def hefr_retrieve_endpoint(body: dict[str, Any]):
    """Phase 4: entity-first retrieval. Body: {query, tenant_id, top_entities, top_chunks}."""
    from src.services.embedding import embed_single
    from src.services.hefr_retrieval import hefr_retrieve

    settings_get = __import__("src.config", fromlist=["get_settings"]).get_settings
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    settings = settings_get()
    clients = clients_get()
    query = body.get("query", "")
    tenant_id = body.get("tenant_id", "default")
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'")

    # Embed query + extract entities
    q_vec = await embed_single(
        clients.http,
        settings.ollama_embed_url,
        settings.ollama_embed_model,
        query,
        timeout=30.0,
    )
    q_entities = []
    if getattr(clients, "entity_extractor", None) is not None:
        try:
            ents, _ = await clients.entity_extractor.extract(query)
            q_entities = [e.name for e in ents]
        except Exception:
            pass

    chunks, entities = await hefr_retrieve(
        q_vec,
        q_entities,
        clients,
        settings,
        tenant_id,
        top_entities=int(body.get("top_entities", 20)),
        top_chunks=int(body.get("top_chunks", 30)),
    )
    return {
        "query_entities_extracted": q_entities,
        "top_entities_found": entities[:10],
        "chunks_returned": len(chunks),
        "sample_chunks": chunks[:5],
    }


# ── Cross-doc ─────────────────────────────────────────────────────────────────


@router.post("/cross_doc/build", tags=["v3"])
async def cross_doc_build(body: dict[str, Any]):
    """
    Build cross-document relationships:
      - (:Document)-[:SHARES_ENTITIES]->(:Document)
      - (:Chunk)-[:SIMILAR_TO {cross_doc: true}]->(:Chunk)
      - (:Document)-[:SIMILAR_DOC]->(:Document)
    """
    from src.services.cross_doc import build_cross_doc_graph

    settings_get = __import__("src.config", fromlist=["get_settings"]).get_settings
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    settings = settings_get()
    clients = clients_get()
    tenant_id = body.get("tenant_id") or "default"

    started = time.monotonic()
    result = await build_cross_doc_graph(
        clients.neo4j,
        clients.qdrant,
        settings.qdrant_collection,
        tenant_id=tenant_id,
        sample_chunks=int(body.get("sample_chunks", 500)),
        min_chunk_score=float(body.get("min_chunk_score", 0.75)),
        min_shared_entities=int(body.get("min_shared_entities", 3)),
        min_entity_jaccard=float(body.get("min_entity_jaccard", 0.10)),
        min_chunk_edges_for_doc=int(body.get("min_chunk_edges_for_doc", 5)),
        min_doc_avg_score=float(body.get("min_doc_avg_score", 0.78)),
    )
    result["duration_seconds"] = time.monotonic() - started
    return result


# ── Community ─────────────────────────────────────────────────────────────────


@router.post("/community/build", tags=["v3"])
async def community_build(body: dict[str, Any]):
    """
    Trigger Leiden clustering + LLM summary build for a tenant.
    Body: {"tenant_id": "default", "levels": 1, "resolution": 1.0, "min_size": 3}
    """
    from src.services.community import build_communities_for_tenant

    settings_get = __import__("src.config", fromlist=["get_settings"]).get_settings
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    settings = settings_get()
    clients = clients_get()

    tenant_id = body.get("tenant_id") or "default"
    levels = int(body.get("levels", settings.community_levels))
    resolution = float(body.get("resolution", settings.community_resolution))
    min_size = int(body.get("min_size", settings.community_min_size))
    vote_passes = int(body.get("vote_passes", settings.community_summary_vote_passes))

    started = time.monotonic()
    stats = await build_communities_for_tenant(
        clients.neo4j,
        clients.llm,
        tenant_id=tenant_id,
        levels=levels,
        resolution=resolution,
        min_size=min_size,
        vote_passes=vote_passes,
        llm_model=settings.ollama_model,
    )
    stats["duration_seconds"] = time.monotonic() - started
    return stats


# ── Rerank L2R test ───────────────────────────────────────────────────────────


@router.post("/rerank/l2r/test", tags=["v3"])
async def rerank_l2r_test(body: dict[str, Any]):
    """Test L2R rerank standalone. Body: {query, tenant_id, top_k}."""
    from src.services.query_understanding import understand_query
    from src.services.rerank_l2r import rerank_l2r
    from src.services.retrieval_v2 import multi_path_retrieve

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
    )
    candidates = await multi_path_retrieve(
        understanding,
        clients,
        tenant_id=tenant_id,
        final_top_k=30,
    )
    qe = candidates[0].get("_query_entities", []) if candidates else []

    reranked = await rerank_l2r(query, candidates, query_entities=qe, top_k=top_k)
    return {
        "query": query,
        "top_k": [
            {
                "chunk_id": c.get("chunk_id"),
                "source": c.get("source"),
                "final_score": round(c.get("final_score", 0), 4),
                "l2r_score": round(c.get("l2r_score", 0), 4),
                "stage2_score": round(c.get("stage2_score", 0), 4),
                "features": {k: round(v, 3) for k, v in (c.get("l2r_features") or {}).items()},
            }
            for c in reranked
        ],
    }
