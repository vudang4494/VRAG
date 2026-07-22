"""Admin endpoints — /gaea/refine, /hefr/populate, /hefr/retrieve,
/cross_doc/build, /community/build, /rerank/l2r/test."""

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger  # noqa: F401 — reserved for future per-endpoint logging

from src.clients import get_clients
from src.config import get_settings

router = APIRouter()


# ── GAEA ──────────────────────────────────────────────────────────────────────


@router.post("/gaea/refine", tags=["admin"])
async def gaea_refine(body: dict[str, Any]):
    """Refine all chunk embeddings for a tenant using GAEA.

    Args:
        tenant_id: tenant to refine
        alpha: graph weight (default 0.35)
        neighbor_cap: max neighbors per chunk (default 20)
        batch_size: Qdrant batch size (default 50)
    """
    from src.services.graph_embeddings import batch_refine_tenant

    settings = get_settings()
    clients = get_clients()
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


@router.post("/hefr/populate", tags=["admin"])
async def hefr_populate(body: dict[str, Any]):
    """Populate HEFR entity collection from Neo4j entities.

    Args:
        tenant_id: tenant to populate
        batch_size: batch size (default 100)
    """
    from src.services.hefr import populate_entity_collection

    settings = get_settings()
    clients = get_clients()
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


@router.post("/hefr/retrieve", tags=["admin"])
async def hefr_retrieve(body: dict[str, Any]):
    """HEFR retrieval: query entity collection + pivot to chunks.

    hefr_retrieve is entity-first: it needs a query EMBEDDING and (optionally) query
    entity names, plus the clients/settings objects — not (qdrant, neo4j, query_str).
    The old call passed positional args that did not match the signature at all.
    """
    from src.services.embedding import embed_single
    from src.services.hefr import hefr_retrieve as _hefr_retrieve

    settings = get_settings()
    clients = get_clients()
    tenant = body.get("tenant_id", "default")
    query = body.get("query", "")
    top_k = int(body.get("top_k", 20))
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    query_embedding = await embed_single(
        clients.http, settings.ollama_embed_url, settings.ollama_embed_model, query
    )
    chunks, entities = await _hefr_retrieve(
        query_embedding=query_embedding,
        query_entity_names=body.get("entity_names") or [],
        clients=clients,
        settings=settings,
        tenant_id=tenant,
        top_entities=top_k,
    )
    return {"results": chunks, "entities": entities, "query": query, "tenant_id": tenant}


# ── Cross-Doc ─────────────────────────────────────────────────────────────────


@router.post("/cross_doc/build", tags=["admin"])
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
        aggregate_document_similarity,
        link_chunks_cross_doc,
        link_documents_by_entities,
    )

    settings = get_settings()
    clients = get_clients()
    started = time.monotonic()
    tenant = body.get("tenant_id", "default")

    t0 = time.monotonic()
    r1 = await link_documents_by_entities(
        clients.neo4j,
        tenant,
        min_shared=int(body.get("min_shared", 2)),
        min_jaccard=float(body.get("min_jaccard", 0.05)),
    )
    t1 = time.monotonic()

    r2 = await link_chunks_cross_doc(
        clients.neo4j,
        clients.qdrant,
        settings.qdrant_collection,
        tenant,
        candidates_per_chunk=int(body.get("candidates_per_chunk", 5)),
        min_score=float(body.get("min_score", 0.75)),
        sample_chunks=int(body.get("sample_chunks", 2000)),
    )
    t2 = time.monotonic()

    r3 = await aggregate_document_similarity(
        clients.neo4j,
        tenant,
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


@router.post("/community/build", tags=["admin"])
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

    settings = get_settings()
    clients = get_clients()
    started = time.monotonic()
    tenant = body.get("tenant_id", "default")
    # lazy=true → LazyGraphRAG cluster-only build (no eager LLM summary); defaults
    # to clustering on the NPMI de-hub co-occurrence graph.
    lazy = bool(body.get("lazy", False))
    default_exclude = [
        x.strip() for x in settings.community_exclude_labels_csv.split(",") if x.strip()
    ]
    result = await build_communities_for_tenant(
        clients.neo4j,
        clients.llm,
        tenant,
        levels=int(body.get("levels", 3)),
        resolution=float(body.get("resolution", 1.0)),
        min_size=int(body.get("min_size", 3)),
        vote_passes=int(body.get("vote_passes", 2)),
        llm_model=settings.ollama_model,
        summarize=not lazy,
        use_npmi=bool(body.get("use_npmi", lazy)),
        npmi_min=float(body.get("npmi_min", settings.ppr_npmi_min)),
        exclude_labels=body.get("exclude_labels", default_exclude),
    )
    result["duration_seconds"] = time.monotonic() - started
    return result


@router.post("/entity_resolution/build", tags=["admin"])
async def entity_resolution_build(body: dict[str, Any]):
    """Embedding-confirmed entity resolution soft-fold for a tenant (pick #3).

    Lexical-proposes + centroid-cosine-disposes near-duplicate merge → ALIAS_OF.
    Soft-fold (both nodes persist); PPR/entity_pivot collapse aliases at read time.

    Args:
        tenant_id: tenant to process
        threshold: centroid cosine floor to confirm an alias (default from config)
    """
    from src.services.kg import resolve_entity_aliases

    settings = get_settings()
    clients = get_clients()
    started = time.monotonic()
    tenant = body.get("tenant_id", "default")
    judge_types = [
        x.strip() for x in settings.entity_resolution_judge_types_csv.split(",") if x.strip()
    ]
    result = await resolve_entity_aliases(
        clients.neo4j,
        clients.qdrant,
        settings.qdrant_collection,
        tenant,
        threshold=float(body.get("threshold", settings.entity_resolution_threshold)),
        judge_enabled=bool(body.get("judge", settings.entity_resolution_judge_enabled)),
        judge_hi=float(body.get("judge_hi", settings.entity_resolution_judge_hi)),
        judge_types=body.get("judge_types", judge_types),
        judge_model=settings.light_llm,
    )
    result["duration_seconds"] = time.monotonic() - started
    return result


@router.post("/entity_resolution/audit", tags=["admin"])
async def entity_resolution_audit(body: dict[str, Any]):
    """LLM-judge audit of WRITTEN ALIAS_OF pairs in the cosine gray zone.

    Pairs of judge_types with centroid cosine < judge_hi get a light-LLM
    "same entity? YES/NO"; judged-NO edges are deleted (delete=false = dry-run).
    Vector/LLM errors keep the edge (fail-safe).

    Args:
        tenant_id: tenant to audit
        judge_hi: auto-keep floor (default from config)
        judge_types: entity types to audit (default from config)
        delete: false = report only, no edge deletion (default true)
    """
    from src.services.kg import audit_alias_gray_zone

    settings = get_settings()
    clients = get_clients()
    started = time.monotonic()
    default_types = [
        x.strip() for x in settings.entity_resolution_judge_types_csv.split(",") if x.strip()
    ]
    result = await audit_alias_gray_zone(
        clients.neo4j,
        clients.qdrant,
        settings.qdrant_collection,
        body.get("tenant_id", "default"),
        judge_hi=float(body.get("judge_hi", settings.entity_resolution_judge_hi)),
        judge_types=body.get("judge_types", default_types),
        judge_model=settings.light_llm,
        delete=bool(body.get("delete", True)),
    )
    result["duration_seconds"] = time.monotonic() - started
    return result


# ── Rerank L2R test ───────────────────────────────────────────────────────────


@router.post("/rerank/l2r/test", tags=["admin"])
async def rerank_l2r_test(body: dict[str, Any]):
    from src.services.query_understanding import understand_query
    from src.services.rerank_l2r import rerank_l2r
    from src.services.retrieval import multi_path_retrieve

    settings = get_settings()
    clients = get_clients()
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


# ── Repair Graph ───────────────────────────────────────────────────────────────


@router.post("/repair/build", tags=["admin"])
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

    Body: {
      "tenant_id": "<required>",
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

    from src.services.community import build_communities_for_tenant
    from src.services.cross_doc import (
        aggregate_document_similarity,
        link_chunks_cross_doc,
        link_documents_by_entities,
    )
    from src.services.entity_extractor import create_entity_extractor
    from src.services.graph_embeddings import batch_refine_tenant
    from src.services.hefr import populate_entity_collection
    from src.services.kg import (
        canonicalize_entities,
        delete_orphan_entities,
        upsert_chunk_and_entities,
    )

    settings = get_settings()
    clients = get_clients()

    # tenant_id is required. It used to default to "rag51" — a tenant with zero points,
    # so a caller who forgot the field got a silent no-op against an empty tenant and read
    # it as "the repair pipeline found nothing".
    tenant = body.get("tenant_id")
    if not tenant:
        raise HTTPException(status_code=400, detail="tenant_id is required")

    # NOTE: no batch_size here. The body used to parse one and never use it, so callers
    # tuning it were tuning nothing. Re-add it only together with the code that reads it.
    # Defaults come from settings so backfill produces the SAME cleaned KG as ingest
    # (0.6 threshold, concept/event dropped); body can still override per call.
    _thr = body.get("gliner_threshold")
    gliner_threshold = float(_thr) if _thr is not None else settings.entity_extractor_threshold
    entity_limit = int(body.get("entity_limit", 0))
    _labels = [
        lbl.strip() for lbl in settings.entity_extractor_labels.split(",") if lbl.strip()
    ] or None  # None → DEFAULT_LABELS (concept/event dropped as noise)

    # Init extractor
    extractor = create_entity_extractor(
        provider="gliner",
        model="urchade/gliner_multi-v2.1",
        labels=_labels,
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

    # Step 1.5: clear this tenant's chunk→entity links before re-extraction. A rebuild
    # is a clean slate — otherwise a chunk that now extracts 0 entities (a cleaner
    # extractor dropped its old concept/event spans) keeps those stale links, and the
    # orphan sweep can't reach them (still degree-1). Nodes left unlinked → swept at 4.5.
    async with clients.neo4j.session() as _s:
        await _s.run(
            "MATCH (c:Chunk {tenant_id: $t})-[r:CONTAINS_ENTITY]->() DELETE r",
            t=tenant,
        )
    logger.info("Repair step1.5: cleared existing chunk→entity links")

    # Step 2: Parallel entity extraction
    all_entities: list[tuple] = []
    lock = asyncio.Lock()

    async def _extract_one(cid: str, text: str):
        if not text or len(text.strip()) < 20:
            return
        try:
            ents, _ = await extractor.extract(text)
            if ents:
                records = [
                    (
                        cid,
                        text,
                        {
                            "name": e.name,
                            "type": e.type,
                            "description": e.description,
                            "confidence": e.confidence,
                        },
                    )
                    for e in ents
                ]
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
    for chunk_id, _text, ent in all_entities:
        canon_name = canonical_map.get(ent["name"], ent["name"])
        chunk_entities.setdefault(chunk_id, []).append({**ent, "name": canon_name})
    written = 0
    for chunk_id, entities in chunk_entities.items():
        try:
            await upsert_chunk_and_entities(
                clients.neo4j,
                chunk_id=chunk_id,
                text=next((c[1] for c in chunks if c[0] == chunk_id), ""),
                source=chunk_source.get(chunk_id, "unknown"),
                metadata={"tenant_id": tenant, "format": "pdf"},
                entities=entities,
                relationships=[],
            )
            written += 1
        except Exception as ex:
            logger.debug(f"Upsert failed {chunk_id}: {ex}")
    logger.info(f"Repair step4: wrote {written} chunks with entities")

    # Step 4.5: sweep entities orphaned by the idempotent re-write (old concept/event
    # spans + \n-broken names dropped by the cleaner extractor no longer link anywhere).
    orphaned = await delete_orphan_entities(clients.neo4j, tenant)
    logger.info(f"Repair step4.5: swept {orphaned} orphaned entities")

    # Step 4.6: entity resolution soft-fold (pick #3) — ALIAS_OF near-dups so the
    # downstream steps (SHARES_ENTITIES, community) see collapsed entities. Gated;
    # needs chunks embedded (done above) to compute centroids.
    if settings.entity_resolution_enabled:
        from src.services.kg import resolve_entity_aliases

        er = await resolve_entity_aliases(
            clients.neo4j,
            clients.qdrant,
            settings.qdrant_collection,
            tenant,
            threshold=settings.entity_resolution_threshold,
        )
        logger.info(f"Repair step4.6: entity resolution {er}")

    # Step 5: SHARES_ENTITIES
    shares = await link_documents_by_entities(
        clients.neo4j,
        tenant,
        min_shared=2,
        min_jaccard=0.05,
    )
    logger.info(f"Repair step5: {shares}")

    # Step 6: cross-doc SIMILAR_TO
    cross_doc = await link_chunks_cross_doc(
        clients.neo4j,
        clients.qdrant,
        settings.qdrant_collection,
        tenant,
        candidates_per_chunk=5,
        min_score=0.70,
        sample_chunks=min(len(chunks), 2000),
    )
    logger.info(f"Repair step6: {cross_doc}")

    # Step 7: SIMILAR_DOC
    sim_doc = await aggregate_document_similarity(
        clients.neo4j,
        tenant,
        min_chunk_edges=3,
        min_avg_score=0.70,
    )
    logger.info(f"Repair step7: {sim_doc}")

    # Step 8: GAEA
    gaea = await batch_refine_tenant(
        clients.neo4j,
        clients.qdrant,
        settings.qdrant_collection,
        tenant,
        alpha=0.35,
        neighbor_cap=20,
        batch_size=50,
    )
    logger.info(f"Repair step8: {gaea}")

    # Step 9: Community detection
    community = await build_communities_for_tenant(
        clients.neo4j,
        clients.llm,
        tenant,
        levels=3,
        resolution=1.0,
        min_size=3,
        vote_passes=2,
        llm_model=settings.ollama_model,
    )
    logger.info(f"Repair step9: {community}")

    # Step 10: HEFR
    hefr = await populate_entity_collection(
        clients.neo4j,
        clients.qdrant,
        settings.qdrant_collection,
        tenant,
        batch_size=100,
    )
    logger.info(f"Repair step10: {hefr}")

    return {
        "chunks": len(chunks),
        "entities_extracted": len(all_entities),
        "chunks_with_entities": written,
        "orphans_swept": orphaned,
        "shares_entities": shares,
        "cross_doc": cross_doc,
        "similar_doc": sim_doc,
        "gaea": gaea,
        "community": community,
        "hefr": hefr,
    }


async def _run_with_sem(sem, coro):
    async with sem:
        return await coro
