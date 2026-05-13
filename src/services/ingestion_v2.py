"""Ingestion V2 — orchestrator for the quality-first pipeline.

Flow:
  bytes → format_router → chunker (hierarchical)
        → pii_mask (consistent placeholders)
        → consistency_simulation (5 views + score per chunk)
        → entity voting (3 LLM passes) [optional]
        → parallel:
            ├─ qdrant upsert (5 named vectors + sparse + payload)
            └─ neo4j upsert (Chunk + VARIANT_OF + CONTAINS_ENTITY)
        → link_semantic_chunks (cross-view SIMILAR_TO)
"""
from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from src.services.consistency import process_batch_consistency
from src.services.chunk_quality import filter_chunks_by_quality
from src.services.format_router import route_and_chunk
from src.services.kg import (
    extract_entities_and_relations,
    link_semantic_chunks,
    upsert_chunk_and_entities,
)
from src.services.pii_mask import mask_chunks
from src.services.vector_v2 import upsert_v2


def _doc_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def _chunk_id(doc_id: str, level: str, index: int) -> str:
    return f"{doc_id}::{level}::{index}"


async def _vote_entities(
    text: str,
    llm: Any,
    model: str,
    passes: int = 3,
    min_votes: int = 2,
) -> tuple[list[dict], list[dict]]:
    """Run entity extraction N times, vote on entities/relationships."""
    runs = await asyncio.gather(*[
        extract_entities_and_relations(text, llm, model=model)
        for _ in range(passes)
    ], return_exceptions=True)
    runs = [r for r in runs if isinstance(r, dict)]
    if not runs:
        return [], []

    ent_votes: Counter[str] = Counter()
    ent_meta: dict[str, dict] = {}
    rel_votes: Counter[tuple[str, str]] = Counter()
    rel_meta: dict[tuple[str, str], dict] = {}

    for run in runs:
        for e in run.get("entities", []):
            name = (e.get("name") or "").strip()
            if not name:
                continue
            key = name.lower()
            ent_votes[key] += 1
            # Prefer the most descriptive entry
            existing = ent_meta.get(key)
            if not existing or len(e.get("description", "")) > len(existing.get("description", "")):
                ent_meta[key] = {"name": name, "type": e.get("type", "OTHER"), "description": e.get("description", "")}
        for rel in run.get("relationships", []):
            src = (rel.get("source") or "").strip().lower()
            tgt = (rel.get("target") or "").strip().lower()
            if not src or not tgt:
                continue
            k = (src, tgt)
            rel_votes[k] += 1
            existing = rel_meta.get(k)
            if not existing or len(rel.get("description", "")) > len(existing.get("description", "")):
                rel_meta[k] = {"source": src, "target": tgt, "description": rel.get("description", "")}

    confirmed_entities = []
    for k, v in ent_votes.items():
        if v >= min_votes:
            meta = ent_meta[k]
            confirmed_entities.append({**meta, "confidence": v / passes, "vote_count": v})

    confirmed_rels = []
    for k, v in rel_votes.items():
        if v >= min_votes:
            meta = rel_meta[k]
            confirmed_rels.append({**meta, "confidence": v / passes, "vote_count": v})

    return confirmed_entities, confirmed_rels


async def ingest_document_v2(
    content: bytes,
    filename: str,
    clients: Any,
    tenant_id: str = "default",
    access_level: str = "INTERNAL",
    department: str | None = None,
    author: str | None = None,
    extra_payload: dict | None = None,
) -> dict[str, Any]:
    """
    Full V2 ingest of a single document.
    Returns IngestResult-like dict with rich metrics + per-stage timings (ms).
    """
    import time as _time
    from src.config import get_settings
    settings = get_settings()

    doc_hash = _doc_hash(content)
    doc_id = f"doc_{doc_hash}"
    started = datetime.now(timezone.utc)
    stage_ms: dict[str, float] = {}
    _stage_t0 = _time.monotonic()

    def _mark(stage: str) -> None:
        nonlocal _stage_t0
        now = _time.monotonic()
        stage_ms[stage] = round((now - _stage_t0) * 1000, 1)
        _stage_t0 = now
        logger.info(f"[V2-timing] {filename}: {stage}={stage_ms[stage]:.0f}ms")

    entity_extractor = getattr(clients, "entity_extractor", None)

    # 1. Format detection + doc-type classification → chunking (Phase 5a+5b)
    fmt, chunk_units = await route_and_chunk(
        content=content,
        filename=filename,
        http_client=clients.http,
        embed_url=settings.ollama_embed_url,
        embed_model=settings.ollama_embed_model,
        entity_extractor=entity_extractor,
        section_max_chars=settings.section_max_chars,
        paragraph_max_chars=settings.paragraph_max_chars,
        sentence_max_chars=settings.sentence_max_chars,
        emit_levels=tuple(settings.chunk_levels_enabled),
    )
    if not chunk_units:
        return {"status": "error", "reason": "no_chunks", "filename": filename, "doc_id": doc_id}

    logger.info(f"[V2] {filename}: format={fmt}, chunks={len(chunk_units)}")
    _mark("parse_chunk")

    # 2. Convert to dict-like chunks for the rest of the pipeline
    chunks = []
    for u in chunk_units:
        chunks.append({
            "id": _chunk_id(doc_id, u.chunk_level, u.chunk_index),
            "text": u.text,
            "chunk_index": u.chunk_index,
            "chunk_level": u.chunk_level,
            "parent_chunk_index": u.parent_index,
            "metadata": {**u.metadata, "format": fmt},
        })
    # Build parent_chunk_id mapping using chunk_index → id
    idx_to_id = {c["chunk_index"]: c["id"] for c in chunks}
    for c in chunks:
        pi = c.pop("parent_chunk_index", None)
        c["parent_chunk_id"] = idx_to_id.get(pi) if pi is not None else None

    # 3. PII masking with consistent placeholders
    if settings.pii_mask_enabled:
        chunks, mask_map = await mask_chunks(
            chunks, llm=clients.llm, model=settings.ollama_model,
            use_llm_ner=settings.pii_llm_ner_enabled,
        )
        logger.info(f"[V2] {filename}: masked {len(mask_map.forward)} PII entities")
    _mark("pii_mask")

    # 4. Consistency Simulation — generate views + embed + score (optional)
    chunks = await process_batch_consistency(
        chunks,
        llm=clients.llm,
        http=clients.http,
        embed_url=settings.ollama_embed_url,
        embed_model=settings.ollama_embed_model,
        llm_model=settings.ollama_model,
        concurrent_limit=settings.embed_concurrent_limit,
        enable_llm_views=settings.consistency_views_enabled,
    )
    _mark("consistency_embed")

    # 5. Filter out very low-quality chunks (only when multi-view consistency is on;
    # with views disabled, score is always 0 and would drop everything).
    pre_filter_avg = sum(c.get("consistency_score", 0.0) for c in chunks) / max(len(chunks), 1)
    if settings.consistency_views_enabled:
        chunks_kept = [c for c in chunks if c.get("consistency_score", 0.0) >= settings.consistency_low_threshold or len(c["text"]) < 200]
        dropped = len(chunks) - len(chunks_kept)
        if dropped:
            logger.info(f"[V2] {filename}: dropped {dropped} low-consistency chunks")
        chunks = chunks_kept
    else:
        dropped = 0
    # Average for telemetry — computed on KEPT chunks if any, else pre-filter avg
    avg_consistency = (
        sum(c.get("consistency_score", 0.0) for c in chunks) / len(chunks)
        if chunks else pre_filter_avg
    )

    # 6. Entity extraction — uses SEPARATE entity extractor (GLiNER / API)
    # NOT the semantic LLM. Falls back to old LLM-vote if extractor unavailable.
    entity_extractor = getattr(clients, "entity_extractor", None)

    async def _entities_for(chunk):
        if entity_extractor is not None:
            try:
                ents_obj, rels_obj = await entity_extractor.extract(chunk["text"])
                # Convert to dict format expected by upsert_chunk_and_entities
                ents = [
                    {
                        "name": e.name,
                        "type": e.type,
                        "description": e.description,
                        "confidence": e.confidence,
                    }
                    for e in ents_obj
                ]
                rels = [
                    {
                        "source": r.source,
                        "target": r.target,
                        "description": r.description,
                        "type": r.type,
                        "confidence": r.confidence,
                    }
                    for r in rels_obj
                ]
                return ents, rels
            except Exception as e:
                logger.warning(f"Entity extractor failed, fallback to LLM: {e}")

        # Fallback path: LLM-vote (kept for backward compat with entity_vote_passes>0)
        if settings.entity_vote_passes > 0:
            return await _vote_entities(
                chunk["text"],
                clients.llm,
                settings.ollama_model,
                passes=settings.entity_vote_passes,
                min_votes=settings.entity_vote_min,
            )
        return [], []

    sem = asyncio.Semaphore(4)  # GLiNER is CPU-bound and fast → higher concurrency OK
    async def _bounded_entities(chunk):
        async with sem:
            return await _entities_for(chunk)

    entity_results = await asyncio.gather(*[_bounded_entities(c) for c in chunks])
    _mark("entity_voting")

    # 6b. Phase 6a: CQC chunk quality filter (after entity extraction so we have entity counts)
    entity_counts: dict[str, int] = {
        c["id"]: len(ents) for c, (ents, _) in zip(chunks, entity_results)
    }
    chunks, cqc_rejected = filter_chunks_by_quality(
        chunks, threshold=0.40, entity_counts=entity_counts,
    )
    cqc_dropped = len(cqc_rejected)
    if cqc_dropped:
        logger.info(f"[V2] {filename}: CQC rejected {cqc_dropped} low-quality chunks")
        for rc in cqc_rejected[:3]:
            logger.debug(f"[V2] CQC reject: {rc.get('id')} — {rc.get('quality_reasons', [])[:2]}")
    # Re-align entity_results with filtered chunks
    kept_ids = {c["id"] for c in chunks}
    entity_results = [
        er for er, c in zip(entity_results, chunks)
        if c["id"] in kept_ids
    ]
    _mark("cqc_filter")
    qdrant_points: list[dict] = []
    for c in chunks:
        payload = {
            "tenant_id": tenant_id,
            "doc_id": doc_id,
            "source": filename,
            "text": c["text"],
            "format": fmt,
            "chunk_level": c["chunk_level"],
            "parent_chunk_id": c.get("parent_chunk_id"),
            "consistency_score": float(c.get("consistency_score", 0.0)),
            "access_level": access_level,
            "department": department,
            "author": author,
            "created_at": started.isoformat(),
            **(c.get("metadata") or {}),
            **(extra_payload or {}),
        }
        # Strip None values
        payload = {k: v for k, v in payload.items() if v is not None}
        qdrant_points.append({
            "id": c["id"],
            "view_embeddings": c.get("view_embeddings", {}),
            "payload": payload,
        })

    qdrant_task = upsert_v2(clients.qdrant, settings.qdrant_collection, qdrant_points)

    async def _neo4j_writes():
        total_e = 0
        total_r = 0
        for c, (ents, rels) in zip(chunks, entity_results):
            try:
                await upsert_chunk_and_entities(
                    clients.neo4j,
                    chunk_id=c["id"],
                    text=c["text"],
                    source=filename,
                    metadata={
                        "tenant_id": tenant_id,
                        "doc_id": doc_id,
                        "format": fmt,
                        "chunk_level": c["chunk_level"],
                        "consistency_score": c.get("consistency_score", 0.0),
                        "parent_chunk_id": c.get("parent_chunk_id"),
                    },
                    entities=ents,
                    relationships=rels,
                )
                total_e += len(ents)
                total_r += len(rels)
            except Exception as e:
                logger.warning(f"Neo4j upsert failed for {c['id']}: {e}")
        return total_e, total_r

    chunks_indexed, (entities_extracted, relationships_extracted) = await asyncio.gather(
        qdrant_task, _neo4j_writes(),
    )
    _mark("upsert_qdrant_neo4j")

    # 8. Cross-chunk SIMILAR_TO links via in-doc cosine
    await _link_in_doc(clients, chunks)
    _mark("link_in_doc")

    return {
        "status": "success",
        "filename": filename,
        "doc_id": doc_id,
        "doc_hash": doc_hash,
        "format": fmt,
        "chunks_total": len(chunk_units),
        "chunks_indexed": chunks_indexed,
        "chunks_dropped_low_quality": dropped,
        "chunks_dropped_cqc": cqc_dropped,
        "avg_consistency_score": avg_consistency,
        "entities_extracted": entities_extracted,
        "relationships_extracted": relationships_extracted,
        "duration_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        "stage_ms": stage_ms,
    }


async def _link_in_doc(clients: Any, chunks: list[dict], min_score: float = 0.75) -> None:
    """Link chunks within same doc by 'dense' view cosine similarity."""
    if len(chunks) < 2:
        return
    from src.services.embedding import cosine_similarity

    embeds = []
    ids = []
    for c in chunks:
        emb = c.get("view_embeddings", {}).get("dense") or c.get("view_embeddings", {}).get("original")
        if emb:
            embeds.append(emb)
            ids.append(c["id"])

    for i in range(len(ids)):
        targets = []
        for j in range(len(ids)):
            if i == j:
                continue
            score = cosine_similarity(embeds[i], embeds[j])
            if score >= min_score:
                targets.append((ids[j], score))
        targets.sort(key=lambda x: x[1], reverse=True)
        if targets:
            try:
                await link_semantic_chunks(clients.neo4j, ids[i], targets[:5])
            except Exception as e:
                logger.debug(f"SIMILAR_TO link failed for {ids[i]}: {e}")
