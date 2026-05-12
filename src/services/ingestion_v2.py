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
    Returns IngestResult-like dict with rich metrics.
    """
    from src.config import get_settings
    settings = get_settings()

    doc_hash = _doc_hash(content)
    doc_id = f"doc_{doc_hash}"
    started = datetime.now(timezone.utc)

    # 1. Format detection + chunking
    fmt, chunk_units = await route_and_chunk(
        content=content,
        filename=filename,
        http_client=clients.http,
        embed_url=settings.ollama_embed_url,
        embed_model=settings.ollama_embed_model,
        section_max_chars=settings.section_max_chars,
        paragraph_max_chars=settings.paragraph_max_chars,
        sentence_max_chars=settings.sentence_max_chars,
        emit_levels=tuple(settings.chunk_levels_enabled),
    )
    if not chunk_units:
        return {"status": "error", "reason": "no_chunks", "filename": filename, "doc_id": doc_id}

    logger.info(f"[V2] {filename}: format={fmt}, chunks={len(chunk_units)}")

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
            chunks, llm=clients.llm, model=settings.ollama_model, use_llm_ner=True,
        )
        logger.info(f"[V2] {filename}: masked {len(mask_map.forward)} PII entities")

    # 4. Consistency Simulation — generate 5 views + embed + score
    chunks = await process_batch_consistency(
        chunks,
        llm=clients.llm,
        http=clients.http,
        embed_url=settings.ollama_embed_url,
        embed_model=settings.ollama_embed_model,
        llm_model=settings.ollama_model,
        concurrent_limit=settings.embed_concurrent_limit,
        enable_llm_views=True,
    )

    # 5. Filter out very low-quality chunks (consistency < low threshold AND not very short)
    pre_filter_avg = sum(c.get("consistency_score", 0.0) for c in chunks) / max(len(chunks), 1)
    chunks_kept = [c for c in chunks if c.get("consistency_score", 0.0) >= settings.consistency_low_threshold or len(c["text"]) < 200]
    dropped = len(chunks) - len(chunks_kept)
    if dropped:
        logger.info(f"[V2] {filename}: dropped {dropped} low-consistency chunks")
    chunks = chunks_kept
    # Average for telemetry — computed on KEPT chunks if any, else pre-filter avg
    avg_consistency = (
        sum(c.get("consistency_score", 0.0) for c in chunks) / len(chunks)
        if chunks else pre_filter_avg
    )

    # 6. Entity voting (3 LLM passes, vote)
    async def _entities_for(chunk):
        return await _vote_entities(
            chunk["text"],
            clients.llm,
            settings.ollama_model,
            passes=settings.entity_vote_passes,
            min_votes=settings.entity_vote_min,
        )

    sem = asyncio.Semaphore(2)  # avoid Ollama saturation
    async def _bounded_entities(chunk):
        async with sem:
            return await _entities_for(chunk)

    entity_results = await asyncio.gather(*[_bounded_entities(c) for c in chunks])

    # 7. Build Qdrant points + Neo4j writes (parallel)
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

    # 8. Cross-chunk SIMILAR_TO links via in-doc cosine
    await _link_in_doc(clients, chunks)

    return {
        "status": "success",
        "filename": filename,
        "doc_id": doc_id,
        "doc_hash": doc_hash,
        "format": fmt,
        "chunks_total": len(chunk_units),
        "chunks_indexed": chunks_indexed,
        "chunks_dropped_low_quality": dropped,
        "avg_consistency_score": avg_consistency,
        "entities_extracted": entities_extracted,
        "relationships_extracted": relationships_extracted,
        "duration_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
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
