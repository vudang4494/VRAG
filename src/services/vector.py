"""Vector V2 — multi-named-vector + sparse upsert/search với format-aware filtering.

Tương thích với Qdrant 1.13+ và schema mới (xem scripts/init-qdrant.sh v2).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm


def to_int_id(chunk_id: str) -> int:
    """Deterministic int from chunk_id (consistent across modules)."""
    return int(hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:15], 16)


def build_tenant_filter(
    tenant_id: str | None,
    format_in: list[str] | None = None,
    chunk_levels: list[str] | None = None,
    access_levels: list[str] | None = None,
    doc_ids: list[str] | None = None,
    extra_must: list[dict] | None = None,
) -> qm.Filter | None:
    """Build a Qdrant Filter with tenant + format + chunk_level + access_level."""
    must: list[qm.FieldCondition] = []
    if tenant_id:
        must.append(qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)))
    if format_in:
        must.append(qm.FieldCondition(key="format", match=qm.MatchAny(any=format_in)))
    if chunk_levels:
        must.append(qm.FieldCondition(key="chunk_level", match=qm.MatchAny(any=chunk_levels)))
    if access_levels:
        must.append(qm.FieldCondition(key="access_level", match=qm.MatchAny(any=access_levels)))
    if doc_ids:
        must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids)))
    if extra_must:
        for e in extra_must:
            must.append(qm.FieldCondition(**e))
    if not must:
        return None
    return qm.Filter(must=must)


async def upsert_v2(
    client: AsyncQdrantClient,
    collection: str,
    points: list[dict],
) -> int:
    """
    Upsert points with 5 named vectors + sparse.

    Each point must have:
      - id (str)
      - view_embeddings: dict[str, list[float]]  with keys subset of
        {dense, paraphrase, question, summary, keywords}
      - sparse: dict {"indices": [...], "values": [...]}  (optional)
      - payload: dict with tenant_id, format, chunk_level, etc.

    Missing dense views default to original 'dense' vector (so all 5 slots filled).
    """
    BATCH_SIZE = 50  # Small batches avoid Qdrant write timeouts

    qdrant_points: list[qm.PointStruct] = []
    for p in points:
        pid = to_int_id(p["id"])
        ve = p.get("view_embeddings") or {}
        dense_vec = ve.get("dense") or ve.get("original") or next(iter(ve.values()), None)
        if not dense_vec:
            logger.warning(f"Point {p['id']} has no dense embedding, skipping")
            continue

        named_vectors: dict[str, Any] = {}
        for view_name in ("dense", "paraphrase", "question", "summary", "keywords"):
            vec = ve.get(view_name)
            if vec:
                named_vectors[view_name] = vec
            elif dense_vec:
                named_vectors[view_name] = dense_vec  # fallback to dense for consistency schema

        if not named_vectors:
            logger.warning(f"Point {p['id']} has no usable embeddings, skipping")
            continue

        if "sparse" in p and p["sparse"]:
            named_vectors["bm25"] = qm.SparseVector(
                indices=p["sparse"]["indices"],
                values=p["sparse"]["values"],
            )

        payload = {**(p.get("payload") or {}), "chunk_id": p["id"]}
        qdrant_points.append(qm.PointStruct(id=pid, vector=named_vectors, payload=payload))

    if not qdrant_points:
        return 0

    # Batch upserts in small chunks; wait=False avoids blocking write timeout.
    total = 0
    for i in range(0, len(qdrant_points), BATCH_SIZE):
        batch = qdrant_points[i : i + BATCH_SIZE]
        try:
            await client.upsert(collection_name=collection, points=batch, wait=False)
            total += len(batch)
        except Exception as exc:
            logger.warning(f"Qdrant batch upsert failed ({len(batch)} points): {exc}")
            raise
    return total


async def search_single_view(
    client: AsyncQdrantClient,
    collection: str,
    query_vector: list[float],
    view: str = "dense",
    limit: int = 30,
    filter_: qm.Filter | None = None,
) -> list[dict]:
    """Search one named vector. Returns standard candidate dicts."""
    try:
        resp = await client.query_points(
            collection_name=collection,
            query=query_vector,
            using=view,
            limit=limit,
            query_filter=filter_,
            with_payload=True,
        )
        results = resp.points
    except Exception as e:
        logger.warning(f"Vector search ({view}) failed: {e}")
        return []

    out: list[dict] = []
    for r in results:
        payload = r.payload or {}
        out.append(
            {
                "chunk_id": payload.get("chunk_id", str(r.id)),
                "text": payload.get("text", ""),
                "source": payload.get("source", "unknown"),
                "format": payload.get("format", "unknown"),
                "chunk_level": payload.get("chunk_level", "paragraph"),
                "consistency_score": float(payload.get("consistency_score", 0.7)),
                "page_num": payload.get("page_num"),
                "sheet_name": payload.get("sheet_name"),
                "thread_id": payload.get("thread_id"),
                "score": float(r.score),
                "retrieval_path": f"vector:{view}",
                "metadata": payload,
                # Phase 8: domain distribution for reward scoring
                "domain_distribution": payload.get("domain_distribution", {}),
                "domain_primary": payload.get("domain_primary", ""),
            }
        )
    return out


async def search_multi_view_rrf(
    client: AsyncQdrantClient,
    collection: str,
    query_vector: list[float],
    views: list[str],
    limit_per_view: int = 50,
    final_limit: int = 50,
    filter_: qm.Filter | None = None,
) -> list[dict]:
    """
    Multi-view search using Qdrant native RRF fusion via prefetch API.
    Faster than fan-out + client-side RRF.
    """
    if not views:
        views = ["dense"]
    try:
        prefetch = [
            qm.Prefetch(query=query_vector, using=v, limit=limit_per_view, filter=filter_)
            for v in views
        ]
        resp = await client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=final_limit,
            query_filter=filter_,
            with_payload=True,
        )
        results = resp.points
    except Exception as e:
        logger.warning(f"Multi-view RRF failed, falling back to single dense: {e}")
        return await search_single_view(
            client, collection, query_vector, "dense", final_limit, filter_
        )

    out: list[dict] = []
    for r in results:
        payload = r.payload or {}
        out.append(
            {
                "chunk_id": payload.get("chunk_id", str(r.id)),
                "text": payload.get("text", ""),
                "source": payload.get("source", "unknown"),
                "format": payload.get("format", "unknown"),
                "chunk_level": payload.get("chunk_level", "paragraph"),
                "consistency_score": float(payload.get("consistency_score", 0.7)),
                "score": float(r.score),
                "retrieval_path": "vector:multi_rrf",
                "metadata": payload,
                "domain_distribution": payload.get("domain_distribution", {}),
                "domain_primary": payload.get("domain_primary", ""),
            }
        )
    return out


async def search_sparse(
    client: AsyncQdrantClient,
    collection: str,
    sparse_indices: list[int],
    sparse_values: list[float],
    limit: int = 30,
    filter_: qm.Filter | None = None,
) -> list[dict]:
    """Sparse vector (BM25-style) search."""
    try:
        resp = await client.query_points(
            collection_name=collection,
            query=qm.SparseVector(indices=sparse_indices, values=sparse_values),
            using="bm25",
            limit=limit,
            query_filter=filter_,
            with_payload=True,
        )
        results = resp.points
    except Exception as e:
        logger.warning(f"Sparse search failed: {e}")
        return []

    out: list[dict] = []
    for r in results:
        payload = r.payload or {}
        out.append(
            {
                "chunk_id": payload.get("chunk_id", str(r.id)),
                "text": payload.get("text", ""),
                "source": payload.get("source", "unknown"),
                "format": payload.get("format", "unknown"),
                "chunk_level": payload.get("chunk_level", "paragraph"),
                "consistency_score": float(payload.get("consistency_score", 0.7)),
                "score": float(r.score),
                "retrieval_path": "sparse:bm25",
                "metadata": payload,
                "domain_distribution": payload.get("domain_distribution", {}),
                "domain_primary": payload.get("domain_primary", ""),
            }
        )
    return out


def normalize_scores_by_format(candidates: list[dict]) -> list[dict]:
    """Z-score normalize within each format group."""
    by_fmt: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_fmt[c.get("format", "unknown")].append(c)
    out: list[dict] = []
    for fmt, lst in by_fmt.items():
        if len(lst) < 2:
            for c in lst:
                c["score_normalized"] = c["score"]
            out.extend(lst)
            continue
        scores = [c["score"] for c in lst]
        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5 or 1.0
        for c in lst:
            c["score_normalized"] = (c["score"] - mean) / std
        out.extend(lst)
    return out


def level_factor(chunk_level: str) -> float:
    return {
        "sentence": 0.8,
        "paragraph": 1.0,
        "section": 1.1,
        "document": 0.7,
    }.get(chunk_level, 1.0)


def consistency_factor(score: float, low: float = 0.6, high: float = 0.85) -> float:
    if score >= high:
        return 1.2
    if score >= low:
        return 1.0
    return 0.8


async def delete_by_doc(
    client: AsyncQdrantClient,
    collection: str,
    tenant_id: str,
    doc_id: str,
) -> None:
    """Delete all points for one document."""
    flt = build_tenant_filter(tenant_id, doc_ids=[doc_id])
    if flt is None:
        return
    await client.delete(collection_name=collection, points_selector=qm.FilterSelector(filter=flt))
