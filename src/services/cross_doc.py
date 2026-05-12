"""Cross-document linker — fill in the Document↔Document and cross-doc Chunk↔Chunk gap.

Current schema has:
  (:Chunk)-[:FROM_DOCUMENT]->(:Document)
  (:Chunk)-[:VARIANT_OF]->(:Chunk)   (hierarchical, same doc)
  (:Chunk)-[:SIMILAR_TO]->(:Chunk)   (within-doc only via _link_in_doc)
  (:Entity)-[:RELATES_TO]->(:Entity)

This module adds:
  (:Chunk)-[:SIMILAR_TO]->(:Chunk)        BUT cross-document (different doc_id)
  (:Document)-[:SHARES_ENTITIES]->(:Document)  shared entity count + Jaccard
  (:Document)-[:SIMILAR_DOC]->(:Document)  based on aggregate chunk similarity

Run via:
  python3 -m src.services.cross_doc --tenant default
or via API:
  POST /api/v3/cross_doc/build  {"tenant_id": "..."}
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from loguru import logger


# ── 1. Document↔Document via shared entities ──────────────────────────────────

async def link_documents_by_entities(
    neo4j_driver,
    tenant_id: str | None = None,
    min_shared: int = 3,
    min_jaccard: float = 0.10,
) -> dict[str, int]:
    """
    Create (:Document)-[:SHARES_ENTITIES {count, jaccard}]->(:Document)
    when two documents share at least `min_shared` entities AND Jaccard ≥ `min_jaccard`.
    """
    where_t = "WHERE c.tenant_id = $tid" if tenant_id else ""
    params: dict[str, Any] = {"min_shared": min_shared, "min_jaccard": min_jaccard}
    if tenant_id:
        params["tid"] = tenant_id

    # Find doc-pair shared entity counts
    cypher = f"""
    MATCH (d1:Document)<-[:FROM_DOCUMENT]-(c1:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
          <-[:CONTAINS_ENTITY]-(c2:Chunk)-[:FROM_DOCUMENT]->(d2:Document)
    {where_t}
    {('AND' if where_t else 'WHERE')} d1.id < d2.id
    WITH d1, d2, count(DISTINCT e) AS shared, collect(DISTINCT e.name) AS shared_names
    WHERE shared >= $min_shared
    RETURN d1.id AS d1_id, d2.id AS d2_id, shared, shared_names
    """
    async with neo4j_driver.session() as s:
        result = await s.run(cypher, **params)
        pairs = await result.data()

    if not pairs:
        return {"pairs_checked": 0, "edges_written": 0}

    # Need per-doc total entities for Jaccard
    doc_entity_count: dict[str, int] = {}
    async with neo4j_driver.session() as s:
        cypher_count = f"""
        MATCH (d:Document)<-[:FROM_DOCUMENT]-(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
        {where_t.replace('c.tenant_id', 'd.tenant_id') if where_t else ''}
        RETURN d.id AS id, count(DISTINCT e) AS cnt
        """
        result = await s.run(cypher_count, **({"tid": tenant_id} if tenant_id else {}))
        for row in await result.data():
            doc_entity_count[row["id"]] = row["cnt"]

    edges = 0
    async with neo4j_driver.session() as s:
        for p in pairs:
            d1_total = doc_entity_count.get(p["d1_id"], 0)
            d2_total = doc_entity_count.get(p["d2_id"], 0)
            union = d1_total + d2_total - p["shared"]
            jaccard = p["shared"] / union if union > 0 else 0.0
            if jaccard < min_jaccard:
                continue
            await s.run(
                """
                MATCH (a:Document {id: $a})
                MATCH (b:Document {id: $b})
                MERGE (a)-[r:SHARES_ENTITIES]->(b)
                SET r.count = $count, r.jaccard = $jacc, r.shared_names = $names, r.updated_at = datetime()
                """,
                a=p["d1_id"], b=p["d2_id"], count=p["shared"], jacc=jaccard,
                names=p["shared_names"][:20],
            )
            # Symmetric edge for easier traversal
            await s.run(
                """
                MATCH (a:Document {id: $a})
                MATCH (b:Document {id: $b})
                MERGE (b)-[r:SHARES_ENTITIES]->(a)
                SET r.count = $count, r.jaccard = $jacc, r.shared_names = $names, r.updated_at = datetime()
                """,
                a=p["d1_id"], b=p["d2_id"], count=p["shared"], jacc=jaccard,
                names=p["shared_names"][:20],
            )
            edges += 2
    logger.info(f"link_documents_by_entities: {edges // 2} doc-pairs linked (Jaccard ≥ {min_jaccard})")
    return {"pairs_checked": len(pairs), "edges_written": edges}


# ── 2. Cross-doc Chunk↔Chunk via cosine ──────────────────────────────────────

async def link_chunks_cross_doc(
    neo4j_driver,
    qdrant_client,
    collection: str,
    tenant_id: str | None = None,
    candidates_per_chunk: int = 5,
    min_score: float = 0.75,
    sample_chunks: int = 500,
) -> dict[str, int]:
    """
    For each chunk (sampled), find top-K most similar chunks in OTHER documents
    via Qdrant vector search, then write SIMILAR_TO edges in Neo4j.

    This complements the in-doc SIMILAR_TO from `_link_in_doc` (ingestion_v2).
    """
    from qdrant_client import models as qm

    # 1. Fetch sample of chunks with their doc_id
    where_t = "WHERE c.tenant_id = $tid" if tenant_id else ""
    cypher = f"""
    MATCH (c:Chunk)
    {where_t}
    RETURN c.id AS chunk_id LIMIT $limit
    """
    async with neo4j_driver.session() as s:
        result = await s.run(cypher, limit=sample_chunks, **({"tid": tenant_id} if tenant_id else {}))
        chunks = [row["chunk_id"] for row in await result.data()]

    if not chunks:
        return {"chunks_processed": 0, "edges_written": 0}

    logger.info(f"link_chunks_cross_doc: processing {len(chunks)} chunks")

    edges_written = 0
    for chunk_id in chunks:
        # Get this chunk's dense vector + doc_id from Qdrant
        try:
            from src.services.vector_v2 import to_int_id
            point_id = to_int_id(chunk_id)
            point = await qdrant_client.retrieve(
                collection_name=collection,
                ids=[point_id],
                with_vectors=["dense"],
                with_payload=True,
            )
            if not point:
                continue
            p = point[0]
            doc_id = (p.payload or {}).get("doc_id")
            vec = (p.vector or {}).get("dense") if isinstance(p.vector, dict) else None
            if not vec or not doc_id:
                continue
        except Exception as e:
            logger.debug(f"Skip chunk {chunk_id}: {e}")
            continue

        # Search for similar chunks in OTHER docs
        try:
            flt = qm.Filter(must_not=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])
            if tenant_id:
                flt = qm.Filter(
                    must=[qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))],
                    must_not=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))],
                )
            resp = await qdrant_client.query_points(
                collection_name=collection,
                query=vec,
                using="dense",
                limit=candidates_per_chunk,
                query_filter=flt,
                with_payload=True,
            )
            similar = resp.points
        except Exception as e:
            logger.debug(f"Search failed for {chunk_id}: {e}")
            continue

        # Write SIMILAR_TO edges
        targets = [
            (
                (s_pt.payload or {}).get("chunk_id"),
                float(s_pt.score),
            )
            for s_pt in similar
            if (s_pt.payload or {}).get("chunk_id") and s_pt.score >= min_score
        ]
        if not targets:
            continue

        async with neo4j_driver.session() as s:
            for target_id, score in targets:
                if target_id == chunk_id:
                    continue
                try:
                    await s.run(
                        """
                        MATCH (a:Chunk {id: $a})
                        MATCH (b:Chunk {id: $b})
                        MERGE (a)-[r:SIMILAR_TO]->(b)
                        SET r.score = $score, r.cross_doc = true, r.view = 'dense', r.updated_at = datetime()
                        """,
                        a=chunk_id, b=target_id, score=score,
                    )
                    edges_written += 1
                except Exception as e:
                    logger.debug(f"Edge write failed {chunk_id}→{target_id}: {e}")

    return {"chunks_processed": len(chunks), "edges_written": edges_written}


# ── 3. Aggregate Document similarity ──────────────────────────────────────────

async def aggregate_document_similarity(
    neo4j_driver,
    tenant_id: str | None = None,
    min_chunk_edges: int = 5,
    min_avg_score: float = 0.78,
) -> dict[str, int]:
    """
    Compute (:Document)-[:SIMILAR_DOC {avg_score, edge_count}]->(:Document)
    by aggregating cross-doc Chunk SIMILAR_TO edges.
    """
    where_t = ""
    params: dict[str, Any] = {"min_edges": min_chunk_edges, "min_avg": min_avg_score}
    if tenant_id:
        where_t = "WHERE c1.tenant_id = $tid AND c2.tenant_id = $tid"
        params["tid"] = tenant_id

    cypher = f"""
    MATCH (d1:Document)<-[:FROM_DOCUMENT]-(c1:Chunk)-[s:SIMILAR_TO {{cross_doc: true}}]->(c2:Chunk)-[:FROM_DOCUMENT]->(d2:Document)
    {where_t}
    {('AND' if where_t else 'WHERE')} d1.id < d2.id
    WITH d1, d2, count(s) AS edges, avg(s.score) AS avg_score
    WHERE edges >= $min_edges AND avg_score >= $min_avg
    RETURN d1.id AS d1_id, d2.id AS d2_id, edges, avg_score
    """
    async with neo4j_driver.session() as s:
        result = await s.run(cypher, **params)
        pairs = await result.data()

    if not pairs:
        return {"pairs": 0, "edges_written": 0}

    written = 0
    async with neo4j_driver.session() as s:
        for p in pairs:
            await s.run(
                """
                MATCH (a:Document {id: $a})
                MATCH (b:Document {id: $b})
                MERGE (a)-[r:SIMILAR_DOC]->(b)
                SET r.avg_score = $avg, r.edge_count = $edges, r.updated_at = datetime()
                """,
                a=p["d1_id"], b=p["d2_id"], avg=float(p["avg_score"]), edges=int(p["edges"]),
            )
            await s.run(
                """
                MATCH (a:Document {id: $a})
                MATCH (b:Document {id: $b})
                MERGE (b)-[r:SIMILAR_DOC]->(a)
                SET r.avg_score = $avg, r.edge_count = $edges, r.updated_at = datetime()
                """,
                a=p["d1_id"], b=p["d2_id"], avg=float(p["avg_score"]), edges=int(p["edges"]),
            )
            written += 2
    logger.info(f"aggregate_document_similarity: {written // 2} doc pairs linked")
    return {"pairs": len(pairs), "edges_written": written}


# ── 4. Orchestrator ───────────────────────────────────────────────────────────

async def build_cross_doc_graph(
    neo4j_driver,
    qdrant_client,
    collection: str,
    tenant_id: str | None = None,
    sample_chunks: int = 500,
    min_chunk_score: float = 0.75,
    min_shared_entities: int = 3,
    min_entity_jaccard: float = 0.10,
    min_chunk_edges_for_doc: int = 5,
    min_doc_avg_score: float = 0.78,
) -> dict[str, Any]:
    """
    Run all 3 cross-doc linking phases in order.
    """
    result = {}
    result["shared_entities"] = await link_documents_by_entities(
        neo4j_driver, tenant_id,
        min_shared=min_shared_entities,
        min_jaccard=min_entity_jaccard,
    )
    result["chunk_cross_doc"] = await link_chunks_cross_doc(
        neo4j_driver, qdrant_client, collection, tenant_id,
        candidates_per_chunk=5,
        min_score=min_chunk_score,
        sample_chunks=sample_chunks,
    )
    result["doc_similarity"] = await aggregate_document_similarity(
        neo4j_driver, tenant_id,
        min_chunk_edges=min_chunk_edges_for_doc,
        min_avg_score=min_doc_avg_score,
    )
    return result
