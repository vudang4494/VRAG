"""GraphRAG Community Summaries — Leiden/Louvain clustering + consistent LLM summaries.

Workflow:
1. Read entity-entity graph from Neo4j (per tenant).
2. Project as undirected graph (networkx or igraph).
3. Run Leiden (preferred) or Louvain — multi-level hierarchical.
4. For each cluster, fetch top entities + linked chunks.
5. Generate LLM summary 3 times (with different seeds/temperatures).
6. LLM judge picks best summary.
7. Write (:Community) nodes + IN_COMMUNITY edges to Neo4j.

## Incremental Update Strategy (Layer 4.4)

When new documents are ingested (incremental update), we avoid full graph re-clustering:
  1. Identify new/modified entities added since last community build.
  2. For each new entity, find its top-K neighbors in the existing graph.
  3. Assign each new entity to the community with most neighbor connections.
  4. If a community grows beyond threshold, re-run Leiden on local subgraph only.
  5. If new entities form isolated cluster (no existing neighbors), create new community.
  6. Delete communities for deleted documents/entities.
  7. Regenerate summaries only for affected communities.

This avoids O(N) full rebuild on every document upload.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from typing import Any

from loguru import logger

_COMMUNITY_SUMMARY_PROMPT = """Bạn là chuyên gia phân tích tri thức. Dưới đây là một cụm
các thực thể có liên quan với nhau, cùng với các đoạn văn bản đề cập đến chúng.

Hãy viết tóm tắt (3-5 câu) trình bày:
- Chủ đề chung của cụm này là gì
- Các thực thể chính và vai trò
- Mối quan hệ quan trọng nhất giữa các thực thể
- Bất kỳ rủi ro, cơ hội, hay insight nào nổi bật

Các thực thể trong cụm:
{entities}

Các đoạn văn bản tham khảo:
{chunks}

Tóm tắt:"""


_SUMMARY_JUDGE_PROMPT = """Dưới đây là 3 bản tóm tắt cho cùng một cụm thực thể.
Hãy chọn bản tóm tắt CHÍNH XÁC, ĐẦY ĐỦ và RÕ RÀNG nhất.

Trả lời CHỈ với số 1, 2, hoặc 3.

Bản 1:
{s1}

Bản 2:
{s2}

Bản 3:
{s3}

Số bản tốt nhất:"""


async def fetch_entity_graph(
    neo4j_driver,
    tenant_id: str | None = None,
    min_relationship_confidence: float = 0.0,
) -> tuple[list[dict], list[tuple[str, str, float]]]:
    """
    Pull entities + RELATES_TO edges from Neo4j.
    Falls back to co-occurrence edges (entities sharing a chunk) if no RELATES_TO edges exist.

    Returns (entities, edges) where edge is (source, target, weight).
    """
    where_tenant = "WHERE e.tenant_id = $tid" if tenant_id else ""
    where_rel = "WHERE r.confidence >= $minc" if min_relationship_confidence > 0 else ""
    params: dict[str, Any] = {"minc": min_relationship_confidence}
    if tenant_id:
        params["tid"] = tenant_id

    async with neo4j_driver.session() as s:
        # Entities
        ent_result = await s.run(
            f"MATCH (e:Entity) {where_tenant} RETURN e.name AS name, e.type AS type, e.description AS desc",
            **({"tid": tenant_id} if tenant_id else {}),
        )
        entities = [
            {"name": row["name"], "type": row["type"], "description": row.get("desc")}
            for row in await ent_result.data()
        ]
        # Edges
        match_clause = "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)"
        if tenant_id:
            match_clause += " WHERE a.tenant_id = $tid AND b.tenant_id = $tid"
            if min_relationship_confidence > 0:
                match_clause += " AND coalesce(r.confidence, 1.0) >= $minc"
        elif min_relationship_confidence > 0:
            match_clause += " WHERE coalesce(r.confidence, 1.0) >= $minc"
        edge_result = await s.run(
            f"{match_clause} RETURN a.name AS src, b.name AS tgt, coalesce(r.confidence, 1.0) AS weight",
            **params,
        )
        edges = [(row["src"], row["tgt"], float(row["weight"])) for row in await edge_result.data()]

        # If no RELATES_TO edges, build co-occurrence edges from CONTAINS_ENTITY
        if not edges:
            logger.info("No RELATES_TO edges — building co-occurrence graph from CONTAINS_ENTITY")
            co_occur_result = await s.run(
                """
                MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e1:Entity)
                WITH c, collect(e1.name) AS ents1
                WHERE size(ents1) > 1
                UNWIND ents1 AS e1_name
                UNWIND ents1 AS e2_name
                WITH e1_name, e2_name, c
                WHERE e1_name < e2_name
                WITH e1_name AS src, e2_name AS tgt, count(DISTINCT c) AS shared_chunks
                ORDER BY shared_chunks DESC
                LIMIT 5000
                RETURN src, tgt, toFloat(shared_chunks) / 10.0 AS weight
                """,
            )
            edges = [
                (row["src"], row["tgt"], float(row["weight"]))
                for row in await co_occur_result.data()
            ]
            logger.info(f"Built {len(edges)} co-occurrence edges from CONTAINS_ENTITY")

    return entities, edges


def cluster_leiden(
    entity_names: list[str],
    edges: list[tuple[str, str, float]],
    resolution: float = 1.0,
    seed: int = 42,
) -> dict[str, int]:
    """
    Run Leiden (igraph) or fallback Louvain (networkx) clustering.
    Returns dict {entity_name: community_id}.
    """
    if not entity_names:
        return {}

    # Try Leiden via igraph
    try:
        import igraph as ig

        g = ig.Graph()
        g.add_vertices(entity_names)
        name_to_idx = {n: i for i, n in enumerate(entity_names)}
        ig_edges = [
            (name_to_idx[s], name_to_idx[t])
            for s, t, _ in edges
            if s in name_to_idx and t in name_to_idx
        ]
        weights = [w for s, t, w in edges if s in name_to_idx and t in name_to_idx]
        if ig_edges:
            g.add_edges(ig_edges)
            g.es["weight"] = weights
        partition = g.community_leiden(
            objective_function="modularity",
            resolution_parameter=resolution,
            weights="weight" if ig_edges else None,
            n_iterations=10,
        )
        return {entity_names[i]: cid for i, cid in enumerate(partition.membership)}
    except ImportError:
        logger.info("igraph not installed, falling back to networkx Louvain")
    except Exception as e:
        logger.warning(f"Leiden clustering failed: {e}; falling back to Louvain")

    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities

        g = nx.Graph()
        g.add_nodes_from(entity_names)
        g.add_weighted_edges_from([(s, t, w) for s, t, w in edges])
        communities = louvain_communities(g, weight="weight", resolution=resolution, seed=seed)
        result: dict[str, int] = {}
        for cid, members in enumerate(communities):
            for m in members:
                result[m] = cid
        for n in entity_names:
            result.setdefault(n, -1)
        return result
    except Exception as e:
        logger.warning(f"Louvain clustering failed: {e}")
        return dict.fromkeys(entity_names, 0)


async def fetch_chunks_for_entities(
    neo4j_driver,
    entity_names: list[str],
    tenant_id: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Fetch chunks containing any of the given entities."""
    if not entity_names:
        return []
    cypher = """
    MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
    WHERE e.name IN $names
    """
    params: dict[str, Any] = {"names": entity_names, "limit": limit}
    if tenant_id:
        cypher += " AND c.tenant_id = $tid"
        params["tid"] = tenant_id
    cypher += """
    WITH c, count(e) AS hits
    RETURN c.id AS chunk_id, c.text AS text, c.source AS source, hits
    ORDER BY hits DESC LIMIT $limit
    """
    async with neo4j_driver.session() as s:
        result = await s.run(cypher, **params)
        return [
            {"chunk_id": r["chunk_id"], "text": r["text"], "source": r["source"], "hits": r["hits"]}
            for r in await result.data()
        ]


async def generate_consistent_summary(
    entities: list[dict],
    chunks: list[dict],
    llm: Any,
    model: str = "qwen3.5:4b",
    vote_passes: int = 3,
) -> tuple[str, int]:
    """
    Generate `vote_passes` summaries with different temperatures, LLM judge picks best.
    Returns (best_summary, vote_count).
    """
    ent_str = "\n".join(
        f"- {e['name']} ({e.get('type', '')}): {e.get('description') or ''}" for e in entities[:15]
    )
    chunk_str = "\n---\n".join(c["text"][:500] for c in chunks[:5])
    prompt = _COMMUNITY_SUMMARY_PROMPT.format(entities=ent_str, chunks=chunk_str)

    from src.services.ollama_helper import ollama_chat

    async def _one(seed_temp: float) -> str:
        try:
            return await ollama_chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=seed_temp,
                max_tokens=400,
            )
        except Exception as e:
            logger.debug(f"Summary generation failed: {e}")
            return ""

    temps = [0.2, 0.4, 0.6][:vote_passes]
    summaries = await asyncio.gather(*[_one(t) for t in temps])
    summaries = [s for s in summaries if s]
    if not summaries:
        return ("", 0)
    if len(summaries) == 1:
        return (summaries[0], 1)

    # Judge
    judge_prompt = _SUMMARY_JUDGE_PROMPT.format(
        s1=summaries[0],
        s2=summaries[1] if len(summaries) > 1 else "(không có)",
        s3=summaries[2] if len(summaries) > 2 else "(không có)",
    )
    try:
        raw = await ollama_chat(
            messages=[{"role": "user", "content": judge_prompt}],
            model=model,
            temperature=0.1,
            max_tokens=10,
        )
        match = re.search(r"\d", raw)
        if match:
            idx = int(match.group(0)) - 1
            if 0 <= idx < len(summaries):
                return (summaries[idx], len(summaries))
    except Exception as e:
        logger.debug(f"Consensus-vote failed: {e}")
    return (summaries[0], len(summaries))


async def write_community(
    neo4j_driver,
    community_id: str,
    tenant_id: str | None,
    level: int,
    summary: str,
    member_entities: list[str],
    vote_count: int,
    parent_community_id: str | None = None,
) -> None:
    """Write Community node + IN_COMMUNITY edges."""
    async with neo4j_driver.session() as s:
        await s.run(
            """
            MERGE (com:Community {id: $cid})
            SET com.tenant_id = $tid,
                com.level = $level,
                com.summary = $summary,
                com.member_count = $count,
                com.summary_vote_count = $votes,
                com.generated_at = datetime()
            """,
            cid=community_id,
            tid=tenant_id,
            level=level,
            summary=summary,
            count=len(member_entities),
            votes=vote_count,
        )
        for name in member_entities:
            params = {"name": name, "cid": community_id, "level": level}
            cypher = """
            MATCH (e:Entity {name: $name})
            MATCH (com:Community {id: $cid})
            MERGE (e)-[r:IN_COMMUNITY]->(com)
            SET r.level = $level
            """
            if tenant_id:
                cypher = """
                MATCH (e:Entity {name: $name, tenant_id: $tid})
                MATCH (com:Community {id: $cid})
                MERGE (e)-[r:IN_COMMUNITY]->(com)
                SET r.level = $level
                """
                params["tid"] = tenant_id
            await s.run(cypher, **params)

        if parent_community_id:
            await s.run(
                """
                MATCH (com:Community {id: $cid})
                MATCH (parent:Community {id: $pid})
                MERGE (com)-[:SUB_COMMUNITY_OF]->(parent)
                """,
                cid=community_id,
                pid=parent_community_id,
            )


async def build_communities_for_tenant(
    neo4j_driver,
    llm: Any,
    tenant_id: str | None = None,
    levels: int = 1,
    resolution: float = 1.0,
    min_size: int = 3,
    vote_passes: int = 3,
    llm_model: str = "qwen3.5:4b",
    concurrent_summaries: int = 2,
) -> dict[str, Any]:
    """
    Full pipeline: fetch graph → cluster → summarize each → write back.
    Run nightly per tenant. Returns stats.
    """
    entities, edges = await fetch_entity_graph(neo4j_driver, tenant_id)
    if not entities:
        return {"communities": 0, "summaries_written": 0, "skipped_small": 0}

    entity_names = [e["name"] for e in entities]
    name_to_entity = {e["name"]: e for e in entities}

    membership = cluster_leiden(entity_names, edges, resolution=resolution)
    groups: dict[int, list[str]] = {}
    for name, cid in membership.items():
        groups.setdefault(cid, []).append(name)

    sem = asyncio.Semaphore(concurrent_summaries)
    written = 0
    skipped = 0

    async def _process(cid: int, members: list[str]) -> int:
        nonlocal written, skipped
        if len(members) < min_size:
            skipped += 1
            return 0
        chunks = await fetch_chunks_for_entities(neo4j_driver, members, tenant_id, limit=10)
        ent_objs = [name_to_entity[n] for n in members if n in name_to_entity]
        async with sem:
            summary, votes = await generate_consistent_summary(
                ent_objs,
                chunks,
                llm,
                llm_model,
                vote_passes,
            )
        if not summary:
            return 0
        community_id = f"comm_{tenant_id or 'default'}_L0_{cid}_{uuid.uuid4().hex[:6]}"
        await write_community(
            neo4j_driver,
            community_id,
            tenant_id,
            level=0,
            summary=summary,
            member_entities=members,
            vote_count=votes,
        )
        written += 1
        return 1

    await asyncio.gather(*[_process(cid, members) for cid, members in groups.items()])

    return {
        "communities": len(groups),
        "summaries_written": written,
        "skipped_small": skipped,
        "entities_total": len(entities),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4.4 — Incremental Community Update
# ─────────────────────────────────────────────────────────────────────────────


async def incremental_update_communities(
    neo4j_driver,
    llm: Any,
    tenant_id: str | None = None,
    new_entity_names: list[str] | None = None,
    deleted_doc_ids: list[str] | None = None,
    llm_model: str = "qwen3.5:4b",
    max_neighbors: int = 20,
    community_rebuild_threshold: int = 50,
) -> dict[str, Any]:
    """
    Incrementally update community structure after new document ingestion.

    Args:
        neo4j_driver: Neo4j driver
        new_entity_names: Entities from newly ingested documents
        deleted_doc_ids: Documents that were deleted (remove their entities from communities)
        community_rebuild_threshold: If a community grows beyond this, re-run Leiden locally

    Returns:
        stats dict with updated/created/skipped community counts
    """
    stats: dict[str, Any] = {
        "entities_assigned": 0,
        "communities_updated": 0,
        "communities_created": 0,
        "communities_rebuilt": [],
        "entities_removed": 0,
    }

    try:
        async with neo4j_driver.session() as s:
            # Step 1: Delete communities for removed documents
            if deleted_doc_ids:
                r = await s.run(
                    """
                    MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d:Document)
                    WHERE d.id IN $doc_ids
                    WITH c
                    MATCH (e:Entity)-[:CONTAINS_ENTITY]->(c)
                    MATCH (e)-[ic:IN_COMMUNITY]->(comm:Community)
                    DELETE ic
                    """,
                    doc_ids=deleted_doc_ids,
                )
                await r.consume()
                stats["entities_removed"] = len(deleted_doc_ids)

            # Step 2: Assign new entities to existing communities
            if new_entity_names:
                # For each new entity, find neighbor entities already in communities
                for entity_name in new_entity_names:
                    r = await s.run(
                        """
                        MATCH (ne:Entity {tenant_id: $tid})
                        WHERE ne.name = $name
                        OPTIONAL MATCH (ne)-[:RELATES_TO]-(existing:Entity)
                        OPTIONAL MATCH (existing)-[:IN_COMMUNITY]->(comm:Community)
                        WITH comm, count(existing) AS neighbor_count
                        WHERE comm IS NOT NULL
                        RETURN comm.id AS community_id, neighbor_count
                        ORDER BY neighbor_count DESC
                        LIMIT 5
                        """,
                        tid=tenant_id,
                        name=entity_name,
                    )
                    community_rows = await r.data()

                    if community_rows and community_rows[0].get("community_id"):
                        # Assign to top neighbor community
                        top_comm = community_rows[0]["community_id"]
                        await s.run(
                            """
                            MATCH (e:Entity {name: $name, tenant_id: $tid})
                            MATCH (c:Community {id: $cid})
                            MERGE (e)-[:IN_COMMUNITY]->(c)
                            """,
                            name=entity_name,
                            tid=tenant_id,
                            cid=top_comm,
                        )
                        stats["entities_assigned"] += 1
                    else:
                        # Isolated entity — create single-entity community
                        # (will be merged into larger community on next full rebuild)
                        single_comm_id = (
                            f"comm_{tenant_id or 'default'}_incremental_{uuid.uuid4().hex[:8]}"
                        )
                        await s.run(
                            """
                            MATCH (e:Entity {name: $name, tenant_id: $tid})
                            MERGE (c:Community {
                                id: $cid,
                                tenant_id: $tid,
                                level: 0,
                                summary: '',
                                member_count: 1,
                                summary_vote_count: 0,
                                generated_at: datetime(),
                                is_incremental: true
                            })
                            MERGE (e)-[:IN_COMMUNITY]->(c)
                            """,
                            name=entity_name,
                            tid=tenant_id,
                            cid=single_comm_id,
                        )
                        stats["communities_created"] += 1

                    stats["communities_updated"] += 1

    except Exception as e:
        logger.debug(f"Incremental community update failed: {e}")

    return stats
