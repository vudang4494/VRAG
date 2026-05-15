"""Retrieval V2 — multi-path retrieval with weighted RRF.

Pipeline:
  query_understanding ─► reformulations + intent
  for each reformulation × view:
      vector_search (Qdrant named vectors)
  + sparse search (BM25 if available)
  + graph traversal (Neo4j entity-based)
  + community search (if intent matches)
  ─► weighted RRF fusion
  ─► return top candidates for reranking
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

import httpx
from loguru import logger

from src.services.embedding import embed_single
from src.services.vector_v2 import (
    build_tenant_filter,
    consistency_factor,
    level_factor,
    normalize_scores_by_format,
    search_multi_view_rrf,
    search_single_view,
)
from src.services.domain_tagger import (
    DOMAIN_AXES,
    domain_reward,
    tag_query,
    DomainDistribution,
)


# Map intent → preferred views + flags.
# "graph_aware" view (Phase 1 GAEA refined) included for all intents — it
# carries cross-chunk context via entity-neighborhood attention.
INTENT_STRATEGY: dict[str, dict[str, Any]] = {
    "factual": {
        "views": ["dense", "graph_aware", "keywords"],
        "use_graph": False,
        "use_community": False,
        "format_preference": None,
        "graph_hops": 0,
    },
    "analytical": {
        "views": ["dense", "graph_aware", "summary", "question"],
        "use_graph": True,
        "use_community": True,
        "format_preference": None,
        "graph_hops": 2,
    },
    "summarization": {
        "views": ["summary", "graph_aware"],
        "use_graph": False,
        "use_community": True,
        "format_preference": None,
        "graph_hops": 1,
    },
    "comparison": {
        "views": ["dense", "graph_aware", "question"],
        "use_graph": True,
        "use_community": False,
        "format_preference": None,
        "graph_hops": 2,
    },
}


def reformulation_weight(kind: str) -> float:
    return {
        "original": 1.0,
        "rewrite": 1.1,
        "hyde": 1.3,
        "step_back": 0.8,
        "keywords": 0.9,
        "decompose": 1.1,
        # Entity-pivot is a high-precision path — boost when entities match.
        # Higher weight means top-matched chunks rank very high in fused output.
        "entity_pivot": 1.5,
        "graph": 1.0,
        "community": 1.2,
    }.get(kind, 1.0)


async def _embed_query(http, embed_url, embed_model, text: str) -> list[float]:
    try:
        return await embed_single(http, embed_url, embed_model, text, timeout=30.0)
    except Exception as e:
        logger.warning(f"Query embed failed for '{text[:60]}': {e}")
        return []


async def _entity_pivot_path(
    neo4j_driver,
    entity_extractor: Any,
    query_text: str,
    tenant_id: str | None,
    top_k: int = 20,
    temporal_filter: str = "",
) -> tuple[list[dict], list[str]]:
    """
    Entity-pivot retrieval — bridge from query to chunks via shared entities.

    This is the path that makes Vector+KG actually deliver value:
      1. Extract entities from the user query (same GLiNER used at ingest).
      2. Cypher: find chunks that CONTAINS_ENTITY any of those entities.
      3. Score by number of matched entities + tenant filter.

    Returns (candidates, query_entity_names) so caller can also include the
    extracted entities in the LLM prompt context.
    """
    if entity_extractor is None or not query_text.strip():
        return [], []

    try:
        ents, _ = await entity_extractor.extract(query_text)
    except Exception as e:
        logger.debug(f"Query entity extraction failed: {e}")
        return [], []

    if not ents:
        return [], []

    # Apply the same normalization as ingest-time entity storage (kg.py _sanitize).
    # "Self-RAG" → "Self_RAG" so Cypher matches the entity node name stored in Neo4j.
    # The raw GLiNER names are NOT applied to Neo4j during ingest — _sanitize is.
    from src.services.kg import _sanitize
    seen: dict[str, str] = {}
    for e in ents:
        key = _sanitize(e.name).strip().lower()
        if key and key not in seen and len(key) >= 2:
            seen[key] = _sanitize(e.name)  # keep original-case for display
    if not seen:
        return [], []
    names_lower = list(seen.keys())
    names_display = [seen[k] for k in names_lower]

    where_clause = "WHERE c.tenant_id = $tid" if tenant_id else ""
    params: dict[str, Any] = {"names": names_lower, "top_k": top_k}
    if tenant_id:
        params["tid"] = tenant_id

    # Match entities: exact case-insensitive OR substring (catches "Leiden" → "Leiden algorithm")
    # temporal_filter from detect_temporal_intent() gets appended to the Entity MATCH
    entity_filter = f" AND ({temporal_filter})" if temporal_filter else ""
    cypher = f"""
    UNWIND $names AS qname
    MATCH (e:Entity)
    WHERE (toLower(e.name) = qname OR toLower(e.name) CONTAINS qname){entity_filter}
    MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e)
    {where_clause}
    WITH c, count(DISTINCT e) AS matches, collect(DISTINCT e.name) AS matched_names
    ORDER BY matches DESC
    LIMIT $top_k
    OPTIONAL MATCH (c)-[:FROM_DOCUMENT]->(d:Document)
    RETURN c.id AS chunk_id, c.text AS text, c.source AS source,
           c.format AS format, c.chunk_level AS chunk_level,
           c.consistency_score AS consistency_score,
           c.domain_distribution AS domain_distribution,
           c.domain_primary AS domain_primary,
           matches, matched_names
    """

    try:
        async with neo4j_driver.session() as s:
            result = await s.run(cypher, **params)
            rows = await result.data()
    except Exception as e:
        logger.warning(f"Entity-pivot Cypher failed: {e}")
        return [], names_display

    candidates: list[dict] = []
    max_possible_matches = max(len(names_lower), 1)
    for r in rows:
        score = float(r["matches"]) / max_possible_matches  # normalized 0..1
        candidates.append({
            "chunk_id": r["chunk_id"],
            "text": r["text"] or "",
            "source": r["source"] or "unknown",
            "format": r["format"] or "unknown",
            "chunk_level": r["chunk_level"] or "paragraph",
            "consistency_score": float(r["consistency_score"] or 0.7),
            "score": score,
            "retrieval_path": "entity_pivot",
            "matched_entities": r["matched_names"] or [],
            "entity_match_count": r["matches"],
            "metadata": {},
            # Phase 8: domain distribution for reward scoring
            "domain_distribution": r.get("domain_distribution") or {},
            "domain_primary": r.get("domain_primary") or "",
        })

    logger.info(
        f"entity_pivot: query entities {names_display} → "
        f"{len(candidates)} chunks (top matches={candidates[0]['entity_match_count'] if candidates else 0})"
    )
    return candidates, names_display


async def _graph_path(neo4j_driver, query_vec, http, embed_url, embed_model, tenant_id, top_k=20):
    """Wrap V1 graph_retrieve with tenant filter."""
    from src.services.kg import graph_retrieve
    try:
        results = await graph_retrieve(
            neo4j_driver, query_vec, http, embed_url, embed_model, top_k=top_k,
        )
        # Add tenant filter post-hoc if needed (graph_retrieve V1 không filter tenant)
        if tenant_id:
            results = [r for r in results if (r.get("metadata") or {}).get("tenant_id") in (None, tenant_id)]
        for r in results:
            r["retrieval_path"] = "graph"
            r.setdefault("format", "graph")
            r.setdefault("chunk_level", "paragraph")
            r.setdefault("consistency_score", 0.7)
            r.setdefault("score", float(r.get("graph_score", 0.0)))
            # Phase 8: domain from payload
            r["domain_distribution"] = r.get("metadata", {}).get("domain_distribution", {})
            r["domain_primary"] = r.get("metadata", {}).get("domain_primary", "")
        return results
    except Exception as e:
        logger.warning(f"Graph path failed: {e}")
        return []


async def _community_path(
    neo4j_driver,
    query_vec,
    http,
    embed_url,
    embed_model,
    tenant_id,
    top_k=5,
) -> list[dict]:
    """
    Vector-search community summaries by embedding similarity.
    Returns up to top_k community summaries as 'chunks' for context.
    """
    if not neo4j_driver:
        return []
    try:
        async with neo4j_driver.session() as s:
            cypher = "MATCH (com:Community) WHERE com.summary IS NOT NULL"
            params: dict[str, Any] = {}
            if tenant_id:
                cypher += " AND com.tenant_id = $tid"
                params["tid"] = tenant_id
            cypher += " RETURN com.id AS id, com.summary AS summary, com.level AS level, com.member_count AS mc LIMIT 200"
            result = await s.run(cypher, **params)
            communities = await result.data()
    except Exception as e:
        logger.debug(f"Community path: fetch failed: {e}")
        return []

    if not communities:
        return []

    from src.services.embedding import embed_batch, cosine_similarity
    summaries = [c["summary"] for c in communities]
    try:
        embeds = await embed_batch(http, embed_url, embed_model, summaries, batch_size=16, timeout=60.0)
    except Exception:
        return []

    scored = [
        (c, cosine_similarity(query_vec, e))
        for c, e in zip(communities, embeds)
        if e and any(v != 0 for v in e)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    out: list[dict] = []
    for com, sim in scored[:top_k]:
        out.append({
            "chunk_id": com["id"],
            "text": com["summary"],
            "source": f"community_L{com['level']}",
            "format": "community",
            "chunk_level": "section",
            "consistency_score": 0.85,
            "score": float(sim),
            "retrieval_path": "community",
            "metadata": {"level": com["level"], "member_count": com["mc"]},
        })
    return out


def weighted_rrf(
    paths: dict[str, list[dict]],
    k: int = 60,
    final_top_k: int = 50,
    query_domain: DomainDistribution | None = None,
    domain_scale: float = 0.3,
) -> list[dict]:
    """
    Weighted RRF fusion across many ranked lists.

    paths = {path_key: [candidate_dicts]}
    candidate dict must have: chunk_id, score, format, chunk_level, consistency_score, retrieval_path

    weight per candidate = path_weight × consistency_factor × level_factor × domain_reward

    Phase 8: domain reward = cosine(chunk_domain_vec, query_domain_vec) × scale
    """
    fused: dict[str, dict] = {}
    for path_key, results in paths.items():
        if not results:
            continue
        # path_key might be "kind:view" or "graph" or "community"; pull path weight
        kind = path_key.split(":", 1)[0] if ":" in path_key else path_key
        path_weight = reformulation_weight(kind)
        for rank, c in enumerate(results, 1):
            cid = c["chunk_id"]
            cs_factor = consistency_factor(c.get("consistency_score", 0.7))
            lvl_factor = level_factor(c.get("chunk_level", "paragraph"))

            # Phase 8: domain reward
            dm_reward = 1.0
            if query_domain is not None:
                chunk_dd = c.get("domain_distribution")
                if chunk_dd:
                    try:
                        chunk_domain = DomainDistribution.from_list([
                            chunk_dd.get(ax, 0.0) for ax in DOMAIN_AXES
                        ])
                        dm_reward = 1.0 + domain_reward(chunk_domain, query_domain, scale=domain_scale)
                    except Exception:
                        dm_reward = 1.0

            base = path_weight / (k + rank)
            contribution = base * cs_factor * lvl_factor * dm_reward
            if cid not in fused:
                fused[cid] = {**c, "rrf_score": 0.0, "matched_paths": [], "domain_reward": 0.0}
            fused[cid]["rrf_score"] += contribution
            fused[cid]["matched_paths"].append(path_key)
            fused[cid]["domain_reward"] = max(fused[cid].get("domain_reward", 0.0), dm_reward - 1.0)

    sorted_results = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_results[:final_top_k]


async def multi_path_retrieve(
    understanding: dict[str, Any],
    clients: Any,
    tenant_id: str | None = None,
    format_filter: list[str] | None = None,
    access_levels: list[str] | None = None,
    top_k_per_path: int = 30,
    final_top_k: int = 50,
) -> list[dict]:
    """
    Run multi-path retrieval based on QueryUnderstanding result.
    """
    from src.config import get_settings
    settings = get_settings()

    intent = understanding.get("intent", "factual")
    strategy = INTENT_STRATEGY.get(intent, INTENT_STRATEGY["factual"])
    views = strategy["views"] or ["dense"]
    reformulations = understanding.get("reformulations", [{"kind": "original", "text": understanding.get("original", ""), "weight": 1.0}])

    # Phase 8: tag query domain for reward scoring
    query_domain = tag_query(understanding.get("original", ""))

    qdrant_filter = build_tenant_filter(
        tenant_id=tenant_id,
        format_in=format_filter,
        access_levels=access_levels,
    )

    # Embed all reformulations in parallel
    embed_tasks = [
        _embed_query(clients.http, settings.ollama_embed_url, settings.ollama_embed_model, r["text"])
        for r in reformulations
    ]
    embeds = await asyncio.gather(*embed_tasks)

    paths: dict[str, list[dict]] = {}

    # Vector search paths: each reformulation × each view
    async def _one_search(reform_kind: str, vec: list[float], view: str) -> tuple[str, list[dict]]:
        if not vec:
            return f"{reform_kind}:{view}", []
        results = await search_single_view(
            clients.qdrant, settings.qdrant_collection, vec, view,
            limit=top_k_per_path, filter_=qdrant_filter,
        )
        return f"{reform_kind}:{view}", results

    search_tasks = []
    for r, vec in zip(reformulations, embeds):
        for v in views:
            search_tasks.append(_one_search(r["kind"], vec, v))

    # Graph path (if intent supports)
    graph_task = None
    if strategy["use_graph"] and embeds and embeds[0]:
        graph_task = _graph_path(
            clients.neo4j, embeds[0], clients.http,
            settings.ollama_embed_url, settings.ollama_embed_model,
            tenant_id, top_k=top_k_per_path,
        )

    # Community path (if intent supports + enabled)
    community_task = None
    if strategy["use_community"] and getattr(settings, "community_enabled", False) and embeds and embeds[0]:
        community_task = _community_path(
            clients.neo4j, embeds[0], clients.http,
            settings.ollama_embed_url, settings.ollama_embed_model,
            tenant_id, top_k=5,
        )

    # Entity-pivot path — always-on when entity_extractor available.
    # Bridges from query→entities→chunks, complements vector cosine.
    # Returns the extracted query entities for later inclusion in context.
    # Phase 6b: detect temporal intent and filter entity lookup by time range.
    from src.services.temporal_entities import detect_temporal_intent
    temporal = detect_temporal_intent(understanding.get("original", ""))
    if temporal.get("type") != "none":
        logger.info(f"  temporal intent: {temporal['type']} — filter: {temporal.get('filter_cypher', '')[:80]}")
    entity_pivot_task = None
    query_entities_holder: dict[str, list[str]] = {"entities": []}
    entity_extractor = getattr(clients, "entity_extractor", None)
    if entity_extractor is not None:
        async def _run_entity_pivot():
            cands, query_ents = await _entity_pivot_path(
                clients.neo4j, entity_extractor,
                understanding.get("original", ""),
                tenant_id, top_k=top_k_per_path,
                temporal_filter=temporal.get("filter_cypher", ""),
            )
            query_entities_holder["entities"] = query_ents
            return cands
        entity_pivot_task = _run_entity_pivot()

    # Run all paths in parallel
    all_tasks = search_tasks \
        + ([graph_task] if graph_task else []) \
        + ([community_task] if community_task else []) \
        + ([entity_pivot_task] if entity_pivot_task else [])
    results = await asyncio.gather(*all_tasks, return_exceptions=True)

    # Collect vector search results
    idx = 0
    for r, _ in zip(reformulations, embeds):
        for v in views:
            res = results[idx]
            idx += 1
            if isinstance(res, Exception):
                continue
            path_key, candidates = res
            paths[path_key] = candidates

    # Graph results
    if graph_task is not None:
        res = results[idx]
        idx += 1
        if not isinstance(res, Exception):
            paths["graph"] = res

    # Community results
    if community_task is not None:
        res = results[idx]
        idx += 1
        if not isinstance(res, Exception):
            paths["community"] = res

    # Entity-pivot results (entity-aware retrieval — KG bridge)
    if entity_pivot_task is not None:
        res = results[idx]
        if not isinstance(res, Exception):
            paths["entity_pivot"] = res

    # Normalize scores per format BEFORE RRF
    all_cands: list[dict] = []
    for v in paths.values():
        all_cands.extend(v)
    all_cands = normalize_scores_by_format(all_cands)

    # Re-bucket back into paths (preserve normalization)
    norm_map = {c["chunk_id"] + "|" + c.get("retrieval_path", ""): c for c in all_cands}
    for k, lst in paths.items():
        paths[k] = [norm_map.get(c["chunk_id"] + "|" + c.get("retrieval_path", ""), c) for c in lst]

    # Weighted RRF — now with domain reward boost (Phase 8)
    fused = weighted_rrf(
        paths, k=settings.rrf_k, final_top_k=final_top_k,
        query_domain=query_domain, domain_scale=0.3,
    )

    # Attach query entities to first result for caller (LLM prompt) to access.
    # Uses a global key on each candidate so it survives downstream processing.
    if fused and query_entities_holder.get("entities"):
        for c in fused:
            c["_query_entities"] = query_entities_holder["entities"]

    return fused
