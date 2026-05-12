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


# Map intent → preferred views + flags
INTENT_STRATEGY: dict[str, dict[str, Any]] = {
    "factual": {
        "views": ["dense", "keywords"],
        "use_graph": False,
        "use_community": False,
        "format_preference": None,
        "graph_hops": 0,
    },
    "analytical": {
        "views": ["dense", "summary", "question"],
        "use_graph": True,
        "use_community": True,
        "format_preference": None,
        "graph_hops": 2,
    },
    "summarization": {
        "views": ["summary"],
        "use_graph": False,
        "use_community": True,
        "format_preference": None,
        "graph_hops": 1,
    },
    "comparison": {
        "views": ["dense", "question"],
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
    }.get(kind, 1.0)


async def _embed_query(http, embed_url, embed_model, text: str) -> list[float]:
    try:
        return await embed_single(http, embed_url, embed_model, text, timeout=30.0)
    except Exception as e:
        logger.warning(f"Query embed failed for '{text[:60]}': {e}")
        return []


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
) -> list[dict]:
    """
    Weighted RRF fusion across many ranked lists.

    paths = {path_key: [candidate_dicts]}
    candidate dict must have: chunk_id, score, format, chunk_level, consistency_score, retrieval_path

    weight per candidate = path_weight × consistency_factor × level_factor
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
            base = path_weight / (k + rank)
            contribution = base * cs_factor * lvl_factor
            if cid not in fused:
                fused[cid] = {**c, "rrf_score": 0.0, "matched_paths": []}
            fused[cid]["rrf_score"] += contribution
            fused[cid]["matched_paths"].append(path_key)

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

    # Run all paths in parallel
    all_tasks = search_tasks + ([graph_task] if graph_task else []) + ([community_task] if community_task else [])
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
        if not isinstance(res, Exception):
            paths["community"] = res

    # Normalize scores per format BEFORE RRF
    all_cands: list[dict] = []
    for v in paths.values():
        all_cands.extend(v)
    all_cands = normalize_scores_by_format(all_cands)

    # Re-bucket back into paths (preserve normalization)
    norm_map = {c["chunk_id"] + "|" + c.get("retrieval_path", ""): c for c in all_cands}
    for k, lst in paths.items():
        paths[k] = [norm_map.get(c["chunk_id"] + "|" + c.get("retrieval_path", ""), c) for c in lst]

    # Weighted RRF
    fused = weighted_rrf(paths, k=settings.rrf_k, final_top_k=final_top_k)
    return fused
