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

## RRF Fusion Algorithm (Layer 7)

RRF Score Formula (per candidate chunk):
  RRF_score = (path_weight × consistency_factor × level_factor × domain_reward) / (k + rank)

Where:
  path_weight         = reformulation_weight(kind) — hand-tuned boost per retrieval path
  consistency_factor  = 1.2 if consistency_score >= 0.85, 1.0 if >= 0.60, else 0.8
  level_factor         = sentence:0.8, paragraph:1.0, section:1.1, document:0.7
  domain_reward       = 1.0 + cosine(chunk_domain, query_domain) × scale (Phase 8)
  k                   = 60 (fixed, Cormack 2009)
  rank                = position in this path's ranked list (1-indexed)

Path weights (hand-tuned):
  entity_pivot: 1.5  (highest — graph-entity match is most precise signal)
  hyde:         1.3  (hypothetical doc captures intent well)
  community:    1.2  (global context boosts coverage)
  rewrite:      1.1  (paraphrase captures query intent)
  decompose:    1.1  (sub-queries cover facets)
  original:     1.0  (baseline)
  keywords:     0.9  (keyword-only less reliable)
  graph:        1.0  (co-occurrence is weak signal alone)
  step_back:    0.8  (abstract query may be too general)

Multiplicative interaction of 4 multipliers: risk of score distortion if any multiplier
is wrong. For production: consider learning weights via LambdaMART (Phase X).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import numpy as np
from loguru import logger

from src.services.domain_tagger import (
    DOMAIN_AXES,
    DomainDistribution,
    domain_reward,
    tag_query,
)
from src.services.embedding import embed_single
from src.services.vector import (
    build_tenant_filter,
    consistency_factor,
    level_factor,
    search_single_view,
)

# Views that are dense-copies by construction: consistency.py only generates
# paraphrase+summary distinctly; the vector store backfills question/keywords with
# the dense vector. Searching them double-counts dense in RRF. Gated-dropped via
# settings.retrieval_real_views_only.
_DENSE_DUPLICATE_VIEWS = frozenset({"question", "keywords"})

# Map intent → preferred views + flags.
# "graph_aware" view (Phase 1 GAEA refined) included for all intents — it
# carries cross-chunk context via entity-neighborhood attention.
INTENT_STRATEGY: dict[str, dict[str, Any]] = {
    "factual": {
        "views": ["dense", "graph_aware", "keywords"],
        "use_graph": False,
        "use_community": False,
        "use_entity_pivot": False,
        "format_preference": None,
        "graph_hops": 0,
    },
    "analytical": {
        "views": ["dense", "graph_aware", "summary", "question"],
        "use_graph": True,
        "use_community": True,
        "use_entity_pivot": True,
        "format_preference": None,
        "graph_hops": 2,
    },
    "summarization": {
        "views": ["summary", "graph_aware", "dense"],
        "use_graph": False,
        "use_community": True,
        "use_entity_pivot": False,
        "format_preference": None,
        "graph_hops": 1,
        "final_top_k_multiplier": 2,
    },
    "comparison": {
        "views": ["dense", "graph_aware", "question"],
        "use_graph": True,
        "use_community": False,
        "use_entity_pivot": True,
        "format_preference": None,
        "graph_hops": 2,
    },
    "multi_hop": {
        "views": ["dense", "graph_aware", "question"],
        "use_graph": True,
        "use_community": True,
        "use_entity_pivot": True,
        "format_preference": None,
        "graph_hops": 2,
    },
    "kg_construction": {
        "views": ["dense", "keywords", "summary"],
        "use_graph": True,
        "use_community": True,
        "use_entity_pivot": True,
        "format_preference": None,
        "graph_hops": 2,
    },
}


# KG retrieval paths whose RRF weight is scaled by rrf_kg_path_weight_scale.
# These are the entity/graph-derived paths (vs the dense/sparse reformulation
# kinds).
_KG_RRF_PATHS = frozenset(
    {"entity_pivot", "graph", "community", "entity_cosine", "ppr", "entity_gate"}
)


def reformulation_weight(kind: str) -> float:
    base = {
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
        # Tier 3: entity_cosine — cross-doc entity-aware retrieval with
        # TF-IDF + MMR. Slightly higher than entity_pivot because L1 weighting
        # already filters hub-entity noise, leaving high-precision signals.
        "entity_cosine": 1.6,
        # Phase 2.1: HippoRAG 2 PPR — best signal for multi-hop because the
        # random walk surfaces destination entities the single-shot paths miss.
        "ppr": 1.7,
        # Primary entity-gate: cross-doc cosine via entity centroids, run first.
        "entity_gate": 1.8,
    }.get(kind, 1.0)
    # De-tune KG paths per config: at 1.0× these >dense weights net-hurt recall@5
    # (they evict dense chunks from the top-5). ~0.2× keeps the recall@1/MRR gain
    # without the recall@5 loss. Called once per path in weighted_rrf, so the
    # cached settings lookup is negligible.
    if kind in _KG_RRF_PATHS:
        from src.config import get_settings

        base *= get_settings().rrf_kg_path_weight_scale
    return base


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
    chunk_ids_filter: list[str] | None = None,
    pre_extracted_entities: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """
    Entity-pivot retrieval — bridge from query to chunks via shared entities.

    This is the path that makes Vector+KG actually deliver value:
      1. Extract entities from the user query (same GLiNER used at ingest).
         Uses pre_extracted_entities if provided (from Step 2 GLiNER extraction),
         otherwise falls back to calling the extractor directly.
      2. Cypher: find chunks that CONTAINS_ENTITY any of those entities.
         Includes alias resolution (ALIAS_OF edges).
      3. Score by number of matched entities + tenant filter.
      4. Optional: apply chunk_ids_filter for hard limit (supernode protection).

    Returns (candidates, query_entity_names) so caller can also include the
    extracted entities in the LLM prompt context.
    """
    if not query_text.strip():
        return [], []

    # Use pre-extracted entities if available (Step 2 GLiNER), otherwise extract now
    if pre_extracted_entities:
        from src.services.kg import _sanitize

        seen: dict[str, str] = {}
        for name in pre_extracted_entities:
            key = _sanitize(name).strip().lower()
            if key and len(key) >= 2 and key not in seen:
                seen[key] = _sanitize(name)
        if not seen:
            return [], []
        names_lower = list(seen.keys())
        names_display = [seen[k] for k in names_lower]
    elif entity_extractor is not None:
        try:
            ents, _ = await entity_extractor.extract(query_text)
        except Exception as e:
            logger.debug(f"Query entity extraction failed: {e}")
            return [], []

        if not ents:
            return [], []

        # Apply the same normalization as ingest-time entity storage (kg.py _sanitize).
        from src.services.kg import _sanitize

        seen = {}
        for e in ents:
            key = _sanitize(e.name).strip().lower()
            if key and key not in seen and len(key) >= 2:
                seen[key] = _sanitize(e.name)
        if not seen:
            return [], []
        names_lower = list(seen.keys())
        names_display = [seen[k] for k in names_lower]
    else:
        return [], []

    where_clause = "WHERE c.tenant_id = $tid" if tenant_id else ""
    params: dict[str, Any] = {"names": names_lower, "top_k": top_k}
    if tenant_id:
        params["tid"] = tenant_id

    # Chunk IDs filter for hard limit (supernode protection)
    chunk_filter = ""
    if chunk_ids_filter:
        chunk_filter = " AND c.id IN $chunk_ids"
        params["chunk_ids"] = chunk_ids_filter

    # Match entities: exact case-insensitive OR substring (catches "Leiden" → "Leiden algorithm")
    # Plus alias resolution: also match via ALIAS_OF edges.
    # **Normalized match** (strip whitespace + underscore): rescues PDF-extract
    # artifacts like "Plan_RAG"/"A STUTE RAG" that share normalized form with
    # query entity "PlanRAG"/"AstuteRAG". Verified +34 chunks for AstuteRAG,
    # +18 chunks for PlanRAG vs exact-only match.
    # temporal_filter from detect_temporal_intent() gets appended to the Entity MATCH
    entity_filter = f" AND ({temporal_filter})" if temporal_filter else ""
    # Pre-compute normalized form of query names (strip whitespace + underscore)
    names_normalized = [re.sub(r"[\s_]+", "", n).lower() for n in names_lower]
    params["names_norm"] = names_normalized
    cypher = f"""
    UNWIND range(0, size($names) - 1) AS i
    WITH $names[i] AS qname, $names_norm[i] AS qnorm
    MATCH (e:Entity)
    WHERE (
        toLower(e.name) = qname
        OR toLower(e.name) CONTAINS qname
        OR toLower(replace(replace(e.name, ' ', ''), '_', '')) = qnorm
        OR EXISTS {{
            MATCH (a:Entity)-[:ALIAS_OF]->(e)
            WHERE toLower(a.name) = qname OR toLower(a.name) CONTAINS qname
        }}
    ){entity_filter}
    MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e)
    {where_clause}{chunk_filter}
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
        candidates.append(
            {
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
            }
        )

    logger.info(
        f"entity_pivot: query entities {names_display} → "
        f"{len(candidates)} chunks (top matches={candidates[0]['entity_match_count'] if candidates else 0})"
    )
    return candidates, names_display


async def _graph_path(neo4j_driver, query_vec, http, embed_url, embed_model, tenant_id, top_k=20):
    """Graph retrieval, tenant-scoped at the Cypher level."""
    from src.services.kg import graph_retrieve

    try:
        # tenant_id now filters inside graph_retrieve's Cypher, so it only fetches and
        # embeds this tenant's entities — no cross-tenant fetch, no wasted embed, no
        # post-hoc metadata filter (which had also leaked rows whose tenant_id was None).
        results = await graph_retrieve(
            neo4j_driver,
            query_vec,
            http,
            embed_url,
            embed_model,
            top_k=top_k,
            tenant_id=tenant_id,
        )
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

    from src.services.embedding import cosine_similarity, embed_batch

    summaries = [c["summary"] for c in communities]
    try:
        embeds = await embed_batch(
            http, embed_url, embed_model, summaries, batch_size=16, timeout=60.0
        )
    except Exception as e:
        logger.debug(f"Community path: embed batch failed: {e}")
        return []

    scored = [
        (c, cosine_similarity(query_vec, e))
        for c, e in zip(communities, embeds, strict=False)
        if e and any(v != 0 for v in e)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    out: list[dict] = []
    for com, sim in scored[:top_k]:
        out.append(
            {
                "chunk_id": com["id"],
                "text": com["summary"],
                "source": f"community_L{com['level']}",
                "format": "community",
                "chunk_level": "section",
                "consistency_score": 0.85,
                "score": float(sim),
                "retrieval_path": "community",
                "metadata": {"level": com["level"], "member_count": com["mc"]},
            }
        )
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
                        chunk_domain = DomainDistribution.from_list(
                            [chunk_dd.get(ax, 0.0) for ax in DOMAIN_AXES]
                        )
                        dm_reward = 1.0 + domain_reward(
                            chunk_domain, query_domain, scale=domain_scale
                        )
                    except Exception as e:
                        logger.debug(f"Domain reward computation failed for chunk {cid}: {e}")

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
    views = strategy.get("views") or ["dense"]
    if settings.retrieval_real_views_only:
        # question/keywords are dense-copies (see config docstring); drop them so RRF
        # stops counting the dense signal 2-3x. Never empty the list.
        views = [v for v in views if v not in _DENSE_DUPLICATE_VIEWS] or ["dense"]

    # summarization intent gets boosted final_top_k for broad coverage
    intent_top_k_multiplier = strategy.get("final_top_k_multiplier", 1)
    effective_final_top_k = final_top_k * intent_top_k_multiplier

    reformulations = understanding.get(
        "reformulations",
        [{"kind": "original", "text": understanding.get("original", ""), "weight": 1.0}],
    )

    # Phase 8: tag query domain for reward scoring
    query_domain = tag_query(understanding.get("original", ""))

    # Embed unique reformulations in parallel to avoid duplicate HTTP calls
    unique_texts = list({r["text"] for r in reformulations if r.get("text")})
    embed_results = await asyncio.gather(
        *[
            _embed_query(clients.http, settings.ollama_embed_url, settings.ollama_embed_model, t)
            for t in unique_texts
        ]
    )
    embed_map = dict(zip(unique_texts, embed_results, strict=False))
    embeds = [embed_map.get(r["text"], []) for r in reformulations]

    # ── Primary entity-gate: discover cross-doc chunk scope via entity cosine.
    # Replaces the lossy doc-gate. The seed chunk_ids_scope is fed to every
    # downstream path so vector + entity_pivot + entity_cosine all operate
    # within the same entity-aware scope.
    entity_gate_scope: list[str] | None = None
    entity_gate_chunks: list[dict] = []
    if (
        getattr(settings, "entity_gate_enabled", False)
        and embeds
        and embeds[0]
        and clients.neo4j is not None
    ):
        try:
            from src.services.entity_vectors import entity_cosine_primary

            entity_gate_chunks, scope, _entities = await entity_cosine_primary(
                query_vec=np.asarray(embeds[0], dtype=np.float32),
                tenant_id=tenant_id or "default",
                neo4j_driver=clients.neo4j,
                qdrant_client=clients.qdrant,
                collection=settings.qdrant_collection,
                top_k_entities=getattr(settings, "entity_gate_top_k_entities", 50),
                seed_chunks=getattr(settings, "entity_gate_seed_chunks", 200),
                score_floor=getattr(settings, "entity_gate_score_floor", 0.20),
                lambda_mmr=getattr(settings, "entity_cosine_mmr_lambda", 0.6),
            )
            if scope:
                entity_gate_scope = scope
                logger.info(
                    f"entity_gate: scope={len(scope)} chunks, "
                    f"primary_path_chunks={len(entity_gate_chunks)}"
                )
        except Exception as e:
            logger.warning(f"entity_gate failed (fallback to no scope): {e!r}")

    qdrant_filter = build_tenant_filter(
        tenant_id=tenant_id,
        format_in=format_filter,
        access_levels=access_levels,
    )

    paths: dict[str, list[dict]] = {}
    if entity_gate_chunks:
        paths["entity_gate"] = entity_gate_chunks

    # Vector search paths: each reformulation × each view
    async def _one_search(reform_kind: str, vec: list[float], view: str) -> tuple[str, list[dict]]:
        if not vec:
            return f"{reform_kind}:{view}", []
        results = await search_single_view(
            clients.qdrant,
            settings.qdrant_collection,
            vec,
            view,
            limit=top_k_per_path,
            filter_=qdrant_filter,
        )
        return f"{reform_kind}:{view}", results

    search_tasks = []
    for r, vec in zip(reformulations, embeds, strict=False):
        for v in views:
            search_tasks.append(_one_search(r["kind"], vec, v))

    # Graph path (if intent supports)
    graph_task = None
    if strategy["use_graph"] and embeds and embeds[0]:
        graph_task = _graph_path(
            clients.neo4j,
            embeds[0],
            clients.http,
            settings.ollama_embed_url,
            settings.ollama_embed_model,
            tenant_id,
            top_k=top_k_per_path,
        )

    # Community path (if intent supports + enabled)
    community_task = None
    if (
        strategy["use_community"]
        and getattr(settings, "community_enabled", False)
        and embeds
        and embeds[0]
    ):
        community_task = _community_path(
            clients.neo4j,
            embeds[0],
            clients.http,
            settings.ollama_embed_url,
            settings.ollama_embed_model,
            tenant_id,
            top_k=5,
        )

    # Entity-pivot: gated by use_entity_pivot flag per intent strategy.
    # Uses entities pre-extracted by GLiNER from understand_query (Step 2).
    # Phase 6b: detect temporal intent and filter entity lookup by time range.
    # Phase 3b: 2-phase retrieval with hard limit for supernode protection.
    from src.services.temporal_entities import detect_temporal_intent

    temporal = detect_temporal_intent(understanding.get("original", ""))
    if temporal.get("type") != "none":
        logger.info(
            f"  temporal intent: {temporal['type']} — filter: {temporal.get('filter_cypher', '')[:80]}"
        )

    # VRAG Tier 2 — Vector-Driven Hard Limit config
    graph_scope_size = getattr(settings, "graph_scope_size", 100)
    use_hard_limit = getattr(settings, "use_hard_limit", True)

    # ── Phase 1: run dense + graph + community in parallel ──────────────────
    phase1_tasks = (
        search_tasks
        + ([graph_task] if graph_task else [])
        + ([community_task] if community_task else [])
    )
    phase1_results = await asyncio.gather(*phase1_tasks, return_exceptions=True)

    # Collect vector search results
    idx = 0
    for _r, _ in zip(reformulations, embeds, strict=False):
        for _v in views:
            res = phase1_results[idx]
            idx += 1
            if isinstance(res, Exception):
                continue
            path_key, candidates = res
            paths[path_key] = candidates

    # Graph results
    if graph_task is not None:
        res = phase1_results[idx]
        idx += 1
        if not isinstance(res, Exception):
            paths["graph"] = res

    # Community results
    if community_task is not None:
        res = phase1_results[idx]
        idx += 1
        if not isinstance(res, Exception):
            paths["community"] = res

    # ── Phase 2: entity_pivot with Hard Limit scope from Phase 1 paths ──────
    # VRAG Tier 2: Neo4j Cypher confined to Top-N chunk IDs from vector retrieval.
    # Prevents supernode traversal explosion (e.g. entity "AI" linking 10k chunks).
    query_entities_holder: dict[str, list[str]] = {"entities": []}
    entity_extractor = getattr(clients, "entity_extractor", None)
    pre_extracted_entities = understanding.get("entities", []) or []
    use_entity_pivot = strategy.get("use_entity_pivot", False) or len(pre_extracted_entities) >= 1
    chunk_ids_scope: list[str] | None = None

    if entity_extractor is not None and use_entity_pivot:
        try:
            if entity_gate_scope:
                chunk_ids_scope = entity_gate_scope[:graph_scope_size]
                logger.info(
                    f"  hard_limit: entity_pivot scope = {len(chunk_ids_scope)} chunks "
                    f"(from entity_gate primary scope)"
                )
            elif use_hard_limit and paths:
                # Tier 2 fix: deterministic ordering by best score across paths.
                # Previous version: set → list ordering was timing-dependent (asyncio
                # task completion order), causing same query to surface different
                # chunks across runs. See TIER1_COMPARISON_20260520.md § s04 case.
                best_score: dict[str, float] = {}
                for path_results in paths.values():
                    for c in path_results:
                        cid = c.get("chunk_id", "")
                        if not cid:
                            continue
                        score = float(c.get("score", 0.0) or 0.0)
                        if score > best_score.get(cid, -1.0):
                            best_score[cid] = score
                # Stable sort: highest score first, ties broken by chunk_id string (deterministic)
                chunk_ids_scope = sorted(
                    best_score.keys(),
                    key=lambda c: (-best_score[c], c),
                )[:graph_scope_size]
                logger.info(
                    f"  hard_limit: entity_pivot scope = {len(chunk_ids_scope)} chunks "
                    f"(from {len(paths)} retrieval paths, sorted by score)"
                )

            cands, query_ents = await _entity_pivot_path(
                clients.neo4j,
                entity_extractor,
                understanding.get("original", ""),
                tenant_id,
                top_k=top_k_per_path,
                temporal_filter=temporal.get("filter_cypher", ""),
                chunk_ids_filter=chunk_ids_scope,
                pre_extracted_entities=pre_extracted_entities,
            )
            query_entities_holder["entities"] = query_ents or pre_extracted_entities
            paths["entity_pivot"] = cands
        except Exception as e:
            logger.debug(f"entity_pivot path failed: {e}")

    # ── Tier 3: entity_cosine path (gated by ENTITY_COSINE_ENABLED) ──────────
    # Cross-document entity-aware retrieval with L1 (TF-IDF), L3 (MMR), L5
    # (sub-graph Hard Limit) supernova guards. Operates on the same
    # chunk_ids_scope as entity_pivot, so latency overhead is bounded.
    if (
        getattr(settings, "entity_cosine_enabled", False)
        and chunk_ids_scope
        and embeds
        and embeds[0]
    ):
        try:
            from src.services.entity_vectors import entity_cosine_retrieve

            query_vec = np.asarray(embeds[0], dtype=np.float32)
            qnorm = float(np.linalg.norm(query_vec))
            if qnorm > 0:
                query_vec = query_vec / qnorm
            ec_chunks, ec_entities = await entity_cosine_retrieve(
                query_vec=query_vec,
                chunk_ids_scope=chunk_ids_scope,
                tenant_id=tenant_id or "default",
                neo4j_driver=clients.neo4j,
                qdrant_client=clients.qdrant,
                collection=settings.qdrant_collection,
                top_k_entities=getattr(settings, "entity_cosine_top_k_entities", 20),
                top_k_chunks=top_k_per_path,
                lambda_mmr=getattr(settings, "entity_cosine_mmr_lambda", 0.6),
            )
            if ec_chunks:
                paths["entity_cosine"] = ec_chunks
                logger.info(
                    f"  entity_cosine: {len(ec_chunks)} chunks via {len(ec_entities)} entities"
                )
        except Exception as e:
            logger.debug(f"entity_cosine path failed: {e}")

    # ── Phase 2.1: HippoRAG 2 — Personalized PageRank over entity graph ────
    # Seeds on query entities, propagates through RELATES_TO edges, surfaces
    # chunks linked to high-scoring entities. Fix for multi-hop queries where
    # destination chunk is 2-3 hops from the source entity.
    if (
        getattr(settings, "ppr_enabled", False)
        and pre_extracted_entities
        and clients.neo4j is not None
    ):
        try:
            from src.services.ppr import ppr_retrieve

            ppr_chunks = await ppr_retrieve(
                neo4j_driver=clients.neo4j,
                query_entities=pre_extracted_entities,
                tenant_id=tenant_id or "default",
                top_k_chunks=top_k_per_path,
                alpha=getattr(settings, "ppr_alpha", 0.5),
                chunk_ids_filter=None,  # PPR walks the full graph, no scope clamp
            )
            if ppr_chunks:
                paths["ppr"] = ppr_chunks
                logger.info(f"  ppr: {len(ppr_chunks)} chunks")
        except Exception as e:
            logger.debug(f"ppr path failed: {e}")

    # Tier 2 fix: filter low-quality / fragment chunks BEFORE RRF.
    # Root cause from RAGAS analysis: chunks with text < 80 chars or matching
    # figure-caption / lone-title patterns surface alongside real content,
    # then the LLM is forced to fill gaps via parametric knowledge (hurts
    # faithfulness). See TIER1_COMPARISON_20260520.md § f10 case.
    _MIN_CHUNK_CHARS = 80
    _FRAGMENT_PATTERNS = [
        re.compile(r"^\s*figure\s*\d+", re.IGNORECASE),
        re.compile(r"^\s*table\s*\d+", re.IGNORECASE),
        re.compile(r"^\s*\d+(?:\.\d+)?\s*[A-Z][\w\s]{0,40}\s*$"),  # lone section heading
    ]

    def _is_fragment(c: dict) -> bool:
        text = (c.get("text") or "").strip()
        if len(text) < _MIN_CHUNK_CHARS:
            return True
        for pat in _FRAGMENT_PATTERNS:
            if pat.match(text):
                return True
        return False

    total_before = sum(len(v) for v in paths.values())
    for k, lst in paths.items():
        paths[k] = [c for c in lst if not _is_fragment(c)]
    total_after = sum(len(v) for v in paths.values())
    if total_before > total_after:
        logger.info(
            f"  fragment filter: dropped {total_before - total_after}/{total_before} chunks"
        )

    # Weighted RRF — fuses by RANK within each path (not raw score), so no
    # pre-normalization is needed. A prior z-score-per-format pass wrote a
    # `score_normalized` field that weighted_rrf never read — pure hot-path waste,
    # removed. (normalize_scores_by_format stays in vector.py for its unit test.)
    fused = weighted_rrf(
        paths,
        k=settings.rrf_k,
        final_top_k=effective_final_top_k,
        query_domain=query_domain,
        domain_scale=0.3,
    )

    # Attach query entities to first result for caller (LLM prompt) to access.
    # Uses a global key on each candidate so it survives downstream processing.
    if fused and query_entities_holder.get("entities"):
        for c in fused:
            c["_query_entities"] = query_entities_holder["entities"]

    return fused
