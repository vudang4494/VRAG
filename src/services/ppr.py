"""HippoRAG 2 — Personalized PageRank over the entity graph.

The graph is the Neo4j `(:Entity)-[:RELATES_TO]->(:Entity)` subgraph, per tenant.
Query entities seed the personalization vector; the random walk propagates to
related entities through co-occurrence edges; chunks linked to high-scoring
entities are returned. This is the canonical fix for multi-hop questions where
single-shot retrieval finds the source chunk but misses the destination chunk
two hops away.

Cache: the per-tenant graph is loaded once and reused for `_CACHE_TTL_S`
seconds — recomputing for every query would be the bottleneck.

Fail-safe behaviour: any error (empty graph, seed not present, Neo4j down)
returns `[]`. PPR is additive to existing paths in `multi_path_retrieve`, never
a hard dependency.
"""

from __future__ import annotations

import importlib.util
import math
import time
from typing import Any

from loguru import logger

from src.config import get_settings

try:
    import networkx as nx

    _HAS_NX = True
except Exception:  # pragma: no cover — networkx is a hard dep but guard anyway
    _HAS_NX = False

# networkx 3.x `nx.pagerank` delegates to a scipy-backed implementation and
# raises ModuleNotFoundError when scipy is absent — which the broad except in
# `ppr_retrieve` would swallow, silently turning PPR into a no-op (return []).
# Detect scipy once and fall back to networkx's pure-python power iteration so
# PPR actually runs everywhere. Fail-loud: we log the fallback the first time.
_HAS_SCIPY = importlib.util.find_spec("scipy") is not None
_PAGERANK_WARNED = [False]


def _run_pagerank(g, alpha, personalization, max_iter, tol, weight):
    """PageRank that works with or without scipy (see `_HAS_SCIPY`)."""
    if _HAS_SCIPY:
        return nx.pagerank(
            g,
            alpha=alpha,
            personalization=personalization,
            max_iter=max_iter,
            tol=tol,
            weight=weight,
        )
    from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_python

    if not _PAGERANK_WARNED[0]:
        logger.warning(
            "PPR: scipy not installed — using pure-python pagerank (correct, slower). "
            "`uv pip install scipy` enables the fast path."
        )
        _PAGERANK_WARNED[0] = True
    return _pagerank_python(
        g,
        alpha=alpha,
        personalization=personalization,
        max_iter=max_iter,
        tol=tol,
        weight=weight,
    )


# Per-tenant graph cache: (tenant_id, weighted) → (loaded_at_ts, DiGraph).
# `weighted` is part of the key so toggling PPR_EDGE_WEIGHTING never serves a
# graph built under the other mode.
_GRAPH_CACHE: dict[tuple[str, bool], tuple[float, Any]] = {}
_CACHE_TTL_S = 600  # 10 min — entity graph changes only on re-ingest

# Default PPR hyper-params (HippoRAG 2 paper defaults).
DEFAULT_ALPHA = 0.5
DEFAULT_MAX_ITER = 50
DEFAULT_TOL = 1.0e-6


async def _load_entity_graph(neo4j_driver, tenant_id: str) -> Any | None:
    """Return a directed entity graph for the tenant, or None if unavailable.

    Default (PPR_EDGE_WEIGHTING=0): edges are `RELATES_TO` (or co-occurrence
    fallback), unweighted. ALIAS_OF edges are folded by mapping each alias to its
    canonical name before adding to the graph.

    De-hub mode (PPR_EDGE_WEIGHTING=1): the graph is rebuilt WEIGHTED by NPMI over
    CONTAINS_ENTITY co-occurrence, regardless of RELATES_TO presence — because the
    stored RELATES_TO edges carry no frequency and so cannot express the hub
    signal. Pairs with NPMI ≤ ppr_npmi_min (hubs: generic terms / dates / geos
    that co-occur with everything) are dropped. See
    # PPR de-hubbing implementation.

    Cached for `_CACHE_TTL_S`, keyed by (tenant, weighted).
    """
    if not _HAS_NX:
        logger.debug("PPR: networkx unavailable")
        return None

    settings = get_settings()
    weighted = bool(getattr(settings, "ppr_edge_weighting", False))
    npmi_min = float(getattr(settings, "ppr_npmi_min", 0.0))
    cache_key = (tenant_id, weighted)

    now = time.time()
    cached = _GRAPH_CACHE.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL_S:
        return cached[1]

    g = nx.DiGraph()
    edge_source = "relates_to"
    try:
        async with neo4j_driver.session() as s:
            # Resolve aliases first → use canonical names as graph nodes.
            alias_map: dict[str, str] = {}
            alias_result = await s.run(
                "MATCH (a:Entity {tenant_id: $tid})-[:ALIAS_OF]->(c:Entity) "
                "RETURN a.name AS alias, c.name AS canonical",
                tid=tenant_id,
            )
            async for record in alias_result:
                alias_map[record["alias"]] = record["canonical"]

            def _canon(name: str) -> str:
                return alias_map.get(name, name)

            if weighted:
                edge_source = "npmi_cooc"
                await _build_npmi_graph(g, s, tenant_id, _canon, npmi_min)
            else:
                # Preferred: explicit RELATES_TO edges (populated when
                # ENTITY_RELATIONS_ENABLED=1 at ingest).
                rel_result = await s.run(
                    "MATCH (e1:Entity {tenant_id: $tid})-[:RELATES_TO]->(e2:Entity {tenant_id: $tid}) "
                    "RETURN e1.name AS src, e2.name AS tgt",
                    tid=tenant_id,
                )
                async for record in rel_result:
                    src = _canon(record["src"])
                    tgt = _canon(record["tgt"])
                    if src == tgt:
                        continue
                    g.add_edge(src, tgt)

                # Fallback: co-occurrence graph derived from CONTAINS_ENTITY. Two
                # entities sharing ≥1 chunk get an edge. This is what HippoRAG 2
                # uses when OpenIE triples are unavailable. Cap chunk degree to
                # avoid quadratic blow-up on supernode chunks.
                if g.number_of_edges() == 0:
                    edge_source = "co_occurrence"
                    co_result = await s.run(
                        """
                        MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e1:Entity {tenant_id: $tid})
                        WITH c, collect(DISTINCT e1.name) AS ents
                        WHERE size(ents) >= 2 AND size(ents) <= 25
                        RETURN ents
                        """,
                        tid=tenant_id,
                    )
                    pair_count = 0
                    async for record in co_result:
                        ents = [_canon(n) for n in record["ents"]]
                        # Build symmetric pairs within the chunk.
                        for i in range(len(ents)):
                            for j in range(i + 1, len(ents)):
                                a, b = ents[i], ents[j]
                                if a == b:
                                    continue
                                g.add_edge(a, b)
                                g.add_edge(b, a)
                                pair_count += 1
                    logger.info(
                        f"PPR: built co-occurrence graph for {tenant_id} from {pair_count} entity pairs"
                    )
                else:
                    # Symmetric walk: add reverse edges to the explicit graph too.
                    g = g.to_undirected().to_directed()
    except Exception as e:
        logger.warning(f"PPR: failed to load entity graph for {tenant_id}: {e!r}")
        return None

    # Stash the alias map on the graph so `ppr_retrieve` can fold SEED entities
    # too — nodes are already canonical, but query entities arrive as raw surface
    # forms (e.g. "CAYMAN ISLANDS") that must map to the canonical node
    # ("Cayman Islands") or the seed silently misses the folded subgraph.
    g.graph["alias_map"] = alias_map
    _GRAPH_CACHE[cache_key] = (now, g)
    logger.info(
        f"PPR: loaded entity graph for {tenant_id} — "
        f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges, "
        f"{len(alias_map)} aliases, source={edge_source}"
    )
    return g


async def _build_npmi_graph(g, session, tenant_id: str, canon, npmi_min: float) -> None:
    """Populate `g` with NPMI-weighted co-occurrence edges (de-hub, pick #1).

    NPMI(a,b) = log(p_ab / (p_a·p_b)) / -log(p_ab), computed over the co-occurrence
    universe = chunks holding 2..25 tenant entities (same cap the unweighted
    fallback uses). Hub pairs — where p_a or p_b is large because the entity
    co-occurs with everything — get NPMI ≤ 0 and are pruned by `npmi_min`,
    so the hair-ball edges never enter the walk. Symmetric (both directions).
    """
    co_result = await session.run(
        """
        MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity {tenant_id: $tid})
        WITH c, collect(DISTINCT e.name) AS ents
        WHERE size(ents) >= 2 AND size(ents) <= 25
        RETURN ents
        """,
        tid=tenant_id,
    )
    df: dict[str, int] = {}  # entity → # co-occurrence contexts it appears in
    pair: dict[tuple[str, str], int] = {}  # (a,b) sorted → shared-context count
    n_ctx = 0
    async for record in co_result:
        ents = sorted({canon(n) for n in record["ents"]})
        if len(ents) < 2:
            continue
        n_ctx += 1
        for e in ents:
            df[e] = df.get(e, 0) + 1
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                key = (ents[i], ents[j])
                pair[key] = pair.get(key, 0) + 1

    if n_ctx == 0 or not pair:
        return

    kept = 0
    for (a, b), cooc in pair.items():
        p_ab = cooc / n_ctx
        neg_log = -math.log(p_ab)
        if neg_log <= 0.0:  # pair co-occurs in every context → NPMI undefined
            continue
        pmi = math.log(p_ab / ((df[a] / n_ctx) * (df[b] / n_ctx)))
        npmi = pmi / neg_log
        if npmi <= npmi_min:
            continue
        w = float(npmi)
        g.add_edge(a, b, weight=w)
        g.add_edge(b, a, weight=w)
        kept += 1
    logger.info(
        f"PPR: NPMI graph for {tenant_id} — {n_ctx} contexts, "
        f"{len(pair)} pairs → {kept} kept (npmi>{npmi_min})"
    )


async def _entities_to_chunks(
    neo4j_driver,
    ranked_entities: list[tuple[str, float]],
    tenant_id: str,
    top_k_chunks: int,
    chunk_ids_filter: list[str] | None = None,
) -> list[dict]:
    """Map ranked entities → chunks via `CONTAINS_ENTITY`, preserving PPR scores.

    For each chunk we keep the max PPR score across its matched entities and
    record the matching entity for downstream debug.
    """
    if not ranked_entities:
        return []
    score_by_entity = dict(ranked_entities)
    names = [e for e, _ in ranked_entities]

    chunk_score: dict[str, float] = {}
    chunk_payload: dict[str, dict] = {}

    try:
        async with neo4j_driver.session() as s:
            params: dict[str, Any] = {
                "names": names,
                "tid": tenant_id,
                "k": max(top_k_chunks * 4, 40),
            }
            cypher = (
                "UNWIND $names AS name "
                "MATCH (e:Entity {name: name, tenant_id: $tid})"
                "<-[:CONTAINS_ENTITY]-(c:Chunk) "
            )
            if chunk_ids_filter:
                cypher += "WHERE c.id IN $chunk_ids "
                params["chunk_ids"] = list(chunk_ids_filter)
            cypher += "RETURN c.id AS chunk_id, c.text AS text, c.source AS source, e.name AS entity LIMIT $k"
            result = await s.run(cypher, **params)
            async for record in result:
                cid = record["chunk_id"]
                ent = record["entity"]
                score = score_by_entity.get(ent, 0.0)
                if score > chunk_score.get(cid, -1.0):
                    chunk_score[cid] = score
                    chunk_payload[cid] = {
                        "id": cid,
                        "chunk_id": cid,
                        "text": record["text"] or "",
                        "source": record["source"] or "",
                        "matched_entity": ent,
                        "score": float(score),
                        "retrieval_path": "ppr",
                    }
    except Exception as e:
        logger.warning(f"PPR: entities→chunks Cypher failed: {e!r}")
        return []

    # Best chunk per (score, chunk_id) order — deterministic.
    ordered = sorted(
        chunk_payload.values(),
        key=lambda c: (-c["score"], c["chunk_id"]),
    )
    return ordered[:top_k_chunks]


async def ppr_retrieve(
    neo4j_driver,
    query_entities: list[str],
    tenant_id: str,
    top_k_chunks: int = 20,
    alpha: float = DEFAULT_ALPHA,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    chunk_ids_filter: list[str] | None = None,
) -> list[dict]:
    """Run Personalized PageRank seeded on `query_entities`, return chunks.

    Returns at most `top_k_chunks` chunks ordered by PPR score descending. Falls
    through to `[]` on any failure — never raises.
    """
    if not _HAS_NX:
        return []
    if not query_entities:
        return []

    g = await _load_entity_graph(neo4j_driver, tenant_id)
    if g is None or g.number_of_nodes() < 2:
        return []

    # Build personalization vector — uniform over query entities that exist in
    # the graph. Fold each raw query entity through the alias map first so an
    # alias surface form ("CAYMAN ISLANDS") seeds its canonical node
    # ("Cayman Islands"); graph nodes are already canonical. If none exist, fall
    # through (random-walk on uniform prior would just re-rank by degree, which
    # entity_pivot already does).
    alias_map = g.graph.get("alias_map", {})
    folded = [alias_map.get(e, e) for e in query_entities]
    seeds = [e for e in folded if e in g]
    if not seeds:
        logger.debug(
            f"PPR: none of {query_entities[:5]} present in graph "
            f"({g.number_of_nodes()} nodes) — skipping"
        )
        return []
    personalization = dict.fromkeys(g.nodes(), 0.0)
    w = 1.0 / len(seeds)
    for s in seeds:
        personalization[s] = w

    settings = get_settings()
    weighted = bool(getattr(settings, "ppr_edge_weighting", False))
    gamma = float(getattr(settings, "ppr_degree_penalty", 0.0))

    try:
        ranked = _run_pagerank(
            g,
            alpha=alpha,
            personalization=personalization,
            max_iter=max_iter,
            tol=tol,
            weight="weight" if weighted else None,
        )
    except Exception as e:
        logger.warning(f"PPR: pagerank failed: {e!r}")
        return []

    # Degree penalty (de-hub, pick #1): divide each entity's PPR mass by
    # (1+deg)^γ so a hub that the walk inevitably reaches does not dominate the
    # chunk mapping. γ=0 leaves ranking untouched. Applied independently of the
    # edge weighting so each lever is ablatable.
    if gamma > 0.0:
        ranked = {e: sc / ((1.0 + g.degree(e)) ** gamma) for e, sc in ranked.items()}

    # Take more entities than top_k_chunks because not every entity has a chunk
    # in the current scope.
    top_n_entities = max(top_k_chunks * 3, 40)
    top_entities = sorted(ranked.items(), key=lambda kv: kv[1], reverse=True)[:top_n_entities]

    chunks = await _entities_to_chunks(
        neo4j_driver,
        top_entities,
        tenant_id,
        top_k_chunks,
        chunk_ids_filter=chunk_ids_filter,
    )

    logger.info(
        f"PPR: seeds={len(seeds)}/{len(query_entities)} "
        f"graph={g.number_of_nodes()}n/{g.number_of_edges()}e "
        f"→ {len(chunks)} chunks"
    )
    return chunks


def invalidate_cache(tenant_id: str | None = None) -> None:
    """Drop the cached graph(s). Call after an ingest run for the tenant.

    Drops both weighted and unweighted variants for the tenant.
    """
    if tenant_id is None:
        _GRAPH_CACHE.clear()
    else:
        for key in [k for k in _GRAPH_CACHE if k[0] == tenant_id]:
            _GRAPH_CACHE.pop(key, None)
