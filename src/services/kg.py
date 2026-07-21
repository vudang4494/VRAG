"""Knowledge graph service — entity extraction, Neo4j storage and retrieval.

## Entity Canonicalization Strategy

After GLiNER extracts entities from a chunk, canonicalize_entities runs a 3-tier
disambiguation pass before writing to Neo4j:

  1. Exact match: name already exists in KG → use existing canonical
  2. Levenshtein similarity >= 0.85 (same type): → create ALIAS_OF edge
  3. No match: → write as new Entity

The canonical form is the one with highest existing confidence or earliest insertion.

This prevents fragmented graphs where "Apple Inc.", "Apple", "AAPL" become 3 separate
entities, breaking entity-pivot traversal and community detection.
"""

import json
import re
from difflib import SequenceMatcher
from typing import Any

import httpx
from loguru import logger

# Canonical Neo4j schema — MUST match scripts/init-neo4j.cypher. Run at startup so
# the graph is self-healing: without these, MERGE on (Chunk{id})/(Entity{name})
# does a full label scan (O(n) per write, risks duplicates) and graph retrieval
# scans instead of using indexes. All IF NOT EXISTS → idempotent, non-destructive.
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    # Entity identity is (name, tenant_id), NOT name alone.
    #
    # `REQUIRE e.name IS UNIQUE` was global, so two tenants ingesting an entity with the
    # same name were forced onto ONE node, and e.tenant_id became last-writer-wins. Since
    # community.py and hefr.py filter `MATCH (e:Entity {tenant_id: $tid})`, the loser
    # tenant simply stopped seeing its own entity. Chunk retrieval was never affected —
    # _entity_pivot filters on c.tenant_id — so this is a correctness bug in the entity
    # paths, not a chunk leak.
    #
    # The old constraint is dropped explicitly: `CREATE ... IF NOT EXISTS` cannot replace
    # it, and while it exists it keeps enforcing the global merge.
    # Composite uniqueness is supported on Neo4j 5.26 Community (verified on this box).
    # It ignores nodes where either property is null, so the 219 legacy entities with
    # tenant_id = null do not block it; they stay invisible to the tenant-scoped paths
    # until re-ingested, which is what they already were.
    "DROP CONSTRAINT entity_name IF EXISTS",
    "CREATE CONSTRAINT entity_name_tenant IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.tenant_id) IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (com:Community) REQUIRE com.id IS UNIQUE",
    "CREATE INDEX chunk_source IF NOT EXISTS FOR (c:Chunk) ON (c.source)",
    "CREATE INDEX chunk_tenant IF NOT EXISTS FOR (c:Chunk) ON (c.tenant_id)",
    "CREATE INDEX chunk_level IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_level)",
    "CREATE INDEX chunk_format IF NOT EXISTS FOR (c:Chunk) ON (c.format)",
    "CREATE INDEX chunk_consistency IF NOT EXISTS FOR (c:Chunk) ON (c.consistency_score)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX entity_tenant IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id)",
    "CREATE INDEX entity_confidence IF NOT EXISTS FOR (e:Entity) ON (e.confidence)",
    "CREATE INDEX document_source IF NOT EXISTS FOR (d:Document) ON (d.source)",
    "CREATE INDEX document_tenant IF NOT EXISTS FOR (d:Document) ON (d.tenant_id)",
    "CREATE INDEX community_tenant_level IF NOT EXISTS FOR (com:Community) ON (com.tenant_id, com.level)",
    "CREATE FULLTEXT INDEX entity_fts IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description]",
    "CREATE FULLTEXT INDEX chunk_fts IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
    "CREATE FULLTEXT INDEX community_summary_fts IF NOT EXISTS FOR (com:Community) ON EACH [com.summary]",
)


async def ensure_schema(driver) -> int:
    """Create Neo4j constraints + indexes if missing. Idempotent, non-destructive.

    Returns the number of statements that ran without error. Mirrors
    scripts/init-neo4j.cypher — keep both in sync. Best-effort per statement so one
    unsupported index (e.g. on an old Neo4j) doesn't block the rest.
    """
    ok = 0
    async with driver.session() as s:
        for stmt in _SCHEMA_STATEMENTS:
            try:
                await (await s.run(stmt)).consume()
                ok += 1
            except Exception as e:
                logger.debug(f"schema stmt skipped ({stmt[:40]}...): {e}")
    logger.info(f"Neo4j schema ensured ({ok}/{len(_SCHEMA_STATEMENTS)} constraints+indexes)")
    return ok


_ENTITY_EXTRACT_PROMPT = """Ban la chuyen gia trich xuat tri thuc tu van ban.
Trich xuat cac thuc the (entities) va moi quan he (relationships) tu van ban duoi day.

Tra loi CHI bang JSON (khong co giai thich gi them):

{{
  "entities": [
    {{
      "name": "ten thuc the",
      "type": "PERSON|ORGANIZATION|LOCATION|EVENT|PRODUCT|CONCEPT|TECHNOLOGY|OTHER",
      "description": "mo ta ngan ve thuc the nay"
    }}
  ],
  "relationships": [
    {{
      "source": "ten thuc the nguon",
      "target": "ten thuc the dich",
      "description": "mo ta moi quan he"
    }}
  ]
}}

Van ban:
{text}
"""


async def canonicalize_entities(
    driver,
    entities: list[dict],
    tenant_id: str,
) -> list[dict]:
    """
    Resolve entity name variants to canonical forms via 3-tier strategy.

    Returns entities with canonical_name added (may differ from input name).
    Creates ALIAS_OF edges in Neo4j for non-exact variants.

    Tier 1 — Exact match: name already exists in KG → use existing canonical
    Tier 2 — Levenshtein >= 0.85 + same type → create ALIAS_OF edge
    Tier 3 — No match → write as new canonical entity
    """

    if not entities:
        return []

    canonical_entities: list[dict] = []
    aliases_created = 0

    try:
        async with driver.session() as s:
            for entity in entities:
                name = _sanitize(entity.get("name", ""))
                if not name:
                    continue
                etype = entity.get("type", "OTHER")

                # Tier 1: exact match
                r = await s.run(
                    """
                    MATCH (e:Entity {name: $name, tenant_id: $tid})
                    RETURN e.name AS canonical_name, e.type AS canonical_type,
                           e.confidence AS confidence, e.tenant_id AS tenant_id
                    LIMIT 1
                    """,
                    name=name,
                    tid=tenant_id,
                )
                rows = await r.data()
                if rows:
                    canonical_entities.append(
                        {**entity, "canonical_name": rows[0]["canonical_name"]}
                    )
                    continue

                # Tier 2: fuzzy match (SequenceMatcher >= 0.85, same type).
                # Candidates come from the entity_fts fulltext index — a SMALL relevant
                # set — instead of scanning EVERY same-type entity in the tenant. The old
                # full-scan was O(n) per entity -> O(n^2) per batch and made large-tenant
                # backfills (125k chunks) take ~20h. Escape Lucene specials in the name.
                fts_q = re.sub(r'[+\-&|!(){}\[\]^"~*?:\\/]', " ", name).strip()
                candidates = []
                if fts_q:
                    try:
                        r = await s.run(
                            """
                            CALL db.index.fulltext.queryNodes('entity_fts', $q)
                            YIELD node, score
                            WHERE node.tenant_id = $tid AND node.type = $etype
                            RETURN node.name AS canonical_name
                            LIMIT 10
                            """,
                            q=fts_q,
                            tid=tenant_id,
                            etype=etype,
                        )
                        candidates = await r.data()
                    except Exception as fe:
                        logger.debug(f"fulltext candidate lookup failed for {name!r}: {fe}")
                best_ratio = 0.0
                best_canonical = None
                for cand in candidates:
                    ratio = SequenceMatcher(
                        None, name.lower(), cand["canonical_name"].lower()
                    ).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_canonical = cand

                if best_canonical and best_ratio >= 0.85:
                    # Create ALIAS_OF edge
                    await s.run(
                        """
                        MATCH (alias:Entity {name: $name, tenant_id: $tid})
                        MATCH (canon:Entity {name: $canon, tenant_id: $tid})
                        MERGE (alias)-[:ALIAS_OF]->(canon)
                        """,
                        name=name,
                        tid=tenant_id,
                        canon=best_canonical["canonical_name"],
                    )
                    aliases_created += 1
                    canonical_entities.append(
                        {**entity, "canonical_name": best_canonical["canonical_name"]}
                    )
                    continue

                # Tier 3: new entity — canonical_name = name
                canonical_entities.append({**entity, "canonical_name": name})

    except Exception as e:
        logger.debug(f"Canonicalization failed: {e}")
        # Fallback: return as-is
        for entity in entities:
            name = _sanitize(entity.get("name", ""))
            if name:
                canonical_entities.append({**entity, "canonical_name": name})

    if aliases_created > 0:
        logger.info(f"Entity canonicalization: {aliases_created} alias(es) resolved")

    return canonical_entities


def _lexically_related(a: str, b: str) -> bool:
    """True if two entity names are plausibly the same entity by surface form.

    Gate for the cosine confirmation in `resolve_entity_aliases`: centroids are
    context means, so cosine ALONE would merge distinct entities that merely
    co-occur (two people named in one report). Requiring a surface link first —
    substring, acronym, or a shared significant token — blocks that while still
    catching what plain Levenshtein misses ('Apple Inc.'/'Apple',
    'large language model'/'LLM').
    """
    na, nb = a.lower().strip(), b.lower().strip()
    if not na or not nb:
        return False
    # Keep alphanumerics only (diacritics fold to nothing consistently on both
    # sides, so case/punct/whitespace variants normalize to the same skeleton).
    ca = re.sub(r"[^a-z0-9]", "", na)
    cb = re.sub(r"[^a-z0-9]", "", nb)
    if not ca or not cb:
        return False
    # Near-equal by normalization: case/whitespace/punct/diacritic variants,
    # possessives ('National Geographicʼs' ⊃ 'National Geographic'), dropped
    # articles ('the Financial Secretary' ⊃ 'Financial Secretary'), digit
    # artifacts ('...y tế1' ⊃ '...y tế'). Guard: the shorter form must be >=4 chars
    # AND >=80% of the longer. The 80% (not 60%) bar is what stops a short
    # meaning-flipping modifier from merging opposites — 'thành phẩm' (finished)
    # ⊄ 'bán thành phẩm' (semi-finished), 'Comptroller' ⊄ 'Comptroller Gould' —
    # while a small article/possessive on a long base still passes. The cosine gate
    # cannot catch these: same-document context puts them at cos >= 0.9.
    # Deliberately NOT a shared-token rule — that over-merges distinct same-domain
    # entities ('X University'/'Y University', 'Ministry of X'/'Ministry of Y').
    short, long = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
    if len(short) >= 4 and short in long and len(short) >= 0.8 * len(long):
        return True
    # Acronym: initials of the multi-word side equal the other compact side
    # ('large language model'/'LLM', 'United States'/'US').
    toks_a = [t for t in re.split(r"[\s_\-/]+", na) if t]
    toks_b = [t for t in re.split(r"[\s_\-/]+", nb) if t]
    if len(toks_a) >= 2 and "".join(t[0] for t in toks_a) == cb:
        return True
    return len(toks_b) >= 2 and "".join(t[0] for t in toks_b) == ca


def _norm_eq(a: str, b: str) -> bool:
    """Provable identity: names equal after dropping case + every non-alphanumeric.
    'MBPP_'=='MBPP', 'T-DRIVE'=='T -DRIVE', 'Direct Prompting'=='direct prompting'."""
    na = re.sub(r"[^0-9a-z]", "", a.lower())
    return bool(na) and na == re.sub(r"[^0-9a-z]", "", b.lower())


async def _judge_same_entity(
    name_a: str, name_b: str, etype: str, model: str | None
) -> bool | None:
    """Light-LLM verdict for a gray-zone alias pair. True/False = verdict;
    None = LLM error (caller decides fail direction: resolve fails CLOSED —
    no fold; audit fails SAFE — keep the edge).

    Callers MUST check _norm_eq first: case/space/punctuation variants are
    provably the same entity — measured (bench500 2026-07-19) the light LLM
    wrongly answers NO for ~half of them ('MBPP_'/'MBPP', 'Direct Prompting'/
    'direct prompting'), so they never reach the judge."""
    from src.services.ollama_helper import ollama_chat

    prompt = (
        f"Are these two names the SAME real-world {etype.lower()} entity — mere spelling, "
        "case, punctuation, plural or abbreviation/acronym variants of ONE thing?\n"
        "Different versions, models, sizes, subtypes, or related-but-distinct things are "
        "NOT the same.\n"
        f'Name A: "{name_a}"\n'
        f'Name B: "{name_b}"\n'
        "Answer with exactly one word: YES or NO."
    )
    try:
        out = await ollama_chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=5,
        )
        return out.strip().upper().startswith("YES")
    except Exception as e:
        logger.debug(f"_judge_same_entity({name_a!r},{name_b!r}): {e!r}")
        return None


async def resolve_entity_aliases(
    driver,
    qdrant_client,
    collection: str,
    tenant_id: str,
    threshold: float = 0.90,
    max_per_type: int = 4000,
    judge_enabled: bool = False,
    judge_hi: float = 0.92,
    judge_types: list[str] | None = None,
    judge_model: str | None = None,
) -> dict:
    """Embedding-confirmed entity resolution → ALIAS_OF soft-fold (pick #3).

    Lexical proposes (`_lexically_related`), the entity centroid cosine (mean of
    dense chunk vectors, from `entity_vectors.get_entity_vector`) disposes at
    >= `threshold`. Catches acronyms/substrings the Levenshtein tier in
    `canonicalize_entities` misses, without the context-centroid trap of merging
    distinct entities that only co-occur. Canonical = higher CONTAINS_ENTITY degree.

    Soft-fold: creates (alias)-[:ALIAS_OF]->(canon); both nodes and their
    CONTAINS_ENTITY edges persist — PPR (`_load_entity_graph`) and `_entity_pivot`
    collapse aliases at read time. No node-count reduction (that needs a hard
    merge). Idempotent: already-aliased entities are excluded and MERGE is a no-op
    on re-run. Runs as a backfill/repair step (needs chunks already embedded).
    """
    import numpy as np

    from src.services.entity_vectors import get_entity_vector

    # 1. Candidate entities (not already aliases) with chunk-degree, grouped by type.
    by_type: dict[str, list[tuple[str, int]]] = {}
    try:
        async with driver.session() as s:
            result = await s.run(
                """
                MATCH (e:Entity {tenant_id: $tid})
                WHERE NOT (e)-[:ALIAS_OF]->()
                OPTIONAL MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e)
                WITH e, count(DISTINCT c) AS deg
                WHERE deg > 0
                RETURN e.name AS name, coalesce(e.type, 'OTHER') AS type, deg
                ORDER BY deg DESC
                """,
                tid=tenant_id,
            )
            async for r in result:
                by_type.setdefault(r["type"], []).append((r["name"], r["deg"]))
    except Exception as e:
        logger.warning(f"resolve_entity_aliases: candidate fetch failed: {e!r}")
        return {"resolved": 0, "error": str(e)}

    resolved = 0
    canonicals = 0
    judged = 0
    judge_rejected = 0
    by_type_stats: dict[str, int] = {}
    _jtypes = {t.strip().lower() for t in (judge_types or []) if t.strip()}

    for etype, ents in by_type.items():
        if len(ents) > max_per_type:
            logger.warning(
                f"resolve_entity_aliases: type {etype!r} has {len(ents)} entities > "
                f"cap {max_per_type} — capping (scale needs approx-NN blocking)"
            )
            ents = ents[:max_per_type]
        names = [n for n, _ in ents]
        # Only entities with >=1 lexical partner can be a duplicate → skip the rest
        # entirely (no centroid computed for singletons — the dominant cost saver).
        ambiguous: set[str] = set()
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if _lexically_related(names[i], names[j]):
                    ambiguous.add(names[i])
                    ambiguous.add(names[j])
        if not ambiguous:
            continue

        canon: list[tuple[str, Any]] = []  # (name, centroid) in degree-DESC order
        for name, _deg in ents:
            if name not in ambiguous:
                continue
            vec = await get_entity_vector(name, tenant_id, driver, qdrant_client, collection)
            if vec is None:
                continue
            related = [(cn, cv) for cn, cv in canon if _lexically_related(name, cn)]
            best_name, best_cos = None, threshold
            for cname, cvec in related:
                cos = float(np.dot(vec, cvec))
                if cos >= best_cos:
                    best_cos, best_name = cos, cname
            if best_name is None:
                # lexically singleton so far, or cosine says distinct → new canonical
                canon.append((name, vec))
                canonicals += 1
                continue
            # Gray zone [threshold, judge_hi): cosine alone confirmed wrong near-string
            # pairs (llms->MLLMs) — ask the LLM. Error = no fold (fail-closed).
            # Norm-equal names are provably the same → never judged.
            if (
                judge_enabled
                and etype.lower() in _jtypes
                and best_cos < judge_hi
                and not _norm_eq(name, best_name)
            ):
                judged += 1
                verdict = await _judge_same_entity(name, best_name, etype, judge_model)
                if verdict is not True:
                    judge_rejected += 1
                    canon.append((name, vec))
                    canonicals += 1
                    continue
            try:
                async with driver.session() as s:
                    await s.run(
                        """
                        MATCH (alias:Entity {name: $name, tenant_id: $tid})
                        MATCH (canon:Entity {name: $canon, tenant_id: $tid})
                        WHERE alias <> canon
                        MERGE (alias)-[:ALIAS_OF]->(canon)
                        """,
                        name=name,
                        canon=best_name,
                        tid=tenant_id,
                    )
                resolved += 1
                by_type_stats[etype] = by_type_stats.get(etype, 0) + 1
            except Exception as e:
                logger.debug(f"resolve_entity_aliases: fold {name!r}->{best_name!r}: {e}")

    logger.info(
        f"resolve_entity_aliases[{tenant_id}]: {resolved} folded, "
        f"{canonicals} canonicals, threshold={threshold}"
    )
    # New ALIAS_OF edges change PPR's alias map → drop its cached graph.
    try:
        from src.services.ppr import invalidate_cache

        invalidate_cache(tenant_id)
    except Exception:
        pass
    return {
        "resolved": resolved,
        "canonicals": canonicals,
        "by_type": by_type_stats,
        "judged": judged,
        "judge_rejected": judge_rejected,
    }


async def audit_alias_gray_zone(
    driver,
    qdrant_client,
    collection: str,
    tenant_id: str,
    judge_hi: float = 0.92,
    judge_types: list[str] | None = None,
    judge_model: str | None = None,
    delete: bool = True,
) -> dict:
    """Re-score already-written ALIAS_OF pairs of `judge_types`; pairs whose centroid
    cosine sits below `judge_hi` get the LLM judge, and judged-NO edges are DELETED
    (soft-fold is reversible by design — deleting the edge fully undoes the fold).

    Fail direction is deliberate and OPPOSITE to resolve: vector/LLM errors KEEP the
    edge (an audit must not destroy data on infrastructure failure).
    """
    import numpy as np

    from src.services.entity_vectors import get_entity_vector

    types = sorted({t.strip().lower() for t in (judge_types or []) if t.strip()})
    pairs: list[tuple[str, str, str]] = []
    async with driver.session() as s:
        result = await s.run(
            """
            MATCH (a:Entity {tenant_id: $tid})-[:ALIAS_OF]->(c:Entity)
            WHERE toLower(coalesce(a.type, 'OTHER')) IN $types
            RETURN a.name AS alias, coalesce(a.type, 'OTHER') AS etype, c.name AS canon
            """,
            tid=tenant_id,
            types=types,
        )
        async for r in result:
            pairs.append((r["alias"], r["etype"], r["canon"]))

    auto_kept = 0
    norm_kept = 0
    judged_kept = 0
    deleted = 0
    errors = 0
    removed: list[dict] = []
    for alias, etype, canonical in pairs:
        if _norm_eq(alias, canonical):
            norm_kept += 1
            continue
        va = await get_entity_vector(alias, tenant_id, driver, qdrant_client, collection)
        vc = await get_entity_vector(canonical, tenant_id, driver, qdrant_client, collection)
        if va is None or vc is None:
            errors += 1
            continue
        cos = float(np.dot(va, vc))
        if cos >= judge_hi:
            auto_kept += 1
            continue
        verdict = await _judge_same_entity(alias, canonical, etype, judge_model)
        if verdict is None:
            errors += 1
            continue
        if verdict:
            judged_kept += 1
            continue
        if delete:
            async with driver.session() as s:
                await s.run(
                    """
                    MATCH (a:Entity {name: $alias, tenant_id: $tid})
                          -[r:ALIAS_OF]->(c:Entity {name: $canon, tenant_id: $tid})
                    DELETE r
                    """,
                    alias=alias,
                    canon=canonical,
                    tid=tenant_id,
                )
        deleted += 1
        if len(removed) < 50:
            removed.append(
                {"alias": alias, "type": etype, "canon": canonical, "cos": round(cos, 3)}
            )

    logger.info(
        f"audit_alias_gray_zone[{tenant_id}]: {len(pairs)} pairs, norm_kept={norm_kept}, "
        f"auto_kept={auto_kept}, judged_kept={judged_kept}, deleted={deleted}, "
        f"errors={errors} (delete={delete})"
    )
    if deleted and delete:
        try:
            from src.services.ppr import invalidate_cache

            invalidate_cache(tenant_id)
        except Exception:
            pass
    return {
        "pairs": len(pairs),
        "norm_kept": norm_kept,
        "auto_kept": auto_kept,
        "judged_kept": judged_kept,
        "deleted": deleted,
        "errors": errors,
        "delete": delete,
        "removed": removed,
    }


def canonicalize_relationships(rels: list[dict], canonical_map: dict[str, str]) -> list[dict]:
    """
    Replace entity names in relationship source/target with canonical forms.

    Args:
        rels: list of relationship dicts with 'source' and 'target' keys.
        canonical_map: mapping from variant name -> canonical name.

    Returns:
        rels with source/target replaced.
    """
    result = []
    for rel in rels:
        new_rel = dict(rel)
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        new_rel["source"] = canonical_map.get(src, src)
        new_rel["target"] = canonical_map.get(tgt, tgt)
        result.append(new_rel)
    return result


async def extract_entities_and_relations(
    text: str,
    llm: Any,  # kept for backward compat; ignored — uses Ollama native helper
    model: str = "gemma3:4b",
    max_chars: int = 2500,
) -> dict:
    """Use LLM to extract entities + relationships from text. Returns dict.

    Uses Ollama native /api/chat (Phase 0a fix) — OpenAI compat drops think:false
    and Qwen3 returns empty content otherwise.
    """
    from src.services.ollama_helper import ollama_chat

    truncated = text[:max_chars]
    prompt = _ENTITY_EXTRACT_PROMPT.format(text=truncated)

    try:
        raw = await ollama_chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.1,
            max_tokens=512,
        )
        if not raw:
            return {"entities": [], "relationships": []}
        # Strip code fences
        raw = re.sub(r"```(?:json)?\s*|\s*```$", "", raw).strip()
        # Extract first {...} block if LLM wrapped JSON in prose
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            raw = match.group(0)
        if not raw or raw == "{}":
            return {"entities": [], "relationships": []}
        data = json.loads(raw)
        return {
            "entities": data.get("entities", []),
            "relationships": data.get("relationships", []),
        }
    except json.JSONDecodeError:
        # LLM produced unparseable output — silent skip (already logged elsewhere)
        return {"entities": [], "relationships": []}
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        return {"entities": [], "relationships": []}


def _sanitize(name: str) -> str:
    """Normalize string for a Neo4j entity name — the final write gate.

    Collapses internal whitespace/newlines FIRST ("Human\\nResources" → "Human
    Resources"). PDFs wrap entity spans across lines, and canonicalize can adopt an
    older dirty form, so this gate must clean names from every source, not just the
    extractor's clean_entity_name. Then map remaining non-word chars to underscore.
    """
    name = re.sub(r"\s+", " ", name).strip()
    return re.sub(r"[^\w\s\-_]", "_", name)[:200]


async def upsert_chunk_and_entities(
    driver,
    chunk_id: str,
    text: str,
    source: str,
    metadata: dict,
    entities: list[dict],
    relationships: list[dict],
) -> None:
    """
    Store chunk + entities + relationships in Neo4j.

    Promotes key metadata fields to top-level properties so Cypher filters
    can match them:  tenant_id, doc_id, chunk_level, format, consistency_score,
    parent_chunk_id, access_level.
    """
    tenant_id = metadata.get("tenant_id")
    doc_id = metadata.get("doc_id") or source
    chunk_level = metadata.get("chunk_level", "paragraph")
    fmt = metadata.get("format")
    consistency_score = metadata.get("consistency_score")
    parent_chunk_id = metadata.get("parent_chunk_id")
    access_level = metadata.get("access_level", "INTERNAL")

    async with driver.session() as s:
        # Document — set tenant_id at top level
        await s.run(
            """
            MERGE (d:Document {id: $doc_id})
            SET d.source = $source,
                d.tenant_id = coalesce($tenant_id, d.tenant_id),
                d.format = coalesce($fmt, d.format)
            """,
            doc_id=doc_id,
            source=source,
            tenant_id=tenant_id,
            fmt=fmt,
        )

        # Chunk — promote filter-relevant properties to top level
        await s.run(
            """
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = $text,
                c.source = $source,
                c.metadata = $metadata,
                c.tenant_id = coalesce($tenant_id, c.tenant_id),
                c.doc_id = coalesce($doc_id, c.doc_id),
                c.chunk_level = $chunk_level,
                c.format = coalesce($fmt, c.format),
                c.consistency_score = coalesce($consistency, c.consistency_score),
                c.parent_chunk_id = $parent_chunk_id,
                c.access_level = $access_level
            WITH c
            MATCH (d:Document {id: $doc_id})
            MERGE (c)-[:FROM_DOCUMENT]->(d)
            """,
            chunk_id=chunk_id,
            text=text,
            source=source,
            metadata=json.dumps(metadata),
            tenant_id=tenant_id,
            doc_id=doc_id,
            chunk_level=chunk_level,
            fmt=fmt,
            consistency=consistency_score,
            parent_chunk_id=parent_chunk_id,
            access_level=access_level,
        )

        # Hierarchical chunk link (child → parent in same doc)
        if parent_chunk_id and parent_chunk_id != chunk_id:
            await s.run(
                """
                MATCH (c:Chunk {id: $chunk_id})
                MATCH (p:Chunk {id: $parent_id})
                MERGE (c)-[:VARIANT_OF]->(p)
                """,
                chunk_id=chunk_id,
                parent_id=parent_chunk_id,
            )

        # Idempotent entity links: drop THIS chunk's existing CONTAINS_ENTITY before
        # re-creating from the current extraction. Without this, re-ingesting or
        # /repair/build with a cleaner extractor MERGEs new entities on top of the
        # old ones, so stale noise (concept/event spans, "\n"-broken names) stays
        # linked forever. Entity nodes left with no chunk are removed by
        # delete_orphan_entities (called at end of ingest / repair).
        await s.run(
            "MATCH (c:Chunk {id: $chunk_id})-[r:CONTAINS_ENTITY]->() DELETE r",
            chunk_id=chunk_id,
        )

        # Entities — set tenant_id + confidence on Entity node
        for entity in entities:
            name = _sanitize(entity.get("name", ""))
            if not name:
                continue
            etype = entity.get("type", "OTHER")
            desc = entity.get("description", "")[:500]
            confidence = float(entity.get("confidence", 1.0))
            vote_count = int(entity.get("vote_count", 1))
            await s.run(
                """
                MERGE (e:Entity {name: $name, tenant_id: $tenant_id})
                SET e.type = $etype,
                    e.description = $desc,
                    e.confidence = $confidence,
                    e.vote_count = $vote_count
                WITH e
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:CONTAINS_ENTITY]->(e)
                """,
                name=name,
                etype=etype,
                desc=desc,
                tenant_id=tenant_id,
                confidence=confidence,
                vote_count=vote_count,
                chunk_id=chunk_id,
            )

        for rel in relationships:
            src = _sanitize(rel.get("source", ""))
            tgt = _sanitize(rel.get("target", ""))
            if not src or not tgt:
                continue
            desc = rel.get("description", "")[:500]
            confidence = float(rel.get("confidence", 1.0))
            vote_count = int(rel.get("vote_count", 1))
            rel_type = rel.get("type", "RELATES_TO")[:50]
            await s.run(
                """
                // These two MERGEs are where the 219 tenant_id-null entities came from:
                // they keyed on name only and never set tenant_id at all, so any entity
                // first seen as a relationship endpoint was born tenant-less and stayed
                // invisible to community.py / hefr.py forever. Keyed on (name, tenant_id)
                // now, matching the entity MERGE above and the composite constraint.
                MERGE (s:Entity {name: $src, tenant_id: $tenant_id})
                MERGE (t:Entity {name: $tgt, tenant_id: $tenant_id})
                MERGE (s)-[r:RELATES_TO]->(t)
                SET r.description = $desc,
                    r.confidence = $confidence,
                    r.vote_count = $vote_count,
                    r.rel_type = $rel_type
                """,
                src=src,
                tgt=tgt,
                tenant_id=tenant_id,
                desc=desc,
                confidence=confidence,
                vote_count=vote_count,
                rel_type=rel_type,
            )

        logger.debug(
            f"Neo4j: {len(entities)} entities, {len(relationships)} rels from chunk {chunk_id}"
        )


async def delete_orphan_entities(driver, tenant_id: str) -> int:
    """Remove tenant Entity nodes no chunk references anymore.

    The idempotent write in upsert_chunk_and_entities detaches a chunk's old
    CONTAINS_ENTITY before re-linking, which can leave an Entity node with zero
    incoming links (a re-ingest / cleaner-extractor dropped it). Those orphans are
    inert for retrieval (entity paths always start from a chunk) but pollute counts
    and name-matching, so sweep them after a batch. Scoped to the tenant and to
    degree-0 nodes — an entity still linked by any chunk is kept.
    """
    async with driver.session() as s:
        r = await s.run(
            """
            MATCH (e:Entity {tenant_id: $tenant_id})
            WHERE NOT (e)<-[:CONTAINS_ENTITY]-()
            DETACH DELETE e
            RETURN count(e) AS deleted
            """,
            tenant_id=tenant_id,
        )
        rec = await r.single()
        return rec["deleted"] if rec else 0


def _load_chunk_metadata(raw: Any) -> dict:
    """Chunk.metadata is persisted as a JSON string (see upsert_chunk_and_entities).

    Callers treat it as a mapping, so decode it here rather than leaking the raw
    string across the boundary.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


async def graph_retrieve(
    driver,
    query_embedding: list[float],
    http_client: httpx.AsyncClient,
    embed_url: str,
    embed_model: str,
    top_k: int = 5,
    tenant_id: str | None = None,
) -> list[dict]:
    """
    Graph-based retrieval via entity description embedding similarity.

    1. Fetch entity descriptions from Neo4j (limit 500 for performance)
    2. Batch-embed descriptions via Ollama
    3. Score by cosine similarity to query
    4. Fetch chunks linked to top entities
    5. Return scored chunks
    """
    from src.services.embedding import cosine_similarity, embed_batch

    # Tenant filter pushed into the query: without it this fetched up to 500 entity
    # descriptions across ALL tenants and embedded every one (~5.7s), then relied on a
    # post-hoc filter in the caller — wasted work plus a cross-tenant description leak.
    where_e = "AND e.tenant_id = $tid" if tenant_id else ""
    async with driver.session() as s:
        result = await s.run(
            f"""
            MATCH (e:Entity)
            WHERE e.description IS NOT NULL AND e.description <> '' {where_e}
            RETURN e.name AS name, e.type AS type, e.description AS description
            LIMIT 500
            """,
            **({"tid": tenant_id} if tenant_id else {}),
        )
        records = await result.data()

    if not records:
        return []

    entities = [
        {"name": r["name"], "type": r["type"], "description": r["description"]} for r in records
    ]

    # Batch embed descriptions
    try:
        embeds = await embed_batch(
            http_client,
            embed_url,
            embed_model,
            [e["description"] for e in entities],
            batch_size=16,
            timeout=120.0,
        )
    except Exception as e:
        logger.warning(f"Graph retrieval embed failed: {e}")
        return []

    scored = [
        (entities[i], cosine_similarity(query_embedding, vec))
        for i, vec in enumerate(embeds)
        if vec and any(v != 0 for v in vec)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_entities = [e for e, _ in scored[: top_k * 3]]

    if not top_entities:
        return []

    names = [e["name"] for e in top_entities]

    where_c = "AND c.tenant_id = $tid" if tenant_id else ""
    params: dict[str, Any] = {"names": names, "top_k": top_k}
    if tenant_id:
        params["tid"] = tenant_id
    async with driver.session() as s:
        result = await s.run(
            f"""
            MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
            WHERE e.name IN $names {where_c}
            WITH c, collect(DISTINCT e.name) AS matched, count(e) AS cnt
            RETURN c.id AS chunk_id, c.text AS text,
                   c.source AS source, c.metadata AS metadata,
                   matched, cnt
            ORDER BY cnt DESC
            LIMIT $top_k
            """,
            **params,
        )
        records = await result.data()

    chunks = []
    for record in records:
        matched = set(record["matched"])
        score = sum(s for e, s in scored if e["name"] in matched) / len(matched) if matched else 0.0
        chunks.append(
            {
                "chunk_id": record["chunk_id"],
                "text": record["text"],
                "source": record["source"],
                "metadata": _load_chunk_metadata(record.get("metadata")),
                "graph_score": score,
                "matched_entities": record["matched"],
                "retrieval_mode": "graph",
            }
        )

    chunks.sort(key=lambda x: x["graph_score"], reverse=True)
    return chunks[:top_k]


async def link_semantic_chunks(
    driver,
    source_chunk_id: str,
    target_chunks: list[tuple[str, float]],
) -> None:
    """
    Tạo liên kết ngữ nghĩa (Semantic Edge) giữa các đoạn văn bản (chunks)
    khác nhau dựa trên độ tương đồng của Vector Embedding.
    Giúp tối ưu GraphRAG bằng cách kết nối tri thức xuyên tài liệu (Cross-Document).
    """
    if not target_chunks:
        return

    async with driver.session() as s:
        for target_id, score in target_chunks:
            if target_id == source_chunk_id or score < 0.70:
                continue

            await s.run(
                """
                MATCH (c1:Chunk {id: $source_id})
                MATCH (c2:Chunk {id: $target_id})
                MERGE (c1)-[r:SIMILAR_TO]->(c2)
                SET r.score = $score
                """,
                source_id=source_chunk_id,
                target_id=target_id,
                score=score,
            )
