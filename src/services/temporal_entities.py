"""Phase 6b — Temporal entity versioning.

Most GraphRAG papers ignore time. Enterprise reality: facts evolve.
  - "CEO" today != CEO 2 years ago
  - "Project X" budget changed each quarter
  - "Company Y" was renamed

Schema:
  (:Entity {name, valid_from, valid_to, version, tenant_id})

Same canonical name → multiple Entity nodes with non-overlapping time ranges.

Query rewriting:
  "current X"       → filter valid_to IS NULL (active version)
  "X in 2021"        → filter valid_from <= '2021-12-31' AND (valid_to IS NULL OR valid_to >= '2021-01-01')
  "latest X"         → ORDER BY valid_from DESC LIMIT 1
  default            → use active (valid_to IS NULL)

This module:
  1. Helper to detect time references in queries
  2. Cypher rewriter to add temporal filter
  3. Migration helper to add valid_from/valid_to to existing entities
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

_TIME_REF_PATTERNS = {
    "current": [r"\b(hiện tại|hiện nay|bây giờ|now|current|today)\b"],
    "latest": [r"\b(mới nhất|gần đây nhất|latest|most recent)\b"],
    "year": [r"\b(năm|in)\s+(\d{4})\b"],
    "quarter": [r"\b(quý|Q)\s*([1-4])\s*(năm)?\s*(\d{4})\b", r"\b(Q[1-4])\s+(\d{4})\b"],
    "month": [r"\b(tháng|month)\s+(\d{1,2})(?:\s*(năm|of)\s+(\d{4}))?\b"],
    "past": [r"\b(năm ngoái|tháng trước|last year|last month)\b"],
}


def detect_temporal_intent(query: str) -> dict[str, Any]:
    """Parse temporal intent from query.

    Returns:
      {
        "type": "current" | "latest" | "specific" | "past" | "none",
        "year": int | None,
        "quarter": int | None,
        "month": int | None,
        "filter_cypher": str  # ready-to-use Cypher WHERE fragment for Entity
      }
    """
    q = query.lower()
    result = {
        "type": "none",
        "year": None,
        "quarter": None,
        "month": None,
        "filter_cypher": "",
    }

    # Check "current"
    if any(re.search(p, q) for p in _TIME_REF_PATTERNS["current"]):
        result["type"] = "current"
        result["filter_cypher"] = "e.valid_to IS NULL"
        return result

    if any(re.search(p, q) for p in _TIME_REF_PATTERNS["latest"]):
        result["type"] = "latest"
        result["filter_cypher"] = "true  /* sort by valid_from DESC */"
        return result

    # Check year
    for p in _TIME_REF_PATTERNS["year"]:
        m = re.search(p, q)
        if m:
            year = int(m.group(2))
            result["type"] = "specific"
            result["year"] = year
            result["filter_cypher"] = (
                f"(e.valid_from IS NULL OR e.valid_from <= date('{year}-12-31')) "
                f"AND (e.valid_to IS NULL OR e.valid_to >= date('{year}-01-01'))"
            )
            return result

    # Check quarter
    for p in _TIME_REF_PATTERNS["quarter"]:
        m = re.search(p, q)
        if m:
            q_num = int(m.group(2)) if m.group(2).isdigit() else int(m.group(1)[1:])
            year_str = next((g for g in m.groups() if g and len(g) == 4 and g.isdigit()), None)
            year = int(year_str) if year_str else datetime.now().year
            q_start_month = 3 * (q_num - 1) + 1
            q_end_month = q_start_month + 2
            result["type"] = "specific"
            result["year"] = year
            result["quarter"] = q_num
            result["filter_cypher"] = (
                f"(e.valid_from IS NULL OR e.valid_from <= date('{year}-{q_end_month:02d}-30')) "
                f"AND (e.valid_to IS NULL OR e.valid_to >= date('{year}-{q_start_month:02d}-01'))"
            )
            return result

    if any(re.search(p, q) for p in _TIME_REF_PATTERNS["past"]):
        result["type"] = "past"
        result["filter_cypher"] = "e.valid_to IS NOT NULL"
        return result

    # Default: no temporal intent, return active version
    result["filter_cypher"] = "e.valid_to IS NULL OR e.valid_to IS NOT NULL"
    return result


async def add_temporal_to_entity(
    neo4j_driver,
    entity_name: str,
    tenant_id: str,
    valid_from: str | None = None,
    valid_to: str | None = None,
    version: int = 1,
):
    """Add or update temporal range on an Entity node."""
    cypher = """
    MATCH (e:Entity {name: $name, tenant_id: $tid})
    SET e.valid_from = date($from),
        e.valid_to = CASE WHEN $to IS NULL THEN NULL ELSE date($to) END,
        e.version = $version
    RETURN e
    """
    async with neo4j_driver.session() as s:
        await s.run(
            cypher,
            name=entity_name,
            tid=tenant_id,
            **{
                "from": valid_from or datetime.now(UTC).date().isoformat(),
                "to": valid_to,
                "version": version,
            },
        )


async def migrate_entities_set_default_validity(neo4j_driver, tenant_id: str) -> dict[str, int]:
    """Migration: set valid_from to created_at, valid_to NULL for existing entities."""
    cypher = """
    MATCH (e:Entity {tenant_id: $tid})
    WHERE e.valid_from IS NULL
    SET e.valid_from = coalesce(date(e.created_at), date()),
        e.valid_to = NULL,
        e.version = 1
    RETURN count(e) AS migrated
    """
    async with neo4j_driver.session() as s:
        r = await s.run(cypher, tid=tenant_id)
        row = await r.single()
    return {"migrated": int(row["migrated"]) if row else 0, "tenant_id": tenant_id}


async def query_entity_at_time(
    neo4j_driver,
    name: str,
    tenant_id: str,
    as_of: str | None = None,  # ISO date "2024-01-15"
) -> list[dict]:
    """Find entity version active at `as_of`. If None, current."""
    if as_of:
        cypher = """
        MATCH (e:Entity {name: $name, tenant_id: $tid})
        WHERE (e.valid_from IS NULL OR e.valid_from <= date($as_of))
          AND (e.valid_to IS NULL OR e.valid_to >= date($as_of))
        RETURN e.name AS name, e.type AS type, e.description AS desc,
               e.valid_from AS valid_from, e.valid_to AS valid_to, e.version AS version
        ORDER BY e.valid_from DESC
        """
    else:
        cypher = """
        MATCH (e:Entity {name: $name, tenant_id: $tid})
        WHERE e.valid_to IS NULL
        RETURN e.name AS name, e.type AS type, e.description AS desc,
               e.valid_from AS valid_from, e.valid_to AS valid_to, e.version AS version
        ORDER BY e.valid_from DESC LIMIT 1
        """
    async with neo4j_driver.session() as s:
        r = await s.run(cypher, name=name, tid=tenant_id, as_of=as_of)
        rows = await r.data()
    return rows
