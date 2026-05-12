#!/usr/bin/env python3
"""
Verify Neo4j graph schema and relationships via HTTP API (no neo4j-driver needed).

Checks:
  1. All expected node labels present
  2. All expected relationship types
  3. Counts per type — flag empty types
  4. Sample 3 edges per type to verify direction + properties
  5. tenant_id propagation
  6. Document↔Document linkage

Usage:
  python3 scripts/verify_graph.py --tenant default
  python3 scripts/verify_graph.py --http http://localhost:7474 --password ...

Reads NEO4J_PASSWORD from env or .env if not provided.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import os
import sys
from pathlib import Path
from typing import Any

import httpx


EXPECTED_LABELS = ["Document", "Chunk", "Entity", "Community"]
EXPECTED_REL_TYPES = {
    "FROM_DOCUMENT":    "(:Chunk)->(:Document)",
    "CONTAINS_ENTITY":  "(:Chunk)->(:Entity)",
    "RELATES_TO":       "(:Entity)->(:Entity)",
    "VARIANT_OF":       "(:Chunk)->(:Chunk) hierarchical, in-doc",
    "SIMILAR_TO":       "(:Chunk)->(:Chunk) in-doc or cross-doc",
    "SHARES_ENTITIES":  "(:Document)->(:Document) cross-doc entity overlap",
    "SIMILAR_DOC":      "(:Document)->(:Document) cross-doc aggregate",
    "IN_COMMUNITY":     "(:Entity)->(:Community)",
    "SUB_COMMUNITY_OF": "(:Community)->(:Community)",
}


def banner(msg: str) -> None:
    print("\n" + "─" * 70)
    print(f"  {msg}")
    print("─" * 70)


def ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def load_password_from_env(env_file: Path = Path(".env")) -> str | None:
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        if line.startswith("NEO4J_PASSWORD="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


async def cypher(
    client: httpx.AsyncClient,
    http_url: str,
    user: str,
    password: str,
    statement: str,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run Cypher via Neo4j HTTP API. Returns list of row dicts."""
    auth = base64.b64encode(f"{user}:{password}".encode()).decode()
    body = {"statements": [{"statement": statement, "parameters": params or {}}]}
    resp = await client.post(
        f"{http_url}/db/neo4j/tx/commit",
        json=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Basic {auth}",
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        raise RuntimeError(f"Cypher error: {data['errors']}")
    results = data.get("results", [])
    if not results:
        return []
    columns = results[0]["columns"]
    return [dict(zip(columns, row["row"])) for row in results[0]["data"]]


async def main(http_url: str, user: str, password: str, tenant: str | None) -> int:
    issues = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        # ── 1. Node labels ───────────────────────────────────────────────────
        banner("1. Node labels")
        for label in EXPECTED_LABELS:
            stmt = f"MATCH (n:{label}) RETURN count(n) AS c"
            params: dict[str, Any] = {}
            if tenant:
                stmt = f"MATCH (n:{label}) WHERE n.tenant_id = $tid RETURN count(n) AS c"
                params["tid"] = tenant
            try:
                rows = await cypher(client, http_url, user, password, stmt, params)
                count = int(rows[0]["c"]) if rows else 0
            except Exception as e:
                fail(f"{label}: Cypher failed: {e}")
                issues += 1
                continue
            if count == 0:
                warn(f"{label}: 0 nodes")
                if label == "Document":
                    issues += 1
            else:
                ok(f"{label}: {count} nodes")

        # ── 2. Relationship types ────────────────────────────────────────────
        banner("2. Relationship types")
        rel_counts: dict[str, int] = {}
        for rel, desc in EXPECTED_REL_TYPES.items():
            try:
                rows = await cypher(
                    client, http_url, user, password,
                    f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c",
                )
                count = int(rows[0]["c"]) if rows else 0
            except Exception as e:
                fail(f"{rel}: Cypher failed: {e}")
                continue
            rel_counts[rel] = count
            if count == 0:
                if rel in ("SHARES_ENTITIES", "SIMILAR_DOC", "IN_COMMUNITY", "SUB_COMMUNITY_OF"):
                    warn(f"{rel}: 0 — run cross_doc/community build to populate ({desc})")
                elif rel == "VARIANT_OF":
                    warn(f"{rel}: 0 — hierarchical chunking may not be active ({desc})")
                else:
                    fail(f"{rel}: 0 — expected non-zero ({desc})")
                    issues += 1
            else:
                ok(f"{rel}: {count}  ({desc})")

        # ── 3. Cross-doc SIMILAR_TO breakdown ────────────────────────────────
        banner("3. Cross-doc SIMILAR_TO breakdown")
        try:
            rows = await cypher(
                client, http_url, user, password,
                "MATCH (a:Chunk)-[s:SIMILAR_TO]->(b:Chunk) "
                "WHERE a.id <> b.id "
                "RETURN s.cross_doc AS cd, count(s) AS c",
            )
            in_doc = sum(r["c"] for r in rows if not r.get("cd"))
            cross_doc = sum(r["c"] for r in rows if r.get("cd"))
            ok(f"SIMILAR_TO in-doc: {in_doc}")
            if cross_doc == 0:
                warn(f"SIMILAR_TO cross-doc: 0 — run `make v2-cross-doc` to populate")
            else:
                ok(f"SIMILAR_TO cross-doc: {cross_doc}")
        except Exception as e:
            warn(f"SIMILAR_TO breakdown failed: {e}")

        # ── 4. Sample edges per type ─────────────────────────────────────────
        banner("4. Sample edges (3 per non-empty type)")
        for rel in EXPECTED_REL_TYPES:
            if rel_counts.get(rel, 0) == 0:
                continue
            try:
                rows = await cypher(
                    client, http_url, user, password,
                    f"MATCH (a)-[r:{rel}]->(b) "
                    f"RETURN labels(a) AS la, labels(b) AS lb, properties(r) AS props "
                    f"LIMIT 3",
                )
            except Exception:
                continue
            if not rows:
                continue
            print(f"\n  {rel}:")
            for row in rows:
                props = {
                    k: (v[:60] + "..." if isinstance(v, str) and len(v) > 60 else v)
                    for k, v in (row.get("props") or {}).items()
                    if k != "updated_at"
                }
                print(f"    {row['la']} → {row['lb']}  props={props}")

        # ── 5. tenant_id propagation ─────────────────────────────────────────
        banner("5. tenant_id propagation")
        for label in ["Chunk", "Entity", "Document"]:
            try:
                rows = await cypher(
                    client, http_url, user, password,
                    f"MATCH (n:{label}) "
                    f"RETURN sum(CASE WHEN n.tenant_id IS NOT NULL THEN 1 ELSE 0 END) AS with_t, "
                    f"       sum(CASE WHEN n.tenant_id IS NULL THEN 1 ELSE 0 END) AS no_t",
                )
            except Exception:
                continue
            if not rows:
                continue
            with_t = int(rows[0]["with_t"] or 0)
            no_t = int(rows[0]["no_t"] or 0)
            if no_t > 0:
                warn(f"{label}: {no_t} nodes WITHOUT tenant_id (potential leak risk)")
                issues += 1
            elif with_t > 0:
                ok(f"{label}: all {with_t} have tenant_id")

        # ── 6. Document↔Document edges ───────────────────────────────────────
        banner("6. Document↔Document linkage")
        try:
            rows = await cypher(
                client, http_url, user, password,
                "MATCH (d1:Document)-[r:SHARES_ENTITIES|SIMILAR_DOC]->(d2:Document) "
                "RETURN type(r) AS t, count(*) AS c",
            )
        except Exception as e:
            warn(f"Doc-Doc check failed: {e}")
            rows = []
        if not rows:
            warn("NO Document↔Document edges — run `POST /api/v3/cross_doc/build`")
            issues += 1
        else:
            for row in rows:
                ok(f"{row['t']}: {row['c']} edges")

        # ── 7. Orphan check ──────────────────────────────────────────────────
        banner("7. Orphan check")
        try:
            rows = await cypher(
                client, http_url, user, password,
                "MATCH (c:Chunk) WHERE NOT (c)-[:FROM_DOCUMENT]->() RETURN count(c) AS c",
            )
            orphan_chunks = int(rows[0]["c"]) if rows else 0
            if orphan_chunks > 0:
                warn(f"{orphan_chunks} Chunks without FROM_DOCUMENT edge")
            else:
                ok("No orphan Chunks")
        except Exception as e:
            warn(f"Orphan-Chunk check failed: {e}")

        try:
            rows = await cypher(
                client, http_url, user, password,
                "MATCH (e:Entity) WHERE NOT (:Chunk)-[:CONTAINS_ENTITY]->(e) RETURN count(e) AS c",
            )
            orphan_entities = int(rows[0]["c"]) if rows else 0
            if orphan_entities > 0:
                warn(f"{orphan_entities} Entities not referenced by any Chunk")
            else:
                ok("No orphan Entities")
        except Exception as e:
            warn(f"Orphan-Entity check failed: {e}")

    banner("Summary")
    if issues == 0:
        print("  All graph invariants OK.")
    else:
        print(f"  {issues} issue(s) flagged above.")
    return 0 if issues == 0 else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--http", default=os.environ.get("NEO4J_HTTP", "http://localhost:7474"))
    p.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    p.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD") or load_password_from_env() or "")
    p.add_argument("--tenant", default=None)
    args = p.parse_args()

    if not args.password:
        print("ERROR: NEO4J_PASSWORD not found. Set env var or pass --password.", file=sys.stderr)
        sys.exit(2)

    sys.exit(asyncio.run(main(args.http, args.user, args.password, args.tenant)))
