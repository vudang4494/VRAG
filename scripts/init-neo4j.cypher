// ============================================================================
// Init Neo4j schema — Hybrid GraphRAG with Pipeline V2 extensions
//
// V1 schema:
//   (:Chunk)-[:CONTAINS_ENTITY]->(:Entity)-[:RELATES_TO]->(:Entity)
//   (:Chunk)-[:FROM_DOCUMENT]->(:Document)
//
// V2 additions:
//   (:Chunk)-[:VARIANT_OF]->(:Chunk)               // hierarchical (child → parent)
//   (:Chunk)-[:SIMILAR_TO {score, view}]->(:Chunk) // cross-doc semantic link
//   (:Entity)-[:IN_COMMUNITY {level}]->(:Community)
//   (:Community)-[:SUB_COMMUNITY_OF]->(:Community)
//
// Run: cat scripts/init-neo4j.cypher | docker exec -i rag-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD
// ============================================================================

// ── Constraints (uniqueness) ──────────────────────────────────────────────
CREATE CONSTRAINT chunk_id IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

// Entity identity is (name, tenant_id), not name alone. A global unique-name constraint
// forced two tenants onto one node and made e.tenant_id last-writer-wins, so the losing
// tenant vanished from community.py / hefr.py, which both filter on e.tenant_id.
// Must stay in sync with kg._SCHEMA_STATEMENTS — startup applies that one, this file is
// the manual mirror. The DROP is required: CREATE ... IF NOT EXISTS will not replace it.
DROP CONSTRAINT entity_name IF EXISTS;

CREATE CONSTRAINT entity_name_tenant IF NOT EXISTS
FOR (e:Entity) REQUIRE (e.name, e.tenant_id) IS UNIQUE;

CREATE CONSTRAINT document_id IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT community_id IF NOT EXISTS
FOR (com:Community) REQUIRE com.id IS UNIQUE;

// ── Indexes cho performance ────────────────────────────────────────────────
CREATE INDEX chunk_source IF NOT EXISTS
FOR (c:Chunk) ON (c.source);

CREATE INDEX chunk_tenant IF NOT EXISTS
FOR (c:Chunk) ON (c.tenant_id);

CREATE INDEX chunk_level IF NOT EXISTS
FOR (c:Chunk) ON (c.chunk_level);

CREATE INDEX chunk_format IF NOT EXISTS
FOR (c:Chunk) ON (c.format);

CREATE INDEX chunk_consistency IF NOT EXISTS
FOR (c:Chunk) ON (c.consistency_score);

CREATE INDEX entity_type IF NOT EXISTS
FOR (e:Entity) ON (e.type);

CREATE INDEX entity_tenant IF NOT EXISTS
FOR (e:Entity) ON (e.tenant_id);

CREATE INDEX entity_confidence IF NOT EXISTS
FOR (e:Entity) ON (e.confidence);

CREATE INDEX entity_description IF NOT EXISTS
FOR (e:Entity) ON (e.description);

CREATE INDEX document_source IF NOT EXISTS
FOR (d:Document) ON (d.source);

CREATE INDEX document_tenant IF NOT EXISTS
FOR (d:Document) ON (d.tenant_id);

CREATE INDEX community_tenant_level IF NOT EXISTS
FOR (com:Community) ON (com.tenant_id, com.level);

// ── Full-text search indexes ───────────────────────────────────────────────
CREATE FULLTEXT INDEX entity_fts IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name, e.description];

CREATE FULLTEXT INDEX chunk_fts IF NOT EXISTS
FOR (c:Chunk) ON EACH [c.text];

CREATE FULLTEXT INDEX community_summary_fts IF NOT EXISTS
FOR (com:Community) ON EACH [com.summary];

// ── Verify ────────────────────────────────────────────────────────────────
SHOW CONSTRAINTS;
SHOW INDEXES;
