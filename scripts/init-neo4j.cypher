// ============================================================================
// Init Neo4j schema cho Hybrid GraphRAG
// Schema: (Chunk)-[:CONTAINS_ENTITY]->(Entity)-[:RELATES_TO]->(Entity)
//         (Chunk)-[:FROM_DOCUMENT]->(Document)
// Chạy: cat scripts/init-neo4j.cypher | docker exec -i rag-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD
// ============================================================================

// ── Constraints (uniqueness) ──────────────────────────────────────────────
CREATE CONSTRAINT chunk_id IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT entity_name IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS UNIQUE;

CREATE CONSTRAINT document_id IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

// ── Indexes cho performance ────────────────────────────────────────────────
CREATE INDEX chunk_source IF NOT EXISTS
FOR (c:Chunk) ON (c.source);

CREATE INDEX chunk_text IF NOT EXISTS
FOR (c:Chunk) ON (c.text);

CREATE INDEX entity_type IF NOT EXISTS
FOR (e:Entity) ON (e.type);

CREATE INDEX entity_description IF NOT EXISTS
FOR (e:Entity) ON (e.description);

CREATE INDEX document_source IF NOT EXISTS
FOR (d:Document) ON (d.source);

// ── Full-text search index cho entity descriptions ────────────────────────
CREATE FULLTEXT INDEX entity_fts IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name, e.description];

CREATE FULLTEXT INDEX chunk_fts IF NOT EXISTS
FOR (c:Chunk) ON EACH [c.text];

// ── Verify ────────────────────────────────────────────────────────────────
SHOW CONSTRAINTS;
SHOW INDEXES;
