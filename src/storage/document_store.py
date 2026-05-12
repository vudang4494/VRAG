"""Multi-tenant document storage — Qdrant + Neo4j + Postgres."""
import asyncio
import hashlib
import json
import uuid
from datetime import datetime
from typing import Any

from loguru import logger


class DocumentStore:
    """
    Unified document storage with multi-tenant isolation.

    - Qdrant: vectors indexed per-tenant (collection naming: {tenant_id})
    - Neo4j: labels partitioned by tenant via relationship prefixes
    - Postgres: structured metadata, audit logs, source configs
    """

    def __init__(self):
        self._clients: dict[str, Any] = {}
        self._initialized = False

    async def _ensure_clients(self) -> None:
        if self._initialized:
            return

        from src.clients import get_clients
        clients = get_clients()
        self._clients = {
            "qdrant": clients.qdrant,
            "neo4j": clients.neo4j,
            "http": clients.http,
        }
        self._initialized = True

    # -------------------------------------------------------------------------
    # Document lifecycle
    # -------------------------------------------------------------------------

    async def ingest_document(
        self,
        tenant_id: str,
        source_id: str,
        doc: Any,  # ParsedDocument
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        chunk_strategy: str = "fixed",
    ) -> dict[str, Any]:
        """
        Full ingestion pipeline for a single document.

        Returns IngestResult dict with chunk/entity/relationship counts.
        """
        await self._ensure_clients()

        doc_id = str(uuid.uuid4())
        doc_hash = hashlib.md5(
            doc.content.encode() if hasattr(doc, "content") else str(doc).encode()
        ).hexdigest()[:16]

        chunks = self._chunk_text(
            doc.content if hasattr(doc, "content") else str(doc),
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            strategy=chunk_strategy,
        )

        from src.services.embedding import embed_batch
        from src.config import get_settings
        settings = get_settings()

        vectors = await embed_batch(
            self._clients["http"],
            settings.ollama_embed_url,
            settings.ollama_embed_model,
            [c["text"] for c in chunks],
            timeout=60.0,
        )

        from src.services.kg import extract_entities_and_relations
        kg_task = extract_entities_and_relations(
            self._clients["neo4j"],
            self._clients["http"],
            settings.ollama_base_url,
            settings.ollama_model,
            chunks,
        )
        upsert_task = self._upsert_chunks(
            tenant_id=tenant_id,
            source_id=source_id,
            doc_id=doc_id,
            doc_hash=doc_hash,
            doc=doc,
            chunks=chunks,
            vectors=vectors,
        )
        kg_result, _ = await asyncio.gather(kg_task, upsert_task)
        kg_entities, kg_relationships = kg_result

        return {
            "document_id": doc_id,
            "chunk_count": len(chunks),
            "entity_count": kg_entities,
            "relationship_count": kg_relationships,
            "failed_chunks": 0,
        }

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 64,
        strategy: str = "fixed",
    ) -> list[dict[str, Any]]:
        if strategy == "sentence":
            return self._chunk_sentence(text, chunk_size, overlap)
        return self._chunk_fixed(text, chunk_size, overlap)

    def _chunk_fixed(
        self, text: str, size: int, overlap: int
    ) -> list[dict[str, Any]]:
        import re
        chunks = []
        pos = 0
        idx = 0
        while pos < len(text):
            end = min(pos + size, len(text))
            chunk_text = text[pos:end]
            if idx > 0 and chunks:
                overlap_text = chunks[-1]["text"][-overlap:]
                if overlap_text in chunk_text:
                    chunk_text = chunk_text.replace(overlap_text, "", 1)
            chunks.append({
                "text": chunk_text.strip(),
                "start": pos,
                "end": end,
                "chunk_index": idx,
            })
            pos += size - overlap
            idx += 1
        return [c for c in chunks if len(c["text"]) >= 50]

    def _chunk_sentence(
        self, text: str, size: int, overlap: int
    ) -> list[dict[str, Any]]:
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""
        idx = 0
        for sent in sentences:
            if len(current) + len(sent) <= size:
                current += (" " + sent).strip()
            else:
                if current.strip():
                    chunks.append({"text": current.strip(), "chunk_index": idx})
                    idx += 1
                overlap_text = current[-overlap:] if overlap > 0 else ""
                current = (overlap_text + " " + sent).strip()
        if current.strip():
            chunks.append({"text": current.strip(), "chunk_index": idx})
        return [c for c in chunks if len(c["text"]) >= 50]

    async def _upsert_chunks(
        self,
        tenant_id: str,
        source_id: str,
        doc_id: str,
        doc_hash: str,
        doc: Any,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> None:
        qdrant = self._clients["qdrant"]
        neo4j = self._clients["neo4j"]

        from src.services.vector import upsert_points, _to_int_id
        from src.config import get_settings
        settings = get_settings()

        points = []
        for c, vec in zip(chunks, vectors):
            chunk_id = f"{tenant_id}_{doc_hash}_{c['chunk_index']}"
            int_id = _to_int_id(chunk_id)
            points.append({
                "id": int_id,
                "vector": vec,
                "payload": {
                    "text": c["text"],
                    "source": doc.url if hasattr(doc, "url") else "unknown",
                    "chunk_index": c["chunk_index"],
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "tenant_id": tenant_id,
                    "source_id": source_id,
                    "title": doc.title if hasattr(doc, "title") else "Untitled",
                    "tags": json.dumps(doc.metadata.get("tags", [])) if hasattr(doc, "metadata") else "[]",
                    "access_level": doc.metadata.get("access_level", "internal") if hasattr(doc, "metadata") else "internal",
                    "department": doc.metadata.get("department", "") if hasattr(doc, "metadata") else "",
                    "author": doc.author if hasattr(doc, "author") else "",
                },
            })

        await upsert_points(qdrant, settings.qdrant_collection, points)

        async with neo4j.session() as session:
            await session.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d.tenant_id = $tenant_id, d.source_id = $source_id,
                    d.title = $title, d.doc_hash = $doc_hash,
                    d.created_at = datetime(),
                    d.updated_at = datetime()
                """,
                doc_id=doc_id, tenant_id=tenant_id, source_id=source_id,
                title=doc.title if hasattr(doc, "title") else "Untitled", doc_hash=doc_hash,
            )
            for c in chunks:
                chunk_id = f"{tenant_id}_{doc_hash}_{c['chunk_index']}"
                await session.run(
                    """
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.tenant_id = $tenant_id, c.text = $text,
                        c.source = $source, c.chunk_index = $idx,
                        c.doc_id = $doc_id
                    """,
                    chunk_id=chunk_id, tenant_id=tenant_id, text=c["text"],
                    source=doc.url if hasattr(doc, "url") else "unknown",
                    idx=c["chunk_index"], doc_id=doc_id,
                )
                await session.run(
                    """
                    MATCH (c:Chunk {id: $chunk_id}), (d:Document {id: $doc_id})
                    MERGE (c)-[:FROM_DOCUMENT]->(d)
                    """,
                    chunk_id=chunk_id, doc_id=doc_id,
                )

    async def delete_document(self, tenant_id: str, doc_id: str) -> bool:
        """Delete a document and all its chunks from all stores."""
        await self._ensure_clients()
        neo4j = self._clients["neo4j"]
        qdrant = self._clients["qdrant"]
        from src.config import get_settings
        settings = get_settings()

        async with neo4j.session() as session:
            result = await session.run(
                """
                MATCH (c:Chunk {doc_id: $doc_id, tenant_id: $tenant_id})
                RETURN c.id as id
                """,
                doc_id=doc_id, tenant_id=tenant_id,
            )
            chunk_ids = [record["id"] async for record in result]

            await session.run(
                """
                MATCH (c:Chunk {doc_id: $doc_id, tenant_id: $tenant_id})
                DETACH DELETE c
                """,
                doc_id=doc_id, tenant_id=tenant_id,
            )
            await session.run(
                """
                MATCH (d:Document {id: $doc_id, tenant_id: $tenant_id})
                DETACH DELETE d
                """,
                doc_id=doc_id, tenant_id=tenant_id,
            )

        for cid in chunk_ids:
            int_id = self._str_to_int(cid)
            try:
                await qdrant.delete(
                    collection_name=settings.qdrant_collection,
                    points=[int_id],
                )
            except Exception as e:
                logger.warning(f"Failed to delete vector {cid}: {e}")

        return True

    def _str_to_int(self, s: str) -> int:
        import struct
        h = hashlib.sha256(s.encode()).digest()
        return int.from_bytes(h[:7], "big", signed=False) % (2**53)

    # -------------------------------------------------------------------------
    # Multi-tenant retrieval
    # -------------------------------------------------------------------------

    async def search(
        self,
        tenant_id: str,
        query_embedding: list[float],
        top_k: int = 8,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search vectors for a tenant, with optional metadata filters.
        """
        await self._ensure_clients()
        from src.services.vector import vector_search
        from src.config import get_settings
        settings = get_settings()

        filter_condition = self._build_filter(tenant_id, filters)
        results = await vector_search(
            self._clients["qdrant"],
            settings.qdrant_collection,
            query_embedding,
            limit=top_k,
            filter_condition=filter_condition,
        )
        return results

    def _build_filter(self, tenant_id: str, filters: dict[str, Any] | None) -> dict[str, Any] | None:
        """Build Qdrant filter for tenant + metadata."""
        must_clauses = [
            {"key": "tenant_id", "match": {"value": tenant_id}}
        ]
        if not filters:
            return {"must": must_clauses}

        if filters.get("source_ids"):
            must_clauses.append({
                "key": "source_id",
                "match": {"any": filters["source_ids"]},
            })
        if filters.get("tags"):
            for tag in filters["tags"]:
                must_clauses.append({
                    "key": "tags",
                    "match": {"value": tag},
                })
        if filters.get("department"):
            must_clauses.append({
                "key": "department",
                "match": {"value": filters["department"]},
            })
        if filters.get("access_levels"):
            must_clauses.append({
                "key": "access_level",
                "match": {"any": [a.value if hasattr(a, "value") else a for a in filters["access_levels"]]},
            })
        if filters.get("document_ids"):
            must_clauses.append({
                "key": "doc_id",
                "match": {"any": filters["document_ids"]},
            })

        return {"must": must_clauses}
