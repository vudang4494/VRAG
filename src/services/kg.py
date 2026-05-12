"""Knowledge graph service — entity extraction, Neo4j storage and retrieval."""
import asyncio
import json
import re
from typing import Any

import httpx
from neo4j import AsyncGraphDatabase
from loguru import logger


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


async def extract_entities_and_relations(
    text: str,
    llm: Any,
    model: str = "qwen3.5:4b",
    max_chars: int = 2500,
) -> dict:
    """Use LLM to extract entities + relationships from text. Returns dict."""
    truncated = text[:max_chars]
    prompt = _ENTITY_EXTRACT_PROMPT.format(text=truncated)

    try:
        response = await llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        raw = response.choices[0].message.content or ""
        raw = re.sub(r"```(?:json)?\s*", "", raw.strip()).strip().strip("`")
        data = json.loads(raw)
        return {
            "entities": data.get("entities", []),
            "relationships": data.get("relationships", []),
        }
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        return {"entities": [], "relationships": []}


def _sanitize(name: str) -> str:
    """Normalize string for Neo4j property."""
    return re.sub(r"[^\w\s\-_]", "_", name.strip())[:200]


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
    Store a chunk + its entities + relationships in Neo4j.
    Uses batched writes for performance.
    """
    async with driver.session() as s:
        await s.run(
            "MERGE (d:Document {id: $source}) SET d.source = $source",
            source=source,
        )

        await s.run(
            """
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = $text, c.metadata = $metadata, c.source = $source
            WITH c MATCH (d:Document {id: $source})
            MERGE (c)-[:FROM_DOCUMENT]->(d)
            """,
            chunk_id=chunk_id,
            text=text,
            source=source,
            metadata=json.dumps(metadata),
        )

        # Batch entity writes
        for entity in entities:
            name = _sanitize(entity.get("name", ""))
            if not name:
                continue
            etype = entity.get("type", "OTHER")
            desc = entity.get("description", "")[:500]
            await s.run(
                """
                MERGE (e:Entity {name: $name})
                SET e.type = $etype, e.description = $desc
                WITH e MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:CONTAINS_ENTITY]->(e)
                """,
                name=name,
                etype=etype,
                desc=desc,
                chunk_id=chunk_id,
            )

        for rel in relationships:
            src = _sanitize(rel.get("source", ""))
            tgt = _sanitize(rel.get("target", ""))
            if not src or not tgt:
                continue
            desc = rel.get("description", "")[:500]
            await s.run(
                """
                MERGE (s:Entity {name: $src})
                MERGE (t:Entity {name: $tgt})
                MERGE (s)-[r:RELATES_TO]->(t)
                SET r.description = $desc
                """,
                src=src,
                tgt=tgt,
                desc=desc,
            )

        logger.debug(
            f"Neo4j: {len(entities)} entities, {len(relationships)} rels "
            f"from chunk {chunk_id}"
        )


async def graph_retrieve(
    driver,
    query_embedding: list[float],
    http_client: httpx.AsyncClient,
    embed_url: str,
    embed_model: str,
    top_k: int = 5,
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

    async with driver.session() as s:
        result = await s.run(
            """
            MATCH (e:Entity)
            WHERE e.description IS NOT NULL AND e.description <> ''
            RETURN e.name AS name, e.type AS type, e.description AS description
            LIMIT 500
            """
        )
        records = await result.data()

    if not records:
        return []

    entities = [
        {"name": r["name"], "type": r["type"], "description": r["description"]}
        for r in records
    ]

    # Batch embed descriptions
    try:
        embeds = await embed_batch(
            http_client, embed_url, embed_model,
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

    async with driver.session() as s:
        result = await s.run(
            """
            MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
            WHERE e.name IN $names
            WITH c, collect(DISTINCT e.name) AS matched, count(e) AS cnt
            RETURN c.id AS chunk_id, c.text AS text,
                   c.source AS source, c.metadata AS metadata,
                   matched, cnt
            ORDER BY cnt DESC
            LIMIT $top_k
            """,
            names=names,
            top_k=top_k,
        )
        records = await result.data()

    chunks = []
    for record in records:
        matched = set(record["matched"])
        score = sum(
            s for e, s in scored if e["name"] in matched
        ) / len(matched) if matched else 0.0
        chunks.append({
            "chunk_id": record["chunk_id"],
            "text": record["text"],
            "source": record["source"],
            "metadata": record.get("metadata", {}),
            "graph_score": score,
            "matched_entities": record["matched"],
            "retrieval_mode": "graph",
        })

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
                score=score
            )

