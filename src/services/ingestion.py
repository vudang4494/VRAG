"""Ingestion service — document pipeline: parse → chunk → extract KG → embed → store."""
import asyncio
import hashlib
import re
from typing import Any

from loguru import logger

from src.config import get_settings
from src.services.embedding import embed_batch
from src.services.kg import extract_entities_and_relations, upsert_chunk_and_entities
from src.services.vector import upsert_points


def recursive_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    min_size: int = 128,
) -> list[dict]:
    """
    Recursive text chunking with overlap and sentence-boundary awareness.
    Returns list of {text, start, end, chunk_index}.
    """
    if not text or len(text.strip()) < min_size:
        return []

    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        if end < len(text):
            window = chunk_text[-80:]
            match = re.search(
                r"[.!?]\s+(?=[A-ZÀ-ỹ])|[.!?]\s*$|\n{2,}", window
            )
            if match:
                actual_end = start + len(chunk_text[: match.start() + 1].rstrip())
                chunk_text = text[start:actual_end]
                end = actual_end

        chunk_text = chunk_text.strip()
        if len(chunk_text) >= min_size:
            chunks.append({
                "text": chunk_text,
                "start": start,
                "end": end,
                "chunk_index": idx,
            })
            idx += 1

        if end >= len(text):
            break
        start = end - overlap

    return chunks


async def parse_document(content: bytes, filename: str) -> str:
    """Parse document content into plain text. Supports: PDF, DOCX, TXT."""
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()

    if ext == "txt":
        return content.decode("utf-8", errors="replace")

    if ext == "pdf":
        try:
            from docling import parse_pdf
            result = await asyncio.to_thread(parse_pdf, content)
            return result.text
        except ImportError:
            logger.warning("docling not installed, using fallback extraction")
            return _fallback_text(content)

    if ext in ("docx", "doc"):
        try:
            from docx import Document
            doc = await asyncio.to_thread(Document, content)
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            return _fallback_text(content)

    return _fallback_text(content)


def _fallback_text(content: bytes) -> str:
    """Strip binary/control chars, collapse whitespace."""
    try:
        text = content.decode("utf-8", errors="ignore")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
        text = re.sub(r" {3,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    except Exception:
        return ""


async def ingest_document(
    file_content: bytes,
    filename: str,
    clients: Any,
) -> dict:
    """
    Full document ingestion pipeline — optimized for parallel processing.

    1. Parse → plain text
    2. Chunk (512 chars / 64 overlap)
    3. Batch embed all chunks (single Ollama call per chunk, batched)
    4. Parallel: extract KG + upsert vector points
    5. Upsert to Neo4j
    """
    settings = get_settings()

    source = filename
    doc_hash = hashlib.md5(file_content).hexdigest()[:16]
    logger.info(f"Ingesting: {filename} ({len(file_content):,} bytes)")

    # 1. Parse
    try:
        text = await parse_document(file_content, filename)
    except Exception as e:
        raise ValueError(f"Parse failed: {e}")

    if not text or len(text.strip()) < 100:
        raise ValueError("File contains insufficient text to index")

    # 2. Chunk
    chunks = recursive_chunk(text)
    if not chunks:
        raise ValueError("Failed to create chunks from document")
    logger.info(f"Created {len(chunks)} chunks from {filename}")

    # 3. Batch embed all chunks at once
    texts = [c["text"] for c in chunks]
    try:
        embeddings = await embed_batch(
            clients.http,
            settings.ollama_embed_url,
            settings.ollama_embed_model,
            texts,
            batch_size=16,
            timeout=120.0,
        )
    except Exception as e:
        logger.warning(f"Batch embedding failed: {e}, using zero vectors")
        embeddings = [[0.0] * settings.embed_dimension for _ in texts]

    # 4 & 5: Parallel KG extraction + Neo4j upsert
    # Giảm từ 4 xuống 2 để tránh httpx.ReadTimeout trên Ollama (tránh overload)
    semaphore = asyncio.Semaphore(2)
    successful = []
    failed = 0

    async def process(chunk: dict, idx: int, embedding: list[float]) -> dict | None:
        async with semaphore:
            chunk_id = f"{doc_hash}_{idx}"
            text = chunk["text"]

            # KG extraction (sequential to avoid overwhelming LLM)
            kg = await extract_entities_and_relations(
                text, clients.llm, settings.ollama_model
            )

            return {
                "chunk_id": chunk_id,
                "text": text,
                "source": source,
                "embedding": embedding,
                "entities": kg.get("entities", []),
                "relationships": kg.get("relationships", []),
                "chunk_index": idx,
            }

    tasks = [process(chunks[i], i, embeddings[i]) for i in range(len(chunks))]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            failed += 1
            logger.warning(f"Chunk processing failed: {result}")
        else:
            successful.append(result)

    if not successful:
        raise ValueError("All chunks failed processing")

    # 4. Upsert to Qdrant
    points = [
        {
            "id": p["chunk_id"],
            "vector": p["embedding"],
            "payload": {
                "text": p["text"],
                "source": p["source"],
                "chunk_index": p["chunk_index"],
            },
        }
        for p in successful
    ]

    indexed = await upsert_points(
        clients.qdrant,
        settings.qdrant_collection,
        points,
        dimension=settings.embed_dimension,
    )
    logger.info(f"Indexed {indexed} points to Qdrant")

    # 5. Upsert to Neo4j
    all_entities = []
    all_rels = []
    for chunk in successful:
        all_entities.extend(chunk.get("entities", []))
        all_rels.extend(chunk.get("relationships", []))

    seen: set[str] = set()
    dedup_entities = []
    for e in all_entities:
        name = e.get("name", "").strip()
        if name and name not in seen:
            seen.add(name)
            dedup_entities.append(e)

    for chunk in successful:
        await upsert_chunk_and_entities(
            clients.neo4j,
            chunk["chunk_id"],
            chunk["text"],
            chunk["source"],
            {},
            chunk.get("entities", []),
            chunk.get("relationships", []),
        )

    # 6. Semantic Graph Linking (Vector-to-Graph Masking)
    # Áp dụng Vector Embedding từ Qdrant để tạo liên kết ngữ nghĩa (SIMILAR_TO) trong Neo4j
    from src.services.vector import vector_search
    from src.services.kg import link_semantic_chunks
    
    semantic_links_created = 0
    for chunk in successful:
        try:
            similar = await vector_search(
                clients.qdrant,
                settings.qdrant_collection,
                chunk["embedding"],
                limit=4
            )
            target_links = [(s["chunk_id"], s["score"]) for s in similar if s["chunk_id"] != chunk["chunk_id"]]
            
            if target_links:
                await link_semantic_chunks(clients.neo4j, chunk["chunk_id"], target_links)
                semantic_links_created += len(target_links)
        except Exception as e:
            logger.warning(f"Semantic linking failed: {e}")

    return {
        "status": "success",
        "filename": filename,
        "doc_hash": doc_hash,
        "chunks_indexed": indexed,
        "entities_extracted": len(dedup_entities),
        "relationships_extracted": len(all_rels),
        "semantic_edges_created": semantic_links_created,
        "failed_chunks": failed,
    }
