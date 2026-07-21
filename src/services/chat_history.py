"""Chat-history semantic cache + memory layer.

Pipeline integration:
  user input
    → embed via EmbeddingGemma (768d)
    → cosine-search per-tenant history in Qdrant `chat_history` collection
    → if top hit ≥ HIT_THRESHOLD → return cached answer (cache hit)
    → else: run full RAG, then store (query_embed, query, answer, citations)

Why separate from semantic_cache (Redis):
  - Redis cache is keyed by exact embedding hash — only hits on near-identical queries.
  - This layer uses Qdrant cosine search — semantic match across paraphrases,
    multi-turn variations, and follow-ups within the same session.

Embedding model: EmbeddingGemma (768d). Falls back to bge-m3 (1024d) when unavailable.
Collection schema is created lazily on first write.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

import httpx
from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

COLLECTION_NAME = "chat_history"
EMBED_MODEL_PRIMARY = "embeddinggemma"
EMBED_MODEL_FALLBACK = "bge-m3"
EMBED_DIM_PRIMARY = 768
EMBED_DIM_FALLBACK = 1024

# Cosine-similarity threshold for "this past chat answers the new query".
# Calibrated against EmbeddingGemma on VN paraphrases (2026-05-20):
#   1.000  exact match
#   0.815  paraphrase "Giải thích X" ↔ "X là gì?"
#   0.427  different topic (ColBERT vs PPR)
# 0.80 captures real paraphrases with a wide margin over unrelated topics.
HIT_THRESHOLD = 0.80
# Look at this many neighbours before deciding; the best one is what we return.
SEARCH_TOP_K = 3
# History retention per (tenant, session_id) — entries older than this are
# evicted on insert (best-effort, batched).
DEFAULT_TTL_S = 86_400  # 24h

# Domain acronyms for the RAG / IR corpus. At lookup time we also embed an
# acronym-expanded version of the query and take the better of the two cosine
# scores — closes the gap between "PPR là gì?" and "Personalized PageRank là gì?"
# without storing duplicate vectors. Storage uses the user's verbatim query.
ACRONYM_EXPANSIONS: dict[str, str] = {
    "PPR": "Personalized PageRank",
    "RAG": "Retrieval-Augmented Generation",
    "GraphRAG": "Graph-based Retrieval-Augmented Generation",
    "KG": "Knowledge Graph",
    "LLM": "Large Language Model",
    "NER": "Named Entity Recognition",
    "BGE": "BAAI General Embedding",
    "HyDE": "Hypothetical Document Embeddings",
    "CoT": "Chain of Thought",
    "ReAct": "Reasoning and Acting",
    "MMR": "Maximal Marginal Relevance",
    "TF-IDF": "Term Frequency Inverse Document Frequency",
    "KNN": "K-Nearest Neighbors",
    "HNSW": "Hierarchical Navigable Small World",
    "MIPS": "Maximum Inner Product Search",
    "BM25": "Best Matching 25",
    "QA": "Question Answering",
    "API": "Application Programming Interface",
    "PII": "Personally Identifiable Information",
    "OOD": "Out of Distribution",
    "OOV": "Out of Vocabulary",
    "MLM": "Masked Language Modeling",
    "MoE": "Mixture of Experts",
    "GNN": "Graph Neural Network",
    "RAGAS": "Retrieval Augmented Generation Assessment",
    "PEFT": "Parameter-Efficient Fine-Tuning",
    "LoRA": "Low-Rank Adaptation",
    "DPR": "Dense Passage Retrieval",
    "ANN": "Approximate Nearest Neighbor",
    "FAISS": "Facebook AI Similarity Search",
}


def _expand_acronyms(text: str) -> str | None:
    """Return an expanded copy if any known acronym is present, else None.

    Token-boundary match only — avoids expanding 'API' inside 'rapid'. Keeps
    the rest of the sentence untouched, so VN connectives and word order
    survive into the cosine comparison.
    """
    if not text:
        return None
    expanded = text
    changed = False
    for acro, full in ACRONYM_EXPANSIONS.items():
        # Word-boundary regex, case-sensitive for acronyms (most are uppercase
        # canonical forms). Lowercase acronym hits would over-trigger.
        pattern = r"\b" + re.escape(acro) + r"\b"
        new = re.sub(pattern, full, expanded)
        if new != expanded:
            expanded = new
            changed = True
    return expanded if changed else None


async def _embed_query(
    http: httpx.AsyncClient, base_url: str, text: str
) -> tuple[list[float], str]:
    """Return (embedding, model_name_used). Tries EmbeddingGemma first, falls back to bge-m3."""
    for model_name in (EMBED_MODEL_PRIMARY, EMBED_MODEL_FALLBACK):
        try:
            resp = await http.post(
                f"{base_url}/api/embed",
                json={"model": model_name, "input": text},
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            embs = data.get("embeddings") or []
            if embs and isinstance(embs[0], list) and len(embs[0]) > 0:
                return embs[0], model_name
        except Exception as e:
            logger.debug(f"chat_history embed: {model_name} failed ({e!r}), trying next")
    raise RuntimeError("chat_history: all embedding models failed")


async def _ensure_collection(qdrant: AsyncQdrantClient, dim: int) -> None:
    """Lazy-create the chat_history collection with the given vector dim."""
    try:
        collections = await qdrant.get_collections()
        names = {c.name for c in collections.collections}
        if COLLECTION_NAME in names:
            return
    except Exception as e:
        logger.debug(f"chat_history: get_collections failed: {e!r}")

    try:
        await qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        # Tenant index for fast per-tenant filtering.
        await qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="tenant_id",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
        await qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="session_id",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
        logger.info(f"chat_history: created collection {COLLECTION_NAME} (dim={dim})")
    except Exception as e:
        logger.warning(f"chat_history: create_collection failed (may already exist): {e!r}")


def _point_id(tenant_id: str, query: str, ts: float) -> str:
    """Deterministic-ish UUID5 — same (tenant, query) within same minute dedupes."""
    seed = f"{tenant_id}::{query}::{int(ts // 60)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


async def _search_for_emb(
    qdrant: AsyncQdrantClient,
    emb: list[float],
    must: list[qm.FieldCondition],
) -> list[Any]:
    """Run one cosine search against chat_history. Empty list on any failure."""
    try:
        result = await qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=emb,
            limit=SEARCH_TOP_K,
            query_filter=qm.Filter(must=must),
            with_payload=True,
        )
        return result.points if hasattr(result, "points") else []
    except Exception as e:
        logger.debug(f"chat_history lookup: query_points failed {e!r}")
        return []


async def lookup(
    qdrant: AsyncQdrantClient,
    http: httpx.AsyncClient,
    embed_base_url: str,
    tenant_id: str,
    query: str,
    session_id: str | None = None,
    threshold: float = HIT_THRESHOLD,
) -> dict[str, Any] | None:
    """Search chat history for a semantically-similar past query.

    Tries the verbatim query AND (if any domain acronym is present) the
    acronym-expanded form. Returns whichever scores higher — closes the gap
    between "PPR là gì?" and "Personalized PageRank là gì?".

    Returns the cached entry dict (with 'answer', 'citations', 'score') if hit,
    else None. Failures are logged and return None — never raise.
    """
    expanded = _expand_acronyms(query)
    variants = [(query, "verbatim")]
    if expanded and expanded != query:
        variants.append((expanded, "expanded"))

    must = [qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))]
    if session_id:
        must.append(qm.FieldCondition(key="session_id", match=qm.MatchValue(value=session_id)))

    best_hit = None
    best_score = -1.0
    best_variant = "verbatim"
    model_used = ""

    for variant_text, variant_kind in variants:
        try:
            emb, model_name = await _embed_query(http, embed_base_url, variant_text)
        except Exception as e:
            logger.debug(f"chat_history lookup: embed failed for {variant_kind} {e!r}")
            continue
        hits = await _search_for_emb(qdrant, emb, must)
        if not hits:
            continue
        top = hits[0]
        if top.score > best_score:
            best_score = float(top.score)
            best_hit = top
            best_variant = variant_kind
            model_used = model_name

    if best_hit is None:
        return None
    if best_score < threshold:
        logger.debug(
            f"chat_history miss: best_score={best_score:.3f} < threshold={threshold} "
            f"(query='{query[:40]}…' variant={best_variant})"
        )
        return None

    payload = best_hit.payload or {}
    logger.info(
        f"chat_history HIT: score={best_score:.3f} variant={best_variant} model={model_used} "
        f"query='{query[:40]}…' matched='{payload.get('query', '')[:40]}…'"
    )
    return {
        "score": best_score,
        "answer": payload.get("answer", ""),
        "citations": payload.get("citations", []),
        "sources": payload.get("sources", []),
        "original_query": payload.get("query", ""),
        "cached_at": payload.get("ts", 0),
        "embed_model": model_used,
        "matched_variant": best_variant,
    }


async def store(
    qdrant: AsyncQdrantClient,
    http: httpx.AsyncClient,
    embed_base_url: str,
    tenant_id: str,
    query: str,
    answer: str,
    citations: list[Any] | None = None,
    sources: list[Any] | None = None,
    session_id: str | None = None,
) -> bool:
    """Persist (query, answer) to chat history. Returns True on success.

    Never raises — logs and returns False on any failure. Idempotent per
    (tenant, query, minute-bucket) via deterministic point IDs.
    """
    if not query.strip() or not answer.strip():
        return False
    # Don't cache refusals — they have nothing useful to replay.
    if "không có đủ thông tin" in answer.lower() or "tôi không" in answer.lower()[:30]:
        return False

    try:
        emb, model_name = await _embed_query(http, embed_base_url, query)
    except Exception as e:
        logger.debug(f"chat_history store: embed failed {e!r}")
        return False

    await _ensure_collection(qdrant, dim=len(emb))

    ts = time.time()
    point = qm.PointStruct(
        id=_point_id(tenant_id, query, ts),
        vector=emb,
        payload={
            "tenant_id": tenant_id,
            "session_id": session_id or "",
            "query": query[:500],
            "answer": answer[:4000],
            "citations": citations or [],
            "sources": sources or [],
            "embed_model": model_name,
            "ts": ts,
        },
    )

    try:
        await qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
        logger.debug(
            f"chat_history stored: tenant={tenant_id} model={model_name} "
            f"query='{query[:40]}…' answer_len={len(answer)}"
        )
        return True
    except Exception as e:
        logger.warning(f"chat_history store: upsert failed {e!r}")
        return False


async def prune_old(
    qdrant: AsyncQdrantClient,
    tenant_id: str | None = None,
    ttl_s: float = DEFAULT_TTL_S,
) -> int:
    """Best-effort eviction of history entries older than ttl_s.

    Returns count attempted-deleted. Safe to call periodically (e.g. once per
    100 inserts) or on a background timer. Failure returns 0.
    """
    cutoff = time.time() - ttl_s
    must: list[qm.FieldCondition | qm.Filter] = [
        qm.FieldCondition(key="ts", range=qm.Range(lt=cutoff))
    ]
    if tenant_id:
        must.append(qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)))

    try:
        await qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=qm.FilterSelector(filter=qm.Filter(must=must)),
        )
        return 1
    except Exception as e:
        logger.debug(f"chat_history prune: failed {e!r}")
        return 0
