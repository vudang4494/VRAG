"""Query type classifier for intelligent routing between retrieval strategies.

## Algorithm: Semantic Matcher (BGE-M3 Centroid) + OOD Guard

1. **Out-of-domain guard** (regex, highest priority): fast short-circuit, no embedding needed
2. **Semantic intent matching** (centroid dot-product): embed query via BGE-M3, compute
   cosine similarity with intent centroids, pick the highest-scoring intent

Why semantic instead of rule-based:
  - Zero maintenance of regex patterns
  - Handles unseen surface forms, typos, code-mixed queries
  - Fast: dot product on 1024-dim vector < 1ms
  - Centroid-based is robust to query phrasing variations

Usage:
    from src.services.query_router import classify_query, should_use_react

    query_type = classify_query("GraphRAG là gì?")
    use_react = should_use_react(query_type)
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import numpy as np

import httpx
from loguru import logger

ROOT = Path(__file__).parent.parent.parent  # repo root (src/services/.. → src → repo)
_CENTROIDS: dict[str, np.ndarray] | None = None
_SEMANTIC_THRESHOLD: float = 0.45

# ── OOD patterns — regex only, no embedding needed ──────────────────────────────

_OOD_PATTERNS = [
    # Real-world queries not in academic corpus
    r"thời tiết",
    r"bitcoin",
    r"giá .*hôm nay",
    r"nấu (phở|canh|bún)",
    r"tin tức",
    r"(?i)news today",
    r"(?i)weather",
    r"(?i)stock price",
    r"(?i)cook (pho|soup|recipe)",
    r"(?i)news",
    r"bóng đá",
    r"(?i)sport",
    r"(?i)football",
    r"(?i)game",
    r"làm bánh",
    r"tập gym",
    r"mua sắm",
    r"du lịch",
]


def _load_centroids() -> dict[str, np.ndarray]:
    global _CENTROIDS
    if _CENTROIDS is not None:
        return _CENTROIDS
    path = ROOT / "config" / "intent_centroids.npy"
    if not path.exists():
        logger.warning(f"intent_centroids.npy not found at {path}, using rule-based fallback")
        _CENTROIDS = {}
        return _CENTROIDS
    data = np.load(path, allow_pickle=True).item()
    for intent, vec in data.items():
        data[intent] = np.asarray(vec, dtype=np.float32)
    _CENTROIDS = data
    logger.info(f"Loaded {len(_CENTROIDS)} intent centroids from {path}")
    return _CENTROIDS


def _embed_query_sync(query: str, embed_url: str, embed_model: str) -> np.ndarray | None:
    """Embed a query via Ollama /api/embeddings. Returns None on failure."""
    try:
        import requests

        resp = requests.post(
            f"{embed_url}/api/embeddings",
            json={"model": embed_model, "prompt": query, "keep_alive": -1},
            timeout=30,
        )
        resp.raise_for_status()
        return np.asarray(resp.json()["embedding"], dtype=np.float32)
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


def _match_ood(query: str) -> bool:
    q = query.strip().lower()
    return any(re.search(pat, q) for pat in _OOD_PATTERNS)


def classify_query(query: str, embed_url: str | None = None, embed_model: str | None = None) -> str:
    """
    Classify query type using semantic centroid matching.

    Returns one of: factual | analytical | comparison | multi_hop | kg_construction | out_of_domain

    If embed_url/embed_model are None, reads from settings (so the call works inside
    a container where Ollama is at host.docker.internal, not localhost).
    """
    if _match_ood(query):
        return "out_of_domain"

    centroids = _load_centroids()
    if not centroids:
        # Fallback: simple keyword heuristics
        return _fallback_rule_based(query)

    if embed_url is None or embed_model is None:
        try:
            from src.config import get_settings
            s = get_settings()
            embed_url = embed_url or s.ollama_embed_url
            embed_model = embed_model or s.ollama_embed_model
        except Exception:
            embed_url = embed_url or "http://localhost:11434"
            embed_model = embed_model or "bge-m3"

    vec = _embed_query_sync(query, embed_url, embed_model)
    if vec is None:
        return _fallback_rule_based(query)

    best_intent = "factual"
    best_score = -1.0
    for intent, centroid in centroids.items():
        # Both are unit-normalized → dot product = cosine similarity
        score = float(np.dot(vec, centroid))
        if score > best_score:
            best_intent, best_score = intent, score

    if best_score < _SEMANTIC_THRESHOLD:
        logger.debug(f"Low centroid score {best_score:.3f} for query: {query[:60]}, defaulting to factual")
        return "factual"

    return best_intent


def _fallback_rule_based(query: str) -> str:
    """Simple keyword fallback when centroid matching is unavailable."""
    q = query.lower()

    if any(kw in q for kw in ["tóm tắt", "tổng hợp", "tổng quan", "liệt kê"]):
        return "summarization"
    if any(kw in q for kw in ["tại sao", "vì sao", "bằng cách nào", "như thế nào"]):
        return "analytical"
    if any(kw in q for kw in ["so sánh", "khác nhau", "giống nhau", "ưu điểm", "nhược điểm"]):
        return "comparison"
    if any(kw in q for kw in ["mối liên hệ", "ảnh hưởng qua lại", "cả a và b"]):
        return "multi_hop"
    if any(kw in q for kw in ["knowledge graph", "schema", "ontology", "xây dựng đồ thị"]):
        return "kg_construction"

    return "factual"


# ── ReAct gating ───────────────────────────────────────────────────────────────

_REACT_INTENTS = {"analytical", "comparison", "multi_hop", "kg_construction"}


def should_use_react(query_type: str) -> bool:
    """
    Decide whether to route to ReAct loop based on query type.

    ReAct is beneficial for:
      - analytical: causal/why questions benefit from step-by-step reasoning
      - comparison: cross-entity comparison across documents
      - multi_hop: cross-doc entity reasoning
      - kg_construction: schema/structure questions

    ReAct is NOT needed for:
      - factual: single-doc lookup, direct answer
      - out_of_domain: short-circuit early
      - summarization: aggregate across docs (handled separately)
    """
    return query_type in _REACT_INTENTS


def describe_routing(query_type: str, use_react: bool) -> str:
    """Human-readable description of routing decision."""
    if use_react:
        return f"ReAct loop ({query_type} query — multi-step reasoning)"
    return f"Standard retrieval ({query_type} query — single-doc lookup)"
