"""Phase 3 — Multi-feature Learning-to-Rank rerank.

Replaces single cross-encoder with a feature-vector scorer using signals from
all previous phases (consistency, graph_aware, entity match, path confidence).

Phase 1: hand-weighted linear combo (start values + tunable via env)
Phase 2 (future): train LightGBM on labeled query-chunk relevance triplets
Phase 3 (future): online learning from user feedback

Features (11 dimensions):
  1. stage2_score           — semantic match against summary view (from Phase 0)
  2. cosine_dense           — query emb vs chunk dense
  3. cosine_graph_aware     — query emb vs chunk graph_aware (Phase 1 GAEA)
  4. consistency_score      — chunk variance score from ingest
  5. chunk_level_factor     — 0.8 sentence / 1.0 paragraph / 1.1 section
  6. entity_match_count_norm — normalized count of query entities matched
  7. path_confidence        — placeholder, 1.0 for now (Phase 4+)
  8. recency_score          — exponential decay from created_at
  9. retrieval_path_weight  — which path surfaced this chunk
 10. format_match_score     — format preference per intent (placeholder)
 11. source_quality_score   — Phase 6a CQC placeholder (1.0 default)
"""
from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from loguru import logger


# ── Feature weights (hand-tuned initial) ──────────────────────────────────────
# Re-tunable via env: FEATURE_WEIGHTS_JSON='{"stage2":0.3,...}'
DEFAULT_WEIGHTS = {
    "stage2_score":            0.25,
    "cosine_dense":            0.10,
    "cosine_graph_aware":      0.20,
    "consistency_score":       0.10,
    "chunk_level_factor":      0.05,
    "entity_match_count_norm": 0.15,
    "path_confidence":         0.05,
    "recency_score":           0.03,
    "retrieval_path_weight":   0.04,
    "format_match_score":      0.01,
    "source_quality_score":    0.02,
}


def _load_weights() -> dict[str, float]:
    """Try env override, else default."""
    raw = os.environ.get("FEATURE_WEIGHTS_JSON", "")
    if raw:
        try:
            import json as _j
            return {**DEFAULT_WEIGHTS, **_j.loads(raw)}
        except Exception as e:
            logger.warning(f"FEATURE_WEIGHTS_JSON parse failed: {e}; using defaults")
    return DEFAULT_WEIGHTS.copy()


# ── Feature extraction ────────────────────────────────────────────────────────


def _chunk_level_factor(level: str | None) -> float:
    return {
        "sentence": 0.8,
        "paragraph": 1.0,
        "section": 1.1,
        "document": 0.7,
    }.get(level or "paragraph", 1.0)


def _retrieval_path_weight(path: str | None) -> float:
    """Boost based on which path surfaced this chunk."""
    if not path:
        return 0.5
    p = path.lower()
    if "graph_aware" in p:
        return 1.0
    if "entity_pivot" in p or "react:entity" in p:
        return 0.95
    if "sparse" in p or "keywords" in p:
        return 0.7
    if "summary" in p:
        return 0.65
    if "step_back" in p:
        return 0.5
    return 0.6


def _recency_score(created_at: str | None) -> float:
    """Exponential decay: 1.0 for new, 0.5 for 1-yr-old, 0.3 for 3-yr-old."""
    if not created_at:
        return 0.7
    try:
        dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except Exception:
        return 0.7
    now = datetime.now(timezone.utc)
    days = max((now - dt.replace(tzinfo=timezone.utc)).days, 0) if dt.tzinfo is None else max((now - dt).days, 0)
    # Half-life 365 days
    return max(0.3, 0.5 ** (days / 365.0))


def _entity_match_norm(chunk: dict, query_entities: list[str]) -> float:
    """Normalized count of query entities matched in chunk."""
    if not query_entities:
        return 0.5  # neutral
    chunk_ents = set()
    matched = chunk.get("matched_entities") or []
    if matched:
        chunk_ents.update(m.lower() for m in matched)
    # Also check chunk text contains entity names
    text = (chunk.get("text") or "").lower()
    hits = sum(1 for e in query_entities if e.lower() in text or e.lower() in chunk_ents)
    return min(hits / max(len(query_entities), 1), 1.0)


def extract_features(
    chunk: dict,
    query_entities: list[str] | None = None,
    query_format_pref: list[str] | None = None,
) -> dict[str, float]:
    """Compute the 11-dim feature vector for a chunk."""
    query_entities = query_entities or []

    # Cosine values come from upstream retrieval if available
    cos_dense = float(chunk.get("cosine_dense") or chunk.get("score") or 0.0)
    cos_graph = float(chunk.get("cosine_graph_aware") or 0.0)
    # If graph_aware not yet computed, fall back to dense
    if cos_graph == 0.0 and chunk.get("retrieval_path", "").endswith("graph_aware"):
        cos_graph = cos_dense

    fmt = chunk.get("format", "")
    fmt_score = 1.0 if (not query_format_pref or fmt in query_format_pref) else 0.5

    return {
        "stage2_score":            float(chunk.get("stage2_score") or chunk.get("score") or 0.0),
        "cosine_dense":            min(cos_dense, 1.0),
        "cosine_graph_aware":      min(cos_graph, 1.0),
        "consistency_score":       float(chunk.get("consistency_score") or 0.7),
        "chunk_level_factor":      _chunk_level_factor(chunk.get("chunk_level")),
        "entity_match_count_norm": _entity_match_norm(chunk, query_entities),
        "path_confidence":         float(chunk.get("path_confidence") or 1.0),
        "recency_score":           _recency_score(chunk.get("created_at") or chunk.get("metadata", {}).get("created_at")),
        "retrieval_path_weight":   _retrieval_path_weight(chunk.get("retrieval_path")),
        "format_match_score":      fmt_score,
        "source_quality_score":    float(chunk.get("source_quality_score") or 1.0),
    }


def score_with_weights(features: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted linear combo, weight normalization included."""
    total_w = sum(weights.values()) or 1.0
    return sum(features.get(k, 0.0) * w for k, w in weights.items()) / total_w


# ── Main rerank function ──────────────────────────────────────────────────────


async def rerank_l2r(
    query: str,
    candidates: list[dict],
    query_entities: list[str] | None = None,
    query_format_pref: list[str] | None = None,
    top_k: int = 5,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Score and sort candidates using multi-feature L2R.

    Cheap (no LLM, no model load) — pure feature arithmetic.
    Replaces or augments stage 3 LLM judge for cost/latency efficiency.
    """
    if not candidates:
        return []

    w = weights or _load_weights()
    started = time.monotonic()

    for c in candidates:
        feats = extract_features(c, query_entities=query_entities, query_format_pref=query_format_pref)
        c["l2r_features"] = feats
        c["l2r_score"] = score_with_weights(feats, w)
        # Final_score: blend stage2 (semantic) with l2r (feature-aware)
        c["final_score"] = 0.4 * c.get("stage2_score", 0.0) + 0.6 * c["l2r_score"]

    candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    elapsed = time.monotonic() - started
    logger.debug(f"L2R rerank: {len(candidates)} chunks in {elapsed*1000:.0f}ms")
    return candidates[:top_k]


async def rerank_full_pipeline_v2(
    query: str,
    candidates: list[dict],
    http: httpx.AsyncClient,
    embed_url: str,
    embed_model: str = "bge-m3",
    query_entities: list[str] | None = None,
    query_format_pref: list[str] | None = None,
    stage1_top_k: int = 30,
    stage2_top_k: int = 15,
    final_top_k: int = 5,
    enable_stage1: bool = False,
) -> list[dict]:
    """Updated rerank pipeline using L2R instead of LLM judge.

    Stage 1 (optional): bge-reranker cross-encoder (heavy, ~500ms)
    Stage 2: semantic match against summary view (~300ms)
    Stage 3: L2R feature scorer (~10ms, replaces LLM judge ~3000ms)
    """
    from src.services.rerank_stages import rerank_stage1, rerank_stage2

    if not candidates:
        return []

    if enable_stage1:
        stage1 = await rerank_stage1(query, candidates, top_k=stage1_top_k)
    else:
        stage1 = [{**c, "stage1_score": float(c.get("score", 0.0))} for c in candidates[:stage1_top_k]]

    stage2 = await rerank_stage2(query, stage1, http, embed_url, embed_model, top_k=stage2_top_k)

    # Stage 3: L2R (replacing LLM judge)
    final = await rerank_l2r(
        query, stage2,
        query_entities=query_entities,
        query_format_pref=query_format_pref,
        top_k=final_top_k,
    )

    return final
