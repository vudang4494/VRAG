"""3-stage rerank pipeline.

Stage 1: Cross-encoder (BAAI/bge-reranker-v2-m3) — top 50 → top 20.
Stage 2: Second-pass semantic match against 'summary' view — top 20 → top 10.
Stage 3: LLM judge per-candidate — top 10 → top 5.

Note: Stage 1 model loaded lazily. Falls back to Stage 2-only nếu model
chưa cài (avoids breaking on first run without `sentence-transformers`).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
from loguru import logger

from src.services.cross_encoder import _load_cross_encoder
from src.services.embedding import cosine_similarity, embed_single


async def rerank_stage1(
    query: str,
    candidates: list[dict],
    model_name: str = "BAAI/bge-reranker-v2-m3",
    top_k: int = 20,
    batch_size: int = 16,
) -> list[dict]:
    """Cross-encoder rerank. Returns top_k candidates with stage1_score."""
    if not candidates:
        return []
    model = await asyncio.to_thread(_load_cross_encoder, model_name)
    if not model:
        # Fallback: copy existing scores
        ranked = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)
        return [{**c, "stage1_score": float(c.get("score", 0.0))} for c in ranked[:top_k]]

    pairs = [(query, c.get("text", "")[:2000]) for c in candidates]
    try:
        scores = await asyncio.to_thread(
            model.predict, pairs, batch_size=batch_size, show_progress_bar=False
        )
    except Exception as e:
        logger.warning(f"Stage 1 rerank failed: {e}")
        return [{**c, "stage1_score": 0.0} for c in candidates[:top_k]]

    scored = [{**c, "stage1_score": float(s)} for c, s in zip(candidates, scores, strict=False)]
    scored.sort(key=lambda x: x["stage1_score"], reverse=True)
    return scored[:top_k]


async def rerank_stage2(
    query: str,
    candidates: list[dict],
    http: httpx.AsyncClient,
    embed_url: str,
    embed_model: str = "bge-m3",
    top_k: int = 10,
) -> list[dict]:
    """
    Second-pass semantic: re-embed query, compare with 'summary' view (or original text).
    Useful as Stage 1 fallback when cross-encoder unavailable.
    """
    if not candidates:
        return []
    try:
        q_vec = await embed_single(http, embed_url, embed_model, query, timeout=30.0)
    except Exception as e:
        logger.warning(f"Stage 2 query embed failed: {e}")
        return candidates[:top_k]

    # Bound concurrent Ollama embeds so a page of candidates without a precomputed
    # 'summary' view can't fire an unbounded burst at Ollama (M-series ~3-4 streams).
    sem = asyncio.Semaphore(4)

    async def _score_one(c: dict) -> dict:
        # Prefer 'summary' view embedding if present, else compute on text
        summary_emb = (
            c.get("view_embeddings", {}).get("summary")
            if isinstance(c.get("view_embeddings"), dict)
            else None
        )
        if summary_emb:
            sim = cosine_similarity(q_vec, summary_emb)
        else:
            txt = c.get("text", "")[:1500]
            try:
                async with sem:
                    emb = await embed_single(http, embed_url, embed_model, txt, timeout=30.0)
                sim = cosine_similarity(q_vec, emb)
            except Exception:
                sim = 0.0
        return {**c, "stage2_score": float(sim)}

    scored = await asyncio.gather(*[_score_one(c) for c in candidates])
    scored.sort(key=lambda x: x["stage2_score"], reverse=True)
    return scored[:top_k]


_LLM_JUDGE_PROMPT = """Cho câu hỏi và đoạn văn bản dưới đây. Đoạn này có chứa thông tin
trả lời cho câu hỏi không?

Trả lời CHỈ với định dạng:
SCORE: <0-10>
REASON: <1 câu>

Trong đó SCORE:
- 0-3: hoàn toàn không liên quan
- 4-6: có chút liên quan nhưng không đủ
- 7-8: trả lời được một phần câu hỏi
- 9-10: trả lời trực tiếp và đầy đủ

Câu hỏi: {query}

Đoạn văn bản:
{text}

Đánh giá:"""


async def _judge_one(
    query: str, candidate: dict, llm: Any, model: str, timeout: float = 5.0
) -> dict:
    from src.services.ollama_helper import ollama_chat

    text = candidate.get("text", "")[:1500]
    prompt = _LLM_JUDGE_PROMPT.format(query=query, text=text)
    try:
        raw = await asyncio.wait_for(
            ollama_chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.1,
                max_tokens=120,
            ),
            timeout=timeout,
        )
    except Exception as e:
        logger.debug(f"Stage 3 judge failed (fallback to stage2 score): {e}")
        return {
            **candidate,
            "stage3_score": float(candidate.get("stage2_score", 0.0)),
            "judge_reason": None,
        }

    # Parse SCORE: N, REASON: ...
    score_match = re.search(r"SCORE\s*:\s*(\d+(?:\.\d+)?)", raw, re.IGNORECASE)
    reason_match = re.search(r"REASON\s*:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE | re.DOTALL)
    score = float(score_match.group(1)) / 10.0 if score_match else 0.5
    reason = reason_match.group(1).strip() if reason_match else None
    return {**candidate, "stage3_score": min(score, 1.0), "judge_reason": reason}


async def rerank_stage3(
    query: str,
    candidates: list[dict],
    llm: Any,
    model: str = "gemma4:e4b",
    top_k: int = 5,
    concurrent_limit: int = 5,
    per_call_timeout: float = 5.0,
) -> list[dict]:
    """LLM judge per-candidate. Parallel với concurrent limit."""
    if not candidates:
        return []
    sem = asyncio.Semaphore(concurrent_limit)

    async def _bounded(c):
        async with sem:
            return await _judge_one(query, c, llm, model, per_call_timeout)

    scored = await asyncio.gather(*[_bounded(c) for c in candidates])
    scored.sort(key=lambda x: x.get("stage3_score", 0.0), reverse=True)
    return scored[:top_k]


# Note: rerank_full_pipeline lives in src/services/rerank.py (canonical).
# This module only exports the individual stage functions, imported by
# rerank_l2r.py and react_loop.py.
