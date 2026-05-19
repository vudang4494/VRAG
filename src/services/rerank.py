"""3-stage rerank pipeline.

Stage 1: Cross-encoder (BAAI/bge-reranker-v2-m3) — top 50 → top 20.
Stage 2: Second-pass semantic match against 'summary' view — top 20 → top 10.
Stage 3: LLM judge per-candidate — top 10 → top 5.
Stage 4: LambdaMART L2R (optional) — uses learned weights for better reranking.

Note: Stage 1 model loaded lazily. Falls back to Stage 2-only nếu model
chưa cài (avoids breaking on first run without `sentence-transformers`).
L2R is optional and uses LightGBM when available.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
from loguru import logger

from src.services.embedding import cosine_similarity, embed_single

# Lazy-loaded cross-encoder
_CROSS_ENCODER = None


def _load_cross_encoder(model_name: str = "BAAI/bge-reranker-v2-m3"):
    global _CROSS_ENCODER
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER
    try:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER = CrossEncoder(model_name, max_length=512)
        logger.info(f"Loaded cross-encoder: {model_name}")
    except Exception as e:
        logger.warning(f"Cross-encoder not available ({e}); stage 1 will be skipped.")
        _CROSS_ENCODER = False
    return _CROSS_ENCODER


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
    model: str = "qwen3.5:9b",
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


async def rerank_full_pipeline(
    query: str,
    candidates: list[dict],
    http: httpx.AsyncClient,
    embed_url: str,
    llm: Any,
    embed_model: str = "bge-m3",
    llm_model: str = "qwen3.5:9b",
    stage1_top_k: int = 20,
    stage2_top_k: int = 10,
    stage3_top_k: int = 5,
    enable_stage1: bool = True,
    enable_stage3: bool = True,
    early_exit_threshold: float = 0.85,
) -> list[dict]:
    """
    VRAG Tier 3b: 3-stage rerank with Dynamic Early-Exit.

    Final score = 0.4 * stage1 + 0.3 * stage2 + 0.3 * stage3.

    Early-Exit: nếu avg(top-stage3_top_k stage1 scores) >= early_exit_threshold,
    auto-skip stage3 LLM judge (gọt 50% rerank time, theo caitien.md plan).
    Set early_exit_threshold=1.1 để tắt early-exit.
    """
    if not candidates:
        return []

    if enable_stage1:
        stage1 = await rerank_stage1(query, candidates, top_k=stage1_top_k)
    else:
        stage1 = [
            {**c, "stage1_score": float(c.get("score", 0.0))} for c in candidates[:stage1_top_k]
        ]

    stage2 = await rerank_stage2(query, stage1, http, embed_url, embed_model, top_k=stage2_top_k)

    # Dynamic Early-Exit: only meaningful when stage1 produced real cross-encoder scores.
    skip_stage3 = False
    if enable_stage1 and enable_stage3 and stage2:
        top_scores = [c.get("stage1_score", 0.0) for c in stage2[:stage3_top_k]]
        if top_scores:
            avg_conf = sum(top_scores) / len(top_scores)
            if avg_conf >= early_exit_threshold:
                skip_stage3 = True
                logger.info(
                    f"  rerank early-exit: stage1 avg_conf={avg_conf:.3f} >= "
                    f"{early_exit_threshold} — skipping stage3 LLM judge"
                )

    if enable_stage3 and not skip_stage3:
        stage3 = await rerank_stage3(query, stage2, llm, llm_model, top_k=stage3_top_k)
    else:
        stage3 = stage2[:stage3_top_k]

    # Compute final
    for c in stage3:
        s1 = c.get("stage1_score", 0.0)
        s2 = c.get("stage2_score", 0.0)
        s3 = c.get("stage3_score", s2)  # fallback to s2 when stage3 skipped
        c["final_score"] = 0.4 * s1 + 0.3 * s2 + 0.3 * s3
    stage3.sort(key=lambda x: x["final_score"], reverse=True)
    return stage3


async def rerank_with_l2r(
    query: str,
    candidates: list[dict],
    tenant_id: str = "default",
    top_k: int = 10,
    enable_l2r: bool = True,
) -> list[dict]:
    """
    Run L2R reranking on candidates (Stage 4).

    This is called after the standard 3-stage pipeline.
    It uses LambdaMART to learn optimal combination of all retrieval signals.
    """
    if not enable_l2r:
        return candidates[:top_k]

    try:
        from src.services.l2r import l2r_rerank
    except ImportError:
        logger.debug("L2R not available, returning candidates as-is")
        return candidates[:top_k]

    return await l2r_rerank(
        candidates=candidates,
        query_info={"query": query},
        tenant_id=tenant_id,
        top_k=top_k,
    )
