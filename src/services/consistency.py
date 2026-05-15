"""Consistency Simulation — 5-view embedding + pairwise variance scoring.

Cho mỗi chunk:
1. Sinh 5 views (LLM): original / paraphrase / question / summary / keywords
2. Embed cả 5 views với bge-m3
3. Compute consistency_score = mean pairwise cosine của 5 vectors
4. Output: { view_text: dict, view_embeddings: dict, consistency_score: float }
"""

from __future__ import annotations

import asyncio
import re
from itertools import combinations
from typing import Any

import httpx
from loguru import logger

from src.services.embedding import cosine_similarity, embed_single


_PARAPHRASE_PROMPT = """Diễn đạt lại đoạn văn bản sau bằng cách dùng từ ngữ khác,
nhưng giữ nguyên ý chính. Trả lời ngắn gọn, không thêm giải thích.

Văn bản:
{text}

Phiên bản diễn đạt lại:"""

_QUESTION_PROMPT = """Đoạn văn bản sau trả lời cho những câu hỏi nào?
Liệt kê 2-3 câu hỏi cụ thể mà đoạn này có thể là câu trả lời.

Văn bản:
{text}

Danh sách câu hỏi (mỗi câu trên một dòng, không đánh số):"""

_SUMMARY_PROMPT = """Tóm tắt ý chính của đoạn văn bản sau bằng 1-2 câu ngắn gọn.

Văn bản:
{text}

Tóm tắt:"""

_KEYWORDS_PROMPT = """Liệt kê 5-10 từ khóa hoặc cụm từ quan trọng nhất trong đoạn văn bản sau.
Trả lời dạng danh sách phân cách bằng dấu phẩy, không giải thích.

Văn bản:
{text}

Từ khóa:"""


async def _llm_call(llm: Any, model: str, prompt: str, max_tokens: int = 256) -> str:
    """Use Ollama native API (Phase 0a fix). The `llm` arg is kept for signature
    compat but unused; helper reads global clients/settings.
    """
    from src.services.ollama_helper import ollama_chat

    return await ollama_chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.3,
        max_tokens=max_tokens,
    )


async def generate_views(
    text: str,
    llm: Any,
    model: str = "qwen3.5:4b",
    enable_llm_views: bool = True,
) -> dict[str, str]:
    """
    Sinh 5 views của 1 chunk. Nếu enable_llm_views=False hoặc text quá ngắn,
    skip LLM views (chỉ trả 'original').
    """
    if not text.strip():
        return {"original": text}

    if not enable_llm_views or len(text) < 100:
        return {"original": text}

    # Truncate input cho LLM
    snippet = text[:2000]

    tasks = {
        "paraphrase": _llm_call(
            llm, model, _PARAPHRASE_PROMPT.format(text=snippet), max_tokens=400
        ),
        "question": _llm_call(llm, model, _QUESTION_PROMPT.format(text=snippet), max_tokens=200),
        "summary": _llm_call(llm, model, _SUMMARY_PROMPT.format(text=snippet), max_tokens=150),
        "keywords": _llm_call(llm, model, _KEYWORDS_PROMPT.format(text=snippet), max_tokens=100),
    }

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    views = {"original": text}
    for (name, _), result in zip(tasks.items(), results, strict=False):
        if isinstance(result, Exception) or not result:
            continue
        views[name] = result
    return views


async def embed_views(
    views: dict[str, str],
    http: httpx.AsyncClient,
    embed_url: str,
    embed_model: str = "bge-m3",
    timeout: float = 60.0,
) -> dict[str, list[float]]:
    """Embed mỗi view. Trả về dict {view_name: embedding}."""
    names = list(views.keys())
    texts = [views[n] for n in names]

    async def _embed_one(t: str) -> list[float]:
        try:
            return await embed_single(http, embed_url, embed_model, t, timeout=timeout)
        except Exception:
            return []

    embeds = await asyncio.gather(*[_embed_one(t) for t in texts])
    return {n: v for n, v in zip(names, embeds, strict=False) if v}


def consistency_score(view_embeddings: dict[str, list[float]]) -> float:
    """Mean pairwise cosine similarity của các view embeddings."""
    vecs = [v for v in view_embeddings.values() if v]
    if len(vecs) < 2:
        return 0.0
    sims = [cosine_similarity(a, b) for a, b in combinations(vecs, 2)]
    return float(sum(sims) / len(sims)) if sims else 0.0


def classify_consistency(score: float, low: float = 0.60, high: float = 0.85) -> str:
    if score >= high:
        return "high"
    if score >= low:
        return "normal"
    return "low"


def consistency_boost(score: float, low: float = 0.60, high: float = 0.85) -> float:
    """Multiplier dùng cho RRF weighting."""
    if score >= high:
        return 1.2
    if score >= low:
        return 1.0
    return 0.8


async def process_chunk_consistency(
    chunk_text: str,
    llm: Any,
    http: httpx.AsyncClient,
    embed_url: str,
    embed_model: str = "bge-m3",
    llm_model: str = "qwen3.5:4b",
    enable_llm_views: bool = True,
) -> dict[str, Any]:
    """
    Full per-chunk consistency: generate views → embed → score.

    Returns dict:
      {
        "views": {name: text},
        "view_embeddings": {name: list[float]},
        "consistency_score": float,
        "consistency_class": "high|normal|low",
      }
    """
    views = await generate_views(chunk_text, llm, llm_model, enable_llm_views)
    embeds = await embed_views(views, http, embed_url, embed_model)
    score = consistency_score(embeds)
    return {
        "views": views,
        "view_embeddings": embeds,
        "consistency_score": score,
        "consistency_class": classify_consistency(score),
    }


async def process_batch_consistency(
    chunks: list[dict],
    llm: Any,
    http: httpx.AsyncClient,
    embed_url: str,
    embed_model: str = "bge-m3",
    llm_model: str = "qwen3.5:4b",
    concurrent_limit: int = 3,
    enable_llm_views: bool = True,
) -> list[dict]:
    """
    Process consistency cho batch chunks. Hạn chế concurrent_limit để
    không saturate Ollama (Metal GPU M4 = 3 streams optimal).
    """
    sem = asyncio.Semaphore(concurrent_limit)

    async def _one(chunk: dict) -> dict:
        async with sem:
            result = await process_chunk_consistency(
                chunk.get("text", ""),
                llm,
                http,
                embed_url,
                embed_model,
                llm_model,
                enable_llm_views,
            )
            return {**chunk, **result}

    return await asyncio.gather(*[_one(c) for c in chunks])
