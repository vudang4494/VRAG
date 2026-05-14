"""Query Understanding — 6 reformulations + intent classification.

Cho mỗi user query, sinh ra:
- original (passthrough)
- rewrite (LLM viết lại rõ ràng hơn)
- decompose (chia thành sub-questions nếu multi-hop)
- HyDE (sinh "câu trả lời giả định" để embed)
- step-back (trừu tượng hóa)
- keywords (cho sparse vector path)

Cộng với intent classifier: factual | analytical | summarization | comparison.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from loguru import logger


_REWRITE_PROMPT = """Viết lại câu hỏi sau cho rõ ràng và đầy đủ hơn, vẫn giữ nguyên ý.
Không thêm thông tin mới. Trả lời CHỈ với câu hỏi đã viết lại, không giải thích.

Câu hỏi: {query}

Câu hỏi đã viết lại:"""

_DECOMPOSE_PROMPT = """Phân tích câu hỏi sau. Nếu nó là câu hỏi đơn giản (single-hop), trả về chính nó.
Nếu nó cần nhiều bước trả lời (multi-hop), chia thành 2-3 sub-questions cụ thể.

Trả lời CHỈ với JSON:
{{"is_multi_hop": true/false, "sub_questions": ["...", "..."]}}

Câu hỏi: {query}

JSON:"""

_HYDE_PROMPT = """Hãy viết một đoạn văn ngắn 2-3 câu trả lời cho câu hỏi sau,
giả định bạn có thông tin (KHÔNG cần đúng sự thật — chỉ cần có cấu trúc và từ vựng đúng).

Câu hỏi: {query}

Câu trả lời giả định:"""

_STEP_BACK_PROMPT = """Lùi lại một bước và đặt một câu hỏi tổng quát hơn,
trừu tượng hơn câu hỏi sau. Mục tiêu là tìm bối cảnh rộng.

Câu hỏi cụ thể: {query}

Câu hỏi tổng quát hơn:"""

_KEYWORDS_PROMPT = """Trích xuất 3-7 từ khóa hoặc cụm từ quan trọng nhất từ câu hỏi sau.
Bao gồm tên riêng, mã số, ngày tháng, thuật ngữ chuyên môn nếu có.

Trả lời dạng danh sách phân cách bằng dấu phẩy, không giải thích.

Câu hỏi: {query}

Từ khóa:"""

_INTENT_PROMPT = """Phân loại ý định của câu hỏi sau vào MỘT trong 4 loại:
- factual: tìm thông tin cụ thể (số liệu, ngày tháng, tên, định nghĩa)
- analytical: phân tích nguyên nhân/kết quả, suy luận multi-hop
- summarization: tóm tắt, tổng hợp nhiều nguồn
- comparison: so sánh 2+ đối tượng

Trả lời CHỈ một từ: factual, analytical, summarization, hoặc comparison.

Câu hỏi: {query}

Loại:"""


async def _llm_text(llm: Any, model: str, prompt: str, max_tokens: int = 200) -> str:
    """Phase 0a fix — Ollama native to bypass Qwen3 thinking-mode content loss."""
    from src.services.ollama_helper import ollama_chat
    return await ollama_chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.2,
        max_tokens=max_tokens,
    )


async def rewrite_query(query: str, llm: Any, model: str = "qwen3.5:4b") -> str:
    text = await _llm_text(llm, model, _REWRITE_PROMPT.format(query=query), max_tokens=200)
    return text or query


async def decompose_query(query: str, llm: Any, model: str = "qwen3.5:4b") -> tuple[bool, list[str]]:
    raw = await _llm_text(llm, model, _DECOMPOSE_PROMPT.format(query=query), max_tokens=300)
    raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        data = json.loads(raw)
        return bool(data.get("is_multi_hop", False)), [
            s.strip() for s in data.get("sub_questions", []) if s.strip()
        ]
    except Exception:
        return False, []


async def hyde_generate(query: str, llm: Any, model: str = "qwen3.5:4b") -> str:
    text = await _llm_text(llm, model, _HYDE_PROMPT.format(query=query), max_tokens=300)
    return text or query


async def step_back_query(query: str, llm: Any, model: str = "qwen3.5:4b") -> str:
    text = await _llm_text(llm, model, _STEP_BACK_PROMPT.format(query=query), max_tokens=120)
    return text or query


async def extract_keywords(query: str, llm: Any, model: str = "qwen3.5:4b") -> str:
    text = await _llm_text(llm, model, _KEYWORDS_PROMPT.format(query=query), max_tokens=100)
    return text or query


async def classify_intent(query: str, llm: Any, model: str = "qwen3.5:4b") -> str:
    raw = await _llm_text(llm, model, _INTENT_PROMPT.format(query=query), max_tokens=20)
    raw = raw.lower().strip()
    for kw in ("factual", "analytical", "summarization", "comparison"):
        if kw in raw:
            return kw
    return "factual"


async def understand_query(
    query: str,
    llm: Any,
    model: str = "qwen3.5:4b",
    timeout: float = 10.0,
) -> dict[str, Any]:
    """
    Run all 6 reformulations + intent in parallel.

    Returns:
      {
        "original": str,
        "rewrite": str,
        "decompose": {"is_multi_hop": bool, "sub_questions": [str]},
        "hyde": str,
        "step_back": str,
        "keywords": str,
        "intent": "factual|analytical|summarization|comparison",
        "reformulations": [{"kind": ..., "text": ..., "weight": ...}],
      }
    """
    # Tasks ordered by importance — front-load the cheap/critical ones so they
    # complete even if later ones time out. With small LLMs (qwen3.5:4b),
    # reducing reformulations saves significant latency.
    from src.config import get_settings as _gs
    _n = _gs().query_reformulations
    all_tasks = [
        ("intent",    classify_intent(query, llm, model)),    # always
        ("rewrite",   rewrite_query(query, llm, model)),       # always
        ("keywords",  extract_keywords(query, llm, model)),    # always
        ("hyde",      hyde_generate(query, llm, model)),       # if n>=4
        ("decompose", decompose_query(query, llm, model)),     # if n>=5
        ("step_back", step_back_query(query, llm, model)),     # if n>=6
    ]
    # Always include intent classifier; cap other reformulations by config
    selected = [all_tasks[0]] + all_tasks[1:1 + max(_n - 1, 0)]
    tasks = dict(selected)

    gathered = asyncio.gather(*tasks.values(), return_exceptions=True)
    try:
        results = await asyncio.wait_for(gathered, timeout=timeout)
    except asyncio.TimeoutError:
        gathered.cancel()
        logger.warning(f"Query understanding timeout for: {query[:80]}")
        results = [None] * len(tasks)

    keys = list(tasks.keys())
    out: dict[str, Any] = {"original": query}
    for k, r in zip(keys, results):
        if isinstance(r, Exception) or r is None:
            out[k] = "" if k != "decompose" else (False, [])
        else:
            out[k] = r

    is_multi_hop, sub_qs = out.get("decompose") if isinstance(out.get("decompose"), tuple) else (False, [])

    reformulations = [{"kind": "original", "text": query, "weight": 1.0}]
    if out.get("rewrite"):
        reformulations.append({"kind": "rewrite",   "text": out["rewrite"], "weight": 1.1})
    if out.get("hyde"):
        reformulations.append({"kind": "hyde",      "text": out["hyde"],    "weight": 1.3})
    if out.get("step_back"):
        reformulations.append({"kind": "step_back", "text": out["step_back"], "weight": 0.8})
    if out.get("keywords"):
        reformulations.append({"kind": "keywords",  "text": out["keywords"], "weight": 0.9})
    if is_multi_hop and sub_qs:
        for sq in sub_qs[:2]:
            reformulations.append({"kind": "decompose", "text": sq, "weight": 1.1})

    out["reformulations"] = reformulations
    out["is_multi_hop"] = is_multi_hop
    # Ensure 'intent' always present (defaults to 'factual')
    out["intent"] = out.get("intent") or "factual"
    return out
