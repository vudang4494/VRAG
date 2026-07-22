"""Query Understanding — reformulations + intent classification + fast entity extraction.

Cho mỗi user query, sinh ra:
- original (passthrough)
- rewrite (LLM viết lại rõ ràng hơn)
- decompose (chia thành sub-questions nếu multi-hop)
- HyDE (sinh "câu trả lời giả định" để embed)
- step-back (trừu tượng hóa)
- keywords (cho sparse vector path)
- intent (factual | analytical | comparison | multi_hop | kg_construction)
- entities (GLiNER — fast, no LLM call)
- reformulations (weighted list for multi-path retrieval)

Cải tiến so với phiên bản cũ:
- GLiNER thay LLM cho entity extraction: <100ms thay vì 3-10s
- Intent classification chuyển sang query_router (semantic centroid matching)
- Giữ nguyên reformulations LLM cho quality vì chúng cần reasoning
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

_INTENT_PROMPT = """Phân loại ý định của câu hỏi sau vào MỘT trong 5 loại:
- factual: tìm thông tin cụ thể (số liệu, ngày tháng, tên, định nghĩa)
- analytical: phân tích nguyên nhân/kết quả, suy luận
- summarization: tóm tắt, tổng hợp nhiều nguồn
- comparison: so sánh 2+ đối tượng
- multi_hop: suy luận qua nhiều bước, liên quan đến nhiều documents
- kg_construction: hỏi về cấu trúc, schema, pipeline của knowledge graph

Trả lời CHỈ một từ: factual, analytical, summarization, comparison, multi_hop, hoặc kg_construction.

Câu hỏi: {query}

Loại:"""


async def _llm_text(llm: Any, model: str, prompt: str, max_tokens: int = 200) -> str:
    """Use Ollama native chat to avoid Qwen3 thinking-mode content loss."""
    from src.services.ollama_helper import ollama_chat

    return await ollama_chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.2,
        max_tokens=max_tokens,
    )


async def rewrite_query(query: str, llm: Any, model: str = "gemma4:e4b") -> str:
    text = await _llm_text(llm, model, _REWRITE_PROMPT.format(query=query), max_tokens=200)
    return text or query


async def decompose_query(
    query: str, llm: Any, model: str = "gemma4:e4b"
) -> tuple[bool, list[str]]:
    raw = await _llm_text(llm, model, _DECOMPOSE_PROMPT.format(query=query), max_tokens=300)
    raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        data = json.loads(raw)
        return bool(data.get("is_multi_hop", False)), [
            s.strip() for s in data.get("sub_questions", []) if s.strip()
        ]
    except Exception:
        return False, []


async def hyde_generate(query: str, llm: Any, model: str = "gemma4:e4b") -> str:
    text = await _llm_text(llm, model, _HYDE_PROMPT.format(query=query), max_tokens=300)
    return text or query


async def step_back_query(query: str, llm: Any, model: str = "gemma4:e4b") -> str:
    text = await _llm_text(llm, model, _STEP_BACK_PROMPT.format(query=query), max_tokens=120)
    return text or query


async def extract_keywords(query: str, llm: Any, model: str = "gemma4:e4b") -> str:
    text = await _llm_text(llm, model, _KEYWORDS_PROMPT.format(query=query), max_tokens=100)
    return text or query


async def classify_intent(query: str, llm: Any, model: str = "gemma4:e4b") -> str:
    raw = await _llm_text(llm, model, _INTENT_PROMPT.format(query=query), max_tokens=20)
    raw = raw.lower().strip()
    for kw in ("factual", "analytical", "summarization", "comparison"):
        if kw in raw:
            return kw
    return "factual"


# ── Fast entity extraction via GLiNER (no LLM call) ──────────────────────────────

# Labels tuned for academic RAG: entities relevant to research papers
_QUERY_ENTITY_LABELS = [
    "technology",
    "algorithm",
    "metric",
    "organization",
    "person",
    "dataset",
    "product",
    "concept",
]

_gliner_extractor: Any = None


async def _get_gliner():
    """Lazy-load GLiNER extractor on first use."""
    global _gliner_extractor
    if _gliner_extractor is not None:
        return _gliner_extractor
    try:
        from src.services.entity_extractor import GLiNERExtractor

        _gliner_extractor = GLiNERExtractor(
            model_name="urchade/gliner_multi-v2.1",
            labels=_QUERY_ENTITY_LABELS,
            threshold=0.3,
            max_chars=1500,
        )
        logger.info("GLiNER loaded for query-time entity extraction")
    except Exception as e:
        logger.warning(f"GLiNER entity extractor unavailable: {e}")
        _gliner_extractor = None
    return _gliner_extractor


# Tier 2 fix: comparative query patterns.
# GLiNER sometimes treats "X và Y" as a single joint entity, missing one of the
# two comparable entities. We supplement GLiNER output with regex-extracted
# entities from comparative phrasing so the entity_pivot path can search for
# both. Patterns ordered most-specific → most-general.
# Entity tokens for comparative regex: CapitalizedWord, ACRONYM, ACRONYM-XXX, hyphenated
_ENT_TOKEN = r"[A-ZĐ][\w\-]+(?:\s+[A-ZĐ][\w\-]+)?"
# Stop words for entity boundary — don't capture these as part of entity name
_ENT_STOP = r"(?:khác|là|có|trong|trên|về|cho|tại|theo|đối|cái|nào|hoạt|sử|xây|kết|cải|giải)"

_COMPARATIVE_PATTERNS: list[str] = [
    # "so sánh X và Y" / "so sánh X với Y" / "so sánh X vs Y"
    rf"so sánh\s+({_ENT_TOKEN})\s+(?:và|vs|với)\s+({_ENT_TOKEN})(?=[\s\?\.,;:]|$)",
    # "X so với Y" (BiXSE so với InfoNCE)
    rf"({_ENT_TOKEN})\s+(?:cải thiện|cải tiến|.+?)?\s*so với\s+({_ENT_TOKEN})(?=[\s\?\.,;:]|$)",
    # "X vs Y" — direct comparison
    rf"\b({_ENT_TOKEN})\s+vs\s+({_ENT_TOKEN})\b",
    # "X và Y khác nhau / khác gì": stop entity name before "khác"
    rf"({_ENT_TOKEN})\s+và\s+({_ENT_TOKEN})\s+khác",
    # "X và Y" + tail like "trong/về/cho" (entities are adjacent, query continues)
    rf"\b({_ENT_TOKEN})\s+và\s+({_ENT_TOKEN})\s+(?:trong|về|cho|đối với|có gì)",
]


_TRAILING_NON_ENTITY = {
    "cải",
    "thiện",
    "tiến",
    "về",
    "trong",
    "trên",
    "cho",
    "tại",
    "theo",
    "đối",
    "với",
    "có",
    "là",
    "nào",
    "khác",
    "nhau",
    "tốt",
    "hơn",
    "phương",
    "pháp",
    "hệ",
    "thống",
    "mô",
    "hình",
    "model",
    "method",
    "system",
    "approach",
    "the",
    "a",
    "an",
}


def _clean_entity(raw: str) -> str:
    """Strip common Vietnamese non-entity trailing tokens."""
    raw = raw.strip(" \t.,;:?!")
    parts = raw.split()
    # Trim trailing non-entity tokens
    while parts and parts[-1].lower() in _TRAILING_NON_ENTITY:
        parts.pop()
    return " ".join(parts).strip()


def _extract_comparative_entities(query: str) -> list[str]:
    """Extract entities from comparative phrasings via regex (supplements GLiNER)."""
    found: list[str] = []
    for pat in _COMPARATIVE_PATTERNS:
        m = re.search(pat, query, flags=re.IGNORECASE)
        if m and m.lastindex and m.lastindex >= 2:
            for i in range(1, m.lastindex + 1):
                cleaned = _clean_entity(m.group(i) or "")
                if 2 < len(cleaned) < 60 and cleaned.lower() not in {
                    "rag",
                    "system",
                    "method",
                    "approach",
                    "model",
                    "paper",
                    "phương pháp",
                    "hệ thống",
                    "mô hình",
                }:
                    found.append(cleaned)
            break  # first matching pattern wins
    # Dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for e in found:
        if e.lower() not in seen:
            seen.add(e.lower())
            out.append(e)
    return out


async def extract_entities_fast(query: str) -> list[str]:
    """
    Extract named entities from a user query via GLiNER + comparative regex.

    Returns a list of entity name strings (deduplicated).
    Runs in <100ms — no LLM call needed.

    Tier 2 fix: supplements GLiNER with regex for comparative queries like
    "X vs Y" / "so sánh X và Y" where GLiNER often misses one side.
    """
    extractor = await _get_gliner()
    gliner_ents: list[str] = []
    if extractor is not None:
        try:
            ents, _ = await extractor.extract(query)
            gliner_ents = [e.name for e in ents if e.confidence >= 0.3]
        except Exception as e:
            logger.debug(f"Fast entity extraction failed: {e}")

    # Supplement with comparative regex
    comp_ents = _extract_comparative_entities(query)
    if comp_ents:
        logger.info(f"  comparative entities: {comp_ents} (supplementing GLiNER {gliner_ents})")

    # Merge — dedupe by lowercase
    seen: set[str] = set()
    out: list[str] = []
    for e in list(gliner_ents) + comp_ents:
        key = e.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(e)
    return out


async def understand_query(
    query: str,
    llm: Any,
    model: str = "gemma4:e4b",
    timeout: float = 10.0,
    intent: str | None = None,
) -> dict[str, Any]:
    """
    VRAG Tier 1: Zero-LLM Pre-processing.

    By default runs ONLY zero-LLM signals in parallel:
      - Task 1: BGE-M3 query embed (handled downstream in retrieval)
      - Task 2: Semantic Router (centroid dot product, <1ms, no LLM)
      - Task 3: GLiNER entity extract (<100ms, no LLM)

    Optional LLM reformulations are opt-in via QUERY_REFORMULATIONS env var:
      - 0 (default): zero-LLM, fastest
      - 1: + rewrite
      - 2: + keywords
      - 3: + hyde
      - 4: + decompose
      - 5: + step_back (full menu)

    Returns:
      {
        "original": str,
        "intent": str,
        "entities": [str],
        "reformulations": [{"kind": ..., "text": ..., "weight": ...}],
        ... (rewrite/keywords/hyde/decompose/step_back only if enabled)
      }
    """
    from src.config import get_settings as _gs
    from src.services.query_router import classify_query

    _n = _gs().query_reformulations

    # Tier 1 always-on: GLiNER entity extraction (zero-LLM)
    tasks: dict[str, Any] = {"entities": extract_entities_fast(query)}

    # Opt-in LLM reformulations — ordered by retrieval impact
    llm_optional = [
        ("rewrite", rewrite_query(query, llm, model)),
        ("keywords", extract_keywords(query, llm, model)),
        ("hyde", hyde_generate(query, llm, model)),
        ("decompose", decompose_query(query, llm, model)),
        ("step_back", step_back_query(query, llm, model)),
    ]
    for kind, coro in llm_optional[: max(_n, 0)]:
        tasks[kind] = coro

    gathered = asyncio.gather(*tasks.values(), return_exceptions=True)
    try:
        results = await asyncio.wait_for(gathered, timeout=timeout)
    except TimeoutError:
        gathered.cancel()
        logger.warning(f"Query understanding timeout for: {query[:80]}")
        results = [None] * len(tasks)

    keys = list(tasks.keys())
    out: dict[str, Any] = {"original": query}
    for k, r in zip(keys, results, strict=False):
        if isinstance(r, Exception) or r is None:
            out[k] = "" if k not in ("decompose", "entities") else ([], [])
        else:
            out[k] = r

    is_multi_hop, sub_qs = (
        out.get("decompose") if isinstance(out.get("decompose"), tuple) else (False, [])
    )

    # Reformulations list — always includes "original"; others only if produced
    reformulations = [{"kind": "original", "text": query, "weight": 1.0}]
    if out.get("rewrite"):
        reformulations.append({"kind": "rewrite", "text": out["rewrite"], "weight": 1.1})
    if out.get("hyde"):
        reformulations.append({"kind": "hyde", "text": out["hyde"], "weight": 1.3})
    if out.get("step_back"):
        reformulations.append({"kind": "step_back", "text": out["step_back"], "weight": 0.8})
    if out.get("keywords"):
        reformulations.append({"kind": "keywords", "text": out["keywords"], "weight": 0.9})
    if is_multi_hop and sub_qs:
        for sq in sub_qs[:2]:
            reformulations.append({"kind": "decompose", "text": sq, "weight": 1.1})

    # Intent from semantic router — zero-LLM, <1ms
    # Reuse the caller's already-computed intent (the router runs classify_query for its
    # routing decision) instead of embedding the query a second time via bge-m3. Falls back
    # to computing it when called without one (e.g. the streaming path) — off the event
    # loop, since classify_query does a blocking embed.
    if intent is not None:
        out["intent"] = intent
    else:
        out["intent"] = await asyncio.to_thread(classify_query, query)
    out["reformulations"] = reformulations
    out["is_multi_hop"] = is_multi_hop
    out["entities"] = out.get("entities") or []
    return out
