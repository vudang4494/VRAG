"""Validation gates — hallucination check + entity validation + citation completeness.

Three gates run in parallel. If any fails, the response should be refused
or retried with broader retrieval (per config.validation_retry_on_fail).
"""

from __future__ import annotations

import asyncio
import json
import re
from difflib import SequenceMatcher
from typing import Any

from loguru import logger


def _entity_fuzzy_match(candidate: str, kg_names: list[str], threshold: float = 0.80) -> str | None:
    """Return the best-matching KG entity name if similarity >= threshold, else None."""
    best_ratio = 0.0
    best_name = None
    cand_lower = candidate.lower()
    for kg_name in kg_names:
        if kg_name.lower() == cand_lower:
            return kg_name  # exact match takes priority
        ratio = SequenceMatcher(None, cand_lower, kg_name.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = kg_name
    if best_ratio >= threshold:
        return best_name
    return None


_CLAIM_EXTRACT_PROMPT = """Trích xuất các "atomic claims" (mệnh đề nguyên tử) từ câu trả lời sau.
Mỗi claim là một sự thật có thể kiểm chứng độc lập.

Trả về CHỈ JSON: {{"claims": ["...", "..."]}}

Câu trả lời:
{answer}

JSON:"""


_VERIFY_CLAIM_PROMPT = """Kiểm tra xem mệnh đề sau có được hỗ trợ bởi văn bản tham khảo hay không.

Mệnh đề: {claim}

Văn bản tham khảo:
{context}

Trả lời CHỈ một trong ba: YES (có hỗ trợ), NO (không hỗ trợ), PARTIAL (hỗ trợ một phần).
Đáp án:"""


_CITATION_PATTERN = re.compile(
    r"\[(?:chunk[_\-]?id\s*[:=]?\s*)?([\w\-]+(?::[\w\-]+)?(?:::[\w\-]+)?)\]",
    re.IGNORECASE,
)
# Also match Vietnamese draft prompt format: [doc_abc::para::5]
_CITATION_PATTERN_V2 = re.compile(r"\[doc[\w\-]+(?:::[\w\-]+)+\]", re.IGNORECASE)
_ENTITY_PATTERN = re.compile(
    r"\b([A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][\wÀ-ỹ]+(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]+){0,4})\b"
)


async def extract_atomic_claims(answer: str, llm: Any, model: str = "qwen3.5:9b") -> list[str]:
    """Use LLM to extract verifiable claims from the answer.

    Phase 0a fix: use Ollama native (think:false) to avoid Qwen3 empty content.
    """
    from src.services.ollama_helper import ollama_chat

    if not answer.strip():
        return []
    try:
        raw = await ollama_chat(
            messages=[{"role": "user", "content": _CLAIM_EXTRACT_PROMPT.format(answer=answer)}],
            model=model,
            temperature=0.1,
            max_tokens=600,
        )
        if not raw:
            raise ValueError("empty content")
        raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            raw = match.group(0)
        data = json.loads(raw)
        return [c.strip() for c in data.get("claims", []) if c.strip()]
    except Exception as e:
        logger.debug(f"Claim extraction failed: {e}")
        # Fallback: split by sentences
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10][:10]


async def verify_claim(claim: str, context: str, llm: Any, model: str = "qwen3.5:9b") -> str:
    """Returns 'YES' | 'NO' | 'PARTIAL'. Phase 0a — Ollama native."""
    from src.services.ollama_helper import ollama_chat

    try:
        raw = await ollama_chat(
            messages=[
                {
                    "role": "user",
                    "content": _VERIFY_CLAIM_PROMPT.format(claim=claim, context=context[:5000]),
                }
            ],
            model=model,
            temperature=0.1,
            max_tokens=20,
        )
        raw = raw.upper()
        for v in ("YES", "PARTIAL", "NO"):
            if v in raw:
                return v
    except Exception as e:
        logger.debug(f"Claim verification failed: {e}")
    return "PARTIAL"


async def hallucination_gate(
    answer: str,
    context: str,
    llm: Any,
    model: str = "qwen3.5:9b",
    min_grounded_ratio: float = 0.80,
    concurrent_limit: int = 4,
) -> dict[str, Any]:
    """
    Extract claims → verify each against context → return grounded_ratio.
    Pass if grounded (YES + 0.5*PARTIAL) ratio >= threshold.
    """
    claims = await extract_atomic_claims(answer, llm, model)
    if not claims:
        return {"passed": True, "grounded_ratio": 1.0, "claims_total": 0, "verdicts": []}

    sem = asyncio.Semaphore(concurrent_limit)

    async def _bounded(c: str) -> str:
        async with sem:
            return await verify_claim(c, context, llm, model)

    verdicts = await asyncio.gather(*[_bounded(c) for c in claims])
    weights = {"YES": 1.0, "PARTIAL": 0.5, "NO": 0.0}
    score = sum(weights.get(v, 0.0) for v in verdicts) / len(claims)
    return {
        "passed": score >= min_grounded_ratio,
        "grounded_ratio": score,
        "claims_total": len(claims),
        "verdicts": list(zip(claims, verdicts, strict=False)),
    }


async def entity_gate(
    answer: str,
    neo4j_driver,
    tenant_id: str | None = None,
    max_invalid: int = 2,
) -> dict[str, Any]:
    """
    Extract Title-Cased entities from answer, check against Neo4j.

    Matching strategy (3-tier):
      1. Exact lowercase match
      2. Fuzzy match (SequenceMatcher ratio >= 0.80) — catches name variants
         (e.g., "GraphRAG" vs "GraphRAG System", "Tim Cook" vs "Timothy Cook")
      3. Substring containment match
    Returns invalid only if all 3 tiers fail.
    """
    candidates = set()
    for match in _ENTITY_PATTERN.finditer(answer):
        ent = match.group(1).strip()
        if len(ent) > 3 and ent.lower() not in ("tôi", "bạn", "anh", "chị", "ông", "bà"):
            candidates.add(ent)

    if not candidates:
        return {"passed": True, "invalid": [], "checked": 0}

    cand_list = list(candidates)
    try:
        async with neo4j_driver.session() as s:
            # UNWIND batch lookup: exact lowercase match
            params: dict[str, Any] = {"names": [n.lower() for n in cand_list]}
            if tenant_id:
                cypher = """
                UNWIND $names AS n_lower
                OPTIONAL MATCH (e:Entity {tenant_id: $tid}) WHERE toLower(e.name) = n_lower
                RETURN n_lower AS cand_lower, collect(e.name) AS matches
                """
                params["tid"] = tenant_id
            else:
                cypher = """
                UNWIND $names AS n_lower
                OPTIONAL MATCH (e:Entity) WHERE toLower(e.name) = n_lower
                RETURN n_lower AS cand_lower, collect(e.name) AS matches
                """
            result = await s.run(cypher, **params)
            records = await result.data()

        # Build lookup: cand_lower → list of KG names that matched exactly
        exact_matches: dict[str, list[str]] = {}
        for row in records:
            if row["matches"]:
                exact_matches[row["cand_lower"]] = [m for m in row["matches"] if m]

        # Fetch all KG names for fuzzy matching (batched, only if needed)
        need_fuzzy = [c for c in cand_list if c.lower() not in exact_matches]
        kg_names: list[str] = []
        if need_fuzzy:
            try:
                async with neo4j_driver.session() as s:
                    if tenant_id:
                        cypher_all = "MATCH (e:Entity {tenant_id: $tid}) RETURN e.name AS name"
                        result = await s.run(cypher_all, tid=tenant_id)
                    else:
                        cypher_all = "MATCH (e:Entity) RETURN e.name AS name"
                        result = await s.run(cypher_all)
                    kg_records = await result.data()
                kg_names = [r["name"] for r in kg_records if r.get("name")]
            except Exception as e:
                logger.debug(f"Fuzzy lookup KG fetch failed: {e}")
                kg_names = []

        invalid: list[str] = []
        fuzzy_matched: list[str] = []
        for candidate in cand_list:
            cand_lower = candidate.lower()
            # Tier 1: exact lowercase match
            if cand_lower in exact_matches:
                continue
            # Tier 2: fuzzy match (ratio >= 0.80)
            matched = _entity_fuzzy_match(candidate, kg_names, threshold=0.80)
            if matched:
                fuzzy_matched.append(f"{candidate} → {matched}")
                continue
            # Tier 3: substring containment
            if any(cand_lower in n.lower() or n.lower() in cand_lower for n in kg_names):
                continue
            invalid.append(candidate)
    except Exception as e:
        logger.debug(f"Entity gate KG check failed: {e}")
        return {"passed": True, "invalid": [], "checked": len(candidates)}

    return {
        "passed": len(invalid) <= max_invalid,
        "invalid": invalid,
        "checked": len(candidates),
        "fuzzy_matched": fuzzy_matched,
    }


_REFUSAL_PATTERNS = [
    r"không có đủ thông tin",
    r"không tìm thấy",
    r"không thể trả lời",
    r"không đủ dữ liệu",
    r"i don'?t have enough information",
    r"insufficient information",
]


def is_refusal_answer(answer: str) -> bool:
    """Detect if LLM produced a refusal-style answer (which by design has no citations)."""
    if len(answer.strip()) < 200:  # short answers often refusals
        a = answer.lower()
        return any(re.search(p, a) for p in _REFUSAL_PATTERNS)
    return False


def citation_gate(answer: str, min_ratio: float = 0.40) -> dict[str, Any]:
    """
    Check that most sentences have a citation [chunk_id] marker.
    Skip check entirely if answer is a refusal (no citations expected).
    """
    # Refusal answers don't need citations — they're an explicit "I don't know"
    if is_refusal_answer(answer):
        return {"passed": True, "citation_ratio": 1.0, "uncited": [], "skipped_refusal": True}

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]
    if not sentences:
        return {"passed": True, "citation_ratio": 1.0, "uncited": []}
    cited = [s for s in sentences if _CITATION_PATTERN.search(s) or _CITATION_PATTERN_V2.search(s)]
    ratio = len(cited) / len(sentences)
    uncited = [
        s for s in sentences if not (_CITATION_PATTERN.search(s) or _CITATION_PATTERN_V2.search(s))
    ]
    return {
        "passed": ratio >= min_ratio,
        "citation_ratio": ratio,
        "uncited": uncited[:5],
    }


async def validate_answer(
    answer: str,
    context: str,
    llm: Any,
    neo4j_driver=None,
    tenant_id: str | None = None,
    model: str = "qwen3.5:9b",
    min_grounded_ratio: float = 0.80,
    max_invalid_entities: int = 2,
    min_citation_ratio: float = 0.70,
) -> dict[str, Any]:
    """
    Run 3 gates in parallel. Return combined result.

    Returns:
      {
        "passed": bool,
        "grounded_ratio": float,
        "invalid_entities": list[str],
        "citation_ratio": float,
        "failure_reason": str | None,
        "confidence": float,
      }
    """
    halluc_task = hallucination_gate(answer, context, llm, model, min_grounded_ratio)
    entity_task = (
        entity_gate(answer, neo4j_driver, tenant_id, max_invalid_entities)
        if neo4j_driver
        else _passthrough_entity()
    )
    cite_result = citation_gate(answer, min_citation_ratio)

    halluc, entity = await asyncio.gather(halluc_task, entity_task)

    reasons = []
    if not halluc["passed"]:
        reasons.append(f"ungrounded_claims({halluc['grounded_ratio']:.2f}<{min_grounded_ratio})")
    if not entity["passed"]:
        reasons.append(f"unknown_entities({len(entity['invalid'])}>{max_invalid_entities})")
    if not cite_result["passed"]:
        reasons.append(f"low_citations({cite_result['citation_ratio']:.2f}<{min_citation_ratio})")

    confidence = (
        halluc["grounded_ratio"]
        * (1.0 - 0.2 * (not entity["passed"]))
        * (1.0 - 0.1 * (not cite_result["passed"]))
    )

    return {
        "passed": halluc["passed"] and entity["passed"] and cite_result["passed"],
        "grounded_ratio": halluc["grounded_ratio"],
        "invalid_entities": entity["invalid"],
        "citation_ratio": cite_result["citation_ratio"],
        "failure_reason": "; ".join(reasons) if reasons else None,
        "confidence": confidence,
        "details": {
            "claims_total": halluc.get("claims_total", 0),
            "verdicts": halluc.get("verdicts", []),
            "entities_checked": entity.get("checked", 0),
            "uncited_examples": cite_result.get("uncited", []),
        },
    }


async def _passthrough_entity() -> dict[str, Any]:
    return {"passed": True, "invalid": [], "checked": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Layer 10.4 — Self-Correction Loop (CRAG-style)
# When a gate fails, apply targeted corrective action before retrying generation.
# ─────────────────────────────────────────────────────────────────────────────

_CORRECTIVE_PROMPTS = {
    "hallucination": (
        "Bạn là một trợ lý nghiêm túc. Trả lời câu hỏi dựa TRÊN VÀO văn bản tham khảo được cung cấp.\n"
        "QUAN TRỌNG:\n"
        "1. CHỈ sử dụng thông tin có trong văn bản tham khảo\n"
        "2. Nếu không chắc chắn, nói rõ 'Không có thông tin trong tài liệu'\n"
        "3. KHÔNG suy đoán hay bịa đặt thông tin\n"
        "4. TRÍCH DẪN cụ thể: [chunk_id] sau mỗi mệnh đề\n\n"
        "Văn bản tham khảo:\n{context}\n\n"
        "Câu hỏi: {query}\n\n"
        "Trả lời (tiếng Việt, có trích dẫn [chunk_id]):"
    ),
    "citation": (
        "Trả lời câu hỏi. SAU MỖI câu, thêm trích dẫn [chunk_id] gốc.\n"
        "QUAN TRỌNG: Nếu thông tin không có trong văn bản tham khảo, ghi rõ 'Không có thông tin'.\n\n"
        "Văn bản tham khảo:\n{context}\n\n"
        "Câu hỏi: {query}\n\n"
        "Trả lời (có trích dẫn [chunk_id] cho TỪNG mệnh đề):"
    ),
    "entity": (
        "Trả lời câu hỏi. Chỉ nhắc đến các thực thể có trong Neo4j graph.\n"
        "Nếu cần dùng tên biến thể, dùng tên chuẩn từ graph.\n\n"
        "Văn bản tham khảo:\n{context}\n\n"
        "Câu hỏi: {query}\n\n"
        "Trả lời (tiếng Việt, dùng đúng tên thực thể từ graph):"
    ),
}


async def correct_and_regenerate(
    query: str,
    context: str,
    llm: Any,
    model: str = "qwen3.5:9b",
    failure_reason: str = "",
) -> str:
    """
    Apply targeted corrective regeneration based on which gate failed.

    Maps failure reasons to prompts that constrain generation to fix the issue.
    """
    from src.services.ollama_helper import ollama_chat

    reason_lower = failure_reason.lower()
    if "ungrounded" in reason_lower or "hallucination" in reason_lower:
        prompt_key = "hallucination"
    elif "citation" in reason_lower or "low_citation" in reason_lower:
        prompt_key = "citation"
    elif "entity" in reason_lower or "unknown_entity" in reason_lower:
        prompt_key = "entity"
    else:
        # Generic fallback — emphasize grounding + citation
        prompt_key = "hallucination"

    prompt = _CORRECTIVE_PROMPTS[prompt_key].format(context=context[:6000], query=query)
    try:
        answer = await ollama_chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.1,
            max_tokens=1024,
        )
        return answer or ""
    except Exception as e:
        logger.debug(f"Corrective regeneration failed: {e}")
        return ""


def suggest_correction_action(failure_reason: str) -> str:
    """
    Suggest which corrective action to take based on gate failure.

    Returns one of: "regenerate_strict" | "expand_context" | "relax_threshold" | "refuse"
    """
    reason_lower = failure_reason.lower()
    if "ungrounded" in reason_lower:
        return "regenerate_strict"
    if "low_citation" in reason_lower:
        return "expand_context"
    if "unknown_entity" in reason_lower:
        return "expand_context"
    return "refuse"
