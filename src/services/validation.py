"""Validation gates — hallucination check + entity validation + citation completeness.

Three gates run in parallel. If any fails, the response should be refused
or retried with broader retrieval (per config.validation_retry_on_fail).
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from loguru import logger


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


_CITATION_PATTERN = re.compile(r"\[(?:chunk[_\-]?id\s*[:=]?\s*)?([\w\-]+)\]")
_ENTITY_PATTERN = re.compile(
    r"\b([A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][\wÀ-ỹ]+(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]+){0,4})\b"
)


async def extract_atomic_claims(answer: str, llm: Any, model: str = "qwen3.5:4b") -> list[str]:
    """Use LLM to extract verifiable claims from the answer."""
    if not answer.strip():
        return []
    try:
        resp = await llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _CLAIM_EXTRACT_PROMPT.format(answer=answer)}],
            temperature=0.1,
            max_tokens=600,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        data = json.loads(raw)
        return [c.strip() for c in data.get("claims", []) if c.strip()]
    except Exception as e:
        logger.debug(f"Claim extraction failed: {e}")
        # Fallback: split by sentences
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10][:10]


async def verify_claim(claim: str, context: str, llm: Any, model: str = "qwen3.5:4b") -> str:
    """Returns 'YES' | 'NO' | 'PARTIAL'."""
    try:
        resp = await llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _VERIFY_CLAIM_PROMPT.format(claim=claim, context=context[:5000])}],
            temperature=0.1,
            max_tokens=20,
        )
        raw = (resp.choices[0].message.content or "").strip().upper()
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
    model: str = "qwen3.5:4b",
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
        "verdicts": list(zip(claims, verdicts)),
    }


async def entity_gate(
    answer: str,
    neo4j_driver,
    tenant_id: str | None = None,
    max_invalid: int = 2,
) -> dict[str, Any]:
    """
    Extract Title-Cased entities from answer (heuristic), check against Neo4j.
    Returns invalid entities count.
    """
    candidates = set()
    for match in _ENTITY_PATTERN.finditer(answer):
        ent = match.group(1).strip()
        if len(ent) > 3 and not ent.lower() in ("tôi", "bạn", "anh", "chị", "ông", "bà"):
            candidates.add(ent)

    if not candidates:
        return {"passed": True, "invalid": [], "checked": 0}

    invalid: list[str] = []
    try:
        async with neo4j_driver.session() as s:
            params = {"names": list(candidates)}
            cypher = """
            UNWIND $names AS n
            OPTIONAL MATCH (e:Entity) WHERE toLower(e.name) = toLower(n)
            RETURN n AS name, count(e) AS found
            """
            if tenant_id:
                cypher = """
                UNWIND $names AS n
                OPTIONAL MATCH (e:Entity {tenant_id: $tid}) WHERE toLower(e.name) = toLower(n)
                RETURN n AS name, count(e) AS found
                """
                params["tid"] = tenant_id
            result = await s.run(cypher, **params)
            data = await result.data()
            for row in data:
                if row["found"] == 0:
                    invalid.append(row["name"])
    except Exception as e:
        logger.debug(f"Entity gate KG check failed: {e}")
        return {"passed": True, "invalid": [], "checked": len(candidates)}

    return {
        "passed": len(invalid) <= max_invalid,
        "invalid": invalid,
        "checked": len(candidates),
    }


def citation_gate(answer: str, min_ratio: float = 0.70) -> dict[str, Any]:
    """
    Check that most sentences have a citation [chunk_id] marker.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]
    if not sentences:
        return {"passed": True, "citation_ratio": 1.0, "uncited": []}
    cited = [s for s in sentences if _CITATION_PATTERN.search(s)]
    ratio = len(cited) / len(sentences)
    uncited = [s for s in sentences if not _CITATION_PATTERN.search(s)]
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
    model: str = "qwen3.5:4b",
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
    entity_task = entity_gate(answer, neo4j_driver, tenant_id, max_invalid_entities) if neo4j_driver else _passthrough_entity()
    cite_result = citation_gate(answer, min_citation_ratio)

    halluc, entity = await asyncio.gather(halluc_task, entity_task)

    reasons = []
    if not halluc["passed"]:
        reasons.append(f"ungrounded_claims({halluc['grounded_ratio']:.2f}<{min_grounded_ratio})")
    if not entity["passed"]:
        reasons.append(f"unknown_entities({len(entity['invalid'])}>{max_invalid_entities})")
    if not cite_result["passed"]:
        reasons.append(f"low_citations({cite_result['citation_ratio']:.2f}<{min_citation_ratio})")

    confidence = halluc["grounded_ratio"] * (1.0 - 0.2 * (not entity["passed"])) * (1.0 - 0.1 * (not cite_result["passed"]))

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
