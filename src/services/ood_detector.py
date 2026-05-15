"""Phase 7b — Out-of-Domain (OOD) Detection.

Detects queries that are outside the knowledge corpus BEFORE generation.
This fixes the "0% refusal accuracy" issue: without OOD detection, the LLM
always tries to answer (and hallucinates) even when retrieved docs are irrelevant.

Approach:
  1. Compute average retrieval score of top candidates.
     If top candidates have very low scores → query is OOD.
  2. Check keyword overlap: query keywords vs top doc sources.
     If zero overlap → query is OOD.
  3. Optional LLM-based check: ask LLM to self-assess relevance of retrieved context.

Thresholds tuned for BGE-M3 + Qdrant cosine similarity on a 0..1 scale.
BGE-M3 cosine scores in range ~0.5-0.85 for relevant docs.
"""

from __future__ import annotations

import re
from typing import Any

# BGE-M3 cosine similarity thresholds for relevance detection.
# These are empirically derived from eval runs on the academic corpus.
_RELEVANCE_HIGH = 0.70  # Top score above this → definitely in-domain
_RELEVANCE_LOW = 0.50  # Top score below this → likely OOD
_RELEVANCE_MARGINAL = 0.60  # Between LOW and HIGH → check keyword overlap


def detect_ood_by_scores(candidates: list[dict], threshold: float = 0.50) -> bool:
    """
    OOD if ALL top candidates have scores below threshold.

    Uses the retrieval `score` field which is the Qdrant cosine similarity
    for the best-ranked path. This is the single most reliable OOD signal.
    """
    if not candidates:
        return True

    top_scores = sorted([c.get("score", 0.0) for c in candidates], reverse=True)
    # Check top-3 scores: if ALL are below threshold, likely OOD
    for s in top_scores[:3]:
        if s >= threshold:
            return False
    return True


def detect_ood_by_keyword_overlap(query: str, candidates: list[dict]) -> bool:
    """
    Extract key terms from query, check if they appear in retrieved doc sources.

    If query is about "GraphRAG" but retrieved docs are about "biology",
    the keyword mismatch strongly suggests OOD.

    Returns True if overlap is too low → likely OOD.
    """
    if not candidates:
        return True

    # Extract significant terms from query (3+ chars, not common stop words)
    stop_words = {
        "của",
        "là",
        "gì",
        "cái",
        "có",
        "không",
        "và",
        "trong",
        "như",
        "thế",
        "nào",
        "với",
        "để",
        "cho",
        "từ",
        "hay",
        "theo",
        "về",
        "được",
        "các",
        "một",
        "này",
        "đó",
        "ra",
        "what",
        "is",
        "the",
        "and",
        "of",
        "to",
        "a",
        "in",
        "for",
        "how",
        "does",
        "why",
        "when",
        "where",
        "which",
    }
    query_lower = query.lower()
    # Split on spaces and punctuation, filter stop words and short terms
    query_terms = set()
    for tok in re.split(r"[\s\W]+", query_lower):
        if len(tok) >= 3 and tok not in stop_words:
            query_terms.add(tok)

    if not query_terms:
        return False

    # Collect text from top candidates
    doc_text = " ".join((c.get("text") or c.get("source") or "").lower() for c in candidates[:5])

    # Count how many query terms appear in docs
    overlap = sum(1 for t in query_terms if t in doc_text)
    overlap_ratio = overlap / len(query_terms)

    # If < 30% of query terms appear in retrieved docs → OOD
    return overlap_ratio < 0.30


def detect_ood_mixed(candidates: list[dict], query: str) -> dict[str, Any]:
    """
    Combined OOD detection using multiple signals.

    Returns {
        "is_ood": bool,
        "top_score": float,
        "keyword_overlap_ratio": float,
        "confidence": float,   # 0..1, how sure we are
        "reason": str,
    }
    """
    if not candidates:
        return {
            "is_ood": True,
            "top_score": 0.0,
            "keyword_overlap_ratio": 0.0,
            "confidence": 0.95,
            "reason": "no_retrieval_candidates",
        }

    top_score = max(c.get("score", 0.0) for c in candidates)
    top3_avg = sum(sorted([c.get("score", 0.0) for c in candidates], reverse=True)[:3]) / 3

    # Keyword overlap
    stop_words = {
        "của",
        "là",
        "gì",
        "cái",
        "có",
        "không",
        "và",
        "trong",
        "như",
        "thế",
        "nào",
        "với",
        "để",
        "cho",
        "từ",
        "hay",
        "theo",
        "về",
        "được",
        "các",
        "một",
        "này",
        "đó",
        "ra",
        "what",
        "is",
        "the",
        "and",
        "of",
        "to",
        "a",
        "in",
        "for",
        "how",
        "does",
        "why",
        "when",
        "where",
        "which",
    }
    query_terms = set()
    for tok in re.split(r"[\s\W]+", query.lower()):
        if len(tok) >= 3 and tok not in stop_words:
            query_terms.add(tok)

    doc_text = " ".join((c.get("text") or c.get("source") or "").lower() for c in candidates[:5])
    overlap = sum(1 for t in query_terms if t in doc_text)
    kw_ratio = overlap / len(query_terms) if query_terms else 1.0

    # Decision logic
    is_ood = False
    reason = "in_domain"

    if top_score < _RELEVANCE_LOW:
        if kw_ratio < 0.30:
            is_ood = True
            reason = "low_score_no_keyword_overlap"
        else:
            reason = "low_score_but_keyword_match"
    elif top_score < _RELEVANCE_MARGINAL:
        if kw_ratio < 0.30:
            is_ood = True
            reason = "marginal_score_no_keyword_overlap"
        else:
            reason = "marginal_score_with_keyword_match"
    else:
        reason = "high_score_in_domain"

    # Confidence: how sure are we?
    if is_ood:
        confidence = 0.90 if top_score < _RELEVANCE_LOW else 0.75
    else:
        confidence = 0.90 if top_score >= _RELEVANCE_MARGINAL else 0.65

    return {
        "is_ood": is_ood,
        "top_score": round(top_score, 4),
        "top3_avg_score": round(top3_avg, 4),
        "keyword_overlap_ratio": round(kw_ratio, 3),
        "confidence": round(confidence, 2),
        "reason": reason,
        "num_candidates": len(candidates),
        "query_terms_found": overlap,
        "query_terms_total": len(query_terms),
    }
