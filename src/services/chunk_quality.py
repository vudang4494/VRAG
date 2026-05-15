"""Phase 6a — Chunk Quality Classifier (CQC).

Pre-ingest filter to reject low-quality chunks (chat fluff, draft markers,
promotional content, outdated info). Critical for ingesting chat logs and
heterogeneous content where blind ingestion ruins KG.

Multi-signal score (no LLM, all regex/counting):
  + length within bounds      → +
  + entity density            → +
  + structured content (code/table/list) → +
  - hedge / uncertainty words  → -
  - marketing / promotional    → -
  - draft markers (TODO/DRAFT) → -
  - excessive filler            → -

Returns score 0-1. Drop if < threshold (default 0.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


HEDGE_WORDS_VI = [
    r"\bcó thể\b",
    r"\bchắc(?: chắn)?\b",
    r"\bcó lẽ\b",
    r"\bhình như\b",
    r"\bgiả định\b",
    r"\bdường như\b",
    r"\bnghi ngờ\b",
]
HEDGE_WORDS_EN = [
    r"\bmaybe\b",
    r"\bpossibly\b",
    r"\bperhaps\b",
    r"\bI think\b",
    r"\bsupposedly\b",
    r"\bseems like\b",
]

MARKETING_VI = [
    r"\btuyệt vời\b",
    r"\bđột phá\b",
    r"\b(siêu|cực kỳ)\s+\w+",
    r"\b#1 (về|cho)\b",
    r"\b(tốt|hàng đầu|đẳng cấp)\s+\w+",
]
MARKETING_EN = [
    r"\bamazing\b",
    r"\bgame[- ]?chang(?:er|ing)\b",
    r"\brevolution(?:ary)?\b",
    r"\b(best|world[- ]class)\s+\w+",
    r"\bbreakthrough\b",
    r"\b(unique|unmatched)\b",
]

DRAFT_MARKERS = [
    r"\bTODO\b",
    r"\bFIXME\b",
    r"\bDRAFT\b",
    r"\bWIP\b",
    r"\bXXX\b",
    r"\b\(\?\?\?\)\b",
    r"\[\s*placeholder\s*\]",
]

CODE_PATTERN = re.compile(r"```[\w]*\n[\s\S]+?```|`[^`]+`")
TABLE_PATTERN = re.compile(r"^\|.*\|\n\|[\s\-:|]+\|", re.MULTILINE)
LIST_PATTERN = re.compile(r"^\s*[-*+•]\s+\S+", re.MULTILINE)


@dataclass
class QualityScore:
    overall: float  # 0-1
    length_ok: bool
    entity_density: float
    has_structure: bool
    hedge_count: int
    marketing_count: int
    draft_markers: int
    accept: bool
    reasons: list[str]


def assess_chunk_quality(
    text: str,
    entity_count: int = 0,
    min_length: int = 50,
    max_length: int = 5000,
    threshold: float = 0.4,
) -> QualityScore:
    """Compute chunk quality. Returns QualityScore with accept verdict."""
    text = text.strip()
    L = len(text)
    words = len(text.split())

    reasons: list[str] = []
    score = 1.0  # start optimistic, deduct

    # ── Length checks ──
    length_ok = min_length <= L <= max_length
    if L < min_length:
        score -= 0.5
        reasons.append(f"too_short ({L}<{min_length})")
    elif L > max_length:
        score -= 0.2
        reasons.append(f"too_long ({L}>{max_length})")

    # ── Entity density ──
    density = entity_count / max(words, 1)
    if density < 0.005 and L > 200:  # < 1 entity per 200 words
        score -= 0.15
        reasons.append(f"low_entity_density ({density:.4f})")
    elif density > 0.05:  # rich in entities
        score += 0.05

    # ── Structure presence (boost) ──
    has_struct = bool(
        CODE_PATTERN.search(text) or TABLE_PATTERN.search(text) or LIST_PATTERN.search(text)
    )
    if has_struct:
        score += 0.10  # structured content is valuable
    elif L < 100:
        score -= 0.10
        reasons.append("no_structure_short")

    # ── Hedge words (uncertainty signal) ──
    hedge_total = 0
    for p in HEDGE_WORDS_VI + HEDGE_WORDS_EN:
        hedge_total += len(re.findall(p, text, re.IGNORECASE))
    hedge_ratio = hedge_total / max(words, 1)
    if hedge_ratio > 0.05:  # > 5% hedge words = uncertain content
        score -= min(0.3, hedge_ratio * 2)
        reasons.append(f"high_hedge ({hedge_ratio:.2%})")

    # ── Marketing / promotional ──
    marketing_total = 0
    for p in MARKETING_VI + MARKETING_EN:
        marketing_total += len(re.findall(p, text, re.IGNORECASE))
    if marketing_total > 2:
        score -= 0.2
        reasons.append(f"marketing ({marketing_total} hits)")

    # ── Draft markers (auto-reject if found) ──
    draft_count = 0
    for p in DRAFT_MARKERS:
        draft_count += len(re.findall(p, text, re.IGNORECASE))
    if draft_count > 0:
        score -= 0.5
        reasons.append(f"draft_markers ({draft_count})")

    # Final clamp
    score = max(0.0, min(1.0, score))
    accept = score >= threshold

    return QualityScore(
        overall=score,
        length_ok=length_ok,
        entity_density=density,
        has_structure=has_struct,
        hedge_count=hedge_total,
        marketing_count=marketing_total,
        draft_markers=draft_count,
        accept=accept,
        reasons=reasons,
    )


def filter_chunks_by_quality(
    chunks: list[dict],
    threshold: float = 0.4,
    entity_counts: dict[str, int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Split chunks into (kept, rejected) by quality score.

    entity_counts: optional dict {chunk_id: int} from prior NER pass.
    """
    entity_counts = entity_counts or {}
    kept: list[dict] = []
    rejected: list[dict] = []
    for c in chunks:
        cid = c.get("id") or c.get("chunk_id", "")
        ec = entity_counts.get(cid, 0)
        score = assess_chunk_quality(c.get("text", ""), entity_count=ec, threshold=threshold)
        c["quality_score"] = score.overall
        c["quality_reasons"] = score.reasons
        if score.accept:
            kept.append(c)
        else:
            rejected.append(c)
    return kept, rejected
