"""Phase 5a — Multi-signal chunker.

Replaces single-cosine topic boundary detection with 5-signal ensemble:
  S1. Semantic drop: bge-m3 cosine between consecutive sentence embeddings
  S2. Discourse marker: Vietnamese/English transition words ("Tuy nhiên", "However"...)
  S3. Entity shift: Jaccard distance between sentence entity sets (GLiNER)
  S4. Structural: markdown headings, blank lines, list items
  S5. Mode shift: fact ↔ opinion ↔ instruction (lightweight regex)

Weighted vote — split if score > 0.55 (tunable).
"""
from __future__ import annotations

import re
from typing import Any

import httpx
from loguru import logger

from src.services.chunkers.base import BaseChunker, ChunkUnit


# ── Vietnamese + English discourse markers ────────────────────────────────────
_DISCOURSE_MARKERS_VI = [
    r"\bTuy nhiên\b", r"\bNgoài ra\b", r"\bMặt khác\b", r"\bDo đó\b", r"\bVì vậy\b",
    r"\bNgược lại\b", r"\bĐồng thời\b", r"\bThứ nhất\b", r"\bThứ hai\b", r"\bThứ ba\b",
    r"\bTóm lại\b", r"\bCuối cùng\b", r"\bBên cạnh đó\b", r"\bHơn nữa\b",
    r"\bChẳng hạn\b", r"\bVí dụ\b",
]
_DISCOURSE_MARKERS_EN = [
    r"\bHowever\b", r"\bMoreover\b", r"\bFurthermore\b", r"\bIn contrast\b",
    r"\bOn the other hand\b", r"\bConversely\b", r"\bIn addition\b", r"\bTherefore\b",
    r"\bThus\b", r"\bFirstly?\b", r"\bSecondly?\b", r"\bThirdly?\b", r"\bFinally\b",
    r"\bIn conclusion\b", r"\bFor example\b", r"\bFor instance\b",
]
_DISCOURSE_RE = re.compile(
    "|".join(_DISCOURSE_MARKERS_VI + _DISCOURSE_MARKERS_EN),
    flags=re.IGNORECASE,
)


# ── Mode classification ──────────────────────────────────────────────────────
_FACT_PATTERNS = [r"\d+%", r"\d+\s*(tỷ|triệu|nghìn|million|billion)", r"\d{4}", r"là\s+\w+"]
_OPINION_PATTERNS = [r"\b(rất|cực kỳ|tuyệt|excellent|amazing|terrible)\b",
                      r"\b(theo tôi|theo chúng tôi|in my opinion)\b",
                      r"\b(có thể|chắc chắn|definitely|maybe|possibly)\b"]
_INSTRUCTION_PATTERNS = [r"^(Bước \d|Step \d|First[,\.]|Next[,\.])",
                          r"\b(hãy|please|let us|let's)\b"]


def _detect_mode(text: str) -> str:
    if any(re.search(p, text, re.IGNORECASE) for p in _INSTRUCTION_PATTERNS):
        return "INSTRUCTION"
    if any(re.search(p, text, re.IGNORECASE) for p in _OPINION_PATTERNS):
        return "OPINION"
    if any(re.search(p, text, re.IGNORECASE) for p in _FACT_PATTERNS):
        return "FACT"
    return "NARRATIVE"


def _is_structural_boundary(text: str) -> bool:
    """Heading, list start, code block, blank line."""
    s = text.strip()
    if not s:
        return True
    return bool(
        re.match(r"^#{1,6}\s", s)        # markdown heading
        or re.match(r"^[-*+]\s", s)      # list bullet
        or re.match(r"^\d+\.\s", s)      # numbered list
        or s.startswith("```")           # code block
    )


def _entity_jaccard_distance(ents_a: set[str], ents_b: set[str]) -> float:
    """1 - Jaccard. 0 = same entities, 1 = totally different."""
    if not ents_a and not ents_b:
        return 0.0
    union = ents_a | ents_b
    inter = ents_a & ents_b
    return 1.0 - (len(inter) / len(union)) if union else 0.0


class MultiSignalChunker(BaseChunker):
    """5-signal ensemble chunker. Falls back to SemanticChunker style when
    embeddings/entity-extractor unavailable.

    Signal weights (tunable):
      semantic_drop:     0.30
      discourse_marker:  0.20
      entity_shift:      0.25
      structural:        0.15
      mode_shift:        0.10
    """
    name = "multi_signal"

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        embed_url: str = "",
        embed_model: str = "bge-m3",
        entity_extractor: Any = None,
        boundary_threshold: float = 0.55,
        weights: dict[str, float] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.http = http_client
        self.embed_url = embed_url
        self.embed_model = embed_model
        self.entity_extractor = entity_extractor
        self.boundary_threshold = boundary_threshold
        self.weights = weights or {
            "semantic_drop": 0.30,
            "discourse_marker": 0.20,
            "entity_shift": 0.25,
            "structural": 0.15,
            "mode_shift": 0.10,
        }

    async def chunk(self, content: bytes | str, filename: str = "") -> list[ChunkUnit]:
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
        text = text.strip()
        if not text:
            return []

        sentences = self.split_sentences(text)
        if not sentences:
            return [ChunkUnit(text=text, chunk_index=0, chunk_level="paragraph")]

        # Compute signals
        boundaries = await self._compute_boundaries(sentences)

        # Group into paragraphs
        paragraphs: list[list[str]] = [[sentences[0]]]
        for i, sent in enumerate(sentences[1:], start=1):
            if boundaries[i] > self.boundary_threshold:
                paragraphs.append([sent])
            else:
                if sum(len(s) for s in paragraphs[-1]) > self.paragraph_max_chars:
                    paragraphs.append([sent])
                else:
                    paragraphs[-1].append(sent)

        # Pack into ChunkUnits
        units: list[ChunkUnit] = []
        for idx, para in enumerate(paragraphs):
            units.append(ChunkUnit(
                text=" ".join(para),
                chunk_index=idx,
                chunk_level="paragraph",
                parent_index=None,
                metadata={"filename": filename, "chunker": self.name},
            ))
        return units

    async def _compute_boundaries(self, sentences: list[str]) -> list[float]:
        """Return list of boundary scores, len(sentences). scores[0] = 0 (start)."""
        n = len(sentences)
        scores = [0.0] * n

        # S1: semantic drop (if embeddings available)
        if self.http and self.embed_url:
            try:
                from src.services.embedding import embed_batch, cosine_similarity
                embs = await embed_batch(
                    self.http, self.embed_url, self.embed_model, sentences, batch_size=32, timeout=60.0,
                )
                for i in range(1, n):
                    if embs[i-1] and embs[i]:
                        drop = 1.0 - cosine_similarity(embs[i-1], embs[i])
                        scores[i] += self.weights["semantic_drop"] * drop
            except Exception as e:
                logger.debug(f"MultiSignal embed failed, skip semantic_drop: {e}")

        # S2-S5: per-sentence pairwise
        prev_ents: set[str] = set()
        prev_mode = ""
        for i in range(n):
            sent = sentences[i]

            # S2 discourse marker (on current sentence)
            if _DISCOURSE_RE.search(sent[:50]):  # check first 50 chars
                scores[i] += self.weights["discourse_marker"] * 1.0

            # S3 entity shift
            if self.entity_extractor is not None and i > 0:
                try:
                    ents, _ = await self.entity_extractor.extract(sent)
                    curr = {e.name.lower() for e in ents}
                    dist = _entity_jaccard_distance(prev_ents, curr)
                    scores[i] += self.weights["entity_shift"] * dist
                    prev_ents = curr
                except Exception:
                    pass

            # S4 structural
            if _is_structural_boundary(sent):
                scores[i] += self.weights["structural"] * 1.0

            # S5 mode shift
            mode = _detect_mode(sent)
            if i > 0 and prev_mode and mode != prev_mode:
                scores[i] += self.weights["mode_shift"] * 1.0
            prev_mode = mode

        return scores
