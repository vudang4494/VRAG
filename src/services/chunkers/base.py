"""Base chunker contract — all format chunkers inherit from this.

## Layer 1.1 — Chunk Context Generation (Anthropic Contextual Retrieval)

After hierarchical chunking, each chunk is standalone (no document context).
Anthropic Contextual Retrieval (2024) shows prepending 50-100 token context to each chunk
before embedding reduces retrieval failure by 35-49%.

Strategy: Generate a "context sentence" per chunk that answers:
  "In what document/section is this chunk found, and what is its purpose?"

This context is prepended to the chunk text before embedding, improving retrieval precision
without changing the chunk itself.

Pattern is orthogonal with Late Chunking (Jina 2024) — they can be combined.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkUnit:
    """A single chunk produced by a chunker. Format-agnostic shape."""

    text: str
    chunk_index: int
    chunk_level: str  # "sentence" | "paragraph" | "section" | "document"
    parent_index: int | None = None
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "chunk_level": self.chunk_level,
            "parent_index": self.parent_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }


class BaseChunker(ABC):
    """
    Subclass and implement `chunk()`.

    Returns a flat list of ChunkUnit ordered by chunk_index. Hierarchy is
    expressed via `parent_index` referring to another unit in the same list.
    """

    name: str = "base"

    def __init__(
        self,
        section_max_chars: int = 4000,
        paragraph_max_chars: int = 800,
        sentence_max_chars: int = 200,
        emit_levels: tuple[str, ...] = ("paragraph", "section"),
    ):
        self.section_max_chars = section_max_chars
        self.paragraph_max_chars = paragraph_max_chars
        self.sentence_max_chars = sentence_max_chars
        self.emit_levels = emit_levels

    @abstractmethod
    async def chunk(self, content: bytes | str, filename: str = "") -> list[ChunkUnit]:
        """Produce chunks from raw content."""

    # ── helpers shared by subclasses ────────────────────────────────────────

    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ỹ])")

    # Abbreviations that should NOT end a sentence even when followed by capital.
    # Trailing dot is part of the abbreviation, not a sentence terminator.
    _ABBREVIATIONS = (
        "et al",
        "i.e",
        "e.g",
        "etc",
        "vs",
        "cf",
        "viz",
        "Fig",
        "Eq",
        "Sec",
        "Ch",
        "Tab",
        "Eqs",
        "Figs",
        "Mr",
        "Mrs",
        "Ms",
        "Dr",
        "Prof",
        "Inc",
        "Ltd",
        "Co",
        "Corp",
        "Jr",
        "Sr",
        "St",
        "No",
        "vol",
        "Vol",
        "pp",
        "p",
        "ed",
        "Eds",
        # Vietnamese honorifics + abbreviations
        "TS",
        "PGS",
        "GS",
        "ThS",
        "KS",
        "BS",
        "CN",
    )
    _PLACEHOLDER = "\x00"

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """Vietnamese + English sentence splitter with citation/URL/decimal guards.

        Avoids three common over-splits seen on the academic PDF corpus:
        - "Nguyen et al. 2024. Mintaka..."  → don't split after "al."
        - "12.35", "abs/2408.04259"          → don't split inside numbers
        - "https://example.com/path"         → don't split inside URLs
        """
        text = text.strip()
        if not text:
            return []

        # Mask dots that should NOT signal sentence end.
        masked = text

        # 1) Abbreviations: replace "abbr." with abbr + PLACEHOLDER
        for abbr in BaseChunker._ABBREVIATIONS:
            masked = re.sub(
                r"\b" + re.escape(abbr) + r"\.",
                abbr + BaseChunker._PLACEHOLDER,
                masked,
            )

        # 2) Decimal / version numbers: digit-dot-digit
        masked = re.sub(
            r"(\d)\.(\d)", lambda m: m.group(1) + BaseChunker._PLACEHOLDER + m.group(2), masked
        )

        # 3) URLs: replace dots inside the URL token
        def _mask_url(m: re.Match[str]) -> str:
            return m.group(0).replace(".", BaseChunker._PLACEHOLDER)

        masked = re.sub(r"https?://\S+", _mask_url, masked)
        masked = re.sub(r"www\.\S+", _mask_url, masked)

        # 4) Single uppercase initial like "F. Scott" — protect "F."
        masked = re.sub(
            r"\b([A-Z])\.(?=\s+[A-Z])",
            lambda m: m.group(1) + BaseChunker._PLACEHOLDER,
            masked,
        )

        sentences = BaseChunker._SENTENCE_RE.split(masked)
        return [s.replace(BaseChunker._PLACEHOLDER, ".").strip() for s in sentences if s.strip()]

    @staticmethod
    def strip_page_artifact(text: str) -> str:
        """Strip trailing PDF page-number / running-footer artifacts.

        Patterns seen on the academic corpus:
        - trailing "\n12", "\n26" (lone page number, 1-4 digits)
        - "\nPreprint."           (running header)
        - "\nPage 7 of 14"
        """
        if not text:
            return text
        t = text.rstrip()
        for _ in range(3):  # repeat a few times for stacked artifacts
            new = re.sub(r"\n\s*\d{1,4}\s*$", "", t)
            new = re.sub(r"\n\s*Page\s+\d+\s+of\s+\d+\s*$", "", new, flags=re.IGNORECASE)
            new = re.sub(r"\n\s*Preprint\.?\s*$", "", new, flags=re.IGNORECASE)
            if new == t:
                break
            t = new
        return t

    @staticmethod
    def pack_units(
        units: list[str],
        max_chars: int,
        joiner: str = " ",
    ) -> list[str]:
        """Greedy pack units (sentences/lines) into chunks ≤ max_chars."""
        packed: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for u in units:
            u = u.strip()
            if not u:
                continue
            if buf_len + len(u) + len(joiner) > max_chars and buf:
                packed.append(joiner.join(buf))
                buf = [u]
                buf_len = len(u)
            else:
                buf.append(u)
                buf_len += len(u) + len(joiner)
        if buf:
            packed.append(joiner.join(buf))
        return packed
