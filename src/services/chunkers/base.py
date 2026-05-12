"""Base chunker contract — all format chunkers inherit from this."""
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

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """Vietnamese + English sentence splitter."""
        text = text.strip()
        if not text:
            return []
        sentences = BaseChunker._SENTENCE_RE.split(text)
        return [s.strip() for s in sentences if s.strip()]

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
