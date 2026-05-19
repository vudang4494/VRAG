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


_CONTEXT_PROMPT = """Generate a 1-sentence context description for the following chunk
from a document. This context helps retrieve the correct chunk later.

The context should answer: "In which document and section does this appear, and what is it about?"

Format: Return ONLY the context sentence, no explanation.

Example: "In the GraphRAG paper, the Methods section on entity extraction, this paragraph discusses ..."

Document: {doc_name}
Section: {section}
Chunk: {chunk_text[:200]}

Context:"""


async def generate_chunk_context(
    chunk_text: str,
    doc_name: str = "document",
    section: str = "section",
    llm: Any = None,
    model: str = "qwen3.5:9b",
    max_context_tokens: int = 100,
) -> str:
    """
    Generate contextual context for a chunk (Anthropic Contextual Retrieval pattern).

    Args:
        chunk_text: The chunk text to generate context for
        doc_name: Document filename/title
        section: Section or heading name
        llm: LLM client (if None, returns template-based context)
        model: LLM model name
        max_context_tokens: Approximate max tokens for context (default 100 ≈ 50-80 words)

    Returns:
        Context string to prepend to chunk text before embedding
    """
    if llm is None:
        # Fallback: simple template-based context (no LLM call)
        return f"[Document: {doc_name} | Section: {section}] {chunk_text[:200]}"

    from src.services.ollama_helper import ollama_chat

    try:
        prompt = _CONTEXT_PROMPT.format(
            doc_name=doc_name,
            section=section,
            chunk_text=chunk_text,
        )
        context = await ollama_chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.1,
            max_tokens=max_context_tokens,
        )
        if context and context.strip():
            return context.strip()
    except Exception:
        pass

    # Fallback on error
    return f"[Document: {doc_name} | Section: {section}]"
