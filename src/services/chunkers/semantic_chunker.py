"""Semantic chunker — topic-shift detection cho MD/TXT/HTML.

Strategy:
1. Split thành sentences.
2. (Optional) Embed sentence-by-sentence, detect topic shift bằng cosine drop.
3. Pack sentences thành paragraphs (cùng topic).
4. Pack paragraphs thành sections nếu max_chars cho phép.
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from src.services.chunkers.base import BaseChunker, ChunkUnit


class SemanticChunker(BaseChunker):
    name = "semantic"

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        embed_url: str = "",
        embed_model: str = "bge-m3",
        topic_shift_threshold: float = 0.55,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.http = http_client
        self.embed_url = embed_url
        self.embed_model = embed_model
        self.topic_shift_threshold = topic_shift_threshold

    async def chunk(self, content: bytes | str, filename: str = "") -> list[ChunkUnit]:
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
        text = text.strip()
        if not text:
            return []

        sentences = self.split_sentences(text)
        if not sentences:
            return [ChunkUnit(text=text, chunk_index=0, chunk_level="paragraph")]

        # Step 1: group sentences thành paragraphs theo topic shift hoặc max_chars
        paragraphs = await self._group_by_topic(sentences)

        # Step 2: group paragraphs thành sections
        section_groups = self._group_into_sections(paragraphs)

        # Step 3: emit ChunkUnit theo emit_levels
        units: list[ChunkUnit] = []
        idx = 0
        for section_idx, section in enumerate(section_groups):
            section_text = "\n\n".join(section)
            section_parent_idx = idx if "section" in self.emit_levels else None
            if "section" in self.emit_levels:
                units.append(
                    ChunkUnit(
                        text=section_text,
                        chunk_index=idx,
                        chunk_level="section",
                        parent_index=None,
                        metadata={"section_index": section_idx},
                    )
                )
                idx += 1
            for para_idx, para in enumerate(section):
                if "paragraph" in self.emit_levels:
                    units.append(
                        ChunkUnit(
                            text=para,
                            chunk_index=idx,
                            chunk_level="paragraph",
                            parent_index=section_parent_idx,
                            metadata={"section_index": section_idx, "paragraph_index": para_idx},
                        )
                    )
                    idx += 1
        return units

    async def _group_by_topic(self, sentences: list[str]) -> list[str]:
        """Group sentences thành paragraphs. Topic shift detection optional."""
        if not self.http or not self.embed_url:
            return self.pack_units(sentences, self.paragraph_max_chars)

        try:
            from src.services.embedding import embed_batch, cosine_similarity

            embeds = await embed_batch(
                self.http,
                self.embed_url,
                self.embed_model,
                sentences,
                batch_size=32,
                timeout=60.0,
            )
        except Exception as e:
            logger.warning(f"Sentence embedding failed, fallback to greedy pack: {e}")
            return self.pack_units(sentences, self.paragraph_max_chars)

        paragraphs: list[list[str]] = [[sentences[0]]]
        buf_len = len(sentences[0])
        for i in range(1, len(sentences)):
            sim = cosine_similarity(embeds[i - 1], embeds[i])
            sent = sentences[i]
            split_now = (
                sim < self.topic_shift_threshold or buf_len + len(sent) > self.paragraph_max_chars
            )
            if split_now:
                paragraphs.append([sent])
                buf_len = len(sent)
            else:
                paragraphs[-1].append(sent)
                buf_len += len(sent) + 1

        return [" ".join(p) for p in paragraphs]

    def _group_into_sections(self, paragraphs: list[str]) -> list[list[str]]:
        sections: list[list[str]] = [[]]
        cur_len = 0
        for para in paragraphs:
            if cur_len + len(para) > self.section_max_chars and sections[-1]:
                sections.append([para])
                cur_len = len(para)
            else:
                sections[-1].append(para)
                cur_len += len(para)
        return [s for s in sections if s]
