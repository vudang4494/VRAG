"""PDF chunker — layout-aware via docling, fallback pypdf, with page metadata.

Output: ChunkUnit list with page_num + heading_path metadata when possible.
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from src.services.chunkers.base import BaseChunker, ChunkUnit
from src.services.chunkers.semantic_chunker import SemanticChunker


class PDFChunker(BaseChunker):
    name = "pdf"

    def __init__(
        self,
        http_client=None,
        embed_url: str = "",
        embed_model: str = "bge-m3",
        prefer_docling: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.http_client = http_client
        self.embed_url = embed_url
        self.embed_model = embed_model
        self.prefer_docling = prefer_docling

    async def chunk(self, content: bytes | str, filename: str = "") -> list[ChunkUnit]:
        if isinstance(content, str):
            content = content.encode("utf-8")

        pages = await self._parse(content, filename)
        if not pages:
            return []

        all_units: list[ChunkUnit] = []
        next_index = 0

        for page_num, page_text in pages:
            if not page_text.strip():
                continue
            sub_chunker = SemanticChunker(
                http_client=self.http_client,
                embed_url=self.embed_url,
                embed_model=self.embed_model,
                section_max_chars=self.section_max_chars,
                paragraph_max_chars=self.paragraph_max_chars,
                sentence_max_chars=self.sentence_max_chars,
                emit_levels=self.emit_levels,
            )
            page_units = await sub_chunker.chunk(page_text, filename=filename)
            for u in page_units:
                u.chunk_index = next_index
                u.metadata.setdefault("page_num", page_num)
                u.metadata.setdefault("filename", filename)
                u.metadata["format"] = "pdf"
                if u.parent_index is not None:
                    u.parent_index += next_index - page_units[0].chunk_index
                next_index += 1
                all_units.append(u)
        return all_units

    async def _parse(self, content: bytes, filename: str) -> list[tuple[int, str]]:
        """Return list of (page_num, text). Try docling → pypdf → fitz."""
        if self.prefer_docling:
            try:
                return await self._parse_docling(content, filename)
            except Exception as e:
                logger.debug(f"docling parse failed, falling back: {e}")

        try:
            return self._parse_pypdf(content)
        except Exception as e:
            logger.debug(f"pypdf parse failed: {e}")

        try:
            return self._parse_fitz(content)
        except Exception as e:
            logger.warning(f"All PDF parsers failed for {filename}: {e}")
            return []

    async def _parse_docling(self, content: bytes, filename: str) -> list[tuple[int, str]]:
        import asyncio
        import tempfile
        from pathlib import Path

        def _run():
            from docling.document_converter import DocumentConverter
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(content)
                tmp_path = Path(f.name)
            try:
                converter = DocumentConverter()
                result = converter.convert(tmp_path)
                md = result.document.export_to_markdown()
                return [(1, md)]
            finally:
                tmp_path.unlink(missing_ok=True)

        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=60.0)

    def _parse_pypdf(self, content: bytes) -> list[tuple[int, str]]:
        from io import BytesIO
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]

    def _parse_fitz(self, content: bytes) -> list[tuple[int, str]]:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=content, filetype="pdf")
        return [(i + 1, doc.load_page(i).get_text("text")) for i in range(len(doc))]
