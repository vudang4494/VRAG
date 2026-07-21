"""PDF chunker — layout-aware via docling, fallback pypdf, with page metadata.

Output: ChunkUnit list with page_num + heading_path metadata when possible.
"""

from __future__ import annotations

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

        # F1+F3: clean each page (strip trailing page-number artifact), then JOIN
        # all pages into one document text before chunking. The previous per-page
        # loop broke any sentence that straddled a page boundary — observed at
        # 32% on section-level chunks. Track char-offset → page mapping so each
        # output chunk still gets its starting page_num.
        offset_to_page: list[tuple[int, int]] = []  # (start_char, page_num)
        parts: list[str] = []
        cursor = 0
        sep = "\n\n"
        for page_num, page_text in pages:
            cleaned = BaseChunker.strip_page_artifact(page_text or "").strip()
            if not cleaned:
                continue
            offset_to_page.append((cursor, page_num))
            parts.append(cleaned)
            cursor += len(cleaned) + len(sep)
        if not parts:
            return []
        full_text = sep.join(parts)

        sub_chunker = SemanticChunker(
            http_client=self.http_client,
            embed_url=self.embed_url,
            embed_model=self.embed_model,
            section_max_chars=self.section_max_chars,
            paragraph_max_chars=self.paragraph_max_chars,
            sentence_max_chars=self.sentence_max_chars,
            emit_levels=self.emit_levels,
        )
        units = await sub_chunker.chunk(full_text, filename=filename)

        def _page_for_offset(offset: int) -> int:
            page = offset_to_page[0][1] if offset_to_page else 1
            for start, p in offset_to_page:
                if start <= offset:
                    page = p
                else:
                    break
            return page

        cursor_in_doc = 0
        for u in units:
            sample = (u.text or "")[:80]
            found = full_text.find(sample, cursor_in_doc) if sample else -1
            if found < 0 and sample:
                found = full_text.find(sample)
            offset = found if found >= 0 else cursor_in_doc
            page_num = _page_for_offset(offset)
            u.metadata.setdefault("page_num", page_num)
            u.metadata.setdefault("filename", filename)
            u.metadata["format"] = "pdf"
            cursor_in_doc = max(cursor_in_doc, offset + max(len(u.text or ""), 1))
        return units

    @staticmethod
    def _has_text(pages: list[tuple[int, str]]) -> bool:
        """True if any page carries real characters, not just whitespace."""
        return any(text and text.strip() for _, text in pages)

    async def _parse(self, content: bytes, filename: str) -> list[tuple[int, str]]:
        """Return list of (page_num, text). Try docling → pypdf → fitz.

        A parser not raising is NOT the same as a parser extracting text. pypdf returns
        pages of empty strings for a scanned PDF rather than raising, so the chain used to
        stop at the first parser that did not throw and never reach fitz. Those documents
        ingested as chunks_total=0 and were recorded as SUCCESS — 35 of 526 files in the
        corpus500 run, mostly Vietnamese scans (~27.5% of VN docs are scans).

        Each parser now has to produce text to win. If none does, the caller gets [] and a
        WARNING that says the PDF has no text layer — the honest answer, and the signal
        that OCR is the missing capability rather than the chunker being broken.
        """

        async def _try(name: str) -> list[tuple[int, str]] | None:
            """Run one parser. None = it failed or found no text; caller moves on."""
            try:
                if name == "docling":
                    pages = await self._parse_docling(content, filename)
                elif name == "pypdf":
                    pages = self._parse_pypdf(content)
                else:
                    pages = self._parse_fitz(content)
            except Exception as e:
                logger.debug(f"{name} parse failed for {filename}, falling back: {e}")
                return None
            if self._has_text(pages):
                return pages
            logger.debug(f"{name} returned {len(pages)} page(s) with no text for {filename}")
            return None

        names = ["docling", "pypdf", "fitz"] if self.prefer_docling else ["pypdf", "fitz"]
        for name in names:
            pages = await _try(name)
            if pages is not None:
                return pages

        logger.warning(
            f"No text layer found in {filename} by any parser (docling/pypdf/fitz). "
            f"This is almost always a scanned PDF; VRAG has no OCR, so it will ingest "
            f"as 0 chunks."
        )
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
