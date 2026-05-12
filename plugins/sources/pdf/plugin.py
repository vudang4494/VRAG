"""PDF source plugin — extracts text from PDF files using docling."""
import hashlib
import io
import re
import time
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdf_backend import PyPdfDocumentBackend
from docling.chunking import HybridChunker
from docling.datamodel.settings import Settings

from plugins.base import (
    BaseSourcePlugin,
    ParsedDocument,
    PluginCapability,
    PluginConfig,
    SourceCredentials,
    IngestResult,
)


class PDFSourcePlugin(BaseSourcePlugin):
    """
    Plugin for extracting text from PDF files.

    Uses Docling for high-quality PDF parsing with layout awareness.
    Falls back to pypdf for simple text extraction.
    """

    name = "pdf"
    version = "1.0.0"
    capabilities = [
        PluginCapability.INGEST_FILE,
        PluginCapability.INGEST_URL,
    ]
    supported_types = ["pdf"]

    async def initialize(self) -> None:
        self._converter = None

    @property
    def converter(self) -> DocumentConverter:
        if self._converter is None:
            pdf_format = PdfFormatOption(
                backend=PyPdfDocumentBackend,
                eager_mode=False,
            )
            settings = Settings(
                chunking={
                    "tokenizer": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                }
            )
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: pdf_format},
                settings=settings,
            )
        return self._converter

    async def fetch(self, url_or_path: str, **kwargs: Any) -> ParsedDocument:
        """
        Extract text from a PDF file.

        Args:
            url_or_path: Local file path or HTTP URL to a PDF.
            **kwargs: Extra options (password for encrypted PDFs).

        Returns:
            ParsedDocument with extracted text and metadata.
        """
        import httpx
        from src.parsers.pdf_parser import parse_pdf_fallback

        pdf_bytes: bytes
        file_name = Path(url_or_path).name if "/" in url_or_path else url_or_path

        # Fetch if URL
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.get(url_or_path)
                r.raise_for_status()
                pdf_bytes = r.content
                if not file_name or file_name == url_or_path:
                    content_disp = r.headers.get("content-disposition", "")
                    match = re.search(r'filename="?([^";]+)"?', content_disp)
                    if match:
                        file_name = match.group(1)
        else:
            path = Path(url_or_path)
            if path.exists():
                pdf_bytes = path.read_bytes()
            else:
                raise FileNotFoundError(f"PDF not found: {url_or_path}")

        # Try Docling first
        try:
            result = await self._extract_docling(pdf_bytes, file_name)
            if result:
                return result
        except Exception as e:
            import traceback
            traceback.print_exc()

        # Fallback: pypdf
        return await self._extract_pypdf(pdf_bytes, file_name)

    async def _extract_docling(self, pdf_bytes: bytes, file_name: str) -> ParsedDocument | None:
        """Extract using Docling with hybrid chunking."""
        import asyncio

        async def _convert():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self.converter.convert(io.BytesIO(pdf_bytes))
            )

        conv_res = await _convert()
        if not conv_res or not conv_res.documents:
            return None

        doc = conv_res.documents[0]

        # Export as markdown for clean text
        md_text = doc.export_to_markdown()

        # Also get metadata
        meta = {}
        if hasattr(doc, "metadata") and doc.metadata:
            meta = {
                "title": getattr(doc.metadata, "title", None),
                "author": getattr(doc.metadata, "authors", None),
                "subject": getattr(doc.metadata, "subject", None),
                "creation_date": getattr(doc.metadata, "creation_date", None),
            }

        # Chunk using Docling's HybridChunker
        chunker = HybridChunker(
            tokenizer=self.config.get("tokenizer", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            max_tokens=self.config.get("chunk_size", 512),
            merge_same_paragraph=True,
        )

        # Build full text for content field
        content = md_text
        if hasattr(doc, "text") and doc.text:
            content = doc.text

        doc_hash = hashlib.md5(pdf_bytes).hexdigest()[:16]

        return ParsedDocument(
            title=meta.get("title") or file_name.replace(".pdf", ""),
            content=self._normalize_text(content),
            raw_content=pdf_bytes,
            file_type="pdf",
            file_size_bytes=len(pdf_bytes),
            metadata={
                **meta,
                "doc_hash": doc_hash,
                "parsing_backend": "docling",
                "page_count": getattr(doc, "page_count", 0),
                "reading_time_min": self._estimate_reading_time(content),
                "language": self._detect_language(content),
            },
        )

    async def _extract_pypdf(self, pdf_bytes: bytes, file_name: str) -> ParsedDocument:
        """Fallback: simple pypdf text extraction."""
        import asyncio

        def _extract() -> tuple[str, dict]:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = []
            meta = {}
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append(f"--- Page {i + 1} ---\n{text}")

            if reader.metadata:
                meta = {
                    "title": reader.metadata.get("/Title"),
                    "author": reader.metadata.get("/Author"),
                    "subject": reader.metadata.get("/Subject"),
                }

            return "\n\n".join(pages), meta

        content, meta = await asyncio.get_event_loop().run_in_executor(None, _extract)

        return ParsedDocument(
            title=meta.get("title") or file_name.replace(".pdf", ""),
            content=self._normalize_text(content),
            raw_content=pdf_bytes,
            file_type="pdf",
            file_size_bytes=len(pdf_bytes),
            metadata={
                **meta,
                "parsing_backend": "pypdf",
                "page_count": len(content.split("--- Page")) - 1,
                "reading_time_min": self._estimate_reading_time(content),
                "language": self._detect_language(content),
            },
        )
