"""DOCX/Word source plugin — extracts text from .docx files."""
import hashlib
import io
from pathlib import Path
from typing import Any

from plugins.base import ParsedDocument, PluginCapability, PluginConfig, SourceCredentials
from plugins.base import BaseSourcePlugin


class DOCXSourcePlugin(BaseSourcePlugin):
    """Plugin for extracting text from DOCX and DOC files."""

    name = "docx"
    version = "1.0.0"
    capabilities = [PluginCapability.INGEST_FILE]
    supported_types = ["docx", "doc"]

    async def fetch(self, url_or_path: str, **kwargs: Any) -> ParsedDocument:
        """Extract text and metadata from a DOCX file."""
        import asyncio
        import httpx

        docx_bytes: bytes
        file_name = Path(url_or_path).name

        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.get(url_or_path)
                r.raise_for_status()
                docx_bytes = r.content
        else:
            path = Path(url_or_path)
            if not path.exists():
                raise FileNotFoundError(f"DOCX not found: {url_or_path}")
            docx_bytes = path.read_bytes()

        def _extract() -> tuple[str, dict]:
            from docx import Document
            import docx2txt

            doc = Document(io.BytesIO(docx_bytes))

            # Extract text
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)

            # Extract from tables
            for table in doc.tables:
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        full_text.append(" | ".join(cells))

            # Extract styles/headings
            headings = []
            for para in doc.paragraphs:
                if para.style.name and para.style.name.startswith("Heading"):
                    level = para.style.name.replace("Heading ", "")
                    headings.append(f"[H{level}] {para.text}")

            content = "\n\n".join(full_text)
            if not content.strip():
                content = docx2txt.process(io.BytesIO(docx_bytes))

            # Metadata
            core_props = doc.core_properties
            meta = {
                "title": core_props.title or file_name.replace(".docx", ""),
                "author": core_props.author,
                "subject": core_props.subject,
                "keywords": core_props.keywords,
                "created": str(core_props.created) if core_props.created else None,
                "modified": str(core_props.modified) if core_props.modified else None,
            }

            return content, meta, headings

        result = await asyncio.get_event_loop().run_in_executor(None, _extract)
        content, meta, headings = result

        return ParsedDocument(
            title=meta.get("title") or file_name.replace(".docx", "").replace(".doc", ""),
            content=self._normalize_text(content),
            raw_content=docx_bytes,
            file_type=file_name.split(".")[-1].lower(),
            file_size_bytes=len(docx_bytes),
            metadata={
                **meta,
                "headings": headings,
                "doc_hash": hashlib.md5(docx_bytes).hexdigest()[:16],
                "parsing_backend": "python-docx",
                "reading_time_min": self._estimate_reading_time(content),
                "language": self._detect_language(content),
            },
        )
