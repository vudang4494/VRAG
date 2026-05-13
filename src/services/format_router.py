"""Format router — detect file format and dispatch to right chunker."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from src.services.chunkers.base import BaseChunker, ChunkUnit
from src.services.chunkers.chat_chunker import ChatChunker
from src.services.chunkers.docx_chunker import DocxChunker
from src.services.chunkers.multi_signal_chunker import MultiSignalChunker
from src.services.chunkers.pdf_chunker import PDFChunker
from src.services.chunkers.semantic_chunker import SemanticChunker
from src.services.chunkers.xlsx_chunker import XlsxChunker
from src.services.doc_type_classifier import classify_doc_type, get_strategy


def detect_format(filename: str, content: bytes | None = None) -> str:
    """Return canonical format name: pdf|docx|xlsx|csv|md|txt|html|chat|email|unknown."""
    ext = Path(filename).suffix.lower().lstrip(".")
    extension_map = {
        "pdf": "pdf",
        "docx": "docx",
        "doc": "docx",
        "xlsx": "xlsx",
        "xls": "xlsx",
        "xlsm": "xlsx",
        "csv": "csv",
        "tsv": "csv",
        "md": "md",
        "markdown": "md",
        "txt": "txt",
        "text": "txt",
        "html": "html",
        "htm": "html",
        "json": "chat",
        "jsonl": "chat",
        "eml": "email",
        "msg": "email",
    }
    if ext in extension_map:
        return extension_map[ext]

    # Magic bytes
    if content:
        if content.startswith(b"%PDF"):
            return "pdf"
        if content.startswith(b"PK\x03\x04"):
            # docx/xlsx are zip — disambiguate by filename usually, fallback xlsx if has xl/ folder
            if b"word/" in content[:2048]:
                return "docx"
            if b"xl/" in content[:2048]:
                return "xlsx"
        if content.startswith(b"<!DOCTYPE html") or content.startswith(b"<html"):
            return "html"
        if content.lstrip().startswith(b"["):
            return "chat"

    return "txt"


def get_chunker(
    fmt: str,
    http_client: httpx.AsyncClient | None = None,
    embed_url: str = "",
    embed_model: str = "bge-m3",
    **chunker_kwargs,
) -> BaseChunker:
    """Return chunker instance for the given format."""
    common = dict(chunker_kwargs)
    if fmt == "pdf":
        return PDFChunker(
            http_client=http_client,
            embed_url=embed_url,
            embed_model=embed_model,
            **common,
        )
    if fmt in ("docx", "doc"):
        return DocxChunker(**common)
    if fmt in ("xlsx", "xls", "csv", "tsv"):
        return XlsxChunker(**common)
    if fmt in ("chat", "email"):
        return ChatChunker(**common)
    # md, txt, html, unknown → semantic
    return SemanticChunker(
        http_client=http_client,
        embed_url=embed_url,
        embed_model=embed_model,
        **common,
    )


async def route_and_chunk(
    content: bytes,
    filename: str,
    http_client: httpx.AsyncClient | None = None,
    embed_url: str = "",
    embed_model: str = "bge-m3",
    entity_extractor: Any = None,
    **chunker_kwargs,
) -> tuple[str, list[ChunkUnit]]:
    """One-call API: detect format → classify doc type → chunk → return (format, chunks).

    For md/txt/html: uses MultiSignalChunker with doc-type-aware parameters.
    For other formats: uses format-specific chunker.
    """
    fmt = detect_format(filename, content)
    logger.info(f"Format detected for {filename}: {fmt}")

    # Phase 5b: classify doc type for md/txt/html to tune chunking strategy
    text_preview = content[:8000].decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)[:8000]
    doc_type = classify_doc_type(text_preview, filename)
    logger.info(f"Doc type for {filename}: {doc_type}")
    strat = get_strategy(doc_type)

    # md/txt/html → Phase 5a MultiSignalChunker with doc-type-aware params
    if fmt in ("md", "txt", "html"):
        chunker = MultiSignalChunker(
            http_client=http_client,
            embed_url=embed_url,
            embed_model=embed_model,
            entity_extractor=entity_extractor,
            boundary_threshold=strat["boundary_threshold"],
            section_max_chars=strat["chunk_size"] * 5,
            paragraph_max_chars=strat["chunk_size"],
            **chunker_kwargs,
        )
    else:
        chunker = get_chunker(fmt, http_client, embed_url, embed_model, **chunker_kwargs)

    chunks = await chunker.chunk(content, filename=filename)
    return fmt, chunks
