"""Base classes for all data source plugins."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, ClassVar

import pydantic


class PluginCapability(str, Enum):
    INGEST_FILE = "ingest:file"
    INGEST_URL = "ingest:url"
    INGEST_STREAM = "ingest:stream"
    INGEST_CRAWL = "ingest:crawl"
    INGEST_WEBHOOK = "ingest:webhook"
    INGEST_SCHEDULED = "ingest:scheduled"
    QUERY = "query"
    DELETE = "delete"
    SYNC_STATUS = "sync:status"


@dataclass
class SourceCredentials:
    """Encrypted credentials passed to a plugin."""
    encrypted_blob: str


@dataclass
class PluginConfig:
    """Configuration for a plugin instance (from Source.custom_config)."""
    raw: dict[str, Any]
    tenant_id: str = ""
    source_id: str = ""
    source_name: str = ""
    access_level: str = "internal"

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.raw:
            raise ValueError(f"Plugin config requires '{key}'")
        return self.raw[key]


@dataclass
class ParsedDocument:
    """Normalized output from any source plugin."""
    title: str
    content: str
    raw_content: bytes | None = None
    url: str | None = None
    author: str | None = None
    created_date: datetime | None = None
    file_type: str | None = None
    file_size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text_preview(self) -> str:
        return self.content[:200].replace("\n", " ")


@dataclass
class SyncResult:
    """Result of a sync/crawl operation."""
    source_id: str
    documents: list[ParsedDocument]
    errors: list[str] = field(default_factory=list)
    crawled_urls: int = 0
    skipped_urls: int = 0
    duration_seconds: float = 0.0


@dataclass
class IngestResult:
    """Result of document ingestion through a plugin."""
    document_id: str
    chunk_count: int
    entity_count: int
    relationship_count: int
    failed_chunks: int = 0
    error_message: str | None = None
    processing_time_ms: float = 0.0


class BaseSourcePlugin(ABC):
    """
    Abstract base class for all source plugins.

    Plugins are responsible for:
    1. Fetching/raw content from their source
    2. Converting it into ParsedDocument format
    3. Optionally: handling auth, pagination, rate limiting
    """

    name: ClassVar[str] = "base"
    version: ClassVar[str] = "1.0.0"
    capabilities: ClassVar[list[PluginCapability]] = []
    supported_types: ClassVar[list[str]] = []

    def __init__(self, config: PluginConfig, credentials: SourceCredentials | None = None):
        self.config = config
        self.credentials = credentials
        self._client: Any = None

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """Called once when the plugin is first used. Use for setup."""
        pass

    async def close(self) -> None:
        """Called when the plugin is destroyed. Use for cleanup."""
        pass

    # -------------------------------------------------------------------------
    # Core ingestion methods
    # -------------------------------------------------------------------------

    @abstractmethod
    async def fetch(self, url_or_path: str, **kwargs: Any) -> ParsedDocument:
        """Fetch a single document from a URL or file path."""
        ...

    async def ingest(self, url_or_path: str, **kwargs: Any) -> IngestResult:
        """
        Full ingest pipeline: fetch -> parse -> chunk -> embed -> store.
        Override this for custom processing logic.
        """
        doc = await self.fetch(url_or_path, **kwargs)
        from src.storage.document_store import DocumentStore
        store = DocumentStore()
        result = await store.ingest_document(
            tenant_id=self.config.tenant_id,
            source_id=self.config.source_id,
            doc=doc,
            chunk_size=self.config.get("chunk_size", 512),
            chunk_overlap=self.config.get("chunk_overlap", 64),
        )
        return result

    async def stream(self, query: str, **kwargs: Any) -> AsyncIterator[ParsedDocument]:
        """Stream documents (for large crawls). Yield one ParsedDocument at a time."""
        doc = await self.fetch(query, **kwargs)
        yield doc

    async def sync(self, **kwargs: Any) -> SyncResult:
        """
        Sync all documents from the source (used for scheduled/crawl sources).
        Default implementation calls fetch() once.
        """
        import time
        start = time.monotonic()
        docs: list[ParsedDocument] = []
        errors: list[str] = []
        try:
            doc = await self.fetch("sync://default", **kwargs)
            docs.append(doc)
        except Exception as e:
            errors.append(str(e))
        return SyncResult(
            source_id=self.config.source_id,
            documents=docs,
            errors=errors,
            crawled_urls=len(docs),
            duration_seconds=time.monotonic() - start,
        )

    async def health_check(self) -> dict[str, Any]:
        """Check if the source is reachable/configured correctly."""
        return {"status": "ok", "plugin": self.name, "version": self.version}

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        """Clean and normalize text content."""
        import re
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _estimate_reading_time(self, text: str) -> int:
        """Estimate reading time in minutes (avg 200 wpm)."""
        words = len(text.split())
        return max(1, words // 200)

    def _detect_language(self, text: str) -> str:
        """Simple language detection using character patterns."""
        vietnamese_chars = sum(1 for c in text if '\u00C0' <= c <= '\u024F')
        total_chars = len(text.replace(" ", ""))
        if total_chars == 0:
            return "unknown"
        vi_ratio = vietnamese_chars / total_chars
        if vi_ratio > 0.15:
            return "vi"
        elif vi_ratio > 0.05:
            return "mixed"
        return "en"
