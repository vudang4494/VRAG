"""Email / Gmail source plugin."""
import asyncio
import hashlib
from datetime import datetime
from typing import Any

from plugins.base import (
    BaseSourcePlugin,
    ParsedDocument,
    PluginCapability,
    PluginConfig,
    SourceCredentials,
    SyncResult,
)


class EmailSourcePlugin(BaseSourcePlugin):
    """Ingest email threads (Gmail, generic IMAP)."""

    name = "email"
    version = "1.0.0"
    capabilities = [PluginCapability.INGEST_SCHEDULED]
    supported_types = ["email", "gmail"]

    def __init__(self, config: PluginConfig, credentials: SourceCredentials | None = None):
        super().__init__(config, credentials)

    async def fetch(self, thread_id: str, **kwargs: Any) -> ParsedDocument:
        messages = kwargs.get("messages", [])
        subject = kwargs.get("subject", f"Thread {thread_id}")
        thread_content = "\n\n".join(
            f"From: {m.get('from_','')} | Date: {m.get('date','')}\n{m.get('body','')}"
            for m in messages
        )
        return ParsedDocument(
            title=subject,
            content=thread_content,
            author=messages[0].get("from_") if messages else None,
            created_date=datetime.fromisoformat(messages[0]["date"]) if messages and messages[0].get("date") else None,
            metadata={
                "thread_id": thread_id,
                "message_count": len(messages),
                "doc_hash": hashlib.md5(thread_content.encode()).hexdigest()[:16],
                "ingested_via": "email_plugin",
            },
        )

    async def sync(self, **kwargs: Any) -> SyncResult:
        import time
        start = time.monotonic()
        threads = kwargs.get("threads", [])
        docs = []
        for thread in threads:
            doc = await self.fetch(
                thread_id=thread.get("id", "unknown"),
                messages=thread.get("messages", []),
                subject=thread.get("subject", ""),
            )
            docs.append(doc)
        return SyncResult(
            source_id=self.config.source_id,
            documents=docs,
            crawled_urls=len(docs),
            duration_seconds=time.monotonic() - start,
        )
