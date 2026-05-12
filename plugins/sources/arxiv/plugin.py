"""ArXiv source plugin."""
import asyncio
import hashlib
import re
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


class ArxivSourcePlugin(BaseSourcePlugin):
    """Fetch and parse ArXiv papers."""

    name = "arxiv"
    version = "1.0.0"
    capabilities = [
        PluginCapability.INGEST_URL,
        PluginCapability.INGEST_SCHEDULED,
    ]
    supported_types = ["arxiv", "pdf"]

    def __init__(self, config: PluginConfig, credentials: SourceCredentials | None = None):
        super().__init__(config, credentials)
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    def _extract_arxiv_id(self, identifier: str) -> str:
        match = re.search(r"(\d+\.\d+)", identifier)
        return match.group(1) if match else identifier.strip()

    async def fetch(self, arxiv_ref: str, **kwargs: Any) -> ParsedDocument:
        client = await self._get_client()
        aid = self._extract_arxiv_id(arxiv_ref)

        r = await client.get(f"http://export.arxiv.org/api/query?id_list={aid}")
        r.raise_for_status()
        xml_text = r.text

        title = re.search(r"<title>(.*?)</title>", xml_text, re.DOTALL)
        summary = re.search(r"<summary>(.*?)</summary>", xml_text, re.DOTALL)
        authors_match = re.findall(r"<author>.*?<name>(.*?)</name>.*?</author>", xml_text, re.DOTALL)
        published = re.search(r"<published>(.*?)</published>", xml_text)

        text = ""
        if title:
            text += f"# {title.group(1).strip()}\n\n"
        if authors_match:
            text += f"Authors: {', '.join(authors_match)}\n\n"
        if summary:
            text += summary.group(1).strip() + "\n"

        return ParsedDocument(
            title=title.group(1).strip() if title else f"arXiv:{aid}",
            content=text,
            author=", ".join(authors_match) if authors_match else None,
            created_date=datetime.fromisoformat(published.group(1).replace("Z", "+00:00")) if published else None,
            file_type="xml",
            metadata={
                "arxiv_id": aid,
                "doc_hash": hashlib.md5(text.encode()).hexdigest()[:16],
                "ingested_via": "arxiv_plugin",
            },
        )

    async def sync(self, **kwargs: Any) -> SyncResult:
        import time
        start = time.monotonic()
        arxiv_ids = self.config.get("arxiv_ids", [])
        docs = []
        errors = []

        for aid in arxiv_ids:
            try:
                doc = await self.fetch(aid)
                docs.append(doc)
            except Exception as e:
                errors.append(f"arXiv {aid}: {e}")

        return SyncResult(
            source_id=self.config.source_id,
            documents=docs,
            crawled_urls=len(docs),
            errors=errors,
            duration_seconds=time.monotonic() - start,
        )
