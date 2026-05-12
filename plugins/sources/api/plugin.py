"""API / Webhook source plugin."""
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


class APIKeySourcePlugin(BaseSourcePlugin):
    """Poll REST APIs and convert responses to documents."""

    name = "api"
    version = "1.0.0"
    capabilities = [
        PluginCapability.INGEST_URL,
        PluginCapability.INGEST_SCHEDULED,
        PluginCapability.INGEST_WEBHOOK,
    ]
    supported_types = ["api", "webhook", "rest"]

    def __init__(self, config: PluginConfig, credentials: SourceCredentials | None = None):
        super().__init__(config, credentials)
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            import httpx
            headers = {"Accept": "application/json"}
            if self.credentials:
                import json
                try:
                    creds = json.loads(self.credentials.encrypted_blob)
                    headers.update(creds.get("headers", {}))
                except Exception:
                    pass
            limits = httpx.Limits(max_connections=10)
            self._client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(30.0), headers=headers)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    async def fetch(self, url: str, **kwargs: Any) -> ParsedDocument:
        client = await self._get_client()
        method = kwargs.get("method", "GET").upper()
        params = kwargs.get("params", {})
        body = kwargs.get("body")
        extractor_path = kwargs.get("extractor_path", "")

        req_kwargs: dict[str, Any] = {"params": params}
        if body and method != "GET":
            req_kwargs["json" if isinstance(body, dict) else "content"] = body

        r = await client.request(method, url, **req_kwargs)
        r.raise_for_status()
        data = r.json()

        if extractor_path:
            for key in extractor_path.split("."):
                data = data[key]

        import json
        content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        return ParsedDocument(
            title=kwargs.get("title", url),
            content=content,
            url=url,
            created_date=datetime.now(),
            file_type="json",
            file_size_bytes=len(content.encode()),
            metadata={
                "api_endpoint": url,
                "method": method,
                "doc_hash": hashlib.md5(content.encode()).hexdigest()[:16],
                "ingested_via": "api_plugin",
            },
        )

    async def sync(self, **kwargs: Any) -> SyncResult:
        import time
        start = time.monotonic()
        endpoints = self.config.get("endpoints", [])
        docs = []
        errors = []

        for ep in endpoints:
            try:
                doc = await self.fetch(
                    url=ep["url"],
                    method=ep.get("method", "GET"),
                    params=ep.get("params", {}),
                    body=ep.get("body"),
                    extractor_path=ep.get("extractor_path", ""),
                    title=ep.get("title", ep["url"]),
                )
                docs.append(doc)
            except Exception as e:
                errors.append(f"{ep.get('url')}: {e}")

        return SyncResult(
            source_id=self.config.source_id,
            documents=docs,
            crawled_urls=len(docs),
            errors=errors,
            duration_seconds=time.monotonic() - start,
        )
