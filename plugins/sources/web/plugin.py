"""Web source plugin — crawl URLs, extract text from webpages."""
import asyncio
import hashlib
import re
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlparse

from loguru import logger

from plugins.base import (
    BaseSourcePlugin,
    ParsedDocument,
    PluginCapability,
    PluginConfig,
    SourceCredentials,
    SyncResult,
)


class WebSourcePlugin(BaseSourcePlugin):
    """Crawl and extract text from webpages."""

    name = "webpage"
    version = "1.0.0"
    capabilities = [
        PluginCapability.INGEST_URL,
        PluginCapability.INGEST_CRAWL,
        PluginCapability.INGEST_STREAM,
    ]
    supported_types = ["webpage", "html"]

    def __init__(self, config: PluginConfig, credentials: SourceCredentials | None = None):
        super().__init__(config, credentials)
        self._visited: set[str] = set()
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            import httpx
            limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
            self._client = httpx.AsyncClient(
                limits=limits,
                timeout=httpx.Timeout(30.0, connect=10.0),
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; RAGBot/2.0; +http://example.com/bot)",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _extract_main_content(self, html: str, url: str) -> str:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
                tag.decompose()

            article = soup.find("article") or soup.find("main") or soup.find("div", class_=re.compile(r"content|article|post|entry"))
            if article:
                soup = article

            title = soup.find("title")
            title_text = title.get_text().strip() if title else ""
            paragraphs = soup.find_all("p")
            text_parts = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
            return self._normalize_text(f"{title_text}\n\n{' '.join(text_parts)}")
        except ImportError:
            pass

        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return self._normalize_text(text.strip())

    def _extract_metadata(self, html: str, url: str) -> dict[str, Any]:
        meta: dict[str, Any] = {"url": url}
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            meta["title"] = soup.find("title").get_text().strip() if soup.find("title") else ""
            for attr in [("meta", "name", "description"), ("meta", "property", "og:description")]:
                tag = soup.find(*attr)
                if tag and tag.get("content"):
                    meta["description"] = tag["content"]
                    break
            author_tag = soup.find("meta", attrs={"name": "author"}) or soup.find("meta", attrs={"property": "article:author"})
            if author_tag:
                meta["author"] = author_tag.get("content")
            date_tag = soup.find("meta", attrs={"property": "article:published_time"}) or soup.find("meta", attrs={"name": "date"})
            if date_tag and date_tag.get("content"):
                try:
                    meta["created_date"] = datetime.fromisoformat(date_tag["content"].replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass
        except ImportError:
            pass
        return meta

    def _is_valid_url(self, url: str, base: str) -> bool:
        if not url or url.startswith("#") or url.startswith("mailto:") or url.startswith("tel:"):
            return False
        parsed = urlparse(urljoin(base, url))
        return bool(parsed.scheme in ("http", "https") and parsed.netloc)

    async def fetch(self, url: str, **kwargs: Any) -> ParsedDocument:
        client = await self._get_client()
        try:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text
            content_type = response.headers.get("content-type", "")

            if "text/html" not in content_type and "application/xhtml" not in content_type:
                text = response.text
                return ParsedDocument(
                    title=urlparse(url).path.split("/")[-1] or url,
                    content=text,
                    url=url,
                    file_type=content_type.split(";")[0].strip(),
                    file_size_bytes=len(response.content),
                )

            content = self._extract_main_content(html, url)
            meta = self._extract_metadata(html, url)
            doc_hash = hashlib.md5(html.encode()).hexdigest()[:16]

            return ParsedDocument(
                title=meta.get("title", urlparse(url).path),
                content=content,
                url=url,
                author=meta.get("author"),
                created_date=meta.get("created_date"),
                file_type="html",
                file_size_bytes=len(response.content),
                metadata={
                    "description": meta.get("description"),
                    "doc_hash": doc_hash,
                    "ingested_via": "web_plugin",
                },
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            raise

    async def crawl(
        self,
        start_urls: list[str],
        max_depth: int = 1,
        max_pages: int = 50,
        restrict_domains: list[str] | None = None,
    ) -> list[ParsedDocument]:
        """Crawl a set of starting URLs up to max_depth."""
        import httpx
        self._visited.clear()
        results: list[ParsedDocument] = []
        queue: list[tuple[str, int]] = [(url, 0) for url in start_urls]
        if restrict_domains is None:
            restrict_domains = [urlparse(u).netloc for u in start_urls]

        semaphore = asyncio.Semaphore(5)

        async def fetch_page(url: str, depth: int) -> None:
            if len(results) >= max_pages:
                return
            normalized = url.rstrip("/")
            if normalized in self._visited or depth > max_depth:
                return
            self._visited.add(normalized)

            async with semaphore:
                try:
                    doc = await self.fetch(url)
                    results.append(doc)
                    if depth < max_depth:
                        links = self._extract_links(results[-1].metadata.get("_html", ""), url)
                        for link in links:
                            if urlparse(link).netloc in restrict_domains:
                                queue.append((link, depth + 1))
                except Exception as e:
                    logger.warning(f"Crawl error at {url}: {e}")

        tasks = [fetch_page(u, d) for u, d in queue[:max_pages]]
        await asyncio.gather(*tasks, return_exceptions=True)
        return results

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            return [a["href"] for a in soup.find_all("a", href=True) if self._is_valid_url(a["href"], base_url)]
        except ImportError:
            pass
        urls = re.findall(r'href=["\'](https?://[^"\']+)["\']', html)
        return [u for u in urls if self._is_valid_url(u, base_url)]

    async def sync(self, **kwargs: Any) -> Any:
        import time
        start = time.monotonic()
        start_urls = kwargs.get("start_urls", [])
        max_depth = kwargs.get("crawl_depth", self.config.get("crawl_depth", 1))
        docs = await self.crawl(start_urls, max_depth=max_depth, max_pages=kwargs.get("max_pages", 50))
        return SyncResult(
            source_id=self.config.source_id,
            documents=docs,
            crawled_urls=len(docs),
            duration_seconds=time.monotonic() - start,
        )
