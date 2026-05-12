"""GitHub source plugin — issues, PRs, repos, code."""
import asyncio
import hashlib
import re
from datetime import datetime
from typing import Any

from loguru import logger

from plugins.base import (
    BaseSourcePlugin,
    ParsedDocument,
    PluginCapability,
    PluginConfig,
    SourceCredentials,
    SyncResult,
)


class GitHubSourcePlugin(BaseSourcePlugin):
    """Ingest GitHub repositories, issues, PRs, discussions."""

    name = "github"
    version = "1.0.0"
    capabilities = [
        PluginCapability.INGEST_URL,
        PluginCapability.INGEST_CRAWL,
        PluginCapability.INGEST_SCHEDULED,
    ]
    supported_types = ["github"]

    def __init__(self, config: PluginConfig, credentials: SourceCredentials | None = None):
        super().__init__(config, credentials)
        self._token = credentials.encrypted_blob if credentials else None
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            import httpx
            headers = {"Accept": "application/vnd.github+json"}
            if self._token:
                import json
                try:
                    creds = json.loads(self._token)
                    token = creds.get("token") or creds.get("api_key")
                except Exception:
                    token = self._token
                headers["Authorization"] = f"Bearer {token}"
            limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
            self._client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(60.0), headers=headers)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    def _extract_code_snippets(self, text: str, max_snippets: int = 20) -> str:
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        return "\n\n".join(code_blocks[:max_snippets])

    async def fetch(self, url_or_path: str, **kwargs: Any) -> ParsedDocument:
        """Fetch a GitHub resource. URL format: github://owner/repo/path"""
        client = await self._get_client()
        parts = url_or_path.replace("github://", "").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub path: {url_or_path}")
        owner, repo = parts[0], parts[1]
        path = "/".join(parts[2:]) if len(parts) > 2 else ""

        resource_type = kwargs.get("resource", "repo")
        content_text = ""

        if resource_type == "issues":
            issues = await self._fetch_issues(client, owner, repo)
            content_text = "\n\n".join(issues)
        elif resource_type == "prs":
            prs = await self._fetch_prs(client, owner, repo)
            content_text = "\n\n".join(prs)
        elif resource_type == "readme":
            content_text = await self._fetch_readme(client, owner, repo)
        else:
            content_text = await self._fetch_repo_summary(client, owner, repo)

        return ParsedDocument(
            title=f"github/{owner}/{repo}",
            content=content_text,
            url=f"https://github.com/{owner}/{repo}",
            metadata={
                "owner": owner,
                "repo": repo,
                "resource_type": resource_type,
                "doc_hash": hashlib.md5(content_text.encode()).hexdigest()[:16],
                "ingested_via": "github_plugin",
            },
        )

    async def _fetch_readme(self, client: Any, owner: str, repo: str) -> str:
        r = await client.get(f"https://api.github.com/repos/{owner}/{repo}/readme")
        r.raise_for_status()
        import base64
        content = base64.b64decode(r.json()["content"]).decode("utf-8", errors="replace")
        return f"# README: {owner}/{repo}\n\n{content}"

    async def _fetch_issues(self, client: Any, owner: str, repo: str, max_count: int = 100) -> list[str]:
        issues: list[str] = []
        page = 1
        while len(issues) < max_count:
            r = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/issues",
                params={"state": "all", "per_page": 100, "page": page},
            )
            if r.status_code == 403:
                logger.warning("GitHub rate limited")
                break
            r.raise_for_status()
            items = r.json()
            if not items:
                break
            for issue in items:
                if "pull_request" in issue:
                    continue
                issues.append(
                    f"## Issue #{issue['number']}: {issue['title']}\n"
                    f"Author: {issue['user']['login']} | Labels: {[l['name'] for l in issue['labels']]}\n"
                    f"Created: {issue['created_at']}\n\n{issue['body'] or ''}"
                )
            page += 1
        return issues[:max_count]

    async def _fetch_prs(self, client: Any, owner: str, repo: str, max_count: int = 50) -> list[str]:
        prs: list[str] = []
        page = 1
        while len(prs) < max_count:
            r = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/pulls",
                params={"state": "all", "per_page": 100, "page": page},
            )
            if r.status_code == 403:
                break
            r.raise_for_status()
            items = r.json()
            if not items:
                break
            for pr in items:
                prs.append(
                    f"## PR #{pr['number']}: {pr['title']}\n"
                    f"Author: {pr['user']['login']} | Status: {pr['state']}\n"
                    f"Created: {pr['created_at']} | Merged: {pr.get('merged_at')}\n\n{pr['body'] or ''}"
                )
            page += 1
        return prs[:max_count]

    async def _fetch_repo_summary(self, client: Any, owner: str, repo: str) -> str:
        r = await client.get(f"https://api.github.com/repos/{owner}/{repo}")
        r.raise_for_status()
        info = r.json()
        readme = await self._fetch_readme(client, owner, repo)
        return (
            f"# {info['full_name']}\n\n"
            f"Description: {info['description'] or ''}\n"
            f"Stars: {info['stargazers_count']} | Forks: {info['forks_count']}\n"
            f"Language: {info['language']} | Topics: {', '.join(info.get('topics', []))}\n"
            f"License: {info.get('license', {}).get('name', 'None')}\n\n"
            f"{readme}"
        )

    async def sync(self, **kwargs: Any) -> SyncResult:
        import time
        start = time.monotonic()
        repo_url = self.config.require("repo_url")
        owner, repo = repo_url.replace("https://github.com/", "").replace("github://", "").split("/")[:2]
        client = await self._get_client()
        docs = []
        errors = []

        try:
            summary = await self._fetch_repo_summary(client, owner, repo)
            docs.append(ParsedDocument(
                title=f"github/{owner}/{repo}",
                content=summary,
                url=f"https://github.com/{owner}/{repo}",
                metadata={"resource": "repo", "ingested_via": "github_plugin"},
            ))
        except Exception as e:
            errors.append(f"Repo summary: {e}")

        try:
            issues = await self._fetch_issues(client, owner, repo)
            for issue_text in issues:
                doc_hash = hashlib.md5(issue_text.encode()).hexdigest()[:16]
                docs.append(ParsedDocument(
                    title="GitHub Issue",
                    content=issue_text,
                    metadata={"doc_hash": doc_hash, "ingested_via": "github_plugin"},
                ))
        except Exception as e:
            errors.append(f"Issues: {e}")

        return SyncResult(
            source_id=self.config.source_id,
            documents=docs,
            crawled_urls=len(docs),
            errors=errors,
            duration_seconds=time.monotonic() - start,
        )
