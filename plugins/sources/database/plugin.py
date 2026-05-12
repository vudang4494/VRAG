"""Database source plugin — SQL databases as RAG sources."""
import asyncio
import hashlib
import json
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


class DatabaseSourcePlugin(BaseSourcePlugin):
    """Query SQL databases and convert results to documents."""

    name = "database"
    version = "1.0.0"
    capabilities = [
        PluginCapability.INGEST_QUERY,
        PluginCapability.INGEST_SCHEDULED,
    ]
    supported_types = ["database"]

    def __init__(self, config: PluginConfig, credentials: SourceCredentials | None = None):
        super().__init__(config, credentials)
        self._pool: Any = None

    async def _get_pool(self) -> Any:
        if self._pool is not None:
            return self._pool

        db_type = self.config.require("db_type")
        host = self.config.require("host")
        port = self.config.get("port", 5432 if db_type == "postgresql" else 3306)
        database = self.config.require("database")
        user = self.config.require("user")
        password = self.config.get("password", "")

        if db_type == "postgresql":
            import asyncpg
            self._pool = await asyncpg.create_pool(
                host=host, port=port, user=user, password=password,
                database=database, min_size=1, max_size=5,
            )
        elif db_type == "mysql":
            import aiomysql
            self._pool = await aiomysql.create_pool(
                host=host, port=port, user=user, password=password,
                db=database, minsize=1, maxsize=5,
            )
        elif db_type == "sqlite":
            import aiosqlite
            conn = await aiosqlite.connect(self.config.require("path"))
            self._pool = conn
        else:
            raise ValueError(f"Unsupported db_type: {db_type}")

        return self._pool

    async def close(self) -> None:
        if self._pool:
            if hasattr(self._pool, "close"):
                self._pool.close()
                if hasattr(self._pool, "wait_closed"):
                    await self._pool.wait_closed()
            self._pool = None

    def _rows_to_text(self, columns: list[str], rows: list[tuple], title: str) -> str:
        lines = [f"## {title}\n"]
        lines.append("Columns: " + ", ".join(columns) + "\n")
        for row in rows:
            row_dict = dict(zip(columns, row))
            lines.append(json.dumps(row_dict, ensure_ascii=False, default=str))
        return "\n".join(lines)

    async def _execute_query(self, query: str, pool: Any) -> tuple[list[str], list[tuple]]:
        db_type = self.config.get("db_type", "postgresql")
        if db_type == "postgresql":
            rows = await pool.fetch(query)
            columns = list(rows[0].keys()) if rows else []
            return columns, [tuple(r.values()) for r in rows]
        elif db_type == "mysql":
            async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query)
                    columns = [d[0] for d in cur.description] if cur.description else []
                    rows = await cur.fetchall()
                    return columns, list(rows)
        elif db_type == "sqlite":
            async with pool.execute(query) as cur:
                columns = [d[0] for d in cur.description] if cur.description else []
                rows = await cur.fetchall()
                return columns, list(rows)
        return [], []

    async def fetch(self, query: str, **kwargs: Any) -> ParsedDocument:
        pool = await self._get_pool()
        table_name = kwargs.get("table_name", "query_result")
        columns, rows = await self._execute_query(query, pool)

        if not columns:
            return ParsedDocument(
                title=f"DB Query: {table_name}",
                content="No results returned.",
                metadata={"query": query, "ingested_via": "database_plugin"},
            )

        content = self._rows_to_text(columns, rows, table_name)
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]

        return ParsedDocument(
            title=f"Database: {table_name}",
            content=content,
            metadata={
                "query": query,
                "row_count": len(rows),
                "columns": columns,
                "doc_hash": content_hash,
                "ingested_via": "database_plugin",
            },
        )

    async def sync(self, **kwargs: Any) -> SyncResult:
        import time
        start = time.monotonic()
        pool = await self._get_pool()
        queries = self.config.get("queries", [])
        docs = []
        errors = []

        for q in queries:
            name = q.get("name", "query")
            sql = q.get("sql", "")
            try:
                columns, rows = await self._execute_query(sql, pool)
                text = self._rows_to_text(columns, rows, name)
                doc_hash = hashlib.md5(text.encode()).hexdigest()[:16]
                docs.append(ParsedDocument(
                    title=f"DB: {name}",
                    content=text,
                    metadata={"query": sql, "row_count": len(rows), "doc_hash": doc_hash},
                ))
            except Exception as e:
                errors.append(f"Query '{name}': {e}")
                logger.error(f"Database query error: {e}")

        return SyncResult(
            source_id=self.config.source_id,
            documents=docs,
            crawled_urls=len(docs),
            errors=errors,
            duration_seconds=time.monotonic() - start,
        )
