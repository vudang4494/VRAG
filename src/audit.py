"""Audit logging — records every RAG API operation."""

import json
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AuditEvent(str, Enum):
    TENANT_CREATED = "tenant.created"
    TENANT_UPDATED = "tenant.updated"
    TENANT_DELETED = "tenant.deleted"
    SOURCE_CREATED = "source.created"
    SOURCE_SYNC_STARTED = "source.sync.started"
    SOURCE_SYNC_COMPLETED = "source.sync.completed"
    SOURCE_SYNC_FAILED = "source.sync.failed"
    DOCUMENT_INGESTED = "document.ingested"
    DOCUMENT_DELETED = "document.deleted"
    CHAT_QUERY = "chat.query"
    CHAT_QUERY_CACHE_HIT = "chat.query.cache_hit"
    API_KEY_CREATED = "api_key.created"
    API_KEY_REVOKED = "api_key.revoked"


class AuditLogger:
    """
    Audit logger that writes to multiple backends:
    - PostgreSQL (primary, structured)
    - File (fallback)
    - Redis (recent events cache)
    """

    def __init__(self):
        self._buffer: list[dict[str, Any]] = []
        self._flush_interval = 5.0
        self._last_flush = time.monotonic()

    def log(
        self,
        event: AuditEvent | str,
        tenant_id: str | None = None,
        user_id: str | None = None,
        api_key_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "success",
        error_message: str | None = None,
    ) -> None:
        record = {
            "event": event.value if isinstance(event, AuditEvent) else event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "user_id": user_id,
            "api_key_id": api_key_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "metadata": metadata or {},
            "status": status,
            "error_message": error_message,
        }
        self._buffer.append(record)
        if time.monotonic() - self._last_flush > self._flush_interval:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        records = self._buffer[:]
        self._buffer.clear()
        self._last_flush = time.monotonic()

        self._write_to_file(records)
        self._write_to_redis(records)

    async def _flush_async(self) -> None:
        if not self._buffer:
            return
        records = self._buffer[:]
        self._buffer.clear()
        self._last_flush = time.monotonic()

        await self._write_to_postgres(records)
        self._write_to_file(records)
        await self._write_to_redis(records)

    def _write_to_file(self, records: list[dict[str, Any]]) -> None:
        try:
            import os

            log_dir = os.path.expanduser("~/.rag/audit")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(
                log_dir, f"audit_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
            )
            with open(log_file, "a") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    async def _write_to_postgres(self, records: list[dict[str, Any]]) -> None:
        try:
            from src.clients import get_clients

            clients = get_clients()
            if clients.redis is None:
                return
            key = "rag:audit:recent"
            pipe = clients.redis.pipeline()
            for record in records:
                pipe.lpush(key, json.dumps(record))
                pipe.ltrim(key, 0, 999)
            await pipe.execute()
        except Exception:
            pass

    async def _write_to_redis(self, records: list[dict[str, Any]]) -> None:
        try:
            from src.clients import get_clients

            clients = get_clients()
            if clients.redis is None:
                return
            key = "rag:audit:recent"
            pipe = clients.redis.pipeline()
            for record in records:
                pipe.lpush(key, json.dumps(record))
                pipe.ltrim(key, 0, 999)
            await pipe.execute()
        except Exception:
            pass

    async def get_recent_events(
        self,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        try:
            from src.clients import get_clients

            clients = get_clients()
            if clients.redis is None:
                return []
            raw = await clients.redis.lrange("rag:audit:recent", 0, limit - 1)
            events = [json.loads(r) for r in raw]
            if tenant_id:
                events = [e for e in events if e.get("tenant_id") == tenant_id]
            return events
        except Exception:
            return []

    def log_chat(
        self, tenant_id: str, query: str, sources_count: int, cache_hit: bool, latency_ms: float
    ) -> None:
        self.log(
            event=AuditEvent.CHAT_QUERY_CACHE_HIT if cache_hit else AuditEvent.CHAT_QUERY,
            tenant_id=tenant_id,
            metadata={
                "query_preview": query[:100],
                "sources_returned": sources_count,
                "cache_hit": cache_hit,
                "latency_ms": latency_ms,
            },
        )

    def log_ingestion(
        self, tenant_id: str, source_id: str, doc_id: str, chunks: int, duration_ms: float
    ) -> None:
        self.log(
            event=AuditEvent.DOCUMENT_INGESTED,
            tenant_id=tenant_id,
            resource_type="document",
            resource_id=doc_id,
            metadata={
                "source_id": source_id,
                "chunks_indexed": chunks,
                "duration_ms": duration_ms,
            },
        )

    def log_source_sync(
        self,
        tenant_id: str,
        source_id: str,
        docs_crawled: int,
        duration_ms: float,
        error: str | None = None,
    ) -> None:
        self.log(
            event=AuditEvent.SOURCE_SYNC_FAILED if error else AuditEvent.SOURCE_SYNC_COMPLETED,
            tenant_id=tenant_id,
            resource_type="source",
            resource_id=source_id,
            metadata={
                "documents_crawled": docs_crawled,
                "duration_ms": duration_ms,
            },
            status="error" if error else "success",
            error_message=error,
        )


# Global audit logger
audit = AuditLogger()
