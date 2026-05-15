"""Langfuse tracing integration for RAG operations."""

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from loguru import logger


class RAGTracer:
    """
    Langfuse-based tracing for RAG operations.

    Traces:
    - Query embedding
    - Vector search
    - Graph search
    - RRF fusion
    - LLM generation
    - Semantic cache lookup/set
    - Full RAG pipeline
    """

    def __init__(self):
        self._enabled = False
        self._client = None
        self._check()

    def _check(self) -> None:
        try:
            from langfuse import Langfuse
            from src.config import get_settings

            settings = get_settings()
            if settings.langfuse_public_key and settings.langfuse_secret_key:
                self._client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host or "http://localhost:3000",
                )
                self._enabled = True
                logger.info("Langfuse tracing enabled")
        except Exception as e:
            logger.debug(f"Langfuse not configured: {e}")

    @asynccontextmanager
    async def trace(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
    ) -> AsyncIterator[Any]:
        """Async context manager for a trace span."""
        if not self._enabled:
            yield None
            return

        class SpanContext:
            def __init__(self):
                self.trace_id: str = ""
                self.span_id: str = ""
                self._spans: list[dict] = []

            def add_event(self, name: str, metadata: dict | None = None) -> None:
                self._spans.append(
                    {
                        "name": name,
                        "metadata": metadata or {},
                        "timestamp": time.time(),
                    }
                )

        ctx = SpanContext()
        start = time.monotonic()

        try:
            trace = self._client.trace(
                name=name,
                metadata=metadata,
                user_id=user_id,
                tags=tags,
            )
            ctx.trace_id = trace.id
            yield ctx
            trace.update(
                metadata={
                    **(metadata or {}),
                    "duration_ms": (time.monotonic() - start) * 1000,
                }
            )
        except Exception as e:
            logger.warning(f"Langfuse trace error: {e}")
            yield ctx

    async def trace_retrieval(
        self,
        query: str,
        vector_results: int,
        graph_results: int,
        fusion_time_ms: float,
        cache_hit: bool,
        top_k: int,
    ) -> None:
        if not self._enabled:
            return
        try:
            self._client.span(
                name="retrieval",
                metadata={
                    "query": query[:200],
                    "vector_results": vector_results,
                    "graph_results": graph_results,
                    "fusion_time_ms": fusion_time_ms,
                    "cache_hit": cache_hit,
                    "top_k": top_k,
                },
            )
        except Exception as e:
            logger.debug(f"Langfuse span error: {e}")

    async def trace_generation(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        stream: bool,
    ) -> None:
        if not self._enabled:
            return
        try:
            self._client.span(
                name="generation",
                metadata={
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "latency_ms": latency_ms,
                    "stream": stream,
                    "tok_per_sec": (completion_tokens / latency_ms * 1000) if latency_ms > 0 else 0,
                },
            )
        except Exception as e:
            logger.debug(f"Langfuse span error: {e}")

    async def trace_ingestion(
        self,
        tenant_id: str,
        source_id: str,
        document_id: str,
        chunks: int,
        entities: int,
        relationships: int,
        duration_ms: float,
        file_size: int,
    ) -> None:
        if not self._enabled:
            return
        try:
            self._client.event(
                name="ingestion",
                metadata={
                    "tenant_id": tenant_id,
                    "source_id": source_id,
                    "document_id": document_id,
                    "chunks_indexed": chunks,
                    "entities_extracted": entities,
                    "relationships_extracted": relationships,
                    "duration_ms": duration_ms,
                    "file_size_bytes": file_size,
                    "chunks_per_sec": (chunks / duration_ms * 1000) if duration_ms > 0 else 0,
                },
            )
        except Exception as e:
            logger.debug(f"Langfuse event error: {e}")


# Global tracer
tracer = RAGTracer()
