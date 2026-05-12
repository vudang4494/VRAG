"""Prometheus metrics middleware for the RAG API."""
import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match


class PrometheusMetrics:
    """Thread-safe metrics collector."""

    def __init__(self):
        self._requests_total: dict[str, int] = defaultdict(int)
        self._requests_errors: dict[str, int] = defaultdict(int)
        self._request_latencies: dict[str, list[float]] = defaultdict(list)
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._chunks_indexed: int = 0
        self._entities_extracted: int = 0
        self._active_requests: int = 0
        self._start_time: float = time.time()

    def record_request(self, endpoint: str, status: int, latency: float) -> None:
        self._requests_total[endpoint] += 1
        if status >= 400:
            self._requests_errors[endpoint] += 1
        self._request_latencies[endpoint].append(latency)
        if len(self._request_latencies[endpoint]) > 1000:
            self._request_latencies[endpoint] = self._request_latencies[endpoint][-1000:]

    def record_cache(self, hit: bool) -> None:
        if hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

    def record_ingestion(self, chunks: int, entities: int) -> None:
        self._chunks_indexed += chunks
        self._entities_extracted += entities

    def get_metrics(self) -> dict:
        total = sum(self._requests_total.values())
        errors = sum(self._requests_errors.values())
        cache_total = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / cache_total if cache_total > 0 else 0.0

        p95_latencies = {}
        for endpoint, latencies in self._request_latencies.items():
            if latencies:
                sorted_lat = sorted(latencies)
                p95_idx = int(len(sorted_lat) * 0.95)
                p95_latencies[endpoint] = sorted_lat[p95_idx] if sorted_lat else 0.0

        return {
            "total_requests": total,
            "total_errors": errors,
            "error_rate": errors / total if total > 0 else 0.0,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": round(cache_hit_rate, 4),
            "cache_total": cache_total,
            "chunks_indexed_total": self._chunks_indexed,
            "entities_extracted_total": self._entities_extracted,
            "active_requests": self._active_requests,
            "uptime_seconds": time.time() - self._start_time,
            "requests_by_endpoint": dict(self._requests_total),
            "errors_by_endpoint": dict(self._requests_errors),
            "p95_latency_seconds": p95_latencies,
            "avg_latency_seconds": {
                ep: sum(lats) / len(lats) if lats else 0.0
                for ep, lats in self._request_latencies.items()
            },
        }

    def prometheus_output(self) -> str:
        """Generate Prometheus text format output."""
        m = self.get_metrics()
        lines = [
            "# HELP rag_requests_total Total number of RAG API requests",
            "# TYPE rag_requests_total counter",
            f"rag_requests_total {m['total_requests']}",
            "",
            "# HELP rag_requests_errors_total Total number of errors",
            "# TYPE rag_requests_errors_total counter",
            f"rag_requests_errors_total {m['total_errors']}",
            "",
            "# HELP rag_cache_hits_total Semantic cache hits",
            "# TYPE rag_cache_hits_total counter",
            f"rag_cache_hits_total {m['cache_hits']}",
            f"rag_cache_misses_total {m['cache_misses']}",
            "",
            "# HELP rag_cache_hit_rate Cache hit rate ratio",
            "# TYPE rag_cache_hit_rate gauge",
            f"rag_cache_hit_rate {m['cache_hit_rate']}",
            "",
            "# HELP rag_chunks_indexed_total Total chunks indexed",
            "# TYPE rag_chunks_indexed_total counter",
            f"rag_chunks_indexed_total {m['chunks_indexed_total']}",
            f"rag_entities_extracted_total {m['entities_extracted_total']}",
            "",
            "# HELP rag_request_latency_seconds Request latency",
            "# TYPE rag_request_latency_seconds histogram",
        ]

        for ep, latency in m.get("p95_latency_seconds", {}).items():
            safe_ep = ep.replace("/", "_").replace("-", "_").strip("_")
            lines.append(f'rag_request_latency_seconds{{endpoint="{ep}"}} {latency:.4f}')
        lines.append("")
        lines.append("# HELP rag_uptime_seconds Time since API start")
        lines.append("# TYPE rag_uptime_seconds gauge")
        lines.append(f"rag_uptime_seconds {m['uptime_seconds']:.1f}")

        return "\n".join(lines)


# Global metrics instance
_metrics = PrometheusMetrics()


def get_metrics() -> PrometheusMetrics:
    return _metrics


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        method = request.method
        endpoint = f"{method} {path}"

        _metrics._active_requests += 1
        start = time.monotonic()

        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            raise
        finally:
            latency = time.monotonic() - start
            _metrics.record_request(endpoint, status, latency)
            _metrics._active_requests -= 1

        return response
