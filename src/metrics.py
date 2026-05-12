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
        # Pipeline V2 counters
        self._v2_chats_total: int = 0
        self._v2_refusals_total: int = 0
        self._v2_validation_passes: int = 0
        self._v2_validation_fails: int = 0
        self._v2_consistency_scores: list[float] = []
        self._v2_grounded_ratios: list[float] = []
        self._v2_stage_latencies: dict[str, list[float]] = defaultdict(list)
        self._v2_communities_built: int = 0

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

    # ── V2 recording ─────────────────────────────────────────────────────────
    def record_v2_chat(
        self,
        refused: bool,
        validation_passed: bool,
        grounded_ratio: float,
        stage_latencies_ms: dict[str, float],
    ) -> None:
        self._v2_chats_total += 1
        if refused:
            self._v2_refusals_total += 1
        if validation_passed:
            self._v2_validation_passes += 1
        else:
            self._v2_validation_fails += 1
        self._v2_grounded_ratios.append(grounded_ratio)
        if len(self._v2_grounded_ratios) > 1000:
            self._v2_grounded_ratios = self._v2_grounded_ratios[-1000:]
        for stage, latency in stage_latencies_ms.items():
            self._v2_stage_latencies[stage].append(latency)
            if len(self._v2_stage_latencies[stage]) > 500:
                self._v2_stage_latencies[stage] = self._v2_stage_latencies[stage][-500:]

    def record_v2_ingest(self, avg_consistency: float, communities: int = 0) -> None:
        self._v2_consistency_scores.append(avg_consistency)
        if len(self._v2_consistency_scores) > 1000:
            self._v2_consistency_scores = self._v2_consistency_scores[-1000:]
        self._v2_communities_built += communities

    def get_v2_metrics(self) -> dict:
        total = self._v2_chats_total
        refusal_rate = self._v2_refusals_total / total if total else 0.0
        val_total = self._v2_validation_passes + self._v2_validation_fails
        validation_pass_rate = self._v2_validation_passes / val_total if val_total else 0.0
        avg_grounded = sum(self._v2_grounded_ratios) / len(self._v2_grounded_ratios) if self._v2_grounded_ratios else 0.0
        avg_consistency = sum(self._v2_consistency_scores) / len(self._v2_consistency_scores) if self._v2_consistency_scores else 0.0

        stage_p50 = {}
        stage_p95 = {}
        for stage, lats in self._v2_stage_latencies.items():
            if not lats:
                continue
            sorted_l = sorted(lats)
            stage_p50[stage] = sorted_l[len(sorted_l) // 2]
            stage_p95[stage] = sorted_l[int(len(sorted_l) * 0.95)]

        return {
            "v2_chats_total": total,
            "v2_refusals_total": self._v2_refusals_total,
            "v2_refusal_rate": round(refusal_rate, 4),
            "v2_validation_pass_rate": round(validation_pass_rate, 4),
            "v2_avg_grounded_ratio": round(avg_grounded, 4),
            "v2_avg_consistency_score": round(avg_consistency, 4),
            "v2_communities_built": self._v2_communities_built,
            "v2_stage_p50_ms": stage_p50,
            "v2_stage_p95_ms": stage_p95,
        }

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
            "v2": self.get_v2_metrics(),
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

        # V2 metrics
        v2 = self.get_v2_metrics()
        lines.extend([
            "",
            "# HELP rag_v2_chats_total Pipeline V2 chat completions",
            "# TYPE rag_v2_chats_total counter",
            f"rag_v2_chats_total {v2['v2_chats_total']}",
            "# HELP rag_v2_refusals_total Pipeline V2 refusals (quality gate fail)",
            "# TYPE rag_v2_refusals_total counter",
            f"rag_v2_refusals_total {v2['v2_refusals_total']}",
            "# HELP rag_v2_refusal_rate Refusal ratio",
            "# TYPE rag_v2_refusal_rate gauge",
            f"rag_v2_refusal_rate {v2['v2_refusal_rate']}",
            "# HELP rag_v2_validation_pass_rate Validation gates pass rate",
            "# TYPE rag_v2_validation_pass_rate gauge",
            f"rag_v2_validation_pass_rate {v2['v2_validation_pass_rate']}",
            "# HELP rag_v2_avg_grounded_ratio Avg claims grounded ratio",
            "# TYPE rag_v2_avg_grounded_ratio gauge",
            f"rag_v2_avg_grounded_ratio {v2['v2_avg_grounded_ratio']}",
            "# HELP rag_v2_avg_consistency_score Avg chunk consistency score",
            "# TYPE rag_v2_avg_consistency_score gauge",
            f"rag_v2_avg_consistency_score {v2['v2_avg_consistency_score']}",
            "# HELP rag_v2_communities_built Total communities created",
            "# TYPE rag_v2_communities_built counter",
            f"rag_v2_communities_built {v2['v2_communities_built']}",
        ])
        for stage, lat in v2.get("v2_stage_p95_ms", {}).items():
            lines.append(f'rag_v2_stage_p95_ms{{stage="{stage}"}} {lat:.2f}')

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
