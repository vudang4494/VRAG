"""Health check tests for all RAG stack services."""

import subprocess

import httpx
import pytest

BASE = "http://localhost"

SERVICES = {
    "ollama": {"url": "http://localhost:11434/api/tags"},
    "qdrant": {"url": f"{BASE}:6333/healthz"},
    "neo4j": {"url": f"{BASE}:7474"},
    "rag_api": {"url": f"{BASE}:8800/health"},
    "redis": {"cmd": ["docker", "exec", "rag-redis", "redis-cli", "ping"]},
    "postgres": {
        "cmd": ["docker", "exec", "rag-postgres", "pg_isready", "-U", "raguser", "-d", "ragdb"]
    },
    "langfuse": {"url": f"{BASE}:3000/api/public/health"},
    "prometheus": {"url": f"{BASE}:9090/-/healthy"},
    "grafana": {"url": f"{BASE}:3001/api/health"},
    "nginx": {"url": f"{BASE}:80"},
}


class TestServiceHealth:
    """Verify every service in the stack is reachable."""

    @pytest.mark.parametrize("svc", list(SERVICES.keys()), ids=list(SERVICES.keys()))
    def test_service_reachable(self, svc):
        cfg = SERVICES[svc]
        if "url" in cfg:
            with httpx.Client(timeout=10) as client:
                r = client.get(cfg["url"])
                assert r.status_code < 500, f"{svc} returned {r.status_code}: {r.text[:120]}"
        elif "cmd" in cfg:
            result = subprocess.run(cfg["cmd"], capture_output=True, text=True, timeout=15)
            assert result.returncode == 0, f"{svc} failed: {result.stderr}"


class TestRAGAPIEndpoints:
    """Test RAG API endpoints."""

    def test_health_ok(self):
        r = httpx.get(f"{BASE}:8800/health", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["version"] == "2.0.0"

    def test_health_deep(self):
        r = httpx.get(f"{BASE}:8800/health/deep", timeout=30)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("ok", "degraded")
        checks = data["checks"]
        assert checks["ollama"]["status"] == "ok"
        assert checks["qdrant"]["status"] == "ok"
        assert checks["neo4j"]["status"] == "ok"
        assert checks["redis"]["status"] == "ok"

    def test_metrics(self):
        r = httpx.get(f"{BASE}:8800/metrics", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "total_requests" in data

    def test_models_endpoint(self):
        r = httpx.get(f"{BASE}:8800/v1/models", timeout=15)
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        model_ids = [m["id"] for m in data["data"]]
        assert "qwen3.5:4b" in model_ids

    def test_404_routes(self):
        r = httpx.get(f"{BASE}:8800/nonexistent", timeout=5)
        assert r.status_code == 404

    def test_cache_clear(self):
        r = httpx.post(f"{BASE}:8800/cache/clear", timeout=10)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_ingest_file_too_small(self):
        files = {"file": ("tiny.txt", b"x", "text/plain")}
        r = httpx.post(f"{BASE}:8800/ingest/upload", files=files, timeout=60)
        assert r.status_code == 422
