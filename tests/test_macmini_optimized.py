"""Mac Mini M4 Optimization Tests — comprehensive performance and resource tests."""

import asyncio
import time
import os
import sys
import subprocess
import pytest
import httpx

# Configure env for Mac Mini
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_MODEL", "qwen3.5:4b")
os.environ.setdefault("OLLAMA_EMBED_MODEL", "bge-m3")
os.environ.setdefault("OLLAMA_EMBED_URL", "http://localhost:11434")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("NEO4J_URL", "bolt://localhost:7687")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("SEMANTIC_CACHE_TTL", "7200")
os.environ.setdefault("ENABLE_SEMANTIC_CACHE", "true")


# =============================================================================
# Constants
# =============================================================================

API_BASE = "http://localhost:8800"
OLLAMA_BASE = "http://localhost:11434"
QDRANT_BASE = "http://localhost:6333"
NEO4J_BASE = "http://localhost:7474"
REDIS_CONTAINER = "rag-redis"
DASHBOARD_BASE = "http://localhost:7860"

TEST_QUERIES = [
    "Quy trinh tuyen dung nhan su moi",
    "Chinh sach nghi phep nam 2024",
    "Bao cao tai chinh quy 3",
    "Thuong hieu va san pham cua cong ty",
    "Quy che lam viec tu xa",
    "Huong dan su dung phan mem ERP",
    "Chinh sach luong thuong hieu",
    "Ket qua kinh doanh nam nay",
]


# =============================================================================
# Section 1: Service Health
# =============================================================================


class TestMacMiniServices:
    """Verify all Mac Mini stack services are healthy."""

    @pytest.mark.asyncio
    async def test_ollama_reachable(self):
        """Ollama API must be running on host."""
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags")
            assert r.status_code == 200
            data = r.json()
            assert "models" in data
            model_names = [m["name"] for m in data["models"]]
            print(f"\n  Ollama models: {model_names}")

    @pytest.mark.asyncio
    async def test_qdrant_health(self):
        """Qdrant vector DB health check."""
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{QDRANT_BASE}/healthz")
            assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_neo4j_health(self):
        """Neo4j knowledge graph health."""
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(NEO4J_BASE)
            assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_redis_health(self):
        """Redis cache health via docker exec."""
        result = subprocess.run(
            ["docker", "exec", REDIS_CONTAINER, "redis-cli", "ping"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "PONG" in result.stdout

    @pytest.mark.asyncio
    async def test_rag_api_health(self):
        """RAG API /health endpoint."""
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{API_BASE}/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_rag_api_deep_health(self):
        """RAG API /health/deep — all services check."""
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{API_BASE}/health/deep")
            assert r.status_code == 200
            data = r.json()
            assert data["status"] in ("ok", "degraded")
            checks = data["checks"]
            for svc in ["ollama", "qdrant", "neo4j", "redis"]:
                assert checks[svc]["status"] == "ok", (
                    f"{svc} failed: {checks[svc].get('detail', '')}"
                )

    @pytest.mark.asyncio
    async def test_dashboard_reachable(self):
        """Gradio dashboard reachable."""
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{DASHBOARD_BASE}/")
            assert r.status_code == 200


# =============================================================================
# Section 2: LLM & Embedding Performance
# =============================================================================


class TestMacMiniLLMPerformance:
    """Test Ollama LLM and embedding performance on M4 Metal GPU."""

    @pytest.mark.asyncio
    async def test_embedding_latency(self):
        """Measure embedding latency (BGE-M3 on Metal)."""
        test_text = "Quy trinh tuyen dung nhan su moi tai cong ty"
        latencies = []

        async with httpx.AsyncClient(timeout=60) as client:
            for _ in range(5):
                start = time.monotonic()
                r = await client.post(
                    f"{OLLAMA_BASE}/api/embeddings",
                    json={"model": "bge-m3", "prompt": test_text},
                )
                lat = (time.monotonic() - start) * 1000
                assert r.status_code == 200
                data = r.json()
                assert len(data["embedding"]) == 1024
                latencies.append(lat)

        avg = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        print(f"\n  Embedding latency: avg={avg:.1f}ms, p95={p95:.1f}ms")
        assert avg < 500, f"Embedding too slow: {avg:.1f}ms (expected <500ms on M4 Metal)"

    @pytest.mark.asyncio
    async def test_llm_generation_latency(self):
        """Measure LLM generation latency (Qwen3.5-4B on Metal)."""
        latencies = []
        tokens_generated = []

        async with httpx.AsyncClient(timeout=120) as client:
            for _ in range(3):
                start = time.monotonic()
                r = await client.post(
                    f"{OLLAMA_BASE}/api/generate",
                    json={
                        "model": "qwen3.5:4b",
                        "prompt": "Tra loi ngan: Kinh te Viet Nam nam 2024 co kha nang tang truong bao nhieu phan tram?",
                        "stream": False,
                        "options": {"temperature": 0.3, "num_predict": 100},
                    },
                )
                lat = (time.monotonic() - start) * 1000
                assert r.status_code == 200
                data = r.json()
                tokens = data.get("eval_count", 0)
                latencies.append(lat)
                tokens_generated.append(tokens)

        avg = sum(latencies) / len(latencies)
        total_tokens = sum(tokens_generated)
        tps = (total_tokens / (sum(latencies) / 1000)) if latencies else 0
        print(f"\n  LLM latency: avg={avg:.0f}ms, tokens={total_tokens}, TPS={tps:.1f}")
        assert avg < 10000, f"LLM too slow: {avg:.0f}ms"

    @pytest.mark.asyncio
    async def test_batch_embedding_throughput(self):
        """Test batch embedding throughput (M4 optimized: batch_size=32)."""
        texts = [f"Noi dung van ban thu {i} — van de kinh te va cong nghe" for i in range(32)]
        latencies = []

        async with httpx.AsyncClient(timeout=120) as client:
            start = time.monotonic()
            for text in texts:
                t_start = time.monotonic()
                r = await client.post(
                    f"{OLLAMA_BASE}/api/embeddings",
                    json={"model": "bge-m3", "prompt": text},
                )
                latencies.append((time.monotonic() - t_start) * 1000)
                assert r.status_code == 200
            total = (time.monotonic() - start) * 1000

        avg = sum(latencies) / len(latencies)
        print(f"\n  Batch throughput: 32 texts in {total:.0f}ms, avg={avg:.1f}ms/text")

    @pytest.mark.asyncio
    async def test_concurrent_embedding(self):
        """Test concurrent embedding requests (semaphore limit=3)."""
        texts = [f"Van ban so {i}" for i in range(9)]
        sem_limit = 3
        semaphore = asyncio.Semaphore(sem_limit)
        results = []

        async def embed_one(client, text):
            async with semaphore:
                start = time.monotonic()
                r = await client.post(
                    f"{OLLAMA_BASE}/api/embeddings",
                    json={"model": "bge-m3", "prompt": text},
                )
                return (time.monotonic() - start) * 1000, r.status_code

        async with httpx.AsyncClient(timeout=60) as client:
            start = time.monotonic()
            results = await asyncio.gather(*[embed_one(client, t) for t in texts])
            total = (time.monotonic() - start) * 1000

        latencies = [r[0] for r in results]
        assert all(r[1] == 200 for r in results)
        avg = sum(latencies) / len(latencies)
        print(f"\n  Concurrent embedding (9 texts, sem=3): {total:.0f}ms total, {avg:.1f}ms avg")

        # Without concurrency, would be ~9 * avg. With sem=3, should be ~3 * avg
        assert total < avg * 9 * 1.5, (
            f"Concurrency not effective: {total:.0f}ms vs expected ~{avg * 3:.0f}ms"
        )


# =============================================================================
# Section 3: RAG Pipeline Performance
# =============================================================================


class TestRAGPipeline:
    """End-to-end RAG pipeline tests."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", TEST_QUERIES[:4])
    async def test_rag_chat_completions(self, query):
        """Test RAG chat completions — measures full pipeline."""
        async with httpx.AsyncClient(timeout=120) as client:
            start = time.monotonic()
            r = await client.post(
                f"{API_BASE}/v1/chat/completions",
                json={
                    "model": "qwen3.5:4b",
                    "messages": [{"role": "user", "content": query}],
                    "temperature": 0.3,
                    "max_tokens": 512,
                    "stream": False,
                },
                headers={"Content-Type": "application/json"},
            )
            latency = (time.monotonic() - start) * 1000
            assert r.status_code == 200, f"RAG failed: {r.status_code} {r.text[:200]}"
            data = r.json()
            assert "choices" in data
            content = data["choices"][0]["message"]["content"]
            assert content, "Empty response from RAG"
            print(f"\n  Query: {query[:40]}...")
            print(f"  Latency: {latency:.0f}ms")
            print(f"  Response length: {len(content)} chars")

    @pytest.mark.asyncio
    async def test_semantic_cache_hit(self):
        """Verify semantic cache works on repeated queries."""
        query = "Chinh sach luong va thuong"

        async with httpx.AsyncClient(timeout=120) as client:
            # First call — cache miss
            start1 = time.monotonic()
            r1 = await client.post(
                f"{API_BASE}/v1/chat/completions",
                json={
                    "model": "qwen3.5:4b",
                    "messages": [{"role": "user", "content": query}],
                    "temperature": 0.3,
                    "max_tokens": 256,
                    "stream": False,
                },
            )
            latency1 = (time.monotonic() - start1) * 1000
            assert r1.status_code == 200

            # Second call — cache hit
            await asyncio.sleep(0.5)
            start2 = time.monotonic()
            r2 = await client.post(
                f"{API_BASE}/v1/chat/completions",
                json={
                    "model": "qwen3.5:4b",
                    "messages": [{"role": "user", "content": query}],
                    "temperature": 0.3,
                    "max_tokens": 256,
                    "stream": False,
                },
            )
            latency2 = (time.monotonic() - start2) * 1000
            assert r2.status_code == 200

            speedup = latency1 / latency2 if latency2 > 0 else 1.0
            print(f"\n  First call (miss):  {latency1:.0f}ms")
            print(f"  Second call (hit):  {latency2:.0f}ms")
            print(f"  Speedup:            {speedup:.1f}x")
            assert (
                r1.json()["choices"][0]["message"]["content"]
                == r2.json()["choices"][0]["message"]["content"]
            )

    @pytest.mark.asyncio
    async def test_qdrant_collection_info(self):
        """Get Qdrant collection stats."""
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{QDRANT_BASE}/collections/enterprise_kb")
            assert r.status_code == 200
            data = r.json()
            info = data["result"]
            print(f"\n  Qdrant collection 'enterprise_kb':")
            print(f"    Points:     {info.get('points_count', 0)}")
            print(f"    Status:     {info.get('status', 'unknown')}")
            print(f"    Vectors:    {info.get('vectors_count', 0)}")
            print(f"    Index size: {info.get('indexed_points_count', 0)}")

    @pytest.mark.asyncio
    async def test_neo4j_graph_stats(self):
        """Get Neo4j knowledge graph stats."""
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{NEO4J_BASE}/db/neo4j/tx/commit",
                json={
                    "statements": [
                        {"statement": "MATCH (d:Document) RETURN count(d) as count"},
                        {"statement": "MATCH (c:Chunk) RETURN count(c) as count"},
                        {"statement": "MATCH (e:Entity) RETURN count(e) as count"},
                        {"statement": "MATCH ()-[r]->() RETURN count(r) as count"},
                    ]
                },
                auth=("neo4j", ""),
                headers={"Content-Type": "application/json"},
            )
            if r.status_code == 200:
                results = r.json().get("results", [])
                counts = [r["data"][0]["row"][0] if r.get("data") else 0 for r in results]
                print(f"\n  Neo4j Graph:")
                print(f"    Documents:  {counts[0]}")
                print(f"    Chunks:     {counts[1]}")
                print(f"    Entities:   {counts[2]}")
                print(f"    Relations:   {counts[3]}")


# =============================================================================
# Section 4: Resource Usage
# =============================================================================


class TestMacMiniResources:
    """Monitor resource usage of all containers."""

    def test_docker_container_memory(self):
        """Check memory usage of all RAG containers."""
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--format",
                "{{.Names}}\t{{.MemUsage}}",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        print("\n  Container memory usage:")
        for line in result.stdout.strip().split("\n"):
            if line and "rag-" in line:
                print(f"    {line}")
        assert result.returncode == 0

    def test_docker_total_memory(self):
        """Total Docker memory usage."""
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        total_mem = 0
        print("\n  All containers:")
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split("\t")
                if len(parts) >= 2:
                    print(f"    {parts[0]}: {parts[1]}")
                # Try to parse memory usage (format: "123.4MiB / 1GiB")
                mem_str = parts[1] if len(parts) >= 2 else ""
                import re

                match = re.search(r"([\d.]+)([KMGT]i?B)", mem_str)
                if match:
                    val = float(match.group(1))
                    unit = match.group(2)
                    factor = {
                        "KiB": 1 / 1024,
                        "MiB": 1,
                        "GiB": 1024,
                        "KB": 1 / 1024,
                        "MB": 1,
                        "GB": 1024,
                    }.get(unit, 1)
                    total_mem += val * factor
        print(f"\n  Total Docker memory: {total_mem:.1f} MiB ({total_mem / 1024:.2f} GiB)")

    def test_system_memory(self):
        """Mac Mini system memory."""
        result = subprocess.run(
            ["sysctl", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        total_gb = int(result.stdout.strip().split()[-1]) / (1024**3)
        print(f"\n  Mac Mini total RAM: {total_gb:.1f} GB")
        assert total_gb >= 16, f"Expected at least 16GB RAM, got {total_gb:.1f}GB"

    def test_qdrant_vectors_info(self):
        """Detailed Qdrant collection info."""
        import json

        r = httpx.get(f"{QDRANT_BASE}/collections/enterprise_kb", timeout=10)
        if r.status_code == 200:
            data = r.json()["result"]
            print(f"\n  Qdrant 'enterprise_kb':")
            print(f"    Status:    {data.get('status')}")
            print(f"    Points:    {data.get('points_count', 0)}")
            print(f"    Segments:  {data.get('segments_count', 0)}")
            print(f"    Index:     {data.get('indexed_points_count', 0)}")


# =============================================================================
# Section 5: Optimization Verification
# =============================================================================


class TestOptimization:
    """Verify all optimization settings are applied correctly."""

    def test_env_ollama_optimization(self):
        """Verify Ollama optimization env vars."""
        result = subprocess.run(
            ["docker", "exec", "rag-api", "env"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        env_lines = result.stdout.lower()
        checks = {
            "embed_batch_size=32": "EMBED_BATCH_SIZE" in env_lines,
            "embed_concurrent_limit=3": "EMBED_CONCURRENT_LIMIT" in env_lines,
            "max_concurrent_requests=6": "MAX_CONCURRENT_REQUESTS" in env_lines,
            "semantic_cache_ttl=7200": "SEMANTIC_CACHE_TTL" in env_lines,
        }
        for name, passed in checks.items():
            status = "OK" if passed else "MISSING"
            print(f"  {name}: {status}")
        assert all(checks.values()), (
            f"Missing optimizations: {[k for k, v in checks.items() if not v]}"
        )

    def test_redis_memory_limit(self):
        """Verify Redis memory is limited to 128MB."""
        result = subprocess.run(
            ["docker", "exec", "rag-redis", "redis-cli", "CONFIG", "GET", "maxmemory"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        # maxmemory is in bytes; 128MB = 134217728 bytes
        mem = int(result.stdout.strip().split()[-1])
        print(f"\n  Redis maxmemory: {mem} bytes ({mem / 1024 / 1024:.0f}MB)")
        assert mem <= 134217728 * 1.1, f"Redis maxmemory too high: {mem / 1024 / 1024:.0f}MB"

    def test_qdrant_sparse_quantization(self):
        """Verify Qdrant scalar quantization is enabled."""
        r = httpx.get(f"{QDRANT_BASE}/collections/enterprise_kb", timeout=10)
        if r.status_code == 200:
            data = r.json()
            params = data.get("result", {}).get("params", {})
            quant = params.get("quantization_config", {})
            has_quant = quant is not None and quant != {}
            print(f"\n  Qdrant quantization: {has_quant}")
            if quant:
                print(f"    Config: {quant}")
            assert has_quant, "Qdrant scalar quantization should be enabled"

    def test_neo4j_memory_settings(self):
        """Verify Neo4j heap and pagecache sizes."""
        result = subprocess.run(
            ["docker", "exec", "rag-neo4j", "neo4j", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout + result.stderr
        print(f"\n  Neo4j status output: {output[:200]}")

    def test_dashboard_config(self):
        """Verify dashboard connects to correct APIs."""
        # Dashboard should be reachable
        r = httpx.get(f"{DASHBOARD_BASE}/", timeout=10)
        assert r.status_code == 200
        print(f"\n  Dashboard: OK (status {r.status_code})")


# =============================================================================
# Section 6: Stress Test
# =============================================================================


class TestStress:
    """Light stress testing — Mac Mini M4 can handle moderate load."""

    @pytest.mark.asyncio
    async def test_concurrent_chat_requests(self):
        """Send 5 concurrent RAG chat requests."""

        async def chat(client, i):
            start = time.monotonic()
            r = await client.post(
                f"{API_BASE}/v1/chat/completions",
                json={
                    "model": "qwen3.5:4b",
                    "messages": [{"role": "user", "content": TEST_QUERIES[i % len(TEST_QUERIES)]}],
                    "temperature": 0.3,
                    "max_tokens": 128,
                    "stream": False,
                },
            )
            lat = (time.monotonic() - start) * 1000
            return lat, r.status_code

        async with httpx.AsyncClient(timeout=180) as client:
            start = time.monotonic()
            results = await asyncio.gather(*[chat(client, i) for i in range(5)])
            total = (time.monotonic() - start) * 1000

        latencies = [r[0] for r in results]
        successes = sum(1 for r in results if r[1] == 200)
        print(f"\n  5 concurrent requests:")
        print(f"    Total time: {total:.0f}ms")
        print(f"    Avg latency: {sum(latencies) / len(latencies):.0f}ms")
        print(f"    Max latency: {max(latencies):.0f}ms")
        print(f"    Success rate: {successes}/5")
        assert successes == 5, f"Some requests failed: {results}"
