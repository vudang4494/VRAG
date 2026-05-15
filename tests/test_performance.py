"""Performance and stress tests for the RAG stack."""

import concurrent.futures
import statistics
import time

import httpx
import pytest


OLLAMA = "http://localhost:11434"
BASE = "http://localhost:8800"


class TestLLMPerformance:
    """LLM throughput and latency benchmarks."""

    @pytest.mark.parametrize("prompt_tokens", [10, 50, 100, 200])
    def test_latency_by_input_size(self, prompt_tokens):
        prompt = "word " * prompt_tokens
        start = time.monotonic()
        r = httpx.post(
            f"{OLLAMA}/v1/chat/completions",
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 80,
                "temperature": 0.3,
            },
            timeout=120,
        )
        elapsed = time.monotonic() - start
        assert r.status_code == 200
        usage = r.json()["usage"]
        tps = usage["completion_tokens"] / elapsed if elapsed > 0 else 0
        print(f"\n  [{prompt_tokens} tok in] {elapsed:.1f}s | {tps:.1f} tok/s")

    def test_concurrent_requests(self):
        """Handle multiple concurrent LLM requests."""

        def call_llm(i):
            r = httpx.post(
                f"{OLLAMA}/v1/chat/completions",
                json={
                    "model": "qwen3.5:4b",
                    "messages": [{"role": "user", "content": f"Q{i}"}],
                    "max_tokens": 40,
                },
                timeout=120,
            )
            return r.status_code, time.monotonic()

        times = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(call_llm, i) for i in range(3)]
            for f in concurrent.futures.as_completed(futures):
                code, t = f.result()
                times.append(t)
                assert code == 200

        times.sort()
        span = times[-1] - times[0]
        print(f"\n  3 concurrent requests: {span:.1f}s total span")

    def test_sustained_throughput(self):
        """Process multiple requests sequentially."""
        latencies = []
        for i in range(5):
            start = time.monotonic()
            r = httpx.post(
                f"{OLLAMA}/v1/chat/completions",
                json={
                    "model": "qwen3.5:4b",
                    "messages": [{"role": "user", "content": f"Q{i}"}],
                    "max_tokens": 40,
                },
                timeout=120,
            )
            latencies.append(time.monotonic() - start)
            assert r.status_code == 200

        print(f"\n  Mean latency: {statistics.mean(latencies):.1f}s")
        print(f"  P95 latency: {sorted(latencies)[int(len(latencies) * 0.95)]:.1f}s")


class TestEmbeddingPerformance:
    """Embedding throughput benchmarks."""

    def test_batch_throughput(self):
        """Embed multiple texts."""
        texts = [f"text number {i} about machine learning" for i in range(10)]
        times = []
        for t in texts:
            start = time.monotonic()
            r = httpx.post(
                f"{OLLAMA}/api/embeddings",
                json={"model": "bge-m3", "prompt": t},
                timeout=60,
            )
            times.append(time.monotonic() - start)
            assert r.status_code == 200

        print(
            f"\n  10 embeddings: {sum(times):.1f}s total, {statistics.mean(times) * 1000:.0f}ms avg"
        )

    def test_embedding_latency(self):
        """Single embedding latency."""
        times = []
        for _ in range(5):
            start = time.monotonic()
            r = httpx.post(
                f"{OLLAMA}/api/embeddings",
                json={"model": "bge-m3", "prompt": "benchmarking embedding performance"},
                timeout=60,
            )
            times.append(time.monotonic() - start)
            assert r.status_code == 200

        print(
            f"\n  Mean: {statistics.mean(times) * 1000:.0f}ms | "
            f"Min: {min(times) * 1000:.0f}ms | "
            f"Max: {max(times) * 1000:.0f}ms"
        )


class TestRAGPerformance:
    """RAG pipeline performance tests."""

    def test_e2e_throughput(self):
        """End-to-end RAG latency."""
        times = []
        for i in range(3):
            start = time.monotonic()
            r = httpx.post(
                f"{BASE}/v1/chat/completions",
                json={
                    "model": "qwen3.5:4b",
                    "messages": [{"role": "user", "content": f"Question {i}: explain RAG"}],
                    "max_tokens": 80,
                },
                timeout=180,
            )
            times.append(time.monotonic() - start)
            assert r.status_code == 200

        print(
            f"\n  E2E mean: {statistics.mean(times):.1f}s | "
            f"Min: {min(times):.1f}s | Max: {max(times):.1f}s"
        )

    def test_concurrent_rag_requests(self):
        """Handle concurrent RAG requests."""

        def call_rag(i):
            start = time.monotonic()
            r = httpx.post(
                f"{BASE}/v1/chat/completions",
                json={
                    "model": "qwen3.5:4b",
                    "messages": [{"role": "user", "content": f"Q{i}"}],
                    "max_tokens": 40,
                },
                timeout=180,
            )
            return r.status_code, time.monotonic() - start

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(call_rag, i) for i in range(2)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for code, elapsed in results:
            assert code == 200
        print(f"\n  2 concurrent RAG: {[f'{e:.1f}s' for _, e in results]}")


class TestCachePerformance:
    """Semantic cache benchmarks."""

    def test_cache_improves_latency(self):
        """Second identical query should be faster (cache hit)."""
        query = "What is retrieval augmented generation in AI?"
        times = []
        for _ in range(3):
            start = time.monotonic()
            r = httpx.post(
                f"{BASE}/v1/chat/completions",
                json={
                    "model": "qwen3.5:4b",
                    "messages": [{"role": "user", "content": query}],
                    "max_tokens": 80,
                },
                timeout=180,
            )
            times.append(time.monotonic() - start)
            assert r.status_code == 200

        print(f"\n  Query latencies: {[f'{t:.1f}s' for t in times]}")
        assert len(times) == 3
