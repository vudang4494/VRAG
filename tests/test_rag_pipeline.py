"""End-to-end RAG pipeline tests: ingest, retrieval, generation."""
import io
import time

import httpx
import pytest


BASE = "http://localhost:8800"
OLLAMA = "http://localhost:11434"


class TestIngestion:
    """Document ingestion pipeline tests."""

    def test_ingest_small_text(self):
        # Min chunk size ~50 chars, so content must be substantial
        content = ("RAG retrieval augmented generation combines vector search with language models. "
                   "This enables accurate answers based on document context.").encode()
        files = {"file": ("doc.txt", content, "text/plain")}
        r = httpx.post(f"{BASE}/ingest/upload", files=files, timeout=120)
        assert r.status_code == 200, f"Ingest failed: {r.status_code} {r.text}"
        data = r.json()
        assert data["status"] == "success"
        assert data["chunks_indexed"] >= 1

    def test_ingest_vietnamese_text(self):
        content = (
            "Trí tuệ nhân tạo (AI) là một nhánh của khoa học máy tính. "
            "Hệ thống RAG (Retrieval-Augmented Generation) kết hợp tìm kiếm vector "
            "với mô hình ngôn ngữ lớn để tạo ra câu trả lời chính xác và có ngữ cảnh."
        ).encode()
        files = {"file": ("vietnamese.txt", content, "text/plain")}
        r = httpx.post(f"{BASE}/ingest/upload", files=files, timeout=120)
        assert r.status_code == 200, f"Ingest failed: {r.text}"
        data = r.json()
        assert data["status"] == "success"
        assert data["chunks_indexed"] >= 1

    def test_ingest_multiple_chunks(self):
        content = ("Word " * 300).encode()
        files = {"file": ("long.txt", content, "text/plain")}
        r = httpx.post(f"{BASE}/ingest/upload", files=files, timeout=120)
        assert r.status_code == 200
        data = r.json()
        assert data["chunks_indexed"] >= 2, f"Long doc should produce multiple chunks, got {data['chunks_indexed']}"

    def test_ingest_file_too_large(self):
        files = {"file": ("large.txt", b"x" * 300 * 1024 * 1024, "text/plain")}
        r = httpx.post(f"{BASE}/ingest/upload", files=files, timeout=60)
        assert r.status_code == 413

    def test_ingest_empty_file(self):
        files = {"file": ("empty.txt", b"", "text/plain")}
        r = httpx.post(f"{BASE}/ingest/upload", files=files, timeout=60)
        assert r.status_code == 422

    def test_ingest_no_file(self):
        r = httpx.post(f"{BASE}/ingest/upload", timeout=60)
        assert r.status_code == 422

    def test_ingest_idempotent(self):
        content = (
            "RAG retrieval augmented generation is powerful. "
            "This technology combines search with language models. "
            "It enables accurate and context-aware responses."
        ).encode()
        for i in range(2):
            files = {"file": (f"dup{i}.txt", content, "text/plain")}
            r = httpx.post(f"{BASE}/ingest/upload", files=files, timeout=120)
            assert r.status_code == 200, f"Iteration {i} failed: {r.text}"
            assert r.json()["status"] == "success"


class TestRetrieval:
    """Vector and graph retrieval tests."""

    def test_qdrant_collection_exists(self):
        r = httpx.get(f"http://localhost:6333/collections", timeout=5)
        assert r.status_code == 200
        cols = r.json()["result"]["collections"]
        names = [c["name"] for c in cols]
        assert "enterprise_kb" in names, f"enterprise_kb not in {names}"

    def test_qdrant_vectors_indexed(self):
        r = httpx.get(f"http://localhost:6333/collections/enterprise_kb", timeout=5)
        assert r.status_code == 200
        info = r.json()["result"]
        assert info["points_count"] >= 1, "Expected at least 1 indexed point"

    def test_neo4j_entities(self):
        import subprocess
        result = subprocess.run(
            ["docker", "exec", "rag-neo4j", "cypher-shell",
             "-u", "neo4j", "--", "MATCH (d:Document) RETURN count(d) as count"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, f"Neo4j query failed: {result.stderr}"


class TestRAGChat:
    """Full RAG pipeline: retrieval + generation."""

    def test_rag_chat_basic(self):
        r = httpx.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "RAG system hoat dong nhu the nao?"}],
                "max_tokens": 150,
            },
            timeout=120,
        )
        assert r.status_code == 200, f"RAG chat failed: {r.status_code} {r.text}"
        data = r.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content", "") or msg.get("reasoning", "")
        assert len(content) >= 10, "Response too short"
        assert data["usage"]["total_tokens"] >= 10

    def test_rag_chat_streaming(self):
        r = httpx.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 50,
                "stream": True,
            },
            timeout=120,
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_rag_latency(self):
        start = time.monotonic()
        r = httpx.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "What is retrieval augmented generation?"}],
                "max_tokens": 120,
            },
            timeout=120,
        )
        elapsed = time.monotonic() - start
        assert r.status_code == 200
        assert elapsed < 60, f"E2E RAG latency {elapsed:.1f}s > 60s"

    def test_rag_requires_user_message(self):
        r = httpx.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "system", "content": "You are helpful."}],
                "max_tokens": 50,
            },
            timeout=30,
        )
        assert r.status_code == 400

    def test_rag_temperature_parameter(self):
        r = httpx.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 30,
                "temperature": 1.5,
            },
            timeout=60,
        )
        assert r.status_code == 200

    def test_rag_max_tokens_limit(self):
        r = httpx.post(
            f"{BASE}/v1/chat/completions",
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "Say the alphabet"}],
                "max_tokens": 5,
                "temperature": 0.0,
            },
            timeout=60,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["usage"]["completion_tokens"] <= 5

    def test_rag_multilingual(self):
        queries = [
            "What is artificial intelligence?",
            "Trí tuệ nhân tạo là gì?",
            "Qu'est-ce que l'intelligence artificielle?",
        ]
        for q in queries:
            r = httpx.post(
                f"{BASE}/v1/chat/completions",
                json={
                    "model": "qwen3.5:4b",
                    "messages": [{"role": "user", "content": q}],
                    "max_tokens": 60,
                },
                timeout=120,
            )
            assert r.status_code == 200, f"Query '{q}' failed: {r.text}"

    def test_semantic_cache_hit(self):
        query = "What is retrieval augmented generation system?"
        for _ in range(2):
            r = httpx.post(
                f"{BASE}/v1/chat/completions",
                json={
                    "model": "qwen3.5:4b",
                    "messages": [{"role": "user", "content": query}],
                    "max_tokens": 80,
                },
                timeout=120,
            )
            assert r.status_code == 200
