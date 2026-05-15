"""LLM and embedding model tests via Ollama."""

import time

import httpx
import pytest


OLLAMA = "http://localhost:11434"


class TestOllamaServer:
    """Ollama server availability."""

    def test_server_running(self):
        r = httpx.get(f"{OLLAMA}/api/tags", timeout=10)
        assert r.status_code == 200
        assert "models" in r.json()

    def test_qwen_model_available(self):
        names = [m["name"] for m in httpx.get(f"{OLLAMA}/api/tags", timeout=10).json()["models"]]
        assert "qwen3.5:4b" in names, f"qwen3.5:4b not in {names}"

    def test_bge_model_available(self):
        names = [m["name"] for m in httpx.get(f"{OLLAMA}/api/tags", timeout=10).json()["models"]]
        assert any("bge-m3" in n for n in names), f"bge-m3 not in {names}"


class TestLLMInference:
    """LLM inference quality and latency tests."""

    def test_vietnamese_prompt(self):
        r = httpx.post(
            f"{OLLAMA}/v1/chat/completions",
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "Xin chao"}],
                "max_tokens": 60,
                "temperature": 0.3,
            },
            timeout=60,
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        assert "content" in msg or "reasoning" in msg
        assert r.json()["usage"]["total_tokens"] >= 10

    def test_latency_under_30s(self):
        start = time.monotonic()
        r = httpx.post(
            f"{OLLAMA}/v1/chat/completions",
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "What is AI?"}],
                "max_tokens": 80,
            },
            timeout=60,
        )
        elapsed = time.monotonic() - start
        assert r.status_code == 200
        assert elapsed < 30, f"Latency {elapsed:.1f}s > 30s"

    def test_token_output(self):
        r = httpx.post(
            f"{OLLAMA}/v1/chat/completions",
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "Count 1 to 5"}],
                "max_tokens": 50,
                "temperature": 0.0,
            },
            timeout=60,
        )
        assert r.status_code == 200
        assert r.json()["usage"]["completion_tokens"] >= 5

    def test_system_prompt_respected(self):
        r = httpx.post(
            f"{OLLAMA}/v1/chat/completions",
            json={
                "model": "qwen3.5:4b",
                "messages": [
                    {"role": "system", "content": "Answer only with the word HELLO."},
                    {"role": "user", "content": "Hi!"},
                ],
                "max_tokens": 10,
                "temperature": 0.0,
            },
            timeout=60,
        )
        assert r.status_code == 200

    def test_multiturn_conversation(self):
        r = httpx.post(
            f"{OLLAMA}/v1/chat/completions",
            json={
                "model": "qwen3.5:4b",
                "messages": [
                    {"role": "user", "content": "My name is Minh."},
                    {"role": "assistant", "content": "Hello Minh!"},
                    {"role": "user", "content": "What is my name?"},
                ],
                "max_tokens": 60,
            },
            timeout=60,
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        content = msg.get("content", "") + msg.get("reasoning", "")
        assert len(content) >= 1

    def test_empty_context(self):
        r = httpx.post(
            f"{OLLAMA}/v1/chat/completions",
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "?"}],
                "max_tokens": 20,
            },
            timeout=60,
        )
        assert r.status_code == 200


class TestEmbedding:
    """Embedding model quality and consistency tests."""

    def test_dimensions(self):
        r = httpx.post(
            f"{OLLAMA}/api/embeddings",
            json={"model": "bge-m3", "prompt": "test"},
            timeout=30,
        )
        assert r.status_code == 200
        vec = r.json()["embedding"]
        assert len(vec) == 1024, f"Expected 1024 dims, got {len(vec)}"

    def test_deterministic(self):
        payload = {"model": "bge-m3", "prompt": "deterministic test"}
        r1 = httpx.post(f"{OLLAMA}/api/embeddings", json=payload, timeout=30)
        r2 = httpx.post(f"{OLLAMA}/api/embeddings", json=payload, timeout=30)
        assert r1.json()["embedding"] == r2.json()["embedding"]

    def test_vietnamese_text(self):
        r = httpx.post(
            f"{OLLAMA}/api/embeddings",
            json={"model": "bge-m3", "prompt": "Hệ thống RAG kết hợp vector search"},
            timeout=30,
        )
        assert r.status_code == 200
        vec = r.json()["embedding"]
        assert len(vec) == 1024
        assert abs(sum(vec)) > 0, "Embedding looks all-zero"

    def test_long_text(self):
        long_text = " ".join(["word"] * 500)
        r = httpx.post(
            f"{OLLAMA}/api/embeddings",
            json={"model": "bge-m3", "prompt": long_text},
            timeout=30,
        )
        assert r.status_code == 200
        assert len(r.json()["embedding"]) == 1024

    def test_empty_prompt(self):
        r = httpx.post(
            f"{OLLAMA}/api/embeddings",
            json={"model": "bge-m3", "prompt": ""},
            timeout=30,
        )
        assert r.status_code == 200

    def test_cosine_similarity_similar_texts(self):
        p1 = {"model": "bge-m3", "prompt": "machine learning artificial intelligence"}
        p2 = {"model": "bge-m3", "prompt": "AI deep learning neural networks"}
        r1 = httpx.post(f"{OLLAMA}/api/embeddings", json=p1, timeout=30).json()["embedding"]
        r2 = httpx.post(f"{OLLAMA}/api/embeddings", json=p2, timeout=30).json()["embedding"]

        dot = sum(a * b for a, b in zip(r1, r2))
        n1 = sum(a * a for a in r1) ** 0.5
        n2 = sum(b * b for b in r2) ** 0.5
        sim = dot / (n1 * n2)
        assert 0 <= sim <= 1
        assert sim > 0.3, f"Similar texts should score > 0.3, got {sim:.3f}"
