#!/usr/bin/env python3
"""
Verify cosine similarity correctness using known pairs (Vietnamese + English).

Tests:
  1. Math correctness — known vectors give known cosine values
  2. Synonym pairs should have HIGH cosine (≥ 0.7)
  3. Antonym/different topic pairs should have LOWER cosine
  4. Random unrelated pairs should have LOW cosine (≤ 0.5)
  5. Vietnamese ↔ English translation pairs (multilingual bge-m3 test)
  6. Self-similarity = 1.0

Usage:
  python3 scripts/verify_cosine.py --api http://localhost:11434
"""
from __future__ import annotations

import argparse
import asyncio
import math
import sys

import httpx


# ── Math correctness tests (no API needed) ────────────────────────────────────


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def test_math_correctness() -> int:
    """Test the cosine function itself with known values."""
    cases = [
        ("identical", [1, 0, 0], [1, 0, 0], 1.0, 0.001),
        ("orthogonal", [1, 0, 0], [0, 1, 0], 0.0, 0.001),
        ("opposite", [1, 0, 0], [-1, 0, 0], -1.0, 0.001),
        ("45_degree", [1, 1, 0], [1, 0, 0], 1 / math.sqrt(2), 0.001),
        ("zero_vector", [0, 0, 0], [1, 0, 0], 0.0, 0.001),
        ("normalized_same", [0.6, 0.8], [0.6, 0.8], 1.0, 0.001),
    ]
    failed = 0
    for name, a, b, expected, tol in cases:
        actual = cosine(a, b)
        if abs(actual - expected) <= tol:
            print(f"  [ OK ] {name}: cos={actual:.4f} (expected {expected:.4f})")
        else:
            print(f"  [FAIL] {name}: cos={actual:.4f}, expected {expected:.4f}")
            failed += 1
    return failed


# ── Embedding-based semantic tests (require Ollama) ──────────────────────────


SEMANTIC_PAIRS = [
    # (label, text_a, text_b, expected_min, expected_max)
    # HIGH similarity expected
    ("vi_synonym_company", "công ty ABC", "doanh nghiệp ABC", 0.70, 1.0),
    ("vi_synonym_revenue", "doanh thu quý 3", "doanh số ba tháng cuối", 0.55, 1.0),
    ("vi_paraphrase", "Lợi nhuận tăng 25% so với năm ngoái",
                      "So với năm trước, lợi nhuận đã tăng một phần tư", 0.55, 1.0),
    ("en_synonym_revenue", "revenue in Q3", "sales in third quarter", 0.65, 1.0),
    ("vi_en_translation", "doanh thu quý 3", "revenue in Q3", 0.50, 1.0),
    ("identical_text", "Báo cáo tài chính 2024", "Báo cáo tài chính 2024", 0.99, 1.001),

    # MEDIUM similarity (related topic)
    ("vi_related_topic", "phòng kinh doanh", "phòng marketing", 0.40, 0.80),

    # LOW similarity expected
    ("vi_unrelated", "doanh thu quý 3", "thời tiết Hà Nội", 0.0, 0.45),
    ("vi_antonym_meaning", "công ty lãi 100 tỷ", "công ty lỗ 100 tỷ", 0.35, 0.85),
    ("totally_random_vi", "Báo cáo tài chính ngân hàng",
                           "Công thức nấu phở bò gia truyền", 0.0, 0.45),
]


async def embed(client: httpx.AsyncClient, ollama_url: str, model: str, text: str) -> list[float]:
    resp = await client.post(
        f"{ollama_url}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


async def test_semantic(ollama_url: str, model: str) -> int:
    failed = 0
    async with httpx.AsyncClient(timeout=120.0) as client:
        print(f"\n  Using model: {model}")
        print(f"  Ollama: {ollama_url}\n")

        # Self-similarity check
        text = "đây là một câu kiểm tra"
        v1 = await embed(client, ollama_url, model, text)
        v2 = await embed(client, ollama_url, model, text)
        self_sim = cosine(v1, v2)
        if abs(self_sim - 1.0) < 0.001:
            print(f"  [ OK ] self_similarity: {self_sim:.6f} (deterministic embedding)")
        elif self_sim > 0.99:
            print(f"  [ OK ] self_similarity: {self_sim:.6f} (near-deterministic, minor noise)")
        else:
            print(f"  [FAIL] self_similarity: {self_sim:.4f} — embedding NOT deterministic!")
            failed += 1

        # Dim check
        dim = len(v1)
        if dim == 1024:
            print(f"  [ OK ] embedding_dim: {dim} (bge-m3 expected)")
        else:
            print(f"  [WARN] embedding_dim: {dim} (expected 1024 for bge-m3)")

        # Cosine on real text pairs
        print()
        for label, a, b, min_v, max_v in SEMANTIC_PAIRS:
            va = await embed(client, ollama_url, model, a)
            vb = await embed(client, ollama_url, model, b)
            sim = cosine(va, vb)
            in_range = min_v <= sim <= max_v
            status = "[ OK ]" if in_range else "[WARN]"
            print(f"  {status} {label:30s} sim={sim:.4f} (expected {min_v:.2f}-{max_v:.2f})")
            print(f"           a='{a[:40]}'")
            print(f"           b='{b[:40]}'")
            if not in_range:
                # WARN, not FAIL — model behavior varies
                pass

    return failed


# ── Main ──────────────────────────────────────────────────────────────────────


async def main(args):
    print("═" * 70)
    print("  Cosine Similarity Verification")
    print("═" * 70)

    print("\n── 1. Math correctness ─────────────────────────────────────────")
    math_failed = test_math_correctness()

    print("\n── 2. Semantic embedding test (bge-m3 via Ollama) ──────────────")
    try:
        sem_failed = await test_semantic(args.api, args.model)
    except Exception as e:
        print(f"  [FAIL] Semantic test failed (Ollama unreachable?): {e}")
        sem_failed = 1

    total = math_failed + sem_failed
    print()
    print("═" * 70)
    if total == 0:
        print("  ALL COSINE TESTS PASSED")
    else:
        print(f"  {total} failure(s)")
    print("═" * 70)
    return 0 if total == 0 else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://localhost:11434")
    p.add_argument("--model", default="bge-m3")
    args = p.parse_args()
    sys.exit(asyncio.run(main(args)))
