#!/usr/bin/env python3
"""
Analyze whether retrieval is doing both LOCAL and GLOBAL optimization correctly.

Definitions:
  - LOCAL: top-K results come from same/few documents (precision within doc)
  - GLOBAL: top-K results span many documents (cross-doc reasoning)

A healthy pipeline should:
  - For factual queries: lean LOCAL (1-3 docs)
  - For analytical/comparison queries: balanced (3-7 docs)
  - For summarization queries: lean GLOBAL (5+ docs)

Method:
  - Run a set of probe queries
  - For each, look at top-10 retrieved chunks
  - Compute: doc_diversity (unique docs / total), entropy of source distribution
  - Compare against intent expectation

Usage:
  python3 scripts/local_vs_global_check.py --api http://localhost:8800 --tenant default
"""
from __future__ import annotations

import argparse
import asyncio
import math
import sys
from collections import Counter

import httpx


PROBE_QUERIES = [
    # (query, expected_intent, expected_pattern)
    # Pattern: "local" (≤ 3 docs), "balanced" (3-7 docs), "global" (≥ 5 docs)
    ("Doanh thu Quý 3 năm 2024 là bao nhiêu?", "factual", "local"),
    ("Số nhân viên trong phòng kinh doanh là bao nhiêu?", "factual", "local"),
    ("Mã số thuế của công ty là gì?", "factual", "local"),

    ("Tại sao chi phí marketing tăng năm 2024?", "analytical", "balanced"),
    ("Mối quan hệ giữa dự án X và phòng ban Y?", "analytical", "balanced"),
    ("Vai trò của giám đốc tài chính trong các quyết định đầu tư?", "analytical", "balanced"),

    ("Tóm tắt các rủi ro chính trong báo cáo thường niên.", "summarization", "global"),
    ("Tổng hợp chiến lược kinh doanh giai đoạn 2024-2026.", "summarization", "global"),
    ("Tóm tắt các thay đổi quy định nội bộ trong năm qua.", "summarization", "global"),

    ("So sánh kết quả kinh doanh 2023 và 2024.", "comparison", "balanced"),
]


def shannon_entropy(counter: Counter) -> float:
    """Entropy of source distribution in bits. Higher = more global."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def classify_pattern(doc_count: int, total: int, entropy: float) -> str:
    if total == 0:
        return "empty"
    diversity = doc_count / total
    if diversity < 0.30 and doc_count <= 3:
        return "local"
    if diversity < 0.60 and doc_count <= 7:
        return "balanced"
    return "global"


async def query_v3(client: httpx.AsyncClient, api: str, query: str, tenant: str) -> dict:
    resp = await client.post(
        f"{api}/api/v3/chat",
        json={"query": query, "tenant_id": tenant, "include_sources": True, "max_retries": 0},
        timeout=180.0,
    )
    resp.raise_for_status()
    return resp.json()


async def main(api: str, tenant: str) -> int:
    print("═" * 70)
    print("  Local vs Global Retrieval Pattern Analysis")
    print("═" * 70)
    print(f"  API: {api}   Tenant: {tenant}")

    mismatches = 0
    rows = []

    async with httpx.AsyncClient(timeout=300.0) as client:
        for query, expected_intent, expected_pattern in PROBE_QUERIES:
            try:
                result = await query_v3(client, api, query, tenant)
            except Exception as e:
                print(f"\n  Query: {query}")
                print(f"  [ERR] {e}")
                continue

            sources = result.get("sources", [])
            intent_actual = result.get("intent")

            # Extract source counts
            doc_counter = Counter()
            format_counter = Counter()
            level_counter = Counter()
            for s in sources:
                meta = s if "source" in s else (s.get("metadata") or {})
                doc_counter[s.get("source", "?")] += 1
                format_counter[s.get("format", "?")] += 1
                level_counter[s.get("chunk_level", "?")] += 1

            total = sum(doc_counter.values())
            doc_count = len(doc_counter)
            entropy = shannon_entropy(doc_counter)
            pattern = classify_pattern(doc_count, total, entropy)

            intent_ok = intent_actual == expected_intent
            pattern_ok = pattern == expected_pattern

            print(f"\n  ── Query: {query[:60]}")
            print(f"     Expected: intent={expected_intent}, pattern={expected_pattern}")
            print(f"     Actual:   intent={intent_actual} {'✓' if intent_ok else '✗'}, "
                  f"pattern={pattern} {'✓' if pattern_ok else '✗'}")
            print(f"     Sources: {total} chunks across {doc_count} docs, entropy={entropy:.2f}")
            print(f"     Doc dist: {dict(doc_counter.most_common(5))}")
            print(f"     Format dist: {dict(format_counter)}")
            print(f"     Level dist: {dict(level_counter)}")

            if not pattern_ok:
                mismatches += 1

            rows.append({
                "query": query,
                "expected_intent": expected_intent,
                "actual_intent": intent_actual,
                "expected_pattern": expected_pattern,
                "actual_pattern": pattern,
                "doc_count": doc_count,
                "total_chunks": total,
                "entropy": entropy,
            })

    # Summary
    print()
    print("═" * 70)
    print("  SUMMARY")
    print("═" * 70)
    print(f"  Queries tested: {len(rows)}")
    print(f"  Pattern mismatches: {mismatches}/{len(rows)}")

    by_expected = Counter(r["expected_pattern"] for r in rows)
    by_actual = Counter(r["actual_pattern"] for r in rows)
    print(f"  Expected dist: {dict(by_expected)}")
    print(f"  Actual dist:   {dict(by_actual)}")

    # Insights
    print()
    print("  Insights:")
    actual_local = [r for r in rows if r["actual_pattern"] == "local"]
    if actual_local:
        avg_docs = sum(r["doc_count"] for r in actual_local) / len(actual_local)
        print(f"    - Local queries: avg {avg_docs:.1f} unique docs in top-K")
    actual_global = [r for r in rows if r["actual_pattern"] == "global"]
    if actual_global:
        avg_docs = sum(r["doc_count"] for r in actual_global) / len(actual_global)
        print(f"    - Global queries: avg {avg_docs:.1f} unique docs in top-K")

    if mismatches == 0:
        print("\n  Retrieval pattern matches intent expectations.")
    else:
        print(f"\n  {mismatches} mismatch(es). This MIGHT be OK depending on data —")
        print("  if your KB only has 1-2 docs, all queries will look local.")
        print("  Run with a diverse KB (≥ 20 docs across multiple topics) for real signal.")

    return 0 if mismatches <= len(rows) // 3 else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://localhost:8800")
    p.add_argument("--tenant", default="default")
    args = p.parse_args()
    sys.exit(asyncio.run(main(args.api, args.tenant)))
