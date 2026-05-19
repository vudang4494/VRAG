#!/usr/bin/env python3
"""Quick 10-query benchmark test from vi_benchmark_new10.json."""
from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx

API = "http://localhost:8800"
BENCHMARK = "eval/datasets/vi_benchmark_new10.json"
TENANT = "rag51"


REFUSAL_PATTERNS = [
    r"không có đủ thông tin",
    r"không tìm thấy",
    r"không thể trả lời",
    r"không có thông tin",
    r"i don'?t have enough information",
    r"i don'?t know",
    r"từ chối",
    r"cannot answer",
]


def is_refused(answer: str) -> bool:
    a = answer.lower()
    import re
    return any(re.search(p, a) for p in REFUSAL_PATTERNS)


async def run_query(client: httpx.AsyncClient, q: dict, tenant: str) -> dict:
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{API}/api/v3/chat",
            json={
                "query": q["query"],
                "tenant_id": tenant,
                "max_retries": 0,
                "include_sources": True,
                "disable_validation": True,
                "disable_ood": True,
            },
            timeout=300.0,
        )
        resp.raise_for_status()
        d = resp.json()
        elapsed = time.monotonic() - t0
        answer = d.get("answer", "")
        refused = is_refused(answer)
        routing = d.get("routing", {})
        latency = d.get("latency_breakdown_ms", {})
        return {
            "q_id": q.get("q_id", "unknown"),
            "category": q.get("category", "unknown"),
            "query": q["query"],
            "answer": answer,
            "answer_preview": answer[:200] if answer else "(empty)",
            "latency_s": elapsed,
            "latency_breakdown": latency,
            "refused": refused,
            "query_type": routing.get("query_type", "unknown"),
            "react_used": routing.get("react_used", False),
            "intent": d.get("intent", "unknown"),
            "error": None,
        }
    except Exception as e:
        return {
            "q_id": q.get("q_id", "unknown"),
            "category": q.get("category", "unknown"),
            "query": q["query"],
            "answer": "",
            "answer_preview": "",
            "latency_s": time.monotonic() - t0,
            "latency_breakdown": {},
            "refused": False,
            "query_type": "error",
            "react_used": False,
            "intent": "error",
            "error": str(e)[:200],
        }


async def main():
    with open(BENCHMARK) as f:
        bm = json.load(f)
    queries = bm["queries"]

    print(f"\n{'='*80}")
    print(f"QUICK BENCHMARK — 10 queries — qwen3.5:9b")
    print(f"{'='*80}\n")

    results = []
    async with httpx.AsyncClient(timeout=360.0) as client:
        for i, q in enumerate(queries, 1):
            q_id = q.get("q_id", f"q{i}")
            cat = q.get("category", "unknown")
            qtext = q["query"][:60]

            print(f"[{i:2d}/10] {q_id:<4} ({cat:<18}): {qtext}...")
            sys.stdout.flush()

            r = await run_query(client, q, TENANT)
            results.append(r)

            status = "ERR" if r["error"] else "OK"
            lat = r["latency_s"]
            ref = "REFUSE" if r["refused"] else "OK"
            qt = r["query_type"][:10]
            react = "[R]" if r["react_used"] else "[S]"
            total_ms = r["latency_breakdown"].get("total_ms", 0)

            print(f"        -> {status} | {lat:5.1f}s | ref={ref} | {qt} {react} | total_ms={total_ms:.0f}")
            if r["error"]:
                print(f"        ERROR: {r['error']}")
            else:
                print(f"        Answer preview: {r['answer_preview'][:120]}")
            print()
            sys.stdout.flush()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    ok = [r for r in results if not r["error"]]
    errs = [r for r in results if r["error"]]
    all_lats = sorted([r["latency_s"] for r in ok])
    print(f"  Total: {len(results)} | OK: {len(ok)} | Errors: {len(errs)}")
    if ok:
        p50 = all_lats[len(all_lats) // 2]
        p95 = all_lats[int(len(all_lats) * 0.95)]
        avg = sum(all_lats) / len(all_lats)
        print(f"  Latency: avg={avg:.1f}s p50={p50:.1f}s p95={p95:.1f}s")
        refused = sum(1 for r in ok if r["refused"])
        print(f"  Refused: {refused}/{len(ok)}")
        print()
        print(f"  {'QID':<6} {'CATEGORY':<18} {'TYPE':<12} {'LAT':>6} {'REF':>4}")
        print(f"  {'-'*60}")
        for r in ok:
            print(f"  {r['q_id']:<6} {r['category']:<18} {r['query_type']:<12} {r['latency_s']:>5.1f}s {'YES' if r['refused'] else 'no':>4}")


if __name__ == "__main__":
    asyncio.run(main())
