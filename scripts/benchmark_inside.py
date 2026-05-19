#!/usr/bin/env python3
"""
Benchmark runner — runs inside the Docker container via docker exec.
Bypasses Docker→Ollama networking issue by using localhost inside container.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import httpx

BENCHMARK = "eval/datasets/vi_benchmark_v2.json"
API = "http://localhost:8800"
API_CHAT = "http://localhost:8800/api/v3/chat"

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
    return any(re.search(p, a) for p in REFUSAL_PATTERNS)


async def kw_hit_ratio(answer: str, keywords: list[str]) -> tuple[float, list[dict]]:
    if not keywords:
        return 1.0, []
    answer_lower = answer.lower()
    breakdown = []
    hits = 0
    for kw in keywords:
        hit = kw.lower() in answer_lower
        if hit:
            hits += 1
        breakdown.append({"keyword": kw, "hit": hit})
    return hits / len(keywords), breakdown


async def check_sources(sources: list[dict], expected_docs: list[str]) -> tuple[int, list[str]]:
    found = []
    for s in sources:
        src = s.get("source", "") or ""
        for exp in expected_docs:
            if exp.lower() in src.lower() and exp not in found:
                found.append(exp)
    return len(found), found


async def run_query(client: httpx.AsyncClient, q: dict) -> dict:
    t0 = time.monotonic()
    try:
        resp = await client.post(
            API_CHAT,
            json={
                "query": q["query"],
                "tenant_id": "rag51",
                "max_retries": 0,
                "include_sources": True,
                "disable_validation": True,
                "disable_ood": True,
            },
            timeout=600.0,
        )
        resp.raise_for_status()
        d = resp.json()
        elapsed = time.monotonic() - t0
    except Exception as e:
        return {
            "q_id": q.get("q_id", q.get("id", "unknown")),
            "category": q.get("category", "unknown"),
            "query": q["query"],
            "answer": "",
            "sources": [],
            "latency_ms": (time.monotonic() - t0) * 1000,
            "doc_found": 0,
            "doc_expected": len(set(q.get("expected_docs", []))),
            "doc_recall": 0.0,
            "kw_hit": 0.0,
            "kw_breakdown": [],
            "refused": False,
            "expect_refusal": q.get("expect_refusal", False),
            "refusal_ok": False,
            "query_type": "unknown",
            "react_used": False,
            "error": str(e)[:200],
        }

    answer = d.get("answer", "")
    sources = d.get("sources", [])
    refused = is_refused(answer)
    expect_refusal = q.get("expect_refusal", False)

    expected_docs = q.get("expected_docs", [])
    doc_found, _ = await check_sources(sources, expected_docs)
    doc_expected = len(set(expected_docs))
    doc_recall = doc_found / doc_expected if doc_expected > 0 else 0.0
    refusal_ok = refused == expect_refusal

    keywords = q.get("expected_keywords", [])
    kw_hit, kw_breakdown = await kw_hit_ratio(answer, keywords)

    routing = d.get("routing", {})
    query_type = routing.get("query_type", "unknown")
    react_used = routing.get("react_used", False)

    return {
        "q_id": q.get("q_id", q.get("id", "unknown")),
        "category": q.get("category", "unknown"),
        "query": q["query"],
        "answer": answer,
        "sources": sources,
        "latency_ms": elapsed * 1000,
        "doc_found": doc_found,
        "doc_expected": doc_expected,
        "doc_recall": doc_recall,
        "kw_hit": kw_hit,
        "kw_breakdown": kw_breakdown,
        "refused": refused,
        "expect_refusal": expect_refusal,
        "refusal_ok": refusal_ok,
        "query_type": query_type,
        "react_used": react_used,
        "error": None,
    }


async def main():
    bm_path = Path("/app/" + BENCHMARK)
    if not bm_path.exists():
        # Try from current dir
        bm_path = Path(BENCHMARK)
    with open(bm_path) as f:
        bm = json.load(f)
    queries = bm["queries"]

    n = len(queries)
    print(f"\n{'='*90}")
    print(f"BENCHMARK — vi_benchmark_v2 (42 queries)")
    print(f"  Tenant: rag51 | Model: qwen3.5:9b")
    print(f"  Features: validation=OFF, ood=OFF, retries=0")
    print(f"{'='*90}\n")

    started = time.monotonic()
    results = []

    async with httpx.AsyncClient(timeout=600.0) as client:
        for i, q in enumerate(queries, 1):
            q_id = q.get("q_id", q.get("id", f"q{i}"))
            cat = q.get("category", "unknown")
            qt = q.get("query", "")[:60]
            print(f"[{i:2d}/{n}] {q_id:<12} {cat:<20} {qt}...")

            r = await run_query(client, q)
            results.append(r)

            lat = r["latency_ms"] / 1000
            recall = r["doc_recall"]
            kw = r["kw_hit"]
            ok = "OK" if not r["error"] else "ERR"
            refused_s = "REF" if r["refused"] else ""
            print(f"  -> {ok} lat={lat:.1f}s recall={recall:.0%} kw={kw:.0%} {refused_s}")
            if r.get("error"):
                print(f"  -> ERROR: {r['error']}")

    total = time.monotonic() - started

    # Aggregate
    latencies = sorted([r["latency_ms"] for r in results])
    p50 = latencies[len(latencies) // 2] / 1000
    p95 = latencies[int(len(latencies) * 0.95)] / 1000
    ok_results = [r for r in results if not r["error"]]
    err_results = [r for r in results if r["error"]]

    avg_recall = sum(r["doc_recall"] for r in ok_results) / len(ok_results) if ok_results else 0
    avg_kw = sum(r["kw_hit"] for r in ok_results) / len(ok_results) if ok_results else 0
    avg_refusal = sum(r["refusal_ok"] for r in results) / len(results) if results else 0

    # Per category
    cats: dict[str, list[dict]] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)

    print(f"\n{'='*90}")
    print(f"AGGREGATE RESULTS")
    print(f"{'='*90}")
    print(f"  Queries: {n} total, {len(ok_results)} ok, {len(err_results)} errors")
    print(f"  Avg Doc Recall:   {avg_recall:.1%}")
    print(f"  Avg Keyword Hit:  {avg_kw:.1%}")
    print(f"  Avg Refusal Acc:  {avg_refusal:.1%}")
    print(f"  p50 Latency:      {p50:.1f}s")
    print(f"  p95 Latency:      {p95:.1f}s")
    print(f"  Total Time:       {total/60:.1f}min")
    print(f"\nPer-Category:")
    for cat, rs in sorted(cats.items()):
        ok_rs = [r for r in rs if not r["error"]]
        if not ok_rs:
            continue
        avg_r = sum(r["doc_recall"] for r in ok_rs) / len(ok_rs)
        avg_k = sum(r["kw_hit"] for r in ok_rs) / len(ok_rs)
        avg_ref = sum(r["refusal_ok"] for r in rs) / len(rs)
        avg_lat = sum(r["latency_ms"] for r in ok_rs) / len(ok_rs) / 1000
        react_n = sum(1 for r in ok_rs if r["react_used"])
        print(f"  {cat:<20} N={len(ok_rs):2d} recall={avg_r:.0%} kw={avg_k:.0%} ref={avg_ref:.0%} lat={avg_lat:.1f}s react={react_n}/{len(ok_rs)}")

    # Save results
    out = Path("/app/eval/results/benchmark_v2_caitien.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "results": results,
            "summary": {
                "n": n, "n_ok": len(ok_results), "n_err": len(err_results),
                "avg_doc_recall": avg_recall, "avg_kw_hit": avg_kw,
                "avg_refusal_acc": avg_refusal,
                "p50_latency_s": p50, "p95_latency_s": p95,
                "total_min": total / 60,
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
