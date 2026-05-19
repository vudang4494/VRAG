#!/usr/bin/env python3
"""
Benchmark evaluation for rag51 tenant.
Runs 50 queries and generates comprehensive report.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from datetime import UTC
from pathlib import Path

import httpx

BENCHMARK = "eval/datasets/vi_benchmark_v2.json"
SEMANTIC_THRESHOLD = 0.45
API = "http://localhost:8800"
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
    return any(re.search(p, a) for p in REFUSAL_PATTERNS)


async def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def embed_text(text: str, http: httpx.AsyncClient) -> list[float] | None:
    try:
        resp = await http.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "bge-m3", "prompt": text[:2000]},
            timeout=20.0,
        )
        resp.raise_for_status()
        return resp.json().get("embedding")
    except Exception as e:
        return None


async def kw_hit_ratio(answer: str, keywords: list[str], http: httpx.AsyncClient | None) -> tuple[float, list[dict]]:
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

    if http and keywords:
        kw_embs = {}
        tasks = [embed_text(kw, http) for kw in keywords]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for kw, res in zip(keywords, results):
            kw_embs[kw] = res if not isinstance(res, Exception) and res else None

        answer_emb = await embed_text(answer, http)
        if answer_emb:
            for i, kw in enumerate(keywords):
                kw_emb = kw_embs.get(kw)
                if kw_emb and not breakdown[i]["hit"]:
                    sim = await cosine_sim(kw_emb, answer_emb)
                    if sim > SEMANTIC_THRESHOLD:
                        breakdown[i]["hit"] = True
                        breakdown[i]["semantic"] = True
                        hits += 1

    return hits / len(keywords), breakdown


async def check_sources(sources: list[dict], expected_docs: list[str]) -> tuple[int, list[str]]:
    if not expected_docs:
        return 0, []
    found = []
    for s in sources:
        src = s.get("source", "") or s.get("metadata", {}).get("source", "")
        for exp in expected_docs:
            if exp.lower() in src.lower() and exp not in found:
                found.append(exp)
    return len(found), found


async def run_query(client: httpx.AsyncClient, api: str, q: dict, tenant: str) -> dict:
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{api}/api/v3/chat",
            json={
                "query": q["query"],
                "tenant_id": tenant,
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
            "kw_hit": 0.0,
            "kw_breakdown": [],
            "refused": False,
            "expect_refusal": q.get("expect_refusal", False),
            "refusal_ok": False,
            "query_type": "error",
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
    doc_recall = doc_found / doc_expected if doc_expected > 0 else 1.0 if not expected_docs else 0.0

    refusal_ok = refused == expect_refusal

    keywords = q.get("expected_keywords", [])
    kw_hit, kw_breakdown = await kw_hit_ratio(answer, keywords, None)

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
    with open(BENCHMARK) as f:
        bm = json.load(f)
    queries = bm["queries"]

    n = len(queries)
    print(f"\n{'='*90}")
    print(f"BENCHMARK EVALUATION — rag51 tenant")
    print(f"{'='*90}")
    print(f"  Benchmark: {BENCHMARK}")
    print(f"  Tenant: {TENANT}")
    print(f"  Total queries: {n}")
    print(f"  Model: qwen3.5:9b")
    print(f"  Features: consistency=ON, community=ON")
    print(f"  Eval: validation=OFF, ood=OFF, retries=0")
    print(f"{'='*90}\n")

    started = time.monotonic()
    results = []

    async with httpx.AsyncClient(timeout=600.0) as client:
        for i, q in enumerate(queries, 1):
            q_id = q.get("q_id", q.get("id", f"q{i}"))
            cat = q.get("category", "unknown")
            query_text = q["query"][:60]

            print(f"[{i:2d}/{n}] {q_id:<12} ({cat:<15}): {query_text}...")
            sys.stdout.flush()

            r = await run_query(client, API, q, TENANT)
            results.append(r)

            status = "ERR" if r["error"] else "OK"
            lat = r["latency_ms"] / 1000
            recall = r["doc_recall"]
            kw = r["kw_hit"]
            ref = "REFUSE" if r["refused"] else "OK"
            ref_ok = "Y" if r["refusal_ok"] else "N"
            qt = r["query_type"][:8]
            react = "[R]" if r["react_used"] else "[S]"

            print(f"         -> {status} | recall={recall:.0%} | kw={kw:.0%} | ref={ref} {ref_ok} | {lat:5.0f}s | {qt} {react}")
            if r["error"]:
                print(f"         ERROR: {r['error']}")
            print()
            sys.stdout.flush()

    total_elapsed = time.monotonic() - started

    ok_results = [r for r in results if not r["error"]]
    errors = [r for r in results if r["error"]]

    print(f"\n{'='*90}")
    print("AGGREGATE RESULTS")
    print(f"{'='*90}")
    print(f"  Total queries: {n}")
    print(f"  Successful: {len(ok_results)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Total time: {total_elapsed / 60:.1f} min")
    print()

    all_recalls = [r["doc_recall"] for r in ok_results]
    all_kw = [r["kw_hit"] for r in ok_results]
    all_ref_ok = [r["refusal_ok"] for r in ok_results]
    all_lats = sorted([r["latency_ms"] for r in ok_results])

    if ok_results:
        ov_recall = sum(all_recalls) / len(all_recalls)
        ov_kw = sum(all_kw) / len(all_kw)
        ov_ref = sum(all_ref_ok) / len(all_ref_ok)
        p50 = all_lats[len(all_lats) // 2] / 1000
        p95 = all_lats[int(len(all_lats) * 0.95)] / 1000

        print(f"  OVERALL: p50_lat={p50:5.1f}s  p95_lat={p95:5.1f}s  recall={ov_recall:.1%}  kw_hit={ov_kw:.1%}  ref_acc={ov_ref:.1%}")

    print(f"\n{'='*90}")
    print("PER-CATEGORY BREAKDOWN")
    print(f"{'='*90}")
    cats = {}
    for r in results:
        cat = r["category"]
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(r)

    print(f"  {'CATEGORY':<20} {'N':>3} {'RECALL':>8} {'KW_HIT':>8} {'REF_ACC':>8} {'P50_LAT':>8}")
    print(f"  {'-'*60}")

    for cat in sorted(cats.keys()):
        rs = cats[cat]
        ok = [r for r in rs if not r["error"]]
        if not ok:
            continue
        n_cat = len(ok)
        rec = sum(r["doc_recall"] for r in ok) / n_cat
        kw = sum(r["kw_hit"] for r in ok) / n_cat
        ref = sum(r["refusal_ok"] for r in ok) / n_cat
        lats = sorted([r["latency_ms"] for r in ok])
        p50 = lats[len(lats) // 2] / 1000 if lats else 0

        print(f"  {cat:<20} {n_cat:>3} {rec:>8.1%} {kw:>8.1%} {ref:>8.1%} {p50:>7.1f}s")

    print(f"\n{'='*90}")
    print("QUERY TYPE BREAKDOWN")
    print(f"{'='*90}")
    types = {}
    for r in ok_results:
        qt = r["query_type"]
        if qt not in types:
            types[qt] = {"recalls": [], "kw": [], "react": 0}
        types[qt]["recalls"].append(r["doc_recall"])
        types[qt]["kw"].append(r["kw_hit"])
        if r["react_used"]:
            types[qt]["react"] += 1

    for qt, data in sorted(types.items()):
        n_qt = len(data["recalls"])
        avg_rec = sum(data["recalls"]) / n_qt
        avg_kw = sum(data["kw"]) / n_qt
        print(f"  {qt:<15} N={n_qt:>2}  recall={avg_rec:.1%}  kw={avg_kw:.1%}  ReAct={data['react']}/{n_qt}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"eval/results/benchmark_rag51_{timestamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": BENCHMARK,
        "tenant": TENANT,
        "model": "qwen3.5:9b",
        "config": {
            "consistency_views": True,
            "community_enabled": True,
            "validation_enabled": False,
            "ood_enabled": False,
            "max_retries": 0,
        },
        "queries_total": n,
        "queries_ok": len(ok_results),
        "queries_err": len(errors),
        "total_time_min": total_elapsed / 60,
        "summary": {
            "avg_doc_recall": sum(all_recalls) / len(all_recalls) if all_recalls else 0,
            "avg_kw_hit": sum(all_kw) / len(all_kw) if all_kw else 0,
            "avg_refusal_accuracy": sum(all_ref_ok) / len(all_ref_ok) if all_ref_ok else 0,
            "p50_latency_s": p50 if ok_results else 0,
            "p95_latency_s": p95 if ok_results else 0,
        },
        "per_category": {
            cat: {
                "n": len([r for r in cats[cat] if not r["error"]]),
                "avg_doc_recall": sum(r["doc_recall"] for r in cats[cat] if not r["error"]) / max(len([r for r in cats[cat] if not r["error"]]), 1),
                "avg_kw_hit": sum(r["kw_hit"] for r in cats[cat] if not r["error"]) / max(len([r for r in cats[cat] if not r["error"]]), 1),
                "refusal_accuracy": sum(r["refusal_ok"] for r in cats[cat] if not r["error"]) / max(len([r for r in cats[cat] if not r["error"]]), 1),
            }
            for cat in cats
        },
        "per_query_type": {
            qt: {
                "n": len(data["recalls"]),
                "avg_doc_recall": sum(data["recalls"]) / len(data["recalls"]),
                "avg_kw_hit": sum(data["kw"]) / len(data["kw"]),
                "react_used": data["react"],
            }
            for qt, data in types.items()
        },
        "results": results,
    }

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n{'='*90}")
    print(f"Report saved: {out_path}")
    print(f"{'='*90}")


if __name__ == "__main__":
    asyncio.run(main())
