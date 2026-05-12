#!/usr/bin/env python3
"""Phase 7 — Ablation evaluation runner.

Runs the same query set through multiple pipeline configurations:
  C1. baseline:        /v1/chat/completions (V1, single dense vector)
  C2. v3_chat:          /api/v3/chat (full V3, no GAEA)
  C3. v3_chat_gaea:    /api/v3/chat with graph_aware view (Phase 1)
  C4. v3_chat_react:    /api/v3/chat/react (Phase 2 multi-step)
  C5. hefr:             /api/v3/hefr/retrieve (Phase 4 entity-first)

Metrics per query:
  - correct_doc_in_top:   was expected doc in top-K sources?
  - keyword_hit_ratio:    fraction of expected keywords in answer
  - refusal_correct:      refused iff expect_refusal=True
  - latency_ms:           end-to-end

Output: eval/results/ablation_<date>.json + console summary table.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx


async def query_v1(client: httpx.AsyncClient, api: str, query: str, tenant: str) -> dict:
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{api}/v1/chat/completions",
            json={"model": "qwen3.5:4b", "messages": [{"role": "user", "content": query}],
                  "temperature": 0.3, "max_tokens": 800},
            timeout=300.0,
        )
        resp.raise_for_status()
        d = resp.json()
        answer = d["choices"][0]["message"]["content"]
    except Exception as e:
        return {"error": str(e)[:200], "latency_ms": (time.monotonic() - t0) * 1000}
    return {
        "answer": answer,
        "sources": d.get("sources", []),
        "refused": False,
        "latency_ms": (time.monotonic() - t0) * 1000,
    }


async def query_v3_chat(client: httpx.AsyncClient, api: str, query: str, tenant: str) -> dict:
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{api}/api/v3/chat",
            json={"query": query, "tenant_id": tenant, "max_retries": 0, "include_sources": True},
            timeout=300.0,
        )
        resp.raise_for_status()
        d = resp.json()
    except Exception as e:
        return {"error": str(e)[:200], "latency_ms": (time.monotonic() - t0) * 1000}
    return {
        "answer": d.get("answer", ""),
        "sources": d.get("sources", []),
        "refused": bool(d.get("refused")),
        "latency_ms": (time.monotonic() - t0) * 1000,
    }


async def query_v3_react(client: httpx.AsyncClient, api: str, query: str, tenant: str) -> dict:
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{api}/api/v3/chat/react",
            json={"query": query, "tenant_id": tenant, "max_steps": 4},
            timeout=300.0,
        )
        resp.raise_for_status()
        d = resp.json()
    except Exception as e:
        return {"error": str(e)[:200], "latency_ms": (time.monotonic() - t0) * 1000}
    return {
        "answer": d.get("answer", ""),
        "sources": d.get("sources", []),
        "refused": False,
        "latency_ms": d.get("latency_ms", {}).get("total", (time.monotonic() - t0) * 1000),
        "steps_used": d.get("steps_used"),
    }


async def query_hefr(client: httpx.AsyncClient, api: str, query: str, tenant: str) -> dict:
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{api}/api/v3/hefr/retrieve",
            json={"query": query, "tenant_id": tenant, "top_chunks": 5},
            timeout=120.0,
        )
        resp.raise_for_status()
        d = resp.json()
    except Exception as e:
        return {"error": str(e)[:200], "latency_ms": (time.monotonic() - t0) * 1000}
    chunks = d.get("sample_chunks", [])
    # HEFR doesn't generate answer — synthesize string from top sources
    answer = " ".join((c.get("text") or "")[:200] for c in chunks[:3])
    return {
        "answer": answer,
        "sources": [{"source": c.get("source"), "chunk_id": c.get("chunk_id")} for c in chunks],
        "refused": False,
        "latency_ms": (time.monotonic() - t0) * 1000,
    }


CONFIGS = {
    "C1_baseline_v1": query_v1,
    "C2_v3_chat": query_v3_chat,
    "C3_v3_chat_with_gaea": query_v3_chat,  # same endpoint but graph_aware view active
    "C4_v3_react": query_v3_react,
    "C5_hefr_only": query_hefr,
}


def check_expected_doc(sources: list[dict], expected_doc: str | None, expected_docs: list[str] | None) -> int:
    """Return # of expected docs found in sources."""
    expected = set()
    if expected_doc:
        expected.add(expected_doc)
    if expected_docs:
        expected.update(expected_docs)
    if not expected:
        return 0
    found = set()
    for s in sources:
        src = s.get("source") or s.get("metadata", {}).get("source") or ""
        for exp in expected:
            if exp in src:
                found.add(exp)
    return len(found)


def keyword_hit_ratio(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    a = answer.lower()
    hits = sum(1 for k in keywords if k.lower() in a)
    return hits / len(keywords)


REFUSAL_PATTERNS = [r"không có đủ thông tin", r"không tìm thấy", r"không thể trả lời",
                     r"i don'?t have enough information"]


def is_refused(answer: str) -> bool:
    a = answer.lower()
    return any(re.search(p, a) for p in REFUSAL_PATTERNS)


async def main(args):
    bench = json.loads(Path(args.bench).read_text())
    queries = bench["queries"]
    print(f"Ablation eval — {len(queries)} queries × {len(CONFIGS)} configs = {len(queries) * len(CONFIGS)} runs")
    print(f"Estimated time: {len(queries) * len(CONFIGS) * 20 / 60:.1f} min\n")

    results = {cfg: [] for cfg in CONFIGS}

    async with httpx.AsyncClient(timeout=300.0) as client:
        for q_idx, q in enumerate(queries, 1):
            print(f"[{q_idx}/{len(queries)}] {q['id']} ({q['category']}): {q['query'][:50]}")
            for cfg_name, cfg_fn in CONFIGS.items():
                if args.skip and cfg_name in args.skip:
                    continue
                r = await cfg_fn(client, args.api, q["query"], args.tenant)
                expected_count = len({q.get("expected_doc")} | set(q.get("expected_docs", [])) - {None})
                docs_found = check_expected_doc(r.get("sources", []), q.get("expected_doc"), q.get("expected_docs"))
                kw_hit = keyword_hit_ratio(r.get("answer", ""), q.get("expected_keywords", []))
                refused = is_refused(r.get("answer", "")) or r.get("refused", False)
                expect_refusal = q.get("expect_refusal", False)
                refusal_correct = (refused == expect_refusal)

                results[cfg_name].append({
                    "q_id": q["id"],
                    "category": q["category"],
                    "latency_ms": r.get("latency_ms", 0),
                    "docs_found": docs_found,
                    "docs_expected": expected_count,
                    "kw_hit_ratio": kw_hit,
                    "refused": refused,
                    "expect_refusal": expect_refusal,
                    "refusal_correct": refusal_correct,
                    "answer_excerpt": (r.get("answer") or "")[:200],
                    "error": r.get("error"),
                })
                status = "OK" if not r.get("error") else "ERR"
                print(f"  {cfg_name:<25} {status:<5} {r.get('latency_ms', 0)/1000:>6.1f}s "
                      f"docs={docs_found}/{expected_count} kw={kw_hit:.2f} refused={refused}")
            print()

    # Aggregate per config
    print("\n" + "═" * 90)
    print("AGGREGATE RESULTS")
    print("═" * 90)
    header = f"{'Config':<25} {'N':>4} {'p50 latency':>12} {'avg docs/exp':>14} {'avg kw':>8} {'refusal_acc':>12}"
    print(header)
    print("-" * 90)
    summary: dict[str, dict] = {}
    for cfg, runs in results.items():
        ok_runs = [r for r in runs if not r.get("error")]
        if not ok_runs:
            print(f"{cfg:<25} 0 runs OK")
            continue
        n = len(ok_runs)
        latencies = sorted(r["latency_ms"] for r in ok_runs)
        p50 = latencies[len(latencies) // 2]
        avg_docs = sum(r["docs_found"] / max(r["docs_expected"], 1) for r in ok_runs) / n
        avg_kw = sum(r["kw_hit_ratio"] for r in ok_runs) / n
        refusal_acc = sum(1 for r in ok_runs if r["refusal_correct"]) / n
        summary[cfg] = {
            "n": n, "p50_latency_ms": p50,
            "avg_doc_recall": avg_docs, "avg_kw_hit": avg_kw,
            "refusal_accuracy": refusal_acc,
        }
        print(f"{cfg:<25} {n:>4} {p50/1000:>10.1f}s  {avg_docs:>13.2%}  {avg_kw:>7.2%}  {refusal_acc:>11.2%}")

    # Save full report
    report = {
        "benchmark": args.bench,
        "tenant": args.tenant,
        "configs_tested": list(CONFIGS.keys()),
        "queries_total": len(queries),
        "summary_per_config": summary,
        "results_per_query": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="eval/datasets/vi_benchmark_v1.json")
    p.add_argument("--api", default="http://localhost:8800")
    p.add_argument("--tenant", default="eval")
    p.add_argument("--output", default=None)
    p.add_argument("--skip", nargs="+", default=[])
    args = p.parse_args()
    if not args.output:
        from datetime import datetime as _d
        args.output = f"eval/results/ablation_{_d.now().strftime('%Y%m%d_%H%M%S')}.json"
    asyncio.run(main(args))
