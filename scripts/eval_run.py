#!/usr/bin/env python3
"""
Run an evaluation set against the RAG API (V1 or V3) and compute metrics.

Metrics:
  - Answer Relevance: keyword overlap với expected_keywords
  - Citation Rate: % câu trả lời có citation [chunk_id]
  - Refusal Rate: % câu được refused
  - Refusal Accuracy: refusal đúng khi expect_refusal=True
  - Faithfulness proxy: validation.grounded_ratio (V3 only)
  - Confidence: trung bình validation.confidence (V3 only)
  - P50/P95 latency

Usage:
  python3 scripts/eval_run.py --eval eval/datasets/sample_queries_vi.json \\
      --endpoint v3 --api http://localhost:8800 --tenant default
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx


CITATION_RE = re.compile(r"\[(?:chunk[_\-]?id\s*[:=]?\s*)?[\w\-:.]+\]")


def keyword_overlap(answer: str, expected: list[str]) -> float:
    """Fraction of expected keywords found in answer (case-insensitive substring)."""
    if not expected:
        return 1.0
    answer_low = answer.lower()
    hits = sum(1 for kw in expected if kw.lower() in answer_low)
    return hits / len(expected)


def has_citation(answer: str) -> float:
    """Citation density: ratio of sentences with [chunk_id]."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if CITATION_RE.search(s))
    return cited / len(sentences)


def is_refusal(answer: str, refusal_keywords: list[str]) -> bool:
    answer_low = answer.lower()
    defaults = ["không có đủ thông tin", "không tìm thấy", "không thể trả lời",
                "không liên quan", "không đủ dữ liệu"]
    return any(k.lower() in answer_low for k in (refusal_keywords + defaults))


async def query_v3(client: httpx.AsyncClient, api: str, query: str, tenant: str) -> dict:
    resp = await client.post(
        f"{api}/api/v3/chat",
        json={"query": query, "tenant_id": tenant, "include_sources": True, "max_retries": 0},
        timeout=300.0,
    )
    resp.raise_for_status()
    return resp.json()


async def query_v1(client: httpx.AsyncClient, api: str, query: str, tenant: str) -> dict:
    resp = await client.post(
        f"{api}/v1/chat/completions",
        json={
            "model": "qwen3.5:4b",
            "messages": [{"role": "user", "content": query}],
            "temperature": 0.3,
            "max_tokens": 2048,
        },
        timeout=300.0,
    )
    resp.raise_for_status()
    data = resp.json()
    answer = data["choices"][0]["message"]["content"]
    return {
        "answer": answer,
        "refused": is_refusal(answer, []),
        "intent": None,
        "confidence": None,
        "validation": {"grounded_ratio": None, "citation_ratio": has_citation(answer)},
        "sources": data.get("sources", []),
        "latency_breakdown_ms": {},
    }


async def run(eval_path: Path, api: str, tenant: str, endpoint: str, output: Path | None) -> None:
    data = json.loads(eval_path.read_text())
    queries = data.get("queries", [])
    print(f"Running {len(queries)} queries against {endpoint} ({api})")
    print(f"Tenant: {tenant}\n")

    results: list[dict[str, Any]] = []
    timings: list[float] = []

    query_fn = query_v3 if endpoint == "v3" else query_v1

    async with httpx.AsyncClient(timeout=300.0) as client:
        for i, q in enumerate(queries, 1):
            qid = q["id"]
            query = q["query"]
            expected_keywords = q.get("expected_keywords", [])
            refusal_keywords = q.get("acceptable_refusal_keywords", [])
            expect_refusal = q.get("expect_refusal", False)

            t0 = time.monotonic()
            try:
                response = await query_fn(client, api, query, tenant)
                error = None
            except Exception as e:
                response = {"answer": "", "refused": False, "validation": {}, "sources": []}
                error = str(e)
            elapsed = (time.monotonic() - t0) * 1000
            timings.append(elapsed)

            answer = response.get("answer", "")
            refused = response.get("refused", False) or is_refusal(answer, refusal_keywords)

            relevance = keyword_overlap(answer, expected_keywords)
            citation = response.get("validation", {}).get("citation_ratio")
            if citation is None:
                citation = has_citation(answer)

            refusal_correct = (refused == expect_refusal)
            grounded = response.get("validation", {}).get("grounded_ratio")
            confidence = response.get("confidence")

            results.append({
                "id": qid,
                "query": query,
                "intent_expected": q.get("intent"),
                "intent_classified": response.get("intent"),
                "answer": answer[:500],
                "refused": refused,
                "expect_refusal": expect_refusal,
                "refusal_correct": refusal_correct,
                "relevance": relevance,
                "citation_ratio": citation,
                "grounded_ratio": grounded,
                "confidence": confidence,
                "latency_ms": elapsed,
                "error": error,
                "source_count": len(response.get("sources", [])),
            })

            status = "OK"
            if error:
                status = f"ERR ({error[:40]})"
            elif refused and expect_refusal:
                status = "REFUSED (correct)"
            elif refused:
                status = "REFUSED (unexpected)"
            elif relevance < 0.5:
                status = f"LOW RELEVANCE ({relevance:.2f})"

            print(f"  [{i}/{len(queries)}] {qid:20s} {status:30s} {elapsed:>7.0f}ms")

    # Aggregate
    print("\n" + "═" * 70)
    print("  AGGREGATE METRICS")
    print("═" * 70)

    n = len(results)
    success = [r for r in results if not r["error"]]
    print(f"  Total queries:        {n}")
    print(f"  Successful:           {len(success)}")

    refusals = [r for r in success if r["refused"]]
    print(f"  Refusal rate:         {len(refusals) / max(len(success), 1):.2%}")

    expected_ref = [r for r in success if r["expect_refusal"]]
    correct_ref = [r for r in expected_ref if r["refusal_correct"]]
    if expected_ref:
        print(f"  Refusal accuracy:     {len(correct_ref) / len(expected_ref):.2%} ({len(correct_ref)}/{len(expected_ref)})")

    answering = [r for r in success if not r["refused"]]
    if answering:
        avg_rel = sum(r["relevance"] for r in answering) / len(answering)
        print(f"  Avg relevance:        {avg_rel:.3f}")
        avg_cit = sum(r["citation_ratio"] for r in answering) / len(answering)
        print(f"  Avg citation ratio:   {avg_cit:.3f}")
        grounded_vals = [r["grounded_ratio"] for r in answering if r["grounded_ratio"] is not None]
        if grounded_vals:
            print(f"  Avg grounded ratio:   {sum(grounded_vals) / len(grounded_vals):.3f}")
        confidence_vals = [r["confidence"] for r in answering if r["confidence"] is not None]
        if confidence_vals:
            print(f"  Avg confidence:       {sum(confidence_vals) / len(confidence_vals):.3f}")

    if timings:
        sorted_t = sorted(timings)
        print(f"\n  Latency p50:          {sorted_t[len(sorted_t) // 2]:.0f}ms")
        print(f"  Latency p95:          {sorted_t[int(len(sorted_t) * 0.95)]:.0f}ms")
        print(f"  Latency p99:          {sorted_t[min(int(len(sorted_t) * 0.99), len(sorted_t) - 1)]:.0f}ms")
        print(f"  Latency mean:         {statistics.mean(timings):.0f}ms")

    # Per intent
    print("\n  Per-intent breakdown:")
    intents: dict[str, list] = {}
    for r in success:
        intents.setdefault(r.get("intent_expected") or "unknown", []).append(r)
    for intent, lst in intents.items():
        if not lst:
            continue
        ref_lst = [r for r in lst if r["refused"]]
        ans_lst = [r for r in lst if not r["refused"]]
        avg_rel = sum(r["relevance"] for r in ans_lst) / len(ans_lst) if ans_lst else 0
        print(f"    {intent:20s} n={len(lst):3d}  refusal={len(ref_lst):3d}  avg_relevance={avg_rel:.3f}")

    # Write output
    if output:
        report = {
            "eval_set": str(eval_path),
            "endpoint": endpoint,
            "api": api,
            "tenant": tenant,
            "queries_total": n,
            "successful": len(success),
            "refusals": len(refusals),
            "results": results,
            "summary": {
                "refusal_rate": len(refusals) / max(len(success), 1),
                "avg_relevance": sum(r["relevance"] for r in answering) / max(len(answering), 1) if answering else 0,
                "avg_citation_ratio": sum(r["citation_ratio"] for r in answering) / max(len(answering), 1) if answering else 0,
                "latency_p50_ms": sorted(timings)[len(timings) // 2] if timings else 0,
                "latency_p95_ms": sorted(timings)[int(len(timings) * 0.95)] if timings else 0,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n  Detailed report written to {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", type=Path, default=Path("eval/datasets/sample_queries_vi.json"))
    p.add_argument("--api", default="http://localhost:8800")
    p.add_argument("--tenant", default="default")
    p.add_argument("--endpoint", choices=["v1", "v3"], default="v3")
    p.add_argument("--output", type=Path, default=None, help="Detailed JSON report path")
    args = p.parse_args()

    if not args.eval.exists():
        print(f"Eval set not found: {args.eval}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(args.eval, args.api, args.tenant, args.endpoint, args.output))


if __name__ == "__main__":
    main()
