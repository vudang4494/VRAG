"""Canonical VRAG Benchmark Runner.

Runs benchmark queries against live /api/chat endpoint and outputs latency & quality metrics.

Usage:
    python3 scripts/benchmark.py [--dataset eval/datasets/ragas_groundtruth_corpus500.json] [--tenant corpus500] [--url http://localhost:8800]
"""

import argparse
import json
import time
import urllib.request
from typing import Any


def run_benchmark(dataset_path: str, tenant_id: str, base_url: str) -> dict[str, Any]:
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples") or data
    if isinstance(samples, dict):
        samples = [samples]

    print(f"Loaded {len(samples)} test cases from {dataset_path}")
    print(f"Target API: {base_url}/api/chat (tenant: {tenant_id})")

    results = []
    latencies = []
    success_count = 0

    for idx, item in enumerate(samples, 1):
        query = item.get("user_input") or item.get("query") or ""
        ref_answer = item.get("reference") or item.get("reference_answer") or ""

        payload = {
            "query": query,
            "tenant_id": tenant_id,
            "max_retries": 0,
            "disable_history_cache": True,
        }

        t0 = time.monotonic()
        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                elapsed = time.monotonic() - t0
                body = json.loads(resp.read().decode("utf-8"))
                ans = body.get("answer", "")
                refused = body.get("refused", False)

                latencies.append(elapsed)
                success_count += 1

                print(f"[{idx}/{len(samples)}] Latency: {elapsed:.2f}s | Success")
                results.append(
                    {
                        "query": query,
                        "reference": ref_answer,
                        "answer": ans,
                        "refused": refused,
                        "latency_s": round(elapsed, 2),
                    }
                )
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"[{idx}/{len(samples)}] FAILED ({e}) after {elapsed:.2f}s")
            results.append({"query": query, "error": str(e), "latency_s": round(elapsed, 2)})

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    summary = {
        "total": len(samples),
        "success": success_count,
        "avg_latency_s": round(avg_lat, 2),
        "results": results,
    }

    print("\n" + "=" * 50)
    print(f"BENCHMARK COMPLETED: {success_count}/{len(samples)} Success")
    print(f"Average Latency: {avg_lat:.2f}s")
    print("=" * 50)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VRAG Canonical Benchmark Runner")
    parser.add_argument(
        "--dataset",
        default="eval/datasets/ragas_groundtruth_corpus500.json",
        help="Path to test dataset JSON",
    )
    parser.add_argument("--tenant", default="corpus500", help="Tenant ID to test")
    parser.add_argument("--url", default="http://localhost:8800", help="Base API URL")
    args = parser.parse_args()

    run_benchmark(args.dataset, args.tenant, args.url)
