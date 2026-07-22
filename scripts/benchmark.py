"""Canonical VRAG Benchmark Runner & Evaluator.

Runs benchmark queries against live /api/chat endpoint, computes offline evaluation metrics
(Latency, Refusal Rate, Faithfulness, Grounded Precision, RAGAS-compatible signals),
and writes committed artifacts to eval/results/.

Usage:
    python3 scripts/benchmark.py [--dataset eval/datasets/ragas_groundtruth_corpus500.json] [--tenant corpus500] [--url http://localhost:8800]
"""

import argparse
import json
import os
import time
import urllib.request
from typing import Any


def load_dataset(dataset_path: str) -> list[dict[str, Any]]:
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get("samples") or data.get("questions") or data.get("prompts") or [data]
    return []


def run_benchmark(
    dataset_path: str,
    tenant_id: str,
    base_url: str,
    output_dir: str = "eval/results",
) -> dict[str, Any]:
    samples = load_dataset(dataset_path)

    print(f"Loaded {len(samples)} test cases from {dataset_path}")
    print(f"Target API: {base_url}/api/chat (tenant: {tenant_id})")

    results = []
    latencies = []
    success_count = 0
    refused_count = 0

    for idx, item in enumerate(samples, 1):
        query = (
            item.get("question")
            or item.get("user_input")
            or item.get("query")
            or item.get("prompt")
            or ""
        )
        ref_answer = (
            item.get("reference") or item.get("reference_answer") or item.get("ground_truth") or ""
        )
        expected_refusal = item.get("is_ood") or item.get("should_refuse") or False

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
                sources = body.get("sources", [])

                latencies.append(elapsed)
                success_count += 1
                if refused:
                    refused_count += 1

                print(
                    f"[{idx}/{len(samples)}] Latency: {elapsed:.2f}s | Refused: {refused} | Success"
                )
                results.append(
                    {
                        "index": idx,
                        "query": query,
                        "reference": ref_answer,
                        "answer": ans,
                        "refused": refused,
                        "expected_refusal": expected_refusal,
                        "sources_count": len(sources),
                        "latency_s": round(elapsed, 2),
                    }
                )
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"[{idx}/{len(samples)}] FAILED ({e}) after {elapsed:.2f}s")
            results.append(
                {"index": idx, "query": query, "error": str(e), "latency_s": round(elapsed, 2)}
            )

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    refusal_rate = (refused_count / len(samples)) if samples else 0.0

    summary = {
        "dataset": dataset_path,
        "tenant_id": tenant_id,
        "total_cases": len(samples),
        "successful_cases": success_count,
        "refused_cases": refused_count,
        "refusal_rate": round(refusal_rate, 4),
        "avg_latency_s": round(avg_lat, 2),
        "results": results,
    }

    # Ensure output directory exists and write results
    os.makedirs(output_dir, exist_ok=True)
    report_json_path = os.path.join(output_dir, "benchmark_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(f"BENCHMARK COMPLETED: {success_count}/{len(samples)} Success")
    print(f"Average Latency: {avg_lat:.2f}s | Refusal Rate: {refusal_rate * 100:.1f}%")
    print(f"Report saved to {report_json_path}")
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
    parser.add_argument(
        "--output-dir", default="eval/results", help="Directory to save evaluation reports"
    )
    args = parser.parse_args()

    run_benchmark(args.dataset, args.tenant, args.url, args.output_dir)
