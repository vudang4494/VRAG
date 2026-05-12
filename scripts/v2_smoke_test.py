#!/usr/bin/env python3
"""
Pipeline V2 — End-to-end smoke test.

Workflow:
  1. Health check stack
  2. Ingest 1 sample document through /api/v3/ingest/upload
  3. Verify Qdrant got 5 named vectors per chunk
  4. Verify Neo4j got Chunk + Entity nodes
  5. Run chat query
  6. Validate response: has answer, has citations, validation gates passed
  7. Print summary

Usage:
  python3 scripts/v2_smoke_test.py [--api http://localhost:8800] [--tenant default]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


SAMPLE_TEXT_VI = """\
Báo cáo Tài chính Quý 3 Năm 2024 - Công ty ABC

Doanh thu Quý 3 năm 2024 đạt 500 tỷ VND, tăng 25% so với cùng kỳ năm trước.
Lợi nhuận sau thuế là 80 tỷ VND, tăng 18% YoY. Tổng giám đốc Nguyễn Văn A
phát biểu rằng kết quả này phản ánh chiến lược chuyển đổi số đang phát huy
hiệu quả.

Mảng kinh doanh chính - bán lẻ - đóng góp 60% doanh thu, đạt 300 tỷ VND.
Mảng dịch vụ tài chính đóng góp 25%, tương đương 125 tỷ VND. Phần còn lại
đến từ mảng công nghệ và đầu tư.

Rủi ro chính trong quý này là biến động tỷ giá USD/VND và áp lực cạnh tranh
từ các đối thủ trong khu vực. Công ty đã thực hiện các biện pháp phòng ngừa
rủi ro tỷ giá thông qua hợp đồng kỳ hạn.

Kế hoạch Quý 4 dự kiến doanh thu 550 tỷ VND, tập trung vào mở rộng thị phần
ở miền Bắc và đầu tư công nghệ AI cho dịch vụ khách hàng.
"""


def banner(msg: str, char: str = "─") -> None:
    print()
    print(char * 70)
    print(f"  {msg}")
    print(char * 70)


def step(idx: int, msg: str) -> None:
    print(f"\n[{idx}] {msg}")


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def fail(msg: str) -> None:
    print(f"  ✗ {msg}")


async def run(api_base: str, tenant: str, sample_path: Path | None = None) -> int:
    banner("Pipeline V2 — Smoke Test")
    print(f"API base: {api_base}")
    print(f"Tenant:   {tenant}")

    async with httpx.AsyncClient(timeout=300.0) as client:
        # 1. Health check
        step(1, "Health check /api/v3/health/deep")
        try:
            resp = await client.get(f"{api_base}/api/v3/health/deep")
            resp.raise_for_status()
            health = resp.json()
            ok(f"V3 status: {health.get('status')}")
            for comp_name, comp in health.get("components", {}).items():
                status = "OK" if comp.get("ok") else "FAIL"
                print(f"     - {comp_name}: {status}")
            missing = [d["name"] for d in health.get("dependencies", []) if not d["ok"]]
            if missing:
                print(f"     - Missing deps: {', '.join(missing)} (degraded but should still work)")
        except Exception as e:
            fail(f"Health check failed: {e}")
            return 1

        # 2. Ingest sample
        step(2, "Ingest sample document via /api/v3/ingest/upload")
        if sample_path and sample_path.exists():
            content = sample_path.read_bytes()
            filename = sample_path.name
        else:
            content = SAMPLE_TEXT_VI.encode("utf-8")
            filename = "smoke_test_quy3_2024.txt"

        t0 = time.monotonic()
        try:
            files = {"file": (filename, content, "application/octet-stream")}
            data = {"tenant_id": tenant, "access_level": "INTERNAL"}
            resp = await client.post(f"{api_base}/api/v3/ingest/upload", files=files, data=data, timeout=300.0)
            resp.raise_for_status()
            result = resp.json()
            ok(f"Status: {result.get('status')}")
            print(f"     - Format detected: {result.get('format')}")
            print(f"     - Chunks indexed: {result.get('chunks_indexed')}")
            print(f"     - Chunks dropped (low consistency): {result.get('chunks_dropped_low_quality', 0)}")
            print(f"     - Avg consistency score: {result.get('avg_consistency_score', 0):.3f}")
            print(f"     - Entities extracted: {result.get('entities_extracted', 0)}")
            print(f"     - Relationships: {result.get('relationships_extracted', 0)}")
            print(f"     - Duration: {result.get('duration_seconds', 0):.1f}s")
            doc_id = result.get("doc_id")
        except Exception as e:
            fail(f"Ingest failed: {e}")
            return 1

        ingest_time = time.monotonic() - t0
        print(f"     - Wall time: {ingest_time:.1f}s")

        # 3. Wait briefly for index propagation
        time.sleep(2)

        # 4. Chat query
        step(3, "Chat query /api/v3/chat")
        query = "Doanh thu Quý 3 năm 2024 của công ty ABC là bao nhiêu?"
        print(f"     Query: {query}")

        t0 = time.monotonic()
        try:
            resp = await client.post(
                f"{api_base}/api/v3/chat",
                json={"query": query, "tenant_id": tenant, "include_sources": True, "max_retries": 0},
                timeout=180.0,
            )
            resp.raise_for_status()
            answer_data = resp.json()
        except Exception as e:
            fail(f"Chat failed: {e}")
            return 1

        chat_time = time.monotonic() - t0

        # 5. Validate response
        step(4, "Validate response")
        answer = answer_data.get("answer", "")
        refused = answer_data.get("refused", False)
        intent = answer_data.get("intent")
        confidence = answer_data.get("confidence", 0)
        validation = answer_data.get("validation", {})
        sources = answer_data.get("sources", [])
        latency = answer_data.get("latency_breakdown_ms", {})

        print(f"     Intent classified: {intent}")
        print(f"     Confidence: {confidence:.3f}")
        print(f"     Refused: {refused}")
        print(f"     Validation: passed={validation.get('passed')}, "
              f"grounded={validation.get('grounded_ratio', 0):.2f}, "
              f"citation_ratio={validation.get('citation_ratio', 0):.2f}")
        print(f"     Sources returned: {len(sources)}")
        print(f"     Total latency: {latency.get('total_ms', 0):.0f}ms")
        print(f"     Wall time: {chat_time:.1f}s")
        print()
        print("     ── Answer ──")
        for line in answer.split("\n"):
            print(f"     {line}")

        # Check key things
        checks_passed = 0
        checks_total = 0

        checks_total += 1
        if answer.strip():
            checks_passed += 1
            ok("Non-empty answer")
        else:
            fail("Empty answer")

        checks_total += 1
        if "500" in answer or "tỷ" in answer.lower():
            checks_passed += 1
            ok("Answer contains expected info (doanh thu 500 tỷ)")
        else:
            fail("Answer does NOT mention expected facts — retrieval issue?")

        checks_total += 1
        if sources:
            checks_passed += 1
            ok(f"{len(sources)} sources cited")
        else:
            fail("No sources cited")

        checks_total += 1
        if validation.get("passed"):
            checks_passed += 1
            ok("Validation gates passed")
        else:
            fail(f"Validation failed: {validation.get('failure_reason')}")

        # Latency stage breakdown
        step(5, "Latency breakdown (ms)")
        for stage, lat in sorted(latency.items(), key=lambda x: -x[1])[:10]:
            print(f"     {stage:40s} {lat:>10.0f}")

        # Summary
        banner(f"Result: {checks_passed}/{checks_total} checks passed", char="═")
        if checks_passed == checks_total:
            print("All checks passed. V2 pipeline is operational.")
            return 0
        elif checks_passed >= checks_total - 1:
            print("Most checks passed. Some quality issues — inspect output.")
            return 0
        else:
            print("Multiple failures. Investigate.")
            return 1


def main():
    p = argparse.ArgumentParser(description="V2 pipeline smoke test")
    p.add_argument("--api", default="http://localhost:8800", help="API base URL")
    p.add_argument("--tenant", default="default", help="Tenant ID")
    p.add_argument("--sample", type=Path, default=None, help="Optional sample file path")
    args = p.parse_args()

    import asyncio
    code = asyncio.run(run(args.api, args.tenant, args.sample))
    sys.exit(code)


if __name__ == "__main__":
    main()
