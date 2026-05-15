#!/usr/bin/env python3
"""
Ingest all papers in /tmp/eval_papers/ through Pipeline V2.

Sequential because Ollama is bottleneck. Each ingest ~5 min on M4 with qwen3.5:4b.
Run via: python3 scripts/ingest_eval_papers.py [--api http://localhost:8800] [--tenant eval]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx


async def ingest_one(client: httpx.AsyncClient, api: str, path: Path, tenant: str) -> dict:
    content = path.read_bytes()
    files = {"file": (path.name, content, "text/markdown")}
    data = {"tenant_id": tenant, "access_level": "INTERNAL"}
    resp = await client.post(
        f"{api}/api/v3/ingest/upload",
        files=files,
        data=data,
        timeout=1200.0,  # 20 min ceiling for big papers
    )
    resp.raise_for_status()
    return resp.json()


async def main(args):
    paper_dir = Path(args.dir)
    papers = sorted(paper_dir.glob("*.md"))
    if not papers:
        print(f"No .md files in {paper_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(papers)} papers to ingest into tenant '{args.tenant}'")
    print(f"API: {args.api}")
    print()

    started_total = time.monotonic()
    results: list[dict] = []

    async with httpx.AsyncClient(timeout=1200.0) as client:
        for i, path in enumerate(papers, 1):
            print(f"[{i}/{len(papers)}] {path.name} ({path.stat().st_size / 1024:.1f} KB)")
            t0 = time.monotonic()
            try:
                result = await ingest_one(client, args.api, path, args.tenant)
                elapsed = time.monotonic() - t0
                status = result.get("status", "unknown")
                chunks_total = result.get("chunks_total", 0)
                chunks_indexed = result.get("chunks_indexed", 0)
                dropped = result.get("chunks_dropped_low_quality", 0)
                entities = result.get("entities_extracted", 0)
                rels = result.get("relationships_extracted", 0)
                avg_cons = result.get("avg_consistency_score", 0)
                doc_id = result.get("doc_id", "")
                print(
                    f"    {status}: {chunks_indexed}/{chunks_total} chunks "
                    f"(dropped: {dropped}), entities: {entities}, "
                    f"rels: {rels}, consistency: {avg_cons:.2f}, "
                    f"time: {elapsed / 60:.1f}min, doc_id: {doc_id}"
                )
                results.append(
                    {
                        "file": path.name,
                        "status": status,
                        "doc_id": doc_id,
                        "chunks_total": chunks_total,
                        "chunks_indexed": chunks_indexed,
                        "chunks_dropped": dropped,
                        "entities": entities,
                        "relationships": rels,
                        "avg_consistency": avg_cons,
                        "duration_seconds": elapsed,
                    }
                )
            except Exception as e:
                elapsed = time.monotonic() - t0
                print(f"    FAILED after {elapsed / 60:.1f}min: {e}")
                results.append(
                    {
                        "file": path.name,
                        "status": "error",
                        "error": str(e)[:200],
                        "duration_seconds": elapsed,
                    }
                )

    total_elapsed = time.monotonic() - started_total
    print()
    print("═" * 70)
    print(
        f"  Ingest complete: {sum(1 for r in results if r.get('status') == 'success')}/{len(results)} succeeded"
    )
    print(f"  Total time: {total_elapsed / 60:.1f} min")
    print("═" * 70)

    # Save report
    Path(args.report).write_text(
        json.dumps(
            {
                "tenant": args.tenant,
                "total_files": len(results),
                "total_duration_seconds": total_elapsed,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n  Report: {args.report}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="/tmp/eval_papers")
    p.add_argument("--api", default="http://localhost:8800")
    p.add_argument("--tenant", default="eval")
    p.add_argument("--report", default="/tmp/ingest_report.json")
    args = p.parse_args()
    asyncio.run(main(args))
