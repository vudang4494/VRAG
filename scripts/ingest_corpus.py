"""Bulk-ingest a multi-domain PDF corpus (data/raw/<category>/*.pdf) through the VRAG API.

Recursive glob over category subdirs, resume support (skips files already recorded as
success in the report), calibration mode, and live ETA.

Ingest goes through POST /api/ingest/upload -> ingest_document (canonical entrypoint).
Sequential by design: Ollama embed is the bottleneck and parallel uploads only thrash it.

Usage:
    # calibrate on a size-stratified sample first to measure the real rate
    python3 scripts/ingest_corpus.py --dir data/raw --tenant corpus500 --limit 6 --stratified

    # full run (resumable — re-run the same command after an interrupt)
    python3 scripts/ingest_corpus.py --dir data/raw --tenant corpus500
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
    files = {"file": (path.name, path.read_bytes(), "application/pdf")}
    data = {"tenant_id": tenant, "access_level": "INTERNAL"}
    resp = await client.post(f"{api}/api/ingest/upload", files=files, data=data, timeout=3600.0)
    resp.raise_for_status()
    return resp.json()


def select_files(paper_dir: Path, limit: int, stratified: bool) -> list[Path]:
    papers = sorted(paper_dir.rglob("*.pdf"))
    if not limit:
        return papers
    if not stratified:
        return papers[:limit]
    # Size-stratified sample: spread across the size distribution so the measured
    # rate reflects the real corpus, not just whichever files sort first.
    by_size = sorted(papers, key=lambda p: p.stat().st_size)
    step = max(1, len(by_size) // limit)
    return by_size[::step][:limit]


def load_done(report_path: Path) -> dict[str, dict]:
    if not report_path.exists():
        return {}
    try:
        prev = json.loads(report_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {r["file"]: r for r in prev.get("results", []) if r.get("status") == "success"}


def write_report(report_path: Path, tenant: str, results: list[dict], total: float) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "tenant": tenant,
                "total_files": len(results),
                "total_duration_seconds": total,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


async def main(args):
    paper_dir = Path(args.dir)
    papers = select_files(paper_dir, args.limit, args.stratified)
    if not papers:
        print(f"No PDFs under {paper_dir}", file=sys.stderr)
        sys.exit(1)

    report_path = Path(args.report)
    done = load_done(report_path) if args.resume else {}
    results: list[dict] = list(done.values())
    todo = [p for p in papers if p.name not in done]

    if done:
        print(f"Resume: {len(done)} already ingested, {len(todo)} remaining")
    print(f"Ingesting {len(todo)} PDFs into tenant '{args.tenant}' (dir={paper_dir})")
    print("=" * 78, flush=True)

    started = time.monotonic()
    durations: list[float] = []

    async with httpx.AsyncClient(timeout=3600.0) as client:
        for i, path in enumerate(todo, 1):
            mb = path.stat().st_size / 1e6
            cat = path.parent.name
            t0 = time.monotonic()
            try:
                result = await ingest_one(client, args.api, path, args.tenant)
                elapsed = time.monotonic() - t0
                durations.append(elapsed)
                rec = {
                    "file": path.name,
                    "category": cat,
                    "size_mb": round(mb, 2),
                    "status": result.get("status", "unknown"),
                    "doc_id": result.get("doc_id", ""),
                    "chunks_total": result.get("chunks_total", 0),
                    "chunks_indexed": result.get("chunks_indexed", 0),
                    "chunks_dropped": result.get("chunks_dropped_low_quality", 0),
                    "entities": result.get("entities_extracted", 0),
                    "relationships": result.get("relationships_extracted", 0),
                    "duration_seconds": round(elapsed, 1),
                }
                avg = sum(durations) / len(durations)
                eta_min = avg * (len(todo) - i) / 60
                print(
                    f"[{i}/{len(todo)}] {cat}/{path.name[:44]} ({mb:.1f}MB) "
                    f"-> {rec['status']} {rec['chunks_indexed']}/{rec['chunks_total']} chunks, "
                    f"ents {rec['entities']}, {elapsed:.0f}s | avg {avg:.0f}s/doc, ETA {eta_min:.0f}min",
                    flush=True,
                )
            except Exception as e:
                elapsed = time.monotonic() - t0
                rec = {
                    "file": path.name,
                    "category": cat,
                    "size_mb": round(mb, 2),
                    "status": "error",
                    "error": str(e)[:200],
                    "duration_seconds": round(elapsed, 1),
                }
                print(
                    f"[{i}/{len(todo)}] {cat}/{path.name[:44]} FAILED {elapsed:.0f}s: {str(e)[:120]}",
                    flush=True,
                )
            results.append(rec)
            write_report(report_path, args.tenant, results, time.monotonic() - started)

    total = time.monotonic() - started
    ok = [r for r in results if r.get("status") == "success"]
    chunks = sum(r.get("chunks_indexed", 0) for r in ok)
    ents = sum(r.get("entities", 0) for r in ok)

    print("=" * 78)
    print(f"  {len(ok)}/{len(results)} succeeded | {chunks} chunks indexed | {ents} entities")
    print(f"  total {total / 60:.1f} min", end="")
    if durations:
        print(
            f" | avg {sum(durations) / len(durations):.0f}s/doc | rate {chunks / max(total, 1):.2f} chunks/s"
        )
    else:
        print()
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="data/raw")
    p.add_argument("--api", default="http://localhost:8800")
    p.add_argument("--tenant", default="corpus500")
    p.add_argument("--report", default="eval/results/ingest_corpus500.json")
    p.add_argument("--limit", type=int, default=0, help="Only ingest N files (0 = all)")
    p.add_argument(
        "--stratified",
        action="store_true",
        help="With --limit, sample across the size distribution",
    )
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    asyncio.run(main(p.parse_args()))
