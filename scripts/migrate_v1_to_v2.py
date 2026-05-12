#!/usr/bin/env python3
"""
Migrate V1 documents to V2 pipeline.

Workflow:
  1. List all documents from V1 (read from Neo4j Document nodes or source files).
  2. For each document, locate source bytes (filesystem path).
  3. Re-ingest through /api/v3/ingest/upload.
  4. Track success/failure.

Usage:
  # Re-ingest all docs in a folder
  python3 scripts/migrate_v1_to_v2.py --source-dir /path/to/docs --tenant default

  # Re-ingest from a manifest file (JSON list of paths)
  python3 scripts/migrate_v1_to_v2.py --manifest migrate.json --tenant default
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx


SUPPORTED_EXTS = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown",
                  ".html", ".htm", ".csv", ".tsv", ".xlsx", ".xls",
                  ".json", ".jsonl", ".eml"}


async def ingest_file(client: httpx.AsyncClient, api: str, path: Path, tenant: str, access_level: str) -> dict:
    content = path.read_bytes()
    files = {"file": (path.name, content, "application/octet-stream")}
    data = {"tenant_id": tenant, "access_level": access_level}
    resp = await client.post(
        f"{api}/api/v3/ingest/upload",
        files=files,
        data=data,
        timeout=600.0,
    )
    resp.raise_for_status()
    return resp.json()


async def main_async(args):
    paths: list[Path] = []
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        if isinstance(manifest, list):
            paths = [Path(p) for p in manifest]
        elif isinstance(manifest, dict):
            paths = [Path(p) for p in manifest.get("files", [])]
    elif args.source_dir:
        for p in args.source_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                paths.append(p)

    if not paths:
        print("No files to migrate", file=sys.stderr)
        sys.exit(1)

    print(f"Migrating {len(paths)} files → V2 pipeline")
    print(f"API: {args.api}, tenant: {args.tenant}\n")

    success = 0
    failed: list[tuple[str, str]] = []
    started = time.monotonic()

    async with httpx.AsyncClient(timeout=600.0) as client:
        sem = asyncio.Semaphore(args.concurrent)

        async def _migrate_one(idx, path):
            nonlocal success
            async with sem:
                t0 = time.monotonic()
                try:
                    result = await ingest_file(client, args.api, path, args.tenant, args.access_level)
                    elapsed = time.monotonic() - t0
                    if result.get("status") == "success":
                        success += 1
                        print(f"  [{idx:4d}/{len(paths)}] OK   {path.name:50s} "
                              f"chunks={result.get('chunks_indexed'):3d} "
                              f"consistency={result.get('avg_consistency_score', 0):.2f} "
                              f"{elapsed:.1f}s")
                    else:
                        failed.append((str(path), result.get("reason", "unknown")))
                        print(f"  [{idx:4d}/{len(paths)}] FAIL {path.name}: {result.get('reason')}")
                except Exception as e:
                    failed.append((str(path), str(e)[:200]))
                    print(f"  [{idx:4d}/{len(paths)}] ERR  {path.name}: {e}")

        await asyncio.gather(*[_migrate_one(i, p) for i, p in enumerate(paths, 1)])

    total = time.monotonic() - started
    print("\n" + "═" * 70)
    print(f"  Migration complete: {success}/{len(paths)} succeeded ({total / 60:.1f} min)")
    print("═" * 70)
    if failed:
        print(f"\n  Failed files ({len(failed)}):")
        for path, reason in failed[:20]:
            print(f"    - {path}: {reason}")
        if len(failed) > 20:
            print(f"    ... and {len(failed) - 20} more")

    if args.report:
        report = {
            "total": len(paths),
            "success": success,
            "failed": failed,
            "duration_seconds": total,
        }
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n  Report saved to {args.report}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, help="Directory to recursively re-ingest")
    p.add_argument("--manifest", type=Path, help="JSON file listing paths to re-ingest")
    p.add_argument("--api", default="http://localhost:8800")
    p.add_argument("--tenant", default="default")
    p.add_argument("--access-level", default="INTERNAL")
    p.add_argument("--concurrent", type=int, default=2)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args()
    if not args.source_dir and not args.manifest:
        p.error("Provide --source-dir or --manifest")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
