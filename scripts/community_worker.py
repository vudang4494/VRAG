#!/usr/bin/env python3
"""
Community detection worker — runs Leiden clustering + LLM summary build per tenant.

Run nightly via cron or `make v2-community`.

Usage:
  # Build for all tenants
  python3 scripts/community_worker.py --all --api http://localhost:8800

  # Build for one tenant
  python3 scripts/community_worker.py --tenant default

  # Dry-run: show how many entities per tenant but don't build
  python3 scripts/community_worker.py --all --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx


async def list_tenants(client: httpx.AsyncClient, api: str) -> list[str]:
    try:
        resp = await client.get(f"{api}/api/v2/tenants")
        if resp.status_code == 200:
            data = resp.json()
            return [t["id"] for t in data if t.get("status") == "active"]
    except Exception:
        pass
    return ["default"]


async def build_one(client: httpx.AsyncClient, api: str, tenant: str, levels: int, resolution: float, min_size: int, vote_passes: int) -> dict:
    resp = await client.post(
        f"{api}/api/v3/community/build",
        json={
            "tenant_id": tenant,
            "levels": levels,
            "resolution": resolution,
            "min_size": min_size,
            "vote_passes": vote_passes,
        },
        timeout=3600.0,
    )
    resp.raise_for_status()
    return resp.json()


async def main_async(args):
    started = time.monotonic()

    async with httpx.AsyncClient(timeout=3600.0) as client:
        if args.all:
            tenants = await list_tenants(client, args.api)
        else:
            tenants = [args.tenant]

        print(f"Processing {len(tenants)} tenant(s): {tenants}")

        results = {}
        for tid in tenants:
            print(f"\n[Community] tenant={tid}")
            if args.dry_run:
                print("  (dry-run: skipping actual build)")
                continue
            t0 = time.monotonic()
            try:
                stats = await build_one(
                    client, args.api, tid,
                    args.levels, args.resolution, args.min_size, args.vote_passes,
                )
                elapsed = time.monotonic() - t0
                print(f"  Communities found:    {stats.get('communities')}")
                print(f"  Summaries written:    {stats.get('summaries_written')}")
                print(f"  Skipped (too small):  {stats.get('skipped_small')}")
                print(f"  Entities total:       {stats.get('entities_total')}")
                print(f"  Duration:             {elapsed:.1f}s")
                results[tid] = stats
            except Exception as e:
                print(f"  FAILED: {e}")
                results[tid] = {"error": str(e)[:200]}

    total = time.monotonic() - started
    print(f"\n══ Total duration: {total:.1f}s ══")

    if args.report:
        args.report.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"Report: {args.report}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://localhost:8800")
    p.add_argument("--tenant", default="default")
    p.add_argument("--all", action="store_true", help="Process all active tenants")
    p.add_argument("--levels", type=int, default=1)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--min-size", type=int, default=3)
    p.add_argument("--vote-passes", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
