#!/usr/bin/env python3
"""Phase 7 — Ablation evaluation runner.

Runs the same query set through multiple V3 pipeline configurations:
  C2. v3_chat:          /api/v3/chat (full V3, no GAEA view active)
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
    """ReAct — use /chat endpoint with force_react=True so router enforces query-type guard.

    This is the correct way to test ReAct: the smart router classifies the query
    type first, then forces ReAct for multi-hop/summarization/analytical.
    Using /chat/react directly bypasses routing and would send factual queries
    through ReAct unnecessarily (which is the bug we fixed).
    """
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{api}/api/v3/chat",
            json={"query": query, "tenant_id": tenant, "max_steps": 6, "force_react": True},
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
        "latency_ms": d.get("latency_breakdown_ms", {}).get(
            "total", (time.monotonic() - t0) * 1000
        ),
        "steps_used": d.get("routing", {}).get("steps_used"),
    }


async def query_hefr(client: httpx.AsyncClient, api: str, query: str, tenant: str) -> dict:
    """HEFR (entity-first) retrieval — bypasses standard pipeline, direct to HEFR endpoint."""
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
    answer = " ".join((c.get("text") or "")[:200] for c in chunks[:3])
    return {
        "answer": answer,
        "sources": [{"source": c.get("source"), "chunk_id": c.get("chunk_id")} for c in chunks],
        "refused": False,
        "latency_ms": (time.monotonic() - t0) * 1000,
    }


async def query_smart_router(client: httpx.AsyncClient, api: str, query: str, tenant: str) -> dict:
    """Smart routing via /api/v3/chat — router auto-selects ReAct vs standard pipeline."""
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
        "routing": d.get("routing", {}),
    }


async def query_hefr_entity_pivot(
    client: httpx.AsyncClient, api: str, query: str, tenant: str
) -> dict:
    """HEFR for entity_pivot queries (paper lookup, entity search).
    Falls back to standard pipeline for non-entity queries.
    """
    from src.services.query_router import classify_query
    import re

    q_type = classify_query(query)
    ENTITY_PIVOT_PATTERNS = [
        r"paper nào",
        r"bài báo nào",
        r"doc.*nào",
        r"công trình",
        r"tác giả",
        r"của ai",
        r"những .* đề cập",
    ]
    is_entity_pivot = (
        q_type == "entity_pivot"
        or q_type == "multi_hop"
        and any(re.search(p, query.lower()) for p in ENTITY_PIVOT_PATTERNS)
    )
    if is_entity_pivot:
        return await query_hefr(client, api, query, tenant)
    return await query_v3_chat(client, api, query, tenant)


CONFIGS = {
    "C2_v3_chat": query_v3_chat,
    "C3_v3_chat_with_gaea": query_v3_chat,
    "C4_v3_react": query_v3_react,
    "C5_hefr_only": query_hefr,
    "C6_smart_router": query_smart_router,
    "C7_hefr_entity_pivot": query_hefr_entity_pivot,
}


def check_expected_doc(
    sources: list[dict], expected_doc: str | None, expected_docs: list[str] | None
) -> int:
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


async def keyword_hit_ratio(
    answer: str,
    keywords: list[str],
    http: httpx.AsyncClient | None = None,
    embed_url: str = "http://localhost:11434",
    embed_model: str = "bge-m3",
) -> tuple[float, list[dict]]:
    """
    Compute keyword hit ratio using BGE cosine similarity (paraphrase-aware).

    Returns (ratio, per_keyword_breakdown).
    per_keyword_breakdown = list of {keyword, hit_type: "literal"|"semantic"|"missed", similarity: float}

    A keyword counts as "hit" if either:
      - It appears literally in the answer (case-insensitive): hit_type="literal"
      - Its BGE embedding cosine similarity to the answer > 0.65: hit_type="semantic"
      - Neither: hit_type="missed"
    """
    if not keywords:
        return 1.0, []
    answer_lower = answer.lower()

    # Fast path: count literal hits
    literal_hits = sum(1 for k in keywords if k.lower() in answer_lower)
    literal_ratio = literal_hits / len(keywords)

    # Semantic path: use BGE cosine similarity
    if http is None:
        breakdown = [
            {
                "keyword": kw,
                "hit_type": "literal" if kw.lower() in answer_lower else "missed",
                "similarity": None,
            }
            for kw in keywords
        ]
        return literal_ratio, breakdown

    try:
        # Embed all keywords in parallel
        kw_tasks = []
        for kw in keywords:
            kw_tasks.append(embed_single(http, embed_url, embed_model, kw, timeout=15.0))
        kw_results = await asyncio.gather(*kw_tasks, return_exceptions=True)

        kw_embeds: list[tuple[str, list[float]]] = []
        for kw, result in zip(keywords, kw_results):
            if isinstance(result, Exception):
                kw_embeds.append((kw, []))
            else:
                kw_embeds.append((kw, result))

        if not kw_embeds or not any(e for _, e in kw_embeds):
            breakdown = [
                {
                    "keyword": kw,
                    "hit_type": "literal" if kw.lower() in answer_lower else "missed",
                    "similarity": None,
                }
                for kw in keywords
            ]
            return literal_ratio, breakdown

        # Embed answer (truncated for efficiency)
        answer_emb = await embed_single(http, embed_url, embed_model, answer[:2000], timeout=15.0)
        if not answer_emb:
            breakdown = [
                {
                    "keyword": kw,
                    "hit_type": "literal" if kw.lower() in answer_lower else "missed",
                    "similarity": None,
                }
                for kw in keywords
            ]
            return literal_ratio, breakdown

        from src.services.embedding import cosine_similarity

        semantic_hits = 0
        breakdown: list[dict] = []
        for kw, kw_emb in kw_embeds:
            hit_type = "missed"
            sim_val = 0.0
            if kw_emb and answer_emb:
                sim_val = cosine_similarity(kw_emb, answer_emb)
                if kw.lower() in answer_lower:
                    hit_type = "literal"
                    semantic_hits += 1
                elif sim_val > 0.65:
                    hit_type = "semantic"
                    semantic_hits += 1
            breakdown.append({"keyword": kw, "hit_type": hit_type, "similarity": round(sim_val, 4)})

        return semantic_hits / len(keywords), breakdown
    except Exception:
        breakdown = [
            {
                "keyword": kw,
                "hit_type": "literal" if kw.lower() in answer_lower else "missed",
                "similarity": None,
            }
            for kw in keywords
        ]
        return literal_ratio, breakdown


REFUSAL_PATTERNS = [
    r"không có đủ thông tin",
    r"không tìm thấy",
    r"không thể trả lời",
    r"i don'?t have enough information",
]


def is_refused(answer: str) -> bool:
    a = answer.lower()
    return any(re.search(p, a) for p in REFUSAL_PATTERNS)


async def main(args):
    bench = json.loads(Path(args.bench).read_text())
    queries = bench["queries"]
    print(
        f"Ablation eval — {len(queries)} queries × {len(CONFIGS)} configs = {len(queries) * len(CONFIGS)} runs"
    )
    print(f"Estimated time: {len(queries) * len(CONFIGS) * 20 / 60:.1f} min\n")

    results = {cfg: [] for cfg in CONFIGS}

    async with httpx.AsyncClient(timeout=300.0) as client:
        for q_idx, q in enumerate(queries, 1):
            print(f"[{q_idx}/{len(queries)}] {q['id']} ({q['category']}): {q['query'][:50]}")
            for cfg_name, cfg_fn in CONFIGS.items():
                if args.skip and cfg_name in args.skip:
                    continue
                r = await cfg_fn(client, args.api, q["query"], args.tenant)
                expected_count = len(
                    {q.get("expected_doc")} | set(q.get("expected_docs", [])) - {None}
                )
                docs_found = check_expected_doc(
                    r.get("sources", []), q.get("expected_doc"), q.get("expected_docs")
                )
                kw_hit_ratio, kw_breakdown = await keyword_hit_ratio(
                    r.get("answer", ""),
                    q.get("expected_keywords", []),
                    http=client,
                    embed_url=args.embed_url,
                )
                refused = is_refused(r.get("answer", "")) or r.get("refused", False)
                expect_refusal = q.get("expect_refusal", False)
                refusal_correct = refused == expect_refusal

                results[cfg_name].append(
                    {
                        "q_id": q["id"],
                        "category": q["category"],
                        "latency_ms": r.get("latency_ms", 0),
                        "docs_found": docs_found,
                        "docs_expected": expected_count,
                        "kw_hit_ratio": kw_hit_ratio,
                        "kw_breakdown": kw_breakdown,
                        "refused": refused,
                        "expect_refusal": expect_refusal,
                        "refusal_correct": refusal_correct,
                        "answer_excerpt": (r.get("answer") or "")[:200],
                        "error": r.get("error"),
                        # Routing metadata for C6 smart_router
                        "routing": r.get("routing"),
                    }
                )
                status = "OK" if not r.get("error") else "ERR"
                routing_str = ""
                if cfg_name == "C6_smart_router" and r.get("routing"):
                    rt = r["routing"]
                    routing_str = f" [{rt.get('query_type', '?')}:{'React' if rt.get('react_used') else 'Std'}]"
                print(
                    f"  {cfg_name:<25} {status:<5} {r.get('latency_ms', 0) / 1000:>6.1f}s "
                    f"docs={docs_found}/{expected_count} kw={kw_hit_ratio:.2f} refused={refused}{routing_str}"
                )
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
        # Compute per-keyword hit type breakdown across all runs
        all_bd = [d for r in ok_runs for d in (r.get("kw_breakdown") or [])]
        lit_ok = sum(1 for d in all_bd if d.get("hit_type") == "literal") / max(len(all_bd), 1)
        sem_ok = sum(1 for d in all_bd if d.get("hit_type") == "semantic") / max(len(all_bd), 1)
        miss = sum(1 for d in all_bd if d.get("hit_type") == "missed") / max(len(all_bd), 1)
        kw_ok = lit_ok + sem_ok
        summary[cfg] = {
            "n": n,
            "p50_latency_ms": p50,
            "avg_doc_recall": avg_docs,
            "avg_kw_hit": avg_kw,
            "refusal_accuracy": refusal_acc,
            "kw_breakdown": {
                "literal": round(lit_ok, 3),
                "semantic": round(sem_ok, 3),
                "missed": round(miss, 3),
            },
        }
        print(
            f"{cfg:<25} {n:>4} {p50 / 1000:>10.1f}s  {avg_docs:>13.2%}  {kw_ok:>7.2%}  {refusal_acc:>11.2%}  kw[lit={lit_ok:.0%} sem={sem_ok:.0%} miss={miss:.0%}]"
        )

    # Per-query detailed breakdown
    print("\n" + "═" * 90)
    print("PER-QUERY BREAKDOWN")
    print("═" * 90)
    for cfg, runs in results.items():
        ok_runs = [r for r in runs if not r.get("error")]
        if not ok_runs:
            continue
        print(f"\n{'─' * 90}")
        print(f"  {cfg}")
        print(
            f"  {'q_id':<8} {'category':<18} {'docs':>7} {'kw':>5} {'kw_detail':<40} {'refused':>8} {'lat(s)':>7}"
        )
        print(f"  {'─' * 8} {'─' * 18} {'─' * 7} {'─' * 5} {'─' * 40} {'─' * 8} {'─' * 7}")
        for r in ok_runs:
            kw = r["kw_hit_ratio"]
            bd = r.get("kw_breakdown") or []
            kw_detail = (
                ",".join(f"{d['keyword'][:10]}({d['hit_type'][0]})" for d in bd) if bd else "n/a"
            )
            refused = "YES" if r["refused"] else "no"
            correct_refusal = "✓" if r["refusal_correct"] else "✗"
            print(
                f"  {r['q_id']:<8} {r['category']:<18} {r['docs_found']}/{r['docs_expected']:>4} {kw:>5.2f} {kw_detail[:40]:<40} {refused:>5} {correct_refusal:<3} {r['latency_ms'] / 1000:>6.1f}s"
            )

    # Routing breakdown for C6 smart_router
    if "C6_smart_router" in results and results["C6_smart_router"]:
        print("\nC6 Smart Router — Query Type Breakdown:")
        print("─" * 60)
        from collections import defaultdict

        by_type: dict[str, list] = defaultdict(list)
        for r in results["C6_smart_router"]:
            routing_data = r.get("routing")
            if routing_data and isinstance(routing_data, dict):
                qt = routing_data.get("query_type", "unknown") or "unknown"
            else:
                qt = "unknown"
            by_type[qt].append(r)
        for qt, runs in sorted(by_type.items()):
            n = len(runs)
            avg_docs = sum(r["docs_found"] / max(r["docs_expected"], 1) for r in runs) / n
            avg_kw = sum(r["kw_hit_ratio"] for r in runs) / n
            refused = sum(1 for r in runs if r["refused"]) / n
            print(
                f"  {qt:<20} N={n:>2}  docs={avg_docs:.0%}  kw={avg_kw:.0%}  refused={refused:.0%}"
            )

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
    p.add_argument("--embed-url", default="http://localhost:11434")
    args = p.parse_args()
    if not args.output:
        from datetime import datetime as _d

        args.output = f"eval/results/ablation_{_d.now().strftime('%Y%m%d_%H%M%S')}.json"
    asyncio.run(main(args))
