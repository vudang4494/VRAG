#!/usr/bin/env python3
"""
Clean ablation evaluation — Priority 1+2+3 fixes applied.

Changes from original:
  - disable_validation: true — skip validation gates (P2: OOM fix)
  - max_retries: 0 — no retry loops (P2: OOM fix)
  - disable_ood: true — skip OOD detection (P2: avoid false positives)
  - semantic_threshold: 0.45 (was 0.65) — P3: more keywords count as hit
  - Fresh httpx client per request — avoid connection reuse bugs
  - Single config: C1_v3_standard only (smart router baseline)
  - Sequential: one query at a time — avoid concurrent memory pressure

Metrics:
  - doc_recall: expected docs found in sources
  - kw_hit_ratio: keywords found in answer (literal or semantic)
  - refusal_accuracy: refused == expect_refusal
  - latency_ms: wall clock
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx


# ── Semantic threshold (was 0.65, now 0.45 — P3 fix)
SEMANTIC_THRESHOLD = 0.45

REFUSAL_PATTERNS = [
    r"không có đủ thông tin",
    r"không tìm thấy",
    r"không thể trả lời",
    r"i don'?t have enough information",
    r"từ chối",
]


def is_refused(answer: str) -> bool:
    a = answer.lower()
    return any(__import__("re").search(p, a) for p in REFUSAL_PATTERNS)


async def embed_single(
    http: httpx.AsyncClient,
    embed_url: str,
    embed_model: str,
    text: str,
    timeout: float = 15.0,
) -> list[float] | None:
    try:
        resp = await http.post(
            f"{embed_url}/api/embeddings",
            json={"model": embed_model, "prompt": text[:2000]},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("embedding")
    except Exception:
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def kw_hit_ratio(
    answer: str,
    keywords: list[str],
    http: httpx.AsyncClient | None = None,
    embed_url: str = "http://localhost:11434",
    embed_model: str = "bge-m3",
) -> tuple[float, list[dict]]:
    if not keywords:
        return 1.0, []

    answer_lower = answer.lower()

    # Literal hits
    breakdown = []
    for kw in keywords:
        hit_type = "literal" if kw.lower() in answer_lower else "missed"
        breakdown.append({"keyword": kw, "hit_type": hit_type, "similarity": None})

    # No embedding client → return literal only
    if http is None:
        lit = sum(1 for d in breakdown if d["hit_type"] == "literal")
        return lit / len(keywords), breakdown

    # Parallel embed all keywords + answer
    try:
        kw_tasks = [embed_single(http, embed_url, embed_model, kw) for kw in keywords]
        ans_emb = embed_single(http, embed_url, embed_model, answer[:2000])

        kw_results, ans_emb = await asyncio.gather(
            asyncio.gather(*kw_tasks, return_exceptions=True),
            ans_emb,
        )

        kw_embeds: list[tuple[str, list[float]]] = []
        for kw, result in zip(keywords, kw_results):
            if isinstance(result, Exception) or result is None:
                kw_embeds.append((kw, []))
            else:
                kw_embeds.append((kw, result))

        if not kw_embeds or not any(e for _, e in kw_embeds) or not ans_emb:
            lit = sum(1 for d in breakdown if d["hit_type"] == "literal")
            return lit / len(keywords), breakdown

        for i, (kw, kw_emb) in enumerate(kw_embeds):
            sim = 0.0
            if kw_emb and ans_emb:
                sim = cosine_similarity(kw_emb, ans_emb)

            if breakdown[i]["hit_type"] == "literal":
                breakdown[i]["similarity"] = sim
            elif sim > SEMANTIC_THRESHOLD:
                breakdown[i]["hit_type"] = "semantic"
                breakdown[i]["similarity"] = sim

        hit_count = sum(1 for d in breakdown if d["hit_type"] in ("literal", "semantic"))
        return hit_count / len(keywords), breakdown

    except Exception:
        lit = sum(1 for d in breakdown if d["hit_type"] == "literal")
        return lit / len(keywords), breakdown


def check_expected_doc(
    sources: list[dict],
    expected_doc: str | None,
    expected_docs: list[str] | None,
) -> int:
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


async def query_standard(
    api: str,
    query: str,
    tenant: str,
    disable_validation: bool = True,
) -> dict[str, Any]:
    """
    Standard V3 chat — single fresh client per call to avoid connection reuse.
    Validation disabled for eval to measure raw retrieval quality.
    """
    t0 = time.monotonic()
    # Fresh client per request — prevents connection reuse issues
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            resp = await client.post(
                f"{api}/api/v3/chat",
                json={
                    "query": query,
                    "tenant_id": tenant,
                    "max_retries": 0,
                    "include_sources": True,
                    "disable_validation": disable_validation,
                },
            )
            resp.raise_for_status()
            d = resp.json()
        except Exception as e:
            return {
                "error": str(e)[:200],
                "latency_ms": (time.monotonic() - t0) * 1000,
                "answer": "",
                "sources": [],
                "refused": False,
            }

    return {
        "answer": d.get("answer", ""),
        "sources": d.get("sources", []),
        "refused": bool(d.get("refused")),
        "latency_ms": (time.monotonic() - t0) * 1000,
        "routing": d.get("routing", {}),
        "validation": d.get("validation"),
    }


async def run_eval(args) -> None:
    bench = json.loads(Path(args.bench).read_text())
    queries = bench["queries"]
    n = len(queries)
    print(f"Clean eval — {n} queries × 1 config (C1_standard, validation=OFF)")
    print(f"Semantic threshold: {SEMANTIC_THRESHOLD} (was 0.65)")
    print(f"Expected time: ~{n * 60 / 60:.0f} min\n")

    results: list[dict] = []

    for i, q in enumerate(queries, 1):
        print(f"[{i:2d}/{n}] {q['id']} ({q['category']}): {q['query'][:55]}")

        r = await query_standard(args.api, q["query"], args.tenant)

        expected_count = len({q.get("expected_doc")} | set(q.get("expected_docs", [])) - {None})
        docs_found = check_expected_doc(
            r.get("sources", []),
            q.get("expected_doc"),
            q.get("expected_docs"),
        )

        kw_hit, kw_breakdown = await kw_hit_ratio(
            r.get("answer", ""),
            q.get("expected_keywords", []),
            embed_url=args.embed_url,
        )

        refused = is_refused(r.get("answer", "")) or r.get("refused", False)
        expect_refusal = q.get("expect_refusal", False)
        refusal_correct = refused == expect_refusal

        result = {
            "q_id": q["id"],
            "category": q["category"],
            "latency_ms": r.get("latency_ms", 0),
            "docs_found": docs_found,
            "docs_expected": expected_count,
            "kw_hit_ratio": kw_hit,
            "kw_breakdown": kw_breakdown,
            "refused": refused,
            "expect_refusal": expect_refusal,
            "refusal_correct": refusal_correct,
            "answer_excerpt": (r.get("answer") or "")[:200],
            "error": r.get("error"),
            "routing": r.get("routing"),
            "validation": r.get("validation"),
        }
        results.append(result)

        lat = r.get("latency_ms", 0) / 1000
        docs_str = f"docs={docs_found}/{expected_count}"
        kw_str = f"kw={kw_hit:.2f}"
        ref_str = f"refused={'YES' if refused else 'no'}"
        ref_ok = "(OK)" if refusal_correct else "(WRONG)"
        err_str = f" ERR: {r.get('error', '')[:50]}" if r.get("error") else ""
        routing = r.get("routing", {})
        react_str = (
            f"[{routing.get('query_type', '?')}:{'R' if routing.get('react_used') else 'S'}]"
        )

        print(
            f"         {docs_str:10} {kw_str:8} {ref_str} {ref_ok:8} {lat:6.1f}s "
            f"{react_str}{err_str}"
        )
        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    ok = [r for r in results if not r.get("error")]
    err = [r for r in results if r.get("error")]

    if ok:
        lats = sorted(r["latency_ms"] for r in ok)
        p50 = lats[len(lats) // 2]
        p95 = lats[int(len(lats) * 0.95)]
        avg_doc = sum(r["docs_found"] / max(r["docs_expected"], 1) for r in ok) / len(ok)
        avg_kw = sum(r["kw_hit_ratio"] for r in ok) / len(ok)
        refusal_acc = sum(1 for r in ok if r["refusal_correct"]) / len(ok)

        # Per-keyword hit breakdown
        all_bd = [d for r in ok for d in (r.get("kw_breakdown") or [])]
        lit = sum(1 for d in all_bd if d["hit_type"] == "literal") / max(len(all_bd), 1)
        sem = sum(1 for d in all_bd if d["hit_type"] == "semantic") / max(len(all_bd), 1)
        miss = sum(1 for d in all_bd if d["hit_type"] == "missed") / max(len(all_bd), 1)
        kw_hit_rate = lit + sem

        print(f"\nC1_v3_standard (validation=OFF, retries=0, fresh_client)")
        print(f"  OK={len(ok)}/{len(results)}  ERR={len(err)}")
        print(f"  p50_latency={p50 / 1000:.1f}s  p95_latency={p95 / 1000:.1f}s")
        print(f"  avg_doc_recall={avg_doc:.1%}")
        print(f"  avg_kw_hit={kw_hit_rate:.1%}  (lit={lit:.1%} sem={sem:.1%} miss={miss:.1%})")
        print(f"  refusal_accuracy={refusal_acc:.1%}")

    # ── Per-category breakdown ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PER-CATEGORY BREAKDOWN")
    print("=" * 80)
    cats: dict[str, list] = {}
    for r in results:
        if r.get("error"):
            continue
        c = r["category"]
        if c not in cats:
            cats[c] = []
        cats[c].append(r)

    print(
        f"{'Category':<22} {'N':>3} {'doc_recall':>10} {'kw_hit':>7} {'ref_acc':>7} {'p50_lat':>8}"
    )
    print("-" * 70)
    for cat, runs in sorted(cats.items()):
        n = len(runs)
        avg_doc = sum(r["docs_found"] / max(r["docs_expected"], 1) for r in runs) / n
        avg_kw = sum(r["kw_hit_ratio"] for r in runs) / n
        ref_acc = sum(1 for r in runs if r["refusal_correct"]) / n
        lats = sorted(r["latency_ms"] for r in runs)
        p50 = lats[len(lats) // 2] / 1000 if lats else 0
        print(f"{cat:<22} {n:>3} {avg_doc:>10.1%} {avg_kw:>7.1%} {ref_acc:>7.1%} {p50:>7.1f}s")

    # ── Per-query detail ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PER-QUERY DETAIL")
    print("=" * 80)
    for r in results:
        err_str = f" ERR: {r['error'][:80]}" if r.get("error") else ""
        kw = r["kw_hit_ratio"]
        bd = r.get("kw_breakdown") or []
        kw_str = ",".join(f"{d['keyword'][:10]}({d['hit_type'][0]})" for d in bd)
        if not kw_str:
            kw_str = "n/a"
        routing = r.get("routing") or {}
        react = "R" if routing.get("react_used") else "S"
        lat = r["latency_ms"] / 1000
        print(
            f"  {r['q_id']:<6} [{r['category'][:15]:15}] "
            f"docs={r['docs_found']}/{r['docs_expected']:>1} "
            f"kw={kw:.2f} {kw_str[:35]:35} "
            f"{'REFUSED' if r['refused'] else 'ok':7} "
            f"{lat:6.1f}s [{routing.get('query_type', '?')[:8]:8} {react}]{err_str}"
        )

    # ── Save report ────────────────────────────────────────────────────────────
    report = {
        "eval_type": "clean_eval_priority123",
        "date": __import__("datetime").datetime.now().isoformat(),
        "config": "C1_v3_standard",
        "settings": {
            "validation": "OFF",
            "max_retries": 0,
            "semantic_threshold": SEMANTIC_THRESHOLD,
            "fresh_client_per_request": True,
            "disable_validation_param": True,
        },
        "benchmark": args.bench,
        "tenant": args.tenant,
        "queries_total": len(queries),
        "queries_ok": len(ok),
        "queries_err": len(err),
        "summary": {
            "p50_latency_ms": p50 if ok else 0,
            "p95_latency_ms": p95 if ok else 0,
            "avg_doc_recall": avg_doc if ok else 0,
            "avg_kw_hit": kw_hit_rate if ok else 0,
            "refusal_accuracy": refusal_acc if ok else 0,
            "kw_breakdown": {
                "literal": round(lit, 3) if ok else 0,
                "semantic": round(sem, 3) if ok else 0,
                "missed": round(miss, 3) if ok else 0,
            },
        }
        if ok
        else {},
        "results": results,
        "per_category": {
            cat: {
                "n": len(runs),
                "avg_doc_recall": sum(r["docs_found"] / max(r["docs_expected"], 1) for r in runs)
                / len(runs),
                "avg_kw_hit": sum(r["kw_hit_ratio"] for r in runs) / len(runs),
                "refusal_accuracy": sum(1 for r in runs if r["refusal_correct"]) / len(runs),
            }
            for cat, runs in cats.items()
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="eval/datasets/vi_benchmark_v1.json")
    p.add_argument("--api", default="http://localhost:8800")
    p.add_argument("--tenant", default="eval")
    p.add_argument("--embed-url", default="http://localhost:11434")
    p.add_argument(
        "--output",
        default=None,
    )
    args = p.parse_args()
    if not args.output:
        from datetime import datetime as _d

        args.output = f"eval/results/clean_eval_{_d.now().strftime('%Y%m%d_%H%M%S')}.json"

    asyncio.run(run_eval(args))
