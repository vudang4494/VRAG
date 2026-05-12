#!/usr/bin/env python3
"""
Run 10 evaluation scenarios against /api/v3/chat after ingesting eval papers.

Each scenario has: query, expected_intent, expected_pattern (local/balanced/global),
expected_papers (which papers should appear in sources), check_keywords.

Output: /tmp/eval_scenarios_result.json with per-scenario detail + summary.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import httpx


# 10 scenarios designed for the eval paper set (4 RAG + 3 embedding + 2 graph + 1 unrelated)
SCENARIOS = [
    {
        "id": "S01_factual_local",
        "type": "factual / local",
        "query": "GraphRAG là gì? Tác giả nào đã đề xuất phương pháp này?",
        "expected_intent": "factual",
        "expected_pattern": "local",
        "expected_papers": ["graphrag"],
        "check_keywords": ["graph", "community", "summarization"],
    },
    {
        "id": "S02_cross_paper_synthesis",
        "type": "cross-doc / analytical",
        "query": "So sánh HyDE và GraphRAG: cả hai đều cải thiện RAG, nhưng theo cách nào khác nhau?",
        "expected_intent": "comparison",
        "expected_pattern": "balanced",
        "expected_papers": ["hyde", "graphrag"],
        "check_keywords": ["hypothetical", "graph", "retrieval"],
    },
    {
        "id": "S03_multi_hop",
        "type": "multi-hop / KG traversal",
        "query": "Các phương pháp RAG được trình bày trong các paper sử dụng loại embedding model nào?",
        "expected_intent": "analytical",
        "expected_pattern": "global",
        "expected_papers": ["graphrag", "hyde", "self-rag", "ragas", "bge-m3"],
        "check_keywords": ["embedding", "dense", "vector"],
    },
    {
        "id": "S04_summarization_macro",
        "type": "macro / summarization",
        "query": "Tóm tắt các chủ đề chính được nghiên cứu trong tất cả paper về RAG.",
        "expected_intent": "summarization",
        "expected_pattern": "global",
        "expected_papers": ["graphrag", "hyde", "self-rag", "ragas"],
        "check_keywords": ["RAG", "retrieval", "augmented"],
    },
    {
        "id": "S05_OOD_refusal",
        "type": "out-of-domain (must refuse)",
        "query": "Cơ chế folding của protein insulin diễn ra như thế nào?",
        "expected_intent": "factual",
        "expected_pattern": "local",
        "expected_papers": [],
        "expect_refusal": True,
        "check_keywords": [],
    },
    {
        "id": "S06_embedding_specific",
        "type": "factual / dense embedding paper",
        "query": "BGE-M3 hỗ trợ bao nhiêu ngôn ngữ? Cách hoạt động của multi-functionality là gì?",
        "expected_intent": "factual",
        "expected_pattern": "local",
        "expected_papers": ["bge-m3"],
        "check_keywords": ["multilingual", "multi-functionality", "embedding"],
    },
    {
        "id": "S07_comparison_methods",
        "type": "comparison",
        "query": "So sánh ColBERT, E5, và BGE-M3 — phương pháp nào có ưu điểm gì?",
        "expected_intent": "comparison",
        "expected_pattern": "balanced",
        "expected_papers": ["colbert", "e5", "bge-m3"],
        "check_keywords": ["dense retrieval", "embedding"],
    },
    {
        "id": "S08_graph_algorithm",
        "type": "factual / graph papers",
        "query": "Thuật toán Leiden khác Louvain ở điểm nào và tại sao nó tốt hơn cho community detection?",
        "expected_intent": "analytical",
        "expected_pattern": "local",
        "expected_papers": ["leiden"],
        "check_keywords": ["community", "leiden", "louvain"],
    },
    {
        "id": "S09_cross_topic_link",
        "type": "cross-topic / graph + RAG",
        "query": "Trong GraphRAG, thuật toán phân cụm cộng đồng nào được dùng và tại sao?",
        "expected_intent": "analytical",
        "expected_pattern": "balanced",
        "expected_papers": ["graphrag", "leiden"],
        "check_keywords": ["leiden", "community", "graph"],
    },
    {
        "id": "S10_isolated_topic",
        "type": "isolated topic (biology, no link)",
        "query": "AlphaFold predict cấu trúc protein bằng cách nào?",
        "expected_intent": "factual",
        "expected_pattern": "local",
        "expected_papers": ["alphafold"],
        "check_keywords": ["protein", "folding", "structure"],
    },
]


def classify_pattern(doc_count: int, total: int) -> str:
    if total == 0:
        return "empty"
    if doc_count <= 2:
        return "local"
    if doc_count <= 4:
        return "balanced"
    return "global"


def keyword_hits(answer: str, keywords: list[str]) -> tuple[int, int]:
    if not keywords:
        return (0, 0)
    al = answer.lower()
    hits = sum(1 for k in keywords if k.lower() in al)
    return (hits, len(keywords))


def papers_in_sources(sources: list[dict], expected: list[str]) -> tuple[list[str], list[str]]:
    """Match expected paper-slug against sources by checking source/chunk_id substring."""
    actual_papers: set[str] = set()
    for s in sources:
        src = (s.get("source") or "").lower()
        chunk_id = (s.get("chunk_id") or "").lower()
        for exp in expected:
            if exp.lower() in src or exp.lower() in chunk_id:
                actual_papers.add(exp)
    found = list(actual_papers)
    missing = [e for e in expected if e.lower() not in {a.lower() for a in actual_papers}]
    return found, missing


async def run_one(client: httpx.AsyncClient, api: str, sc: dict, tenant: str) -> dict:
    query = sc["query"]
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{api}/api/v3/chat",
            json={
                "query": query,
                "tenant_id": tenant,
                "max_retries": 0,
                "include_sources": True,
            },
            timeout=600.0,
        )
        elapsed = time.monotonic() - t0
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {
            "id": sc["id"],
            "type": sc["type"],
            "query": query,
            "error": str(e)[:200],
            "elapsed_seconds": time.monotonic() - t0,
        }

    sources = data.get("sources", [])
    answer = data.get("answer", "")
    refused = data.get("refused", False)
    intent_actual = data.get("intent", "")
    validation = data.get("validation", {})

    # Compute pattern
    doc_set = set()
    for s in sources:
        src = s.get("source") or s.get("metadata", {}).get("source")
        if src:
            doc_set.add(src)
    pattern_actual = classify_pattern(len(doc_set), len(sources))

    # Check keyword hits
    hits, total_kw = keyword_hits(answer, sc.get("check_keywords", []))

    # Check papers
    found_papers, missing_papers = papers_in_sources(sources, sc.get("expected_papers", []))

    # Pass criteria
    expected_refusal = sc.get("expect_refusal", False)
    correct_refusal = (refused == expected_refusal)

    return {
        "id": sc["id"],
        "type": sc["type"],
        "query": query,
        "expected_intent": sc.get("expected_intent"),
        "actual_intent": intent_actual,
        "intent_match": intent_actual == sc.get("expected_intent"),
        "expected_pattern": sc.get("expected_pattern"),
        "actual_pattern": pattern_actual,
        "pattern_match": pattern_actual == sc.get("expected_pattern"),
        "expected_refusal": expected_refusal,
        "actual_refusal": refused,
        "correct_refusal": correct_refusal,
        "refusal_reason": data.get("refusal_reason"),
        "confidence": data.get("confidence", 0),
        "validation_passed": validation.get("passed"),
        "grounded_ratio": validation.get("grounded_ratio", 0),
        "citation_ratio": validation.get("citation_ratio", 0),
        "keyword_hits": f"{hits}/{total_kw}",
        "expected_papers": sc.get("expected_papers", []),
        "found_papers": found_papers,
        "missing_papers": missing_papers,
        "source_count": len(sources),
        "doc_count": len(doc_set),
        "elapsed_seconds": elapsed,
        "answer": answer,
        "sources": [
            {
                "chunk_id": s.get("chunk_id"),
                "source": s.get("source"),
                "format": s.get("format"),
                "level": s.get("chunk_level"),
                "score": s.get("final_score"),
            }
            for s in sources
        ],
    }


async def main(args):
    print(f"Running {len(SCENARIOS)} evaluation scenarios against {args.api} (tenant={args.tenant})\n")

    results: list[dict] = []
    async with httpx.AsyncClient(timeout=1200.0) as client:
        for i, sc in enumerate(SCENARIOS, 1):
            print(f"[{i}/{len(SCENARIOS)}] {sc['id']}: {sc['query'][:70]}")
            r = await run_one(client, args.api, sc, args.tenant)
            results.append(r)
            if "error" in r:
                print(f"   ERR: {r['error']}")
            else:
                refused_ind = "REFUSED" if r["actual_refusal"] else "answered"
                print(f"   {refused_ind} | intent: {r['actual_intent']} (exp {r['expected_intent']}) | "
                      f"pattern: {r['actual_pattern']} (exp {r['expected_pattern']}) | "
                      f"sources: {r['source_count']} chunks / {r['doc_count']} docs | "
                      f"validation: {r['validation_passed']} | "
                      f"kw: {r['keyword_hits']} | "
                      f"papers found: {len(r['found_papers'])}/{len(r['expected_papers'])} | "
                      f"time: {r['elapsed_seconds']/60:.1f}min")
            print()

    # Aggregate
    success = [r for r in results if "error" not in r]
    intent_correct = sum(1 for r in success if r.get("intent_match"))
    pattern_correct = sum(1 for r in success if r.get("pattern_match"))
    refusal_correct = sum(1 for r in success if r.get("correct_refusal"))
    validation_passed = sum(1 for r in success if r.get("validation_passed"))

    summary = {
        "total_scenarios": len(results),
        "successful": len(success),
        "errors": len(results) - len(success),
        "intent_accuracy": f"{intent_correct}/{len(success)}",
        "pattern_accuracy": f"{pattern_correct}/{len(success)}",
        "refusal_correctness": f"{refusal_correct}/{len(success)}",
        "validation_pass_rate": f"{validation_passed}/{len(success)}",
        "avg_latency_seconds": sum(r["elapsed_seconds"] for r in success) / max(len(success), 1),
    }

    print("═" * 70)
    print("  AGGREGATE")
    print("═" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    Path(args.output).write_text(json.dumps({
        "summary": summary,
        "scenarios": results,
    }, ensure_ascii=False, indent=2))
    print(f"\n  Report saved: {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://localhost:8800")
    p.add_argument("--tenant", default="eval")
    p.add_argument("--output", default="/tmp/eval_scenarios_result.json")
    args = p.parse_args()
    asyncio.run(main(args))
