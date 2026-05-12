#!/usr/bin/env python3
"""
Build/extend an evaluation set from sample documents.

Modes:
  1. Auto-generate: feeds 5-10 chunks per doc to LLM, asks it to generate
     queries + ground-truth keywords.
  2. Validate: takes user's manual eval set and adds template fields.
  3. Merge: combines multiple eval sets, deduplicates.

Output: JSON file conforming to eval/datasets/sample_queries_vi.json schema.

Usage:
  python3 scripts/eval_build.py auto --source-dir ./docs --output eval/datasets/auto_v1.json --count 50
  python3 scripts/eval_build.py validate --input my_queries.json --output eval/datasets/clean_v1.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


GEN_QUERIES_PROMPT = """Bạn là chuyên gia tạo bộ test cho hệ thống RAG nội bộ doanh nghiệp.
Dựa trên các đoạn văn bản tham khảo dưới đây, hãy sinh ra {count} câu hỏi mà
hệ thống RAG nên trả lời được. Mỗi câu hỏi phải:
- Bằng tiếng Việt tự nhiên
- Có ý định rõ ràng (factual / analytical / summarization / comparison)
- Có thể trả lời được từ văn bản tham khảo
- Bao gồm các thực thể, số liệu, hoặc khái niệm cụ thể

Trả về CHỈ JSON:
{{
  "queries": [
    {{
      "query": "...",
      "intent": "factual|analytical|summarization|comparison",
      "expected_keywords": ["...", "..."],
      "ground_truth_doc_keywords": ["...", "..."]
    }}
  ]
}}

Các đoạn tham khảo:
{context}

JSON:"""


async def auto_generate(source_dir: Path, output: Path, count_per_doc: int, api_base: str) -> None:
    import httpx

    docs = []
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in (".pdf", ".docx", ".txt", ".md", ".html", ".csv", ".xlsx"):
            docs.append(path)

    if not docs:
        print(f"No documents found in {source_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(docs)} documents. Generating {count_per_doc} queries per doc via LLM.")

    out_queries: list[dict[str, Any]] = []
    qid_counter = 0

    async with httpx.AsyncClient(timeout=180.0) as client:
        for path in docs:
            print(f"  - {path.name}")
            try:
                # Read first ~3000 chars
                content = path.read_bytes()[:8000]
                text = content.decode("utf-8", errors="replace")[:3000]
            except Exception as e:
                print(f"    Skipped (read error: {e})")
                continue

            prompt = GEN_QUERIES_PROMPT.format(count=count_per_doc, context=text)
            try:
                resp = await client.post(
                    f"{api_base}/v1/chat/completions",
                    json={
                        "model": "qwen3.5:4b",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.5,
                        "max_tokens": 2000,
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
                # Strip code fences
                import re
                raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
                data = json.loads(raw)
                for q in data.get("queries", []):
                    qid_counter += 1
                    out_queries.append({
                        "id": f"q_auto_{qid_counter:04d}",
                        "query": q.get("query", ""),
                        "intent": q.get("intent", "factual"),
                        "expected_keywords": q.get("expected_keywords", []),
                        "ground_truth_doc_keywords": q.get("ground_truth_doc_keywords", []),
                        "source_doc": path.name,
                    })
            except Exception as e:
                print(f"    Failed: {e}")

    dataset = {
        "name": "Auto-generated Eval Set",
        "description": f"Generated from {len(docs)} documents via LLM.",
        "version": "0.1.0",
        "queries": out_queries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(out_queries)} queries to {output}")


def validate(input_path: Path, output_path: Path) -> None:
    data = json.loads(input_path.read_text())
    queries = data.get("queries", []) if isinstance(data, dict) else data
    cleaned: list[dict[str, Any]] = []
    for i, q in enumerate(queries, 1):
        if not q.get("query"):
            continue
        cleaned.append({
            "id": q.get("id", f"q_clean_{i:04d}"),
            "query": q["query"].strip(),
            "intent": q.get("intent", "factual"),
            "expected_keywords": q.get("expected_keywords", []),
            "acceptable_refusal_keywords": q.get("acceptable_refusal_keywords", []),
            "ground_truth_doc_keywords": q.get("ground_truth_doc_keywords", []),
            "expect_refusal": q.get("expect_refusal", False),
            "source_doc": q.get("source_doc"),
        })
    output = {
        "name": data.get("name", "Cleaned Eval Set"),
        "description": data.get("description", ""),
        "version": data.get("version", "0.1.0"),
        "queries": cleaned,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Wrote {len(cleaned)} validated queries to {output_path}")


def merge(inputs: list[Path], output_path: Path) -> None:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for in_path in inputs:
        data = json.loads(in_path.read_text())
        for q in data.get("queries", []):
            key = q.get("query", "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(q)
    out = {
        "name": "Merged Eval Set",
        "description": f"Merged from {len(inputs)} sources, deduplicated.",
        "version": "0.1.0",
        "queries": merged,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Merged {len(merged)} unique queries → {output_path}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_auto = sub.add_parser("auto", help="Auto-generate queries from docs via LLM")
    p_auto.add_argument("--source-dir", type=Path, required=True)
    p_auto.add_argument("--output", type=Path, required=True)
    p_auto.add_argument("--count", type=int, default=5)
    p_auto.add_argument("--api", default="http://localhost:11434")

    p_validate = sub.add_parser("validate", help="Validate + normalize a user eval set")
    p_validate.add_argument("--input", type=Path, required=True)
    p_validate.add_argument("--output", type=Path, required=True)

    p_merge = sub.add_parser("merge", help="Merge multiple eval sets")
    p_merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    p_merge.add_argument("--output", type=Path, required=True)

    args = p.parse_args()

    if args.cmd == "auto":
        asyncio.run(auto_generate(args.source_dir, args.output, args.count, args.api))
    elif args.cmd == "validate":
        validate(args.input, args.output)
    elif args.cmd == "merge":
        merge(args.inputs, args.output)


if __name__ == "__main__":
    main()
