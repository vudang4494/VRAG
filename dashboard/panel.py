"""VRAG Panel — Gradio UI cho VRAG pipeline (quality-first GraphRAG).

Cung cấp 4 sub-tabs:
  - Chat: gọi /api/v3/chat với view latency breakdown + validation gates
  - Ingest: upload file → /api/v3/ingest/upload với multi-format support
  - Health: /api/v3/health/deep — component + dependencies + metrics
  - Community: trigger /api/v3/community/build

Import vào dashboard.py:
    from dashboard.panel import build_vrag_tab
    with gr.TabItem("VRAG Pipeline"):
        build_vrag_tab(api_base="http://localhost:8800")
"""

from __future__ import annotations

import json
import time
from typing import Any

import gradio as gr
import httpx


_DEFAULT_API = "http://localhost:8800"


# ── HTTP helpers ───────────────────────────────────────────────────────────────


def _get(url: str, timeout: float = 30.0) -> dict:
    with httpx.Client(timeout=timeout) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.json()


def _post_json(url: str, payload: dict, timeout: float = 300.0) -> dict:
    with httpx.Client(timeout=timeout) as c:
        r = c.post(url, json=payload)
        r.raise_for_status()
        return r.json()


def _post_files(url: str, files: dict, data: dict, timeout: float = 600.0) -> dict:
    with httpx.Client(timeout=timeout) as c:
        r = c.post(url, files=files, data=data)
        r.raise_for_status()
        return r.json()


# ── Chat ───────────────────────────────────────────────────────────────────────


def chat_v3_fn(api_base: str, query: str, tenant_id: str, format_filter: str, max_retries: int):
    if not query.strip():
        return "Hãy nhập câu hỏi.", "", "", "", 0.0, ""
    fmt_list = [f.strip() for f in format_filter.split(",") if f.strip()] if format_filter else None
    try:
        result = _post_json(
            f"{api_base}/api/v3/chat",
            {
                "query": query,
                "tenant_id": tenant_id or "default",
                "format_filter": fmt_list,
                "max_retries": int(max_retries),
                "include_sources": True,
            },
            timeout=300.0,
        )
    except Exception as e:
        return f"**Lỗi**: {e}", "", "", "", 0.0, ""

    answer = result.get("answer", "")
    if result.get("refused"):
        answer = f"[REFUSED — {result.get('refusal_reason')}]\n\n{answer}"

    val = result.get("validation", {})
    validation_md = (
        f"**Intent**: `{result.get('intent')}`\n"
        f"**Confidence**: {result.get('confidence', 0):.3f}\n"
        f"**Validation passed**: {val.get('passed')}\n"
        f"**Grounded ratio**: {val.get('grounded_ratio', 0):.3f}\n"
        f"**Citation ratio**: {val.get('citation_ratio', 0):.3f}\n"
        f"**Invalid entities**: {len(val.get('invalid_entities', []))}\n"
        f"**Failure reason**: {val.get('failure_reason') or '(none)'}"
    )

    sources = result.get("sources", [])
    sources_md = (
        "\n\n".join(
            f"**Source [{i + 1}]**: `{s.get('chunk_id')}` "
            f"(format=`{s.get('format')}`, level=`{s.get('chunk_level')}`, "
            f"score=`{s.get('final_score', 0):.3f}`, "
            f"consistency=`{s.get('consistency_score', 0):.2f}`)\n"
            f"> {s.get('text', '')[:300]}"
            for i, s in enumerate(sources)
        )
        or "_no sources_"
    )

    latency = result.get("latency_breakdown_ms", {})
    latency_md = "| Stage | ms |\n|---|---|\n" + "\n".join(
        f"| {k} | {v:.0f} |" for k, v in sorted(latency.items(), key=lambda x: -x[1])[:12]
    )

    total_ms = latency.get("total_ms", 0)
    raw_json = json.dumps(result, ensure_ascii=False, indent=2)[:4000]

    return answer, validation_md, sources_md, latency_md, total_ms / 1000.0, raw_json


# ── Ingest ─────────────────────────────────────────────────────────────────────


def ingest_v3_fn(
    api_base: str, file, tenant_id: str, access_level: str, department: str, author: str
):
    if file is None:
        return "Upload file trước.", ""
    try:
        with open(file.name, "rb") as f:
            content = f.read()
        files = {"file": (file.name.split("/")[-1], content, "application/octet-stream")}
        data = {
            "tenant_id": tenant_id or "default",
            "access_level": access_level or "INTERNAL",
        }
        if department:
            data["department"] = department
        if author:
            data["author"] = author
        result = _post_files(f"{api_base}/api/v3/ingest/upload", files, data, timeout=600.0)
    except Exception as e:
        return f"**Lỗi**: {e}", ""

    summary = (
        f"**Status**: `{result.get('status')}`\n"
        f"**Format detected**: `{result.get('format')}`\n"
        f"**Doc ID**: `{result.get('doc_id')}`\n"
        f"**Chunks indexed**: {result.get('chunks_indexed')}\n"
        f"**Chunks dropped (low quality)**: {result.get('chunks_dropped_low_quality', 0)}\n"
        f"**Avg consistency score**: {result.get('avg_consistency_score', 0):.3f}\n"
        f"**Entities extracted**: {result.get('entities_extracted', 0)}\n"
        f"**Relationships**: {result.get('relationships_extracted', 0)}\n"
        f"**Duration**: {result.get('duration_seconds', 0):.1f}s"
    )
    raw_json = json.dumps(result, ensure_ascii=False, indent=2)
    return summary, raw_json


# ── Health ─────────────────────────────────────────────────────────────────────


def health_v3_fn(api_base: str):
    try:
        data = _get(f"{api_base}/api/v3/health/deep", timeout=15.0)
    except Exception as e:
        return f"**Lỗi**: {e}", "", "", ""

    status_md = f"## Status: `{data.get('status')}`\n\nPipeline enabled: **{data.get('pipeline_v2_enabled')}**"

    components = data.get("components", {})
    comp_lines = ["| Component | OK | Detail |", "|---|---|---|"]
    for name, c in components.items():
        ok = c.get("ok")
        detail = ""
        if name == "qdrant":
            detail = f"collections: {c.get('collections', [])}"
        elif name == "neo4j":
            detail = f"nodes: {c.get('node_count', 0)}"
        elif name == "ollama":
            detail = f"models: {c.get('models', [])}"
        comp_lines.append(f"| {name} | {'OK' if ok else 'FAIL'} | {detail} |")
    components_md = "\n".join(comp_lines)

    deps = data.get("dependencies", [])
    dep_lines = ["| Dependency | OK | Error |", "|---|---|---|"]
    for d in deps:
        dep_lines.append(
            f"| {d['name']} | {'OK' if d.get('ok') else 'FAIL'} | {d.get('error', '')[:60]} |"
        )
    deps_md = "\n".join(dep_lines)

    metrics = data.get("metrics_v2", {})
    metrics_md = (
        f"**Chats total**: {metrics.get('v2_chats_total', 0)}\n"
        f"**Refusal rate**: {metrics.get('v2_refusal_rate', 0):.2%}\n"
        f"**Validation pass rate**: {metrics.get('v2_validation_pass_rate', 0):.2%}\n"
        f"**Avg grounded ratio**: {metrics.get('v2_avg_grounded_ratio', 0):.3f}\n"
        f"**Avg consistency**: {metrics.get('v2_avg_consistency_score', 0):.3f}\n"
        f"**Communities built**: {metrics.get('v2_communities_built', 0)}"
    )

    return status_md, components_md, deps_md, metrics_md


# ── Community ──────────────────────────────────────────────────────────────────


def community_v3_fn(
    api_base: str, tenant_id: str, levels: int, resolution: float, min_size: int, vote_passes: int
):
    try:
        result = _post_json(
            f"{api_base}/api/v3/community/build",
            {
                "tenant_id": tenant_id or "default",
                "levels": int(levels),
                "resolution": float(resolution),
                "min_size": int(min_size),
                "vote_passes": int(vote_passes),
            },
            timeout=3600.0,
        )
    except Exception as e:
        return f"**Lỗi**: {e}", ""

    summary = (
        f"**Communities found**: {result.get('communities', 0)}\n"
        f"**Summaries written**: {result.get('summaries_written', 0)}\n"
        f"**Skipped (too small)**: {result.get('skipped_small', 0)}\n"
        f"**Entities total**: {result.get('entities_total', 0)}\n"
        f"**Duration**: {result.get('duration_seconds', 0):.1f}s"
    )
    raw_json = json.dumps(result, ensure_ascii=False, indent=2)
    return summary, raw_json


# ── Builder ────────────────────────────────────────────────────────────────────


def build_vrag_tab(api_base: str = _DEFAULT_API) -> None:
    """Build all VRAG sub-tabs inside the parent TabItem."""
    gr.Markdown(
        "## Pipeline V2 — Quality-first GraphRAG\n\n"
        "Endpoints: `/api/v3/chat`, `/api/v3/ingest/upload`, `/api/v3/community/build`, `/api/v3/health/deep`.\n"
        "Bật bằng `PIPELINE_V2_ENABLED=1` trong `.env`."
    )

    with gr.Tabs():
        # ── Chat ───────────────────────────────────────────────────────────────
        with gr.TabItem("Chat"):
            gr.Markdown("### Hỏi đáp với deliberation + validation gates")
            with gr.Row():
                with gr.Column(scale=2):
                    chat_query = gr.Textbox(
                        label="Câu hỏi",
                        placeholder="Doanh thu Quý 3 năm 2024 là bao nhiêu?",
                        lines=3,
                    )
                    with gr.Row():
                        chat_tenant = gr.Textbox(label="Tenant", value="default", scale=1)
                        chat_format = gr.Textbox(
                            label="Format filter (csv-separated, optional)",
                            placeholder="pdf,xlsx",
                            scale=1,
                        )
                        chat_retries = gr.Slider(0, 2, value=0, step=1, label="Max retries")
                    chat_btn = gr.Button("Gửi", variant="primary")
                with gr.Column(scale=1):
                    chat_latency = gr.Number(label="Total (s)", value=0)

            with gr.Row():
                with gr.Column(scale=2):
                    chat_answer = gr.Markdown(label="Câu trả lời")
                with gr.Column(scale=1):
                    chat_validation = gr.Markdown(label="Validation")

            with gr.Row():
                chat_sources = gr.Markdown(label="Sources")
            with gr.Accordion("Latency breakdown", open=False):
                chat_latency_md = gr.Markdown()
            with gr.Accordion("Raw JSON", open=False):
                chat_raw = gr.Code(language="json")

            chat_btn.click(
                fn=lambda q, t, f, r: chat_v3_fn(api_base, q, t, f, r),
                inputs=[chat_query, chat_tenant, chat_format, chat_retries],
                outputs=[
                    chat_answer,
                    chat_validation,
                    chat_sources,
                    chat_latency_md,
                    chat_latency,
                    chat_raw,
                ],
            )

        # ── Ingest ─────────────────────────────────────────────────────────────
        with gr.TabItem("Ingest"):
            gr.Markdown(
                "### Upload tài liệu đa định dạng qua Pipeline V2\n"
                "Hỗ trợ: PDF, DOCX, XLSX, CSV, TXT, MD, HTML, JSON/JSONL chat, EML email"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    ingest_file = gr.File(label="File")
                    ingest_tenant = gr.Textbox(label="Tenant", value="default")
                    ingest_access = gr.Dropdown(
                        ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"],
                        value="INTERNAL",
                        label="Access Level",
                    )
                    ingest_dept = gr.Textbox(label="Department (optional)")
                    ingest_author = gr.Textbox(label="Author (optional)")
                    ingest_btn = gr.Button("Ingest", variant="primary")
                with gr.Column(scale=2):
                    ingest_summary = gr.Markdown()
                    with gr.Accordion("Raw JSON", open=False):
                        ingest_raw = gr.Code(language="json")

            ingest_btn.click(
                fn=lambda f, t, a, d, au: ingest_v3_fn(api_base, f, t, a, d, au),
                inputs=[ingest_file, ingest_tenant, ingest_access, ingest_dept, ingest_author],
                outputs=[ingest_summary, ingest_raw],
            )

        # ── Health ─────────────────────────────────────────────────────────────
        with gr.TabItem("Health"):
            gr.Markdown("### Deep health check Pipeline V2")
            health_btn = gr.Button("Refresh", variant="primary")
            health_status = gr.Markdown()
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Components**")
                    health_components = gr.Markdown()
                with gr.Column():
                    gr.Markdown("**Dependencies**")
                    health_deps = gr.Markdown()
            gr.Markdown("**V2 Metrics**")
            health_metrics = gr.Markdown()

            health_btn.click(
                fn=lambda: health_v3_fn(api_base),
                inputs=[],
                outputs=[health_status, health_components, health_deps, health_metrics],
            )

        # ── Community ──────────────────────────────────────────────────────────
        with gr.TabItem("Community Build"):
            gr.Markdown("### Trigger Leiden clustering + LLM community summaries")
            with gr.Row():
                with gr.Column(scale=1):
                    comm_tenant = gr.Textbox(label="Tenant", value="default")
                    comm_levels = gr.Slider(1, 3, value=1, step=1, label="Levels")
                    comm_resolution = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Resolution")
                    comm_min_size = gr.Slider(2, 10, value=3, step=1, label="Min community size")
                    comm_votes = gr.Slider(1, 5, value=3, step=1, label="Summary vote passes")
                    comm_btn = gr.Button("Build", variant="primary")
                with gr.Column(scale=2):
                    comm_summary = gr.Markdown()
                    with gr.Accordion("Raw JSON", open=False):
                        comm_raw = gr.Code(language="json")

            comm_btn.click(
                fn=lambda t, l, r, m, v: community_v3_fn(api_base, t, l, r, m, v),
                inputs=[comm_tenant, comm_levels, comm_resolution, comm_min_size, comm_votes],
                outputs=[comm_summary, comm_raw],
            )
