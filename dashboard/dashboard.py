"""
Enterprise RAG Dashboard — Gradio-based visualization and interaction hub.

Features:
  - Graph visualization (Neo4j entities + relationships)
  - Interactive RAG chat with source citations
  - Document browser with metadata
  - Source management and sync status
  - Performance metrics and charts
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import httpx
import numpy as np

# Graph visualization
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# Charts
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# Dashboard Theme & CSS
# =============================================================================
DASHBOARD_THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="teal",
    neutral_hue="slate",
)
DASHBOARD_CSS = """
    .gradio-container { max-width: 1400px !important; }
    .stats-card { background: #1a1d23; border-radius: 12px; padding: 16px; }
"""

# =============================================================================
# Config
# =============================================================================

API_BASE = "http://localhost:8800"
OLLAMA_BASE = "http://localhost:11434"
NEO4J_URL = "http://localhost:7474"
QDRANT_URL = "http://localhost:6333"

DEFAULT_TENANT = "demo-tenant"
DEFAULT_API_KEY = ""  # Set to your API key


# =============================================================================
# Helpers
# =============================================================================


def get_headers(api_key: str | None = None) -> dict:
    key = api_key or DEFAULT_API_KEY
    if key:
        return {"X-API-Key": key}
    return {}


async def aget(url: str, headers: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, headers=headers or {})
        r.raise_for_status()
        return r.json()


async def apost(url: str, json: dict, headers: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json=json, headers=headers or {})
        r.raise_for_status()
        return r.json()


# =============================================================================
# Neo4j Graph Visualization
# =============================================================================


async def fetch_graph_data(
    tenant_id: str = DEFAULT_TENANT,
    max_entities: int = 100,
    max_relationships: int = 200,
    max_chunks: int = 50,
) -> dict:
    """Fetch graph data from Neo4j for visualization."""
    try:
        query_entities = """
        MATCH (e:Entity)
        WHERE e.tenant_id = $tenant_id
        RETURN e.name as name, e.type as type, e.description as description,
               size((e)<-[:CONTAINS_ENTITY]-()) as chunk_count
        ORDER BY chunk_count DESC
        LIMIT $limit
        """
        query_rels = """
        MATCH (e1:Entity)-[r:RELATES_TO]-(e2:Entity)
        RETURN e1.name as from, e2.name as to, r.description as label
        LIMIT $limit
        """
        query_chunks = """
        MATCH (d:Document)-[:FROM_DOCUMENT]->(c:Chunk)
        WHERE d.tenant_id = $tenant_id
        RETURN d.title as doc, c.text as text, c.id as chunk_id
        LIMIT $limit
        """

        data = {
            "statements": [
                {
                    "statement": query_entities,
                    "parameters": {"tenant_id": tenant_id, "limit": max_entities},
                },
                {"statement": query_rels, "parameters": {"limit": max_relationships}},
                {
                    "statement": query_chunks,
                    "parameters": {"tenant_id": tenant_id, "limit": max_chunks},
                },
            ]
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{NEO4J_URL}/db/neo4j/tx/commit",
                json=data,
                headers={"Content-Type": "application/json"},
                auth=("neo4j", ""),
            )
            if r.status_code != 200:
                return {"error": f"Neo4j returned {r.status_code}: {r.text}"}

            result = r.json()
            results = result.get("results", [])
            if not results:
                return {"error": "No results from Neo4j"}

            entities = [dict(row) for row in results[0].get("data", [])]
            rels = [dict(row) for row in results[1].get("data", [])]
            chunks = [dict(row) for row in results[2].get("data", [])]

            return {"entities": entities, "relationships": rels, "chunks": chunks}

    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def build_graph_image(
    graph_data: dict,
    layout: str = "spring",
    show_labels: bool = True,
    color_by: str = "type",
) -> str:
    """Build a matplotlib visualization of the knowledge graph."""
    if "error" in graph_data:
        return None

    entities = graph_data.get("entities", [])
    relationships = graph_data.get("relationships", [])

    if not entities:
        return None

    G = nx.DiGraph() if color_by == "direction" else nx.Graph()

    type_colors = {
        "PERSON": "#FF6B6B",
        "ORGANIZATION": "#4ECDC4",
        "LOCATION": "#45B7D1",
        "EVENT": "#96CEB4",
        "PRODUCT": "#FFEAA7",
        "CONCEPT": "#DDA0DD",
        "TECHNOLOGY": "#98D8C8",
        "OTHER": "#C0C0C0",
    }

    for e in entities:
        color = type_colors.get(e.get("type", "OTHER"), "#C0C0C0")
        G.add_node(
            e["name"],
            type=e.get("type", "OTHER"),
            color=color,
            description=e.get("description", "")[:100],
        )

    for rel in relationships:
        if rel["from"] in G.nodes() and rel["to"] in G.nodes():
            G.add_edge(
                rel["from"],
                rel["to"],
                label=rel.get("label", "relates")[:30],
            )

    if len(G.nodes()) == 0:
        return None

    fig, ax = plt.subplots(figsize=(16, 12), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")

    if layout == "spring":
        pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "hierarchical":
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, k=2.5, seed=42)
    else:
        pos = nx.spring_layout(G, k=2.5, seed=42)

    colors = [G.nodes[n].get("color", "#C0C0C0") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=800, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#555555", alpha=0.5, arrows=False, ax=ax)

    if show_labels:
        labels = {n: n[:20] for n in G.nodes()}
        nx.draw_networkx_labels(
            G,
            pos,
            labels,
            font_size=7,
            font_color="white",
            ax=ax,
        )

    legend_patches = [
        mpatches.Patch(color=color, label=etype)
        for etype, color in type_colors.items()
        if etype in [G.nodes[n].get("type", "OTHER") for n in G.nodes()]
    ]
    if legend_patches:
        ax.legend(
            handles=legend_patches,
            loc="upper left",
            fontsize=8,
            labelcolor="white",
            facecolor="#161B22",
            edgecolor="#30363D",
        )

    ax.set_title(
        f"Knowledge Graph — {len(G.nodes())} entities, {len(G.edges())} relationships",
        color="white",
        fontsize=14,
        pad=20,
    )
    ax.axis("off")
    plt.tight_layout()

    path = f"/tmp/rag_graph_{uuid.uuid4().hex[:8]}.png"
    plt.savefig(path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    return path


# =============================================================================
# Chat Interface
# =============================================================================

CHAT_HISTORY: list[dict] = []


async def chat_with_rag(
    message: str,
    api_key: str,
    model: str = "gemma4:e4b",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    include_sources: bool = True,
):
    """Send a RAG query and stream the response."""
    global CHAT_HISTORY
    headers = get_headers(api_key)

    payload = {
        "model": model,
        "messages": [{"role": m["role"], "content": m["content"]} for m in CHAT_HISTORY]
        + [{"role": "user", "content": message}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "include_sources": include_sources,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{API_BASE}/v1/chat/completions",
                json=payload,
                headers={**headers, "Accept": "application/json"},
            )
            r.raise_for_status()
            result = r.json()

        msg = result["choices"][0]["message"]
        content = msg.get("content") or msg.get("reasoning") or ""
        sources = result.get("sources", [])
        retrieval_time = result.get("retrieval_time_ms", 0)

        sources_md = ""
        if sources:
            sources_md = "\n\n**Sources:**\n"
            for s in sources[:5]:
                sources_md += f"- `{s['source']}` (score: {s['score']:.3f})\n"

        CHAT_HISTORY.append({"role": "user", "content": message})
        CHAT_HISTORY.append({"role": "assistant", "content": content + sources_md})

        history_tuples = []
        for i in range(0, len(CHAT_HISTORY), 2):
            u_msg = CHAT_HISTORY[i]["content"]
            a_msg = CHAT_HISTORY[i + 1]["content"] if i + 1 < len(CHAT_HISTORY) else None
            history_tuples.append((u_msg, a_msg))

        yield history_tuples, "", f"[{retrieval_time:.0f}ms retrieval]"
    except Exception as e:
        yield f"Error: {e}", "", ""


def clear_chat():
    global CHAT_HISTORY
    CHAT_HISTORY = []
    return [], "", ""


# =============================================================================
# Document Browser
# =============================================================================


async def fetch_documents(api_key: str, tenant_id: str, page: int = 1):
    """Fetch documents list."""
    headers = get_headers(api_key)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                f"{API_BASE}/api/documents",
                params={"tenant_id": tenant_id, "page": page, "page_size": 20},
                headers=headers,
            )
        if r.status_code == 200:
            return r.json()
        return {"documents": [], "total": 0}
    except Exception:
        return {"documents": [], "total": 0}


async def fetch_stats(tenant_id: str = DEFAULT_TENANT):
    """Fetch system stats from all services."""
    stats = {
        "neo4j": {},
        "qdrant": {},
        "ollama": {},
        "redis": {},
        "documents": {},
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                f"{NEO4J_URL}/db/neo4j/tx/commit",
                json={
                    "statements": [
                        {"statement": "MATCH (d:Document) RETURN count(d) as doc_count"},
                        {"statement": "MATCH (c:Chunk) RETURN count(c) as chunk_count"},
                        {"statement": "MATCH (e:Entity) RETURN count(e) as entity_count"},
                        {
                            "statement": "CALL db.labels() YIELD label RETURN collect(label) as labels"
                        },
                    ],
                },
                headers={"Content-Type": "application/json"},
                auth=("neo4j", ""),
            )
            if r.status_code == 200:
                data = r.json().get("results", [])
                if data:
                    stats["neo4j"]["doc_count"] = data[0]["data"][0]["row"][0]
                    stats["neo4j"]["chunk_count"] = data[1]["data"][0]["row"][0]
                    stats["neo4j"]["entity_count"] = data[2]["data"][0]["row"][0]
                    stats["neo4j"]["labels"] = data[3]["data"][0]["row"][0]
    except Exception as e:
        stats["neo4j"]["error"] = str(e)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{QDRANT_URL}/collections/enterprise_kb")
            if r.status_code == 200:
                info = r.json().get("result", {})
                stats["qdrant"]["points"] = info.get("points_count", 0)
                stats["qdrant"]["status"] = info.get("status", "unknown")
    except Exception as e:
        stats["qdrant"]["error"] = str(e)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags")
            if r.status_code == 200:
                models = r.json().get("models", [])
                stats["ollama"]["models"] = [m["name"] for m in models]
    except Exception as e:
        stats["ollama"]["error"] = str(e)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{API_BASE}/metrics")
            if r.status_code == 200:
                stats["redis"] = r.json()
    except Exception:
        pass

    return stats


def make_stats_cards(stats: dict) -> list:
    """Build stat cards for the dashboard."""
    neo4j = stats.get("neo4j", {})
    qdrant = stats.get("qdrant", {})
    ollama = stats.get("ollama", {})
    metrics = stats.get("redis", {})

    return [
        f"### Neo4j\n- Documents: **{neo4j.get('doc_count', '?')}**\n- Chunks: **{neo4j.get('chunk_count', '?')}**\n- Entities: **{neo4j.get('entity_count', '?')}**\n- Labels: {', '.join(neo4j.get('labels', []))}",
        f"### Qdrant\n- Points: **{qdrant.get('points', '?')}**\n- Status: **{qdrant.get('status', '?')}**",
        f"### Ollama\n" + "\n".join(f"- {m}" for m in ollama.get("models", [])),
        f"### API Metrics\n- Total requests: **{metrics.get('total_requests', 0)}**\n- Cache entries: **{metrics.get('cache_entries', 0)}**\n- Cache enabled: **{metrics.get('cache_enabled', False)}**",
    ]


# =============================================================================
# Gradio Interface
# =============================================================================


async def graph_tab(
    tenant_id: str,
    max_entities: int,
    max_rels: int,
    layout: str,
    show_labels: bool,
):
    data = await fetch_graph_data(tenant_id, max_entities, max_rels)
    if "error" in data:
        return None, data["error"]
    if not data.get("entities"):
        return None, "No entities found for this tenant."

    img_path = build_graph_image(data, layout=layout, show_labels=show_labels)
    summary = (
        f"**Graph Summary:** {len(data.get('entities', []))} entities, "
        f"{len(data.get('relationships', []))} relationships, "
        f"{len(data.get('chunks', []))} chunks\n\n"
        f"**Entity Types:** " + ", ".join(f"`{e['type']}`" for e in data.get("entities", [])[:20])
    )
    return img_path, summary


async def sources_tab(api_key: str):
    """List configured sources."""
    headers = get_headers(api_key)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                f"{API_BASE}/api/sources", params={"tenant_id": DEFAULT_TENANT}, headers=headers
            )
        if r.status_code == 200:
            sources = r.json()
            if not sources:
                return "No sources configured. Create one via API or the document upload tab."
            return "\n".join(
                f"**{s['name']}** (`{s['source_type']}`) — {s['status']}"
                + (
                    f"\n  Last sync: {s.get('last_sync_at', 'Never')}"
                    if s.get("last_sync_at")
                    else ""
                )
                + f"\n  Docs: {s.get('document_count', 0)}"
                for s in sources
            )
    except Exception as e:
        return f"Error: {e}"


async def plugins_tab():
    """List available plugins."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{API_BASE}/api/plugins")
        if r.status_code == 200:
            plugins = r.json()
    except Exception:
        plugins = []

    if not plugins:
        plugins = [
            {
                "name": "file",
                "version": "1.0.0",
                "capabilities": ["file", "url"],
                "types": ["pdf", "docx", "txt"],
            },
            {
                "name": "webpage",
                "version": "1.0.0",
                "capabilities": ["crawl", "url"],
                "types": ["html", "webpage"],
            },
            {
                "name": "github",
                "version": "1.0.0",
                "capabilities": ["sync", "scheduled"],
                "types": ["github"],
            },
            {
                "name": "database",
                "version": "1.0.0",
                "capabilities": ["query", "scheduled"],
                "types": ["sql"],
            },
            {
                "name": "api",
                "version": "1.0.0",
                "capabilities": ["scheduled", "webhook"],
                "types": ["rest", "api"],
            },
            {
                "name": "email",
                "version": "1.0.0",
                "capabilities": ["scheduled"],
                "types": ["email", "gmail"],
            },
            {
                "name": "arxiv",
                "version": "1.0.0",
                "capabilities": ["url", "scheduled"],
                "types": ["arxiv"],
            },
        ]

    return "\n\n".join(
        f"### `{p['name']}` v{p['version']}\n"
        f"**Capabilities:** {', '.join(p['capabilities'])}\n"
        f"**Types:** {', '.join(p.get('types', []))}"
        for p in plugins
    )


def build_dashboard():
    """Build the full Gradio dashboard."""
    with gr.Blocks(title="Enterprise RAG Dashboard") as demo:
        gr.Markdown(
            "# Enterprise RAG Dashboard\n"
            "Multi-tenant GraphRAG visualization & interaction hub\n"
            f"| Ollama: `{OLLAMA_BASE}` | Neo4j: `{NEO4J_URL}` | Qdrant: `{QDRANT_URL}` |"
        )

        with gr.Tabs():
            # ── Tab 1: Chat ────────────────────────────────────────────────────
            with gr.TabItem("RAG Chat"):
                gr.Markdown("### Interactive RAG Chat with source citations")
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=500,
                            avatar_images=("👤", "🤖"),
                        )
                        msg_box = gr.Textbox(
                            label="Your question",
                            placeholder="Ask about your documents...",
                            scale=4,
                        )
                        with gr.Row():
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                            clear_btn = gr.Button("Clear", scale=1)
                    with gr.Column(scale=1):
                        api_key_in = gr.Textbox(
                            label="API Key",
                            type="password",
                            placeholder="Optional — uses default if empty",
                        )
                        model_sel = gr.Dropdown(
                            choices=["gemma4:e4b"],
                            value="gemma4:e4b",
                            label="LLM Model",
                        )
                        temp_slider = gr.Slider(0.0, 1.5, value=0.3, step=0.1, label="Temperature")
                        tokens_slider = gr.Slider(64, 4096, value=1024, step=64, label="Max Tokens")
                        sources_cb = gr.Checkbox(True, label="Include Sources")
                        retrieval_info = gr.Markdown("")

                send_btn.click(
                    chat_with_rag,
                    inputs=[msg_box, api_key_in, model_sel, temp_slider, tokens_slider, sources_cb],
                    outputs=[chatbot, msg_box, retrieval_info],
                )
                msg_box.submit(
                    chat_with_rag,
                    inputs=[msg_box, api_key_in, model_sel, temp_slider, tokens_slider, sources_cb],
                    outputs=[chatbot, msg_box, retrieval_info],
                )
                clear_btn.click(clear_chat, outputs=[chatbot, msg_box, retrieval_info])

            # ── Tab 2: Graph ─────────────────────────────────────────────────
            with gr.TabItem("Knowledge Graph"):
                gr.Markdown("### Interactive Knowledge Graph Visualization")
                gr.Markdown(
                    "Nodes = Entities | Edges = Relationships | "
                    "Colors = Entity Types | Size = Connection count"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        tenant_in = gr.Textbox(DEFAULT_TENANT, label="Tenant ID")
                        max_ent = gr.Slider(10, 200, value=50, step=10, label="Max Entities")
                        max_rel = gr.Slider(10, 300, value=100, step=10, label="Max Relationships")
                        layout_sel = gr.Dropdown(
                            ["spring", "kamada", "circular", "hierarchical"],
                            value="spring",
                            label="Graph Layout",
                        )
                        labels_cb = gr.Checkbox(True, label="Show Labels")
                        refresh_btn = gr.Button("Refresh Graph", variant="primary")
                        graph_summary = gr.Markdown("")
                    with gr.Column(scale=2):
                        graph_img = gr.Image(label="Knowledge Graph", height=600, type="filepath")

                refresh_btn.click(
                    graph_tab,
                    inputs=[tenant_in, max_ent, max_rel, layout_sel, labels_cb],
                    outputs=[graph_img, graph_summary],
                )

            # ── Tab 3: Stats ─────────────────────────────────────────────────
            with gr.TabItem("Statistics"):
                gr.Markdown("### System Statistics")
                refresh_stats = gr.Button("Refresh Stats", variant="primary")
                with gr.Row():
                    stat_cards = [gr.Markdown("") for _ in range(4)]

                async def update_stats():
                    stats = await fetch_stats()
                    return make_stats_cards(stats)

                refresh_stats.click(
                    update_stats,
                    inputs=[],
                    outputs=stat_cards,
                )

                gr.Markdown("### Quick Actions")
                with gr.Row():
                    gr.Button("Ingest Test Doc", elem_id="ingest-test").click(None, None, None)
                    gr.Button("Sync All Sources").click(None, None, None)
                    gr.Button("Clear Cache").click(None, None, None)
                    gr.Button("Export Graph").click(None, None, None)

            # ── Tab 4: Sources ────────────────────────────────────────────────
            with gr.TabItem("Sources"):
                gr.Markdown("### Data Source Management")
                gr.Markdown(
                    "Configure and monitor data sources. "
                    "Sources can be synced on-demand or on a schedule."
                )
                refresh_sources = gr.Button("Refresh Sources")
                sources_list = gr.Markdown("Loading...")
                refresh_sources.click(
                    sources_tab,
                    inputs=[api_key_in],
                    outputs=[sources_list],
                )

            # ── Tab 5: Plugins ─────────────────────────────────────────────────
            with gr.TabItem("VRAG Pipeline"):
                try:
                    from dashboard.panel import build_vrag_tab

                    build_vrag_tab(api_base=API_BASE)
                except Exception as _e:
                    gr.Markdown(f"VRAG panel unavailable: {_e}")

            with gr.TabItem("Plugins"):
                gr.Markdown("### Available Source Plugins")
                gr.Markdown(
                    "Each plugin handles a specific data source type. "
                    "They all normalize content into the unified ParsedDocument format."
                )
                plugins_md = gr.Markdown("Loading plugins...")
                demo.load(plugins_tab, outputs=[plugins_md])

    return demo


if __name__ == "__main__":
    demo = build_dashboard()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        share=False,
        show_error=True,
        theme=DASHBOARD_THEME,
        css=DASHBOARD_CSS,
    )
