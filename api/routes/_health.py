"""Health endpoints — /health and /health/deep."""

from typing import Any

from fastapi import APIRouter
from loguru import logger

from src.config import get_settings

router = APIRouter()


@router.get("/health", tags=["v3"])
async def health_v3():
    settings = get_settings()
    return {
        "status": "ok",
        "pipeline_v2_enabled": settings.pipeline_v2_enabled,
        "consistency_num_views": settings.consistency_num_views,
        "rerank_stage3_enabled": settings.rerank_stage3_enabled,
        "validation_enabled": settings.validation_enabled,
        "community_enabled": settings.community_enabled,
    }


@router.get("/health/deep", tags=["v3"])
async def health_v3_deep():
    """
    Detailed health of V2 pipeline components.
    Reports availability of cross-encoder, igraph (Leiden), and dependent libs.
    """
    settings = get_settings()
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    clients = clients_get()

    def _check(name: str, import_path: str) -> dict[str, Any]:
        try:
            __import__(import_path)
            return {"name": name, "ok": True}
        except Exception as e:
            return {"name": name, "ok": False, "error": str(e)[:200]}

    deps = [
        _check("sentence-transformers", "sentence_transformers"),
        _check("python-docx", "docx"),
        _check("openpyxl", "openpyxl"),
        _check("python-igraph", "igraph"),
        _check("leidenalg", "leidenalg"),
        _check("networkx", "networkx"),
        _check("docling", "docling"),
        _check("pypdf", "pypdf"),
    ]

    # Quick component pings
    qdrant_ok = True
    qdrant_collections: list[str] = []
    try:
        cols = await clients.qdrant.get_collections()
        qdrant_collections = [c.name for c in cols.collections]
    except Exception:
        qdrant_ok = False

    neo4j_ok = True
    neo4j_node_count = 0
    try:
        async with clients.neo4j.session() as s:
            r = await s.run("MATCH (n) RETURN count(n) AS c LIMIT 1")
            row = await r.single()
            neo4j_node_count = int(row["c"]) if row else 0
    except Exception:
        neo4j_ok = False

    ollama_ok = True
    ollama_models: list[str] = []
    try:
        resp = await clients.http.get(f"{settings.ollama_base_url}/api/tags", timeout=10.0)
        ollama_models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        ollama_ok = False

    v2_metrics: dict[str, Any] = {}
    try:
        metrics_get = __import__("src.metrics", fromlist=["get_metrics"]).get_metrics
        v2_metrics = metrics_get().get_v2_metrics()
    except Exception as e:
        logger.debug(f"V2 metrics fetch failed: {e}")

    return {
        "status": "ok" if all([qdrant_ok, neo4j_ok, ollama_ok]) else "degraded",
        "pipeline_v2_enabled": settings.pipeline_v2_enabled,
        "components": {
            "qdrant": {"ok": qdrant_ok, "collections": qdrant_collections},
            "neo4j": {"ok": neo4j_ok, "node_count": neo4j_node_count},
            "ollama": {"ok": ollama_ok, "models": ollama_models, "url": settings.ollama_base_url},
        },
        "dependencies": deps,
        "config_summary": {
            "consistency_num_views": settings.consistency_num_views,
            "entity_vote_passes": settings.entity_vote_passes,
            "query_reformulations": settings.query_reformulations,
            "rerank_stage1_enabled": settings.rerank_stage1_enabled,
            "rerank_stage3_enabled": settings.rerank_stage3_enabled,
            "generation_drafts": settings.generation_drafts,
            "generation_refine_enabled": settings.generation_refine_enabled,
            "validation_enabled": settings.validation_enabled,
            "community_enabled": settings.community_enabled,
        },
        "metrics_v2": v2_metrics,
    }
