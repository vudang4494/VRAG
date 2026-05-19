#!/usr/bin/env python3
"""Build intent centroid vectors for semantic query routing.

This script is run OFFLINE once. It:
1. Embeds ~15 anchor queries per intent via BGE-M3
2. Computes the mean centroid per intent
3. Normalizes each centroid to unit length
4. Saves to config/intent_centroids.npy

Usage:
    python scripts/build_intent_centroids.py
"""

import asyncio
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"
CONFIG_DIR.mkdir(exist_ok=True)


# ─── Anchor queries per intent ────────────────────────────────────────────────
_ANCHOR_QUERIES = {
    "factual": [
        "LightRAG là gì và hoạt động như thế nào",
        "BGE-M3 hỗ trợ bao nhiêu ngôn ngữ",
        "E5 embedding model được huấn luyện bằng phương pháp nào",
        "Self-RAG sử dụng reflection tokens để làm gì",
        "ColBERT late interaction mechanism",
        "RAPTOR tree construction method",
        "HippoRAG knowledge graph components",
        "iText2KG module structure",
        "ACORN framework chunking strategy",
        "LongContextRAG document handling approach",
        "Who is the author of the RAPTOR paper",
        "What is the maximum context length supported by BGE-M3",
        "Define HyDE retrieval approach",
        "What is the evaluation metric used in NLL-200 dataset",
        "How many parameters does GLiNER multi-v2.1 have",
    ],
    "analytical": [
        "Tại sao retrieval granularity ảnh hưởng đến chất lượng RAG",
        "Tại sao cross-encoder reranking tốt hơn bi-encoder",
        "Phân tích ưu nhược điểm của late chunking",
        "Đánh giá chiến lược chunking nào hiệu quả nhất cho long documents",
        "RAG systems nên dùng cross-encoder hay bi-encoder để rerank",
        "Late chunking khác gì traditional chunking về mặt embedding quality",
        "Các phương pháp nào dùng LLM để cải thiện retrieval",
        "Tại sao HippoRAG cần knowledge graph trong quá trình indexing",
        "Why does retrieval granularity affect RAG quality",
        "How does late chunking preserve semantic boundaries better",
        "Which chunking strategy works best for code documents",
        "Analyze the trade-offs between dense and sparse retrieval",
        "Why is entity-level indexing more effective than chunk-level",
        "What are the failure modes of vector similarity search",
        "Evaluate the effectiveness of query decomposition techniques",
    ],
    "comparison": [
        "So sánh ColBERT late interaction với BGE-M3 dense retrieval",
        "Self-RAG vs standard RAG: ưu điểm của reflection mechanism",
        "GraphRAG và standard vector RAG khác nhau như thế nào",
        "BiXSE cải thiện dense retrieval bằng cách nào",
        "So sánh KET-RAG với GraphRAG về indexing overhead",
        "EffiR cải thiện efficiency của RAG retrieval như thế nào",
        "LightRAG vs HippoRAG: cái nào tốt hơn cho entity-centric queries",
        "RAPTOR và GraphRAG khác nhau ra sao về cách tổ chức context",
        "ColBERT vs ColBERTv2: improvement in late interaction design",
        "Bi-encoder vs cross-encoder reranking performance comparison",
        "Dense retrieval vs sparse retrieval BM25 trade-offs",
        "GraphRAG vs LightRAG entity discovery capabilities",
        "HyDE vs standard retrieval: when does hypothetical document help",
        "LEGO vs standard RAG modularity comparison",
        "MINTQA vs HotpotQA: different evaluation dimensions",
    ],
    "multi_hop": [
        "Thuật toán Leiden phân cụm cộng đồng khác gì Louvain",
        "LightRAG và HippoRAG đều sử dụng knowledge graph như thế nào",
        "Mối liên hệ giữa entity extraction và graph construction pipeline",
        "RAPTOR và GraphRAG đều dùng tree structure khác nhau thế nào",
        "Các benchmark đánh giá multi-hop QA bao gồm những dataset nào",
        "GeAR cải thiện multi-hop reasoning bằng cách nào",
        "Relationship between HyDE retrieval and query reformulation",
        "How does community detection support global vs local query answering",
        "DocHopQA document hopping mechanism for multi-document reasoning",
        "Leiden vs Louvain community detection in GraphRAG context",
        "Connection between entity linking and knowledge graph completion",
        "Multi-hop reasoning in medical RAG systems",
        "Path finding between entities across different document contexts",
        "Cross-document entity coreference resolution approach",
        "Chain-of-thought prompting for multi-step retrieval tasks",
    ],
    "kg_construction": [
        "Knowledge Graph construction pipeline bao gồm những bước nào",
        "Wikontic khác gì iText2KG trong việc xây dựng knowledge graph",
        "AutoSchemaKG xây dựng schema động bằng cách nào",
        "iText2KG 4 modules hoạt động ra sao",
        "xây dựng schema ontology từ unstructured documents",
        "Entity extraction pipeline cho knowledge graph construction",
        "Ontology learning from scientific papers",
        "Schema extraction from academic literature",
        "Dynamic schema generation for domain-specific knowledge graphs",
        "GraphRAG entity extraction and relationship discovery",
        "iText2KG vs CBEA entity comparison approach",
        "Automatic ontology construction from RAG context",
        "Relationship extraction for knowledge graph population",
        "Coreference resolution in knowledge graph construction",
        "Property validation and schema alignment techniques",
    ],
}


async def embed_texts(texts: list[str], model: str, url: str) -> list[list[float]]:
    """Embed texts via Ollama /api/embeddings."""
    import httpx

    results: list[list[float]] = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for text in texts:
            resp = await client.post(
                f"{url}/api/embeddings",
                json={"model": model, "prompt": text, "keep_alive": -1},
            )
            resp.raise_for_status()
            results.append(resp.json()["embedding"])
    return results


def normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


async def main():
    import httpx

    settings_path = ROOT / ".env"
    embed_url = "http://localhost:11434"
    embed_model = "bge-m3"

    if settings_path.exists():
        for line in settings_path.read_text().splitlines():
            if "=" not in line or line.startswith("#"):
                continue
            k, v = line.split("=", 1)
            if k == "ollama_embed_url":
                embed_url = v.strip()
            elif k == "ollama_embed_model":
                embed_model = v.strip()

    print(f"Embedding model: {embed_model}")
    print(f"Embed URL: {embed_url}")

    intents = list(_ANCHOR_QUERIES.keys())
    all_centroids: dict[str, np.ndarray] = {}

    async with httpx.AsyncClient(timeout=300.0) as client:
        for intent in intents:
            queries = _ANCHOR_QUERIES[intent]
            print(f"\nEmbedding {len(queries)} queries for intent '{intent}'...")

            # Batch embed all queries for this intent
            tasks = [
                client.post(
                    f"{embed_url}/api/embeddings",
                    json={"model": embed_model, "prompt": q, "keep_alive": -1},
                )
                for q in queries
            ]
            import asyncio as _asyncio

            responses = await _asyncio.gather(*tasks, return_exceptions=True)
            vectors = []
            for i, resp in enumerate(responses):
                if isinstance(resp, Exception):
                    print(f"  [WARN] Failed to embed query {i}: {resp}")
                    continue
                resp.raise_for_status()
                vectors.append(np.array(resp.json()["embedding"]))

            if not vectors:
                raise RuntimeError(f"No vectors embedded for intent '{intent}'")

            # Mean centroid
            centroid = np.mean(vectors, axis=0)
            # Normalize to unit length
            centroid = normalize(centroid)
            all_centroids[intent] = centroid
            print(f"  -> centroid norm={np.linalg.norm(centroid):.4f}, shape={centroid.shape}")

    # Save
    out_path = CONFIG_DIR / "intent_centroids.npy"
    np.save(out_path, all_centroids)
    print(f"\nSaved centroids to: {out_path}")

    # Verify
    loaded = np.load(out_path, allow_pickle=True).item()
    for intent, vec in loaded.items():
        print(f"  {intent}: shape={vec.shape}, norm={np.linalg.norm(vec):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
