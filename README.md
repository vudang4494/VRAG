# Enterprise Local RAG Stack v3.0

A production-ready Hybrid GraphRAG system that runs **100% locally** on Apple Silicon (M-series Mac) — no cloud dependency, complete data privacy.

## Key Features

- **Hybrid GraphRAG**: 9 retrieval paths (vector + knowledge graph + community summaries)
- **Multi-tenant**: Strict tenant isolation via Qdrant payload filters + Neo4j property filters
- **Apple Silicon native**: Ollama with Metal GPU acceleration, GLiNER zero-shot NER
- **ReAct agent**: Multi-step reasoning with tool-use for complex multi-hop queries
- **3-stage reranking**: Cross-encoder + semantic match + LLM judge
- **Observability**: Langfuse tracing, Prometheus metrics, Grafana dashboards

## Architecture Overview

```
Query → Router (heuristic) → Query Understanding (6 reformulations)
      → 9-path retrieval (vector/bm25/graph/community/entity-pivot)
      → Weighted RRF fusion
      → OOD detection (score + keyword overlap)
      → ReAct loop OR standard path
      → 3-stage rerank
      → 3 validation gates (hallucination / entity / citation)
      → Answer
```

Full technical details: see [SPEC.md](./SPEC.md).

## Quick Start

### 1. Prerequisites

- Apple Silicon Mac (M-series), 16GB+ Unified Memory
- `brew`, `docker`, `docker-compose`, `make`
- Ollama running on host

### 2. Setup Ollama

```bash
brew install ollama
ollama pull qwen3.5:4b
ollama pull bge-m3
ollama serve
```

### 3. Initialize & Start

```bash
# Generate credentials + build images
make init

# Start all services
make up

# Initialize database schemas
make init-all

# Health check
make health
```

### 4. Try It

```bash
# Health check
curl -s http://localhost:8800/api/v3/health

# Chat with your knowledge base
curl -s -X POST http://localhost:8800/api/v3/chat \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: eval" \
  -d '{"query":"GraphRAG là gì?","max_retries":0}'
```

## Evaluation

30-query Vietnamese benchmark suite in `eval/datasets/`. Run with:

```bash
make v2-eval
# or
python3 scripts/ablation_eval.py --bench eval/datasets/vi_benchmark_v1.json --tenant eval
```

## API Reference

All endpoints at `/api/v3/*` (see `api/routes_v3.py`):

| Endpoint | Method | Description |
|---|---|---|
| `/api/v3/health` | GET | Liveness + dependency check |
| `/api/v3/chat` | POST | Main RAG chat endpoint |
| `/api/v3/search` | POST | Direct retrieval (no generation) |
| `/api/v3/ingest/upload` | POST | Document indexing |
| `/api/v3/tenants` | GET/POST | Tenant management |
| `/api/v3/tenants/{id}/stats` | GET | Tenant statistics |
| `/api/v3/cache/clear` | POST | Clear semantic cache |
| `/metrics` | GET | Prometheus metrics |

## Services

| Component | URL | Notes |
|---|---|---|
| RAG API | `http://localhost:8800` | FastAPI + uvloop |
| Qdrant | `http://localhost:6333` | Vector DB |
| Neo4j Browser | `http://localhost:7474` | Knowledge Graph |
| Redis | `localhost:6379` | Semantic cache |
| Langfuse | `http://localhost:3000` | Tracing |
| Grafana | `http://localhost:3001` | Metrics |

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3.5:4b` | LLM model |
| `OLLAMA_EMBED_MODEL` | `bge-m3` | Embedding model |
| `QDRANT_COLLECTION` | `enterprise_kb` | Vector collection |
| `RETRIEVAL_TOP_K` | `8` | Final chunks returned |
| `QUERY_REFORMULATIONS` | `3` | Query reformulation count |
| `COMMUNITY_ENABLED` | `false` | Enable community summaries |

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](./LICENSE) for details.

### What you can do

- Use, reproduce, and distribute for any purpose (including commercial)
- Create derivative works
- Sublicense to others

### What you must do

- Include the Apache 2.0 license notice
- Include NOTICE file attribution if provided
- Clearly mark any modifications

### What you cannot do

- Use trademarks without permission
- Hold contributors liable (see Section 8)
