# Enterprise RAG Stack — Technical Specification

## 1. Overview

**Name**: Enterprise Local RAG Stack v2.0
**Target**: Apple Silicon Mac (M-series, 16GB+ RAM)
**Purpose**: Production-ready Hybrid GraphRAG system, 100% local, no cloud dependency

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  User Traffic                                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Nginx (port 80) — Rate limiting, reverse proxy          │   │
│  │  ├── Open WebUI (http://localhost)                      │   │
│  │  ├── RAG API (http://localhost:8800)                    │   │
│  │  └── Langfuse (http://localhost:3000)                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
           │
           ├──────────────────────────────────────┐
           ▼                                      ▼
┌───────────────────────┐              ┌───────────────────────┐
│  Ollama (HOST native) │              │  Docker Containers      │
│  Metal GPU accel.    │              │                       │
│  • qwen3.5:4b       │              │  rag-api              │
│  • bge-m3            │              │  rag-qdrant           │
└───────────────────────┘              │  rag-neo4j            │
                                      │  rag-redis            │
                                      │  rag-postgres         │
                                      │  rag-langfuse         │
                                      │  rag-grafana          │
                                      │  rag-prometheus       │
                                      └───────────────────────┘
```

## 3. Component Inventory

### 3.1 LLM Serving — Ollama (Native, Host)
- **Why native**: Docker cannot access Metal GPU directly; Ollama must run on the host
- **Model**: `qwen3.5:4b` (Q4_K_M GGUF, ~2.5GB) — Vietnamese + coding capable
- **Embedding**: `bge-m3` (1024-dim multilingual)
- **Auto-start**: `ollama serve` (or via launchd/systemd)
- **API**: OpenAI-compatible at `http://localhost:11434`
- **GPU**: Metal acceleration via `OLLAMA_METAL=1` (default on Apple Silicon)

### 3.2 RAG API — FastAPI
- **Image**: `rag-rag-api` (Docker, linux/arm64)
- **Port**: 8800
- **Workers**: 1 (Ollama is the bottleneck, not FastAPI)
- **Async**: `uvloop` event loop
- **Endpoints**:
  - `GET /health` — liveness
  - `GET /health/deep` — full dependency check
  - `POST /v1/chat/completions` — RAG-augmented chat (streaming supported)
  - `POST /ingest/upload` — document indexing
  - `GET /v1/models` — available models
  - `GET /metrics` — Prometheus metrics
  - `POST /cache/clear` — clear semantic cache

### 3.3 Vector DB — Qdrant
- **Version**: v1.13.0
- **Port**: 6333 (REST), 6334 (gRPC)
- **Collection**: `enterprise_kb` — 1024-dim, Cosine distance
- **Memory**: 1GB limit (local dev)
- **Auth**: Disabled (internal network only)

### 3.4 Knowledge Graph — Neo4j
- **Version**: 5.26 Community
- **Ports**: 7474 (HTTP), 7687 (Bolt)
- **Plugins**: APOC
- **Auth**: Disabled for local dev
- **Memory**: 1GB heap + 512MB pagecache
- **Schema**: `(Chunk)-[:CONTAINS_ENTITY]->(Entity)-[:RELATES_TO]->(Entity)`
- **Indexes**: Full-text on entity name/description, range on id/source/type

### 3.5 Cache — Redis
- **Version**: 7 Alpine
- **Port**: 6379
- **Memory**: 256MB max, `allkeys-lru` eviction
- **Use**: Semantic query cache (embedding-keyed, TTL 1h)
- **AOF**: Disabled (local dev)

### 3.6 State — PostgreSQL
- **Version**: 16 Alpine
- **Port**: 5432
- **Memory**: 1GB limit
- **Use**: LangGraph checkpointer (future), app state
- **Tuning**: `fsync=off`, `synchronous_commit=off` for local perf

### 3.7 Observability — Langfuse v3
- **DB**: Postgres + ClickHouse
- **Port**: 3000
- **Tracing**: Enabled when `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` set

### 3.8 Monitoring — Prometheus + Grafana
- **Prometheus**: Port 9090, 15d retention
- **Grafana**: Port 3001 (provisioned with Prometheus datasource)

## 4. Data Flow

### 4.1 Document Ingestion
```
File → parse (PDF/DOCX/TXT)
    → chunk (512 chars / 64 overlap, sentence-aware)
    → [CONCURRENT]
        ├→ LLM extract entities + relationships
        └→ batch embed via Ollama
    → [CONCURRENT]
        ├→ upsert vectors to Qdrant
        └→ upsert graph to Neo4j
```

### 4.2 Query Pipeline
```
User Query
    → embed query (BGE-M3)
    → [CONCURRENT]
        ├→ semantic cache check (Redis)
        │   └→ HIT: return cached results
        └→ MISS:
            ├→ vector search (Qdrant, top 20)
            └→ graph search (Neo4j, top 20)
            → RRF fusion (k=60)
            → [CONCURRENT]
                ├→ cache result (Redis, TTL 1h)
                └→ LLM generate (context + system prompt)
            → return response
```

### 4.3 RRF (Reciprocal Rank Fusion)
```
RRF_score(chunk) = Σ weight_i / (k + rank_i)

vector_weight = 1.0
graph_weight = 1.0
k = 60
```

## 5. Configuration

All settings via environment variables (`.env`):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `qwen3.5:4b` | LLM model |
| `OLLAMA_EMBED_MODEL` | `bge-m3` | Embedding model |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `enterprise_kb` | Collection name |
| `NEO4J_URL` | `bolt://neo4j:7687` | Neo4j endpoint |
| `REDIS_URL` | `redis://redis:6379/0` | Redis endpoint |
| `MAX_CONCURRENT_REQUESTS` | `8` | Concurrency limit |
| `SEMANTIC_CACHE_TTL_S` | `3600` | Cache TTL |
| `RETRIEVAL_TOP_K` | `8` | Final retrieved chunks |
| `RETRIEVAL_VECTOR_TOP_K` | `20` | Vector search candidates |
| `RETRIEVAL_GRAPH_TOP_K` | `20` | Graph search candidates |
| `RRF_K` | `60` | RRF constant |

## 6. Resource Allocation

| Container | Memory Limit | CPU | Notes |
|---|---|---|---|
| rag-api | 1 GB | - | FastAPI + async |
| rag-qdrant | 1 GB | - | Vector DB |
| rag-neo4j | 2 GB | - | KG DB |
| rag-redis | 512 MB | - | Cache |
| rag-postgres | 1 GB | - | State |
| rag-langfuse | 1 GB | - | Observability |
| rag-langfuse-clickhouse | 1 GB | - | Traces |
| rag-langfuse-db | 512 MB | - | Langfuse state |
| rag-prometheus | 512 MB | - | Metrics |
| rag-grafana | 512 MB | - | Dashboards |
| rag-open-webui | - | - | Chat UI |
| rag-nginx | - | - | Proxy |

**Total**: ~8 GB RAM requested

## 7. Optimization Summary

| Optimization | Impact | Implementation |
|---|---|---|
| Ollama native (Metal GPU) | 5-10x faster than CPU | Host-native deployment |
| Semantic cache | 2-5x faster for repeated queries | Redis embedding-keyed |
| Concurrent retrieval | ~30% latency reduction | `asyncio.gather` |
| Batch embedding | Fewer LLM calls | `embed_batch()` |
| Connection pooling | Fewer connection setups | httpx limits |
| uvloop | 2-4x faster async | `--loop uvloop` |
| RRF fusion | Better relevance than single-source | `rrf_fuse()` |
| Memory limits | Prevent OOM | `deploy.resources.limits` |
| Async client per service | Non-blocking I/O | All clients are async |

## 8. Ports Summary

| Port | Service | Bind | Public |
|---|---|---|---|
| 80 | Nginx | 0.0.0.0 | Yes |
| 11434 | Ollama | 127.0.0.1 | Host only |
| 6333 | Qdrant | 127.0.0.1 | No |
| 7474 | Neo4j Browser | 127.0.0.1 | No |
| 7687 | Neo4j Bolt | 127.0.0.1 | No |
| 5432 | PostgreSQL | 127.0.0.1 | No |
| 6379 | Redis | 127.0.0.1 | No |
| 3000 | Langfuse | 127.0.0.1 | No |
| 3001 | Grafana | 127.0.0.1 | No |
| 9090 | Prometheus | 127.0.0.1 | No |
| 8800 | RAG API | 127.0.0.1 | No |

## 9. File Structure

```
RAG/
├── docker-compose.yml       # Full stack definition
├── Makefile                # Dev commands
├── README.md               # User guide
├── SPEC.md                 # This file
├── .env                   # Secrets (gitignored)
├── .env.example            # Template
├── .gitignore
│
├── api/
│   ├── Dockerfile          # FastAPI image
│   ├── main.py            # FastAPI app
│   └── requirements.txt    # Python deps
│
├── src/
│   ├── config.py          # Settings from env
│   ├── clients.py         # Global client holders + SemanticCache
│   ├── models.py          # Pydantic schemas
│   ├── __init__.py
│   └── services/
│       ├── retrieval.py   # Hybrid retrieval + RRF fusion
│       ├── ingestion.py   # Document pipeline
│       ├── vector.py     # Qdrant CRUD
│       ├── kg.py         # Neo4j KG operations
│       └── embedding.py   # Ollama embedding utils
│
├── scripts/
│   ├── init-qdrant.sh    # Create Qdrant collection
│   └── init-neo4j.cypher # Neo4j schema
│
├── nginx/
│   └── nginx.conf        # Reverse proxy + rate limit
│
├── prometheus/
│   └── prometheus.yml     # Scrape config
│
└── grafana/
    └── provisioning/
        └── datasources/
            └── prometheus.yml  # Auto-provision Prometheus
```

## 10. Test Suite

```
tests/
├── conftest.py          # pytest config + fixtures
├── test_health.py       # Service + API endpoint health
├── test_models.py       # Ollama LLM + embedding quality
├── test_rag_pipeline.py # Ingest + retrieval + RAG chat
└── test_performance.py  # Latency + throughput benchmarks
```

## 11. API Reference

### POST /v1/chat/completions
OpenAI-compatible RAG endpoint.

**Request**:
```json
{
  "model": "qwen3.5:4b",
  "messages": [{"role": "user", "content": "..."}],
  "temperature": 0.3,
  "max_tokens": 2048,
  "stream": false
}
```

**Response**: Same as OpenAI Chat Completions

### POST /ingest/upload
Upload and index document.

**Request**: `multipart/form-data`
- `file`: File upload (PDF/DOCX/TXT)
- `filename`: Filename string

**Response**:
```json
{
  "status": "success",
  "filename": "doc.pdf",
  "doc_hash": "a1b2c3d4",
  "chunks_indexed": 5,
  "entities_extracted": 12,
  "relationships_extracted": 7,
  "failed_chunks": 0
}
```

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Ollama 500 errors | Model not loaded | `ollama pull qwen3.5:4b` |
| Neo4j auth fails | Old auth.ini volume | Remove volume, restart |
| Qdrant 401 | Old API key volume | Remove volume, restart |
| rag-api crashloop | Import error | Check logs: `docker logs rag-api` |
| Slow embedding | CPU fallback | Verify Metal GPU available |
| Cache always miss | Embedding different | Qdrant hash-based cache key |
| Langfuse no traces | Missing API keys | Set in .env, restart rag-api |

## 13. Multi-Tenant Architecture

### 13.1 Isolation Strategy

| Store | Isolation Method |
|---|---|
| **Qdrant** | `tenant_id` in every point payload + filter on query |
| **Neo4j** | `tenant_id` property on every node |
| **Redis** | Key prefix: `rag:{tenant_id}:cache:*` |
| **API** | API key scoped to tenant, validated per request |

### 13.2 Tenant Data Model

```python
class Tenant:
    id: str           # UUID
    name: str        # "ACME Corp"
    slug: str        # "acme"
    plan: str        # "free" | "pro" | "enterprise"
    settings: dict   # retrieval_top_k, vector_weight, etc.
```

### 13.3 API Key Scoping

Every API key is bound to a `tenant_id`. The `verify_api_key` middleware:
1. Reads `X-API-Key` header
2. Hashes and looks up in key store
3. Attaches `tenant_id` and `scopes` to request context
4. All downstream calls include `tenant_id`

## 14. Source Plugin System

### 14.1 Plugin Architecture

```
PluginRegistry.discover()  →  scans plugins/sources/*/plugin.py
PluginRegistry.create_source_plugin(name)  →  BaseSourcePlugin instance
BaseSourcePlugin.ingest()  →  ParsedDocument  →  DocumentStore.ingest()
```

### 14.2 Plugin Capabilities

| Plugin | Capabilities | Supported Types |
|---|---|---|
| `file` | file, url, stream | pdf, docx, doc, txt, md, csv, xlsx |
| `webpage` | url, crawl, stream | html, webpage |
| `github` | url, crawl, scheduled | github |
| `database` | query, scheduled | postgresql, mysql, sqlite |
| `api` | url, scheduled, webhook | rest, api |
| `email` | scheduled | email, gmail |
| `arxiv` | url, scheduled | arxiv, pdf |

### 14.3 Adding a New Plugin

1. Create `plugins/sources/{name}/plugin.py`
2. Subclass `BaseSourcePlugin`
3. Implement `async def fetch(self, url_or_path, **kwargs) -> ParsedDocument`
4. Optionally implement `async def sync(self, **kwargs) -> SyncResult`
5. The plugin auto-discovers via `PluginRegistry.discover()`

## 15. Reranking

### 15.1 Available Rerankers

| Reranker | Speed | Quality | LLM Calls |
|---|---|---|---|
| `NoOpReranker` | Instant | None | 0 |
| `SemanticReranker` | Fast (~67ms/embed) | Good | Embedding only |
| `OllamaReranker` | Slow (per-candidate LLM) | Best | N x candidates |

### 15.2 Semantic Reranker (default)

Uses cosine similarity between query embedding and candidate text embeddings. No extra LLM call needed — uses BGE-M3.

### 15.3 Configuration

```python
reranker_type: "semantic"  # or "ollama", "none"
reranker_top_k: 10
```

## 16. Observability

### 16.1 Prometheus Metrics

All metrics at `/metrics`:

| Metric | Type | Description |
|---|---|---|
| `rag_requests_total` | Counter | Total API requests |
| `rag_requests_errors_total` | Counter | Total errors |
| `rag_cache_hits_total` | Counter | Cache hits |
| `rag_cache_hit_rate` | Gauge | Hit rate ratio |
| `rag_chunks_indexed_total` | Counter | Chunks indexed |
| `rag_entities_extracted_total` | Counter | Entities extracted |
| `rag_request_latency_seconds` | Histogram | Request latency by endpoint |

### 16.2 Langfuse Tracing

Traces are recorded for:
- Query embedding
- Vector + graph retrieval
- RRF fusion
- LLM generation
- Document ingestion

### 16.3 Audit Logging

Every operation is logged:
- Tenant/source/document CRUD events
- Chat queries with cache hit/miss
- Ingestion jobs with chunk counts
- API key creation/revocation

Logs written to: `~/.rag/audit/audit_{YYYY-MM-DD}.jsonl`
