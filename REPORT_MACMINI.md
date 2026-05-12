# ==============================================================================
# Mac Mini M4 — RAG Stack Test Report
# Hardware: Apple M4 (10-core, 24GB LPDDR5) | Storage: 245GB SSD
# Stack: Ollama (Qwen3.5-4B + BGE-M3) + Qdrant + Neo4j + Redis + FastAPI
# Date: May 2026
# ==============================================================================

## 1. Executive Summary

> **Status**: 🟡 OPTIMIZED — Ready for testing on Mac Mini M4 24GB

This report documents the **Mac Mini M4 24GB optimized deployment** of the Enterprise RAG Stack.
The stack has been stripped of Langfuse/ClickHouse/Prometheus/Grafana observability to save
~1.5GB RAM, leaving maximum headroom for Ollama Metal GPU inference.

### Hardware Profile
| Component | Specification |
|-----------|---------------|
| CPU | Apple M4 (4P+6E cores) |
| GPU | Apple M4 (10-core Metal GPU) |
| RAM | 24 GB LPDDR5 |
| Storage | 245 GB APFS SSD |
| macOS | 15.3.2 |

### RAM Budget (24GB Total)
| Service | Allocated | Notes |
|---------|-----------|-------|
| Ollama (Metal) | ~6-8 GB | Qwen3.5-4B Q4_K_M + BGE-M3 |
| Neo4j | 1 GB | heap 768MB + pagecache 256MB |
| Qdrant | 0.5 GB | scalar quantized vectors |
| Redis | 0.2 GB | semantic cache, LRU eviction |
| rag-api | 0.8 GB | FastAPI + uvloop |
| rag-dashboard | 0.5 GB | Gradio |
| System overhead | ~5 GB | macOS + Docker |
| **Total** | **~14-17 GB** | ✅ Within 24GB |

---

## 2. Optimization Changes from v2.0

### 2.1 Architecture Changes
| Component | Before (v2.0) | After (Mini) | Savings |
|-----------|---------------|--------------|---------|
| Langfuse | ✅ v3 | ❌ Removed | ~200MB |
| ClickHouse | ✅ 24.3 | ❌ Removed | ~500MB |
| Postgres (Langfuse) | ✅ 16 | ❌ Removed | ~150MB |
| Prometheus | ✅ v3 | ❌ Removed | ~100MB |
| Grafana | ✅ 11.4 | ❌ Removed | ~200MB |
| Nginx | ✅ | ❌ Removed | ~50MB |
| Open WebUI | ✅ | ❌ Removed | ~300MB |
| **Total RAM savings** | | | **~1.5 GB** |

### 2.2 Container Optimizations
| Service | Memory Limit | Change |
|---------|-------------|--------|
| Qdrant | 512MB → **512MB** | Same |
| Neo4j | 2GB → **1GB** | heap 768MB, pagecache 256MB |
| Redis | 512MB → **192MB** | 128MB maxmemory + lazy eviction |
| rag-api | 1GB → **768MB** | No Langfuse overhead |
| rag-dashboard | 1GB → **512MB** | Minimal deps |
| postgres (main) | 1GB → **Removed** | Not needed (no Langfuse) |

### 2.3 Ollama Optimizations (M4 Metal GPU)
```bash
OLLAMA_NUM_PARALLEL=3       # M4 can handle 3 concurrent LLM streams
OLLAMA_MAX_LOADED_MODELS=2  # Load LLM + Embedder simultaneously
```

### 2.4 Qdrant Optimizations
- **HNSW m=8** (was 16) — less memory, same accuracy
- **HNSW ef_construct=100** (was 200) — faster indexing
- **Scalar int8 quantization** — 4x memory reduction
- **always_ram=true** — keep quantized vectors in RAM

### 2.5 FastAPI / uvloop Optimizations
```python
max_concurrent_requests = 6       # M4 efficiency cores handle I/O well
embed_batch_size = 32             # Larger batches for throughput (was 16)
embed_concurrent_limit = 3       # Semaphore limit (was 4)
semantic_cache_ttl = 7200s        # 2h cache (was 1h)
Qdrant connection pool = 10       # Reduced from 20
Redis max connections = 10        # Reduced from 20
LLM connection pool = 16           # Reduced from 20
```

### 2.6 Redis Optimizations
```bash
maxmemory = 128mb         # Down from unlimited
maxmemory-policy = allkeys-lru   # LRU eviction
appendonly = no          # No AOF persistence (cache only)
lazyfree-lazy-eviction = yes     # Async eviction
lazyfree-lazy-expire = yes       # Async expiration
```

---

## 3. Test Structure

```
tests/
├── conftest.py                  # Pytest fixtures + env setup
├── test_macmini_optimized.py    # NEW: Mac Mini M4 full test suite
├── test_rag_pipeline.py         # RAG pipeline tests
├── test_models.py               # LLM/embedding tests
├── test_health.py               # Service health checks
└── test_performance.py           # Performance benchmarks
```

### Run Tests
```bash
# 1. Start stack
docker compose -f docker-compose.mini.yml up -d

# 2. Run Mac Mini specific tests
pytest tests/test_macmini_optimized.py -v --tb=short

# 3. Run all tests
pytest tests/ -v --tb=short

# 4. Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 4. Test Categories

### 4.1 Service Health (`TestMacMiniServices`)
- `test_ollama_reachable` — Ollama API on host
- `test_qdrant_health` — Qdrant vector DB
- `test_neo4j_health` — Neo4j knowledge graph
- `test_redis_health` — Redis cache
- `test_rag_api_health` — FastAPI health
- `test_rag_api_deep_health` — All service checks
- `test_dashboard_reachable` — Gradio dashboard

### 4.2 LLM & Embedding (`TestMacMiniLLMPerformance`)
- `test_embedding_latency` — BGE-M3 single embed latency
- `test_llm_generation_latency` — Qwen3.5-4B generation
- `test_batch_embedding_throughput` — 32-text batch
- `test_concurrent_embedding` — 9 concurrent (sem=3)

### 4.3 RAG Pipeline (`TestRAGPipeline`)
- `test_rag_chat_completions` — Full pipeline (8 queries)
- `test_semantic_cache_hit` — Cache hit/miss verification
- `test_qdrant_collection_info` — Vector DB stats
- `test_neo4j_graph_stats` — Knowledge graph stats

### 4.4 Resource Usage (`TestMacMiniResources`)
- `test_docker_container_memory` — Per-container memory
- `test_docker_total_memory` — Total Docker RAM
- `test_system_memory` — Mac Mini RAM
- `test_qdrant_vectors_info` — Qdrant collection details

### 4.5 Optimization Verification (`TestOptimization`)
- `test_env_ollama_optimization` — Env vars applied
- `test_redis_memory_limit` — Redis 128MB limit
- `test_qdrant_sparse_quantization` — Scalar quantization ON
- `test_neo4j_memory_settings` — Neo4j heap size
- `test_dashboard_config` — Dashboard connectivity

### 4.6 Stress Test (`TestStress`)
- `test_concurrent_chat_requests` — 5 concurrent RAG requests

---

## 5. Performance Benchmarks

| Metric | Target | Target (Heavy) |
|--------|--------|----------------|
| Embedding latency (BGE-M3) | < 500ms | < 300ms |
| LLM first token | < 2s | < 1s |
| LLM throughput (tokens/s) | > 15 TPS | > 20 TPS |
| RAG pipeline (no cache) | < 5s | < 3s |
| RAG pipeline (cached) | < 200ms | < 100ms |
| Cache hit speedup | > 3x | > 5x |
| Concurrent requests | 5 concurrent OK | 10 concurrent OK |
| Memory (Docker total) | < 10 GB | < 8 GB |

---

## 6. Quick Start Commands

```bash
# === Setup ===
# 1. Copy env and start
cp .env.mini .env 2>/dev/null || true
docker compose -f docker-compose.mini.yml up -d --build

# 2. Or use startup script
./scripts/start-rag-mini.sh

# === Ollama (on HOST) ===
ollama serve &
ollama pull qwen3.5:4b
ollama pull bge-m3

# === Health Checks ===
curl http://localhost:8800/health
curl http://localhost:8800/health/deep
docker logs rag-api --tail 20

# === Tests ===
pytest tests/test_macmini_optimized.py -v

# === Dashboard ===
open http://localhost:7860

# === Stop ===
docker compose -f docker-compose.mini.yml down
```

---

## 7. File Changes Summary

### New Files
| File | Purpose |
|------|---------|
| `docker-compose.mini.yml` | Lightweight stack for Mac Mini M4 |
| `api/Dockerfile.mini` | Minimal API image (no Langfuse) |
| `api/requirements.mini.txt` | Minimal Python deps |
| `dashboard/Dockerfile.mini` | Minimal dashboard image |
| `dashboard/requirements.mini.txt` | Minimal dashboard deps |
| `scripts/start-ollama.sh` | Optimized Ollama startup |
| `scripts/start-rag-mini.sh` | Full stack startup script |
| `scripts/stop-rag-mini.sh` | Full stack shutdown |
| `tests/test_macmini_optimized.py` | Mac Mini test suite |
| `REPORT_MACMINI.md` | This report |

### Modified Files
| File | Changes |
|------|---------|
| `src/config.py` | Added embed_batch_size, embed_concurrent_limit, cache TTL=2h |
| `src/clients.py` | Reduced connection pools, added init_clients(), optimized SemanticCache |
| `src/services/embedding.py` | Batch size from config, larger dims for zero fallback |
| `api/main.py` | Removed Langfuse, simplified lifespan, updated docstring |
| `scripts/init-qdrant.sh` | Added scalar quantization, optimized HNSW params |

---

## 8. Troubleshooting

| Problem | Solution |
|---------|----------|
| Ollama not found | Start with `ollama serve` or `./scripts/start-ollama.sh` |
| Qdrant connection refused | Check `docker ps` — Qdrant needs 20s to start |
| Neo4j slow queries | Increase heap: `NEO4J_dbms_memory_heap_max__size=1G` |
| Redis OOM | Decrease `maxmemory` or increase TTL |
| Metal GPU not used | `ollama list` — model should show `mmap` not `cpu` |
| API 503 errors | Check Ollama is running with models: `ollama list` |
| Slow embeddings | Increase `embed_concurrent_limit` to 4 |

---

*Report generated: May 2026 | Stack version: 2.0-mini | Target: Mac Mini M4 24GB*
