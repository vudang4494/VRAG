# VRAG — Bản đồ cấu trúc (structure map)

Map thư mục + gom `src/services/` theo **giai đoạn pipeline** để dễ tìm đúng module khi phát triển.
Cấu trúc file vẫn **phẳng, canonical** (theo convention CLAUDE.md — không có v1/v2/v3, không subpackage);
tài liệu này là "lớp logic" chồng lên, không phải layout vật lý.

## Top-level

```
api/            FastAPI app — main.py + routes/_*.py (endpoints under /api)
src/            App core + services
  config.py clients.py models.py metrics.py tracing.py audit.py   ← app infrastructure
  services/     ← RAG pipeline logic (table below)
  services/chunkers/   pdf/docx/xlsx/chat/semantic/multi_signal
dashboard/      UI (open-webui / gradio)
scripts/        operational scripts: backup.sh, start/stop, ingest, db init
docs/           stable reference documents (ARCHITECTURE, STRUCTURE, PIPELINE_FLOW, PRODUCTION_PIPELINE)
config/          intent_centroids.npy (1024-dim, bge-m3)
tests/          pytest test suite
.agents/        agents workspace (AGENTS.md, skills/, memory/)
```

## `src/services/` theo giai đoạn pipeline query

Thứ tự khớp luồng `/api/chat` (xem [PIPELINE_FLOW.md](PIPELINE_FLOW.md) + [ARCHITECTURE.md](ARCHITECTURE.md)):

| Giai đoạn | Module | Vai trò |
|---|---|---|
| **0. Pre-RAG / routing** | `intent_classifier` | greeting/ood/question/follow_up — short-circuit sớm |
| | `chat_history` | semantic cache + memory layer (cosine ≥ 0.80) |
| | `query_router` | centroid intent → chọn strategy / ReAct |
| | `query_understanding` | rewrite + keywords + fast entity |
| **1. Retrieval** | `retrieval` | **orchestrator** `multi_path_retrieve` + weighted RRF |
| | `vector` | Qdrant multi-named-vector + sparse upsert/search |
| | `embedding` | embed (Metal GPU) |
| | `entity_vectors` | entity-cosine cross-doc centroids |
| | `ppr` | HippoRAG-2 Personalized PageRank (multi-hop) |
| | `hefr` | Hierarchical Entity-First Retrieval |
| | `cross_doc` | liên kết cross-document (SHARES_ENTITIES / SIMILAR_DOC) |
| | `graph_embeddings` | GAEA graph-aware embeddings |
| | `temporal_entities` | versioning entity theo thời gian |
| **2. Rerank** | `rerank` / `rerank_stages` | 3-stage rerank pipeline |
| | `rerank_l2r` | learning-to-rank final (entity-aware) |
| | `cross_encoder` | loader bge-reranker-v2-m3 |
| | `consistency` | consistency score (5-view variance) |
| **3. Generation / reasoning** | `react_loop` | ReAct Thought→Action→Observation |
| | `sufficient_context` | Sufficient-Context Gate (light-LLM check before generation) |
| | `global_query` | query-time map-reduce cho câu global/thematic (gated `GLOBAL_QUERY_ENABLED`) |
| | `context_compress` | LLMLingua-2 nén context |
| | `ollama_helper` | LLM call wrapper (keep_alive, think=False) |
| | `validation` | grounding + entity + citation gates |
| **4. Knowledge Graph** | `kg` | Neo4j entity/graph ops |
| | `entity_extractor` | GLiNER NER (tách khỏi LLM) |
| | `community` | Leiden/Louvain community + summaries |
| | `domain_tagger` | domain-axis tagging |
| **5. Ingestion** | `ingestion` | **orchestrator** `ingest_document` |
| | `chunkers/*` | chunk theo format |
| | `format_router` / `doc_type_classifier` | detect format/type → chunker |
| | `chunk_quality` | Chunk Quality Classifier |
| | `pii_mask` | PII masking (regex + LLM-NER) |
 
**Điểm vào bắt buộc (convention):** LLM → `ollama_helper.ollama_chat`; retrieval → `retrieval.multi_path_retrieve`;
ingestion → `ingestion.ingest_document`; upsert → `vector.upsert`.
 
## Chạy / test
 
Launch + drive: skill **`/run-vrag`** (`.agents/skills/run-vrag/driver.sh smoke`). Xem [.agents/AGENTS.md](../.agents/AGENTS.md) cho lệnh thường dùng.
