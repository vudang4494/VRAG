# Enterprise RAG Stack — Technical Specification v3.0

> **ĐÂY LÀ TÀI LIỆU V3.** Nếu bạn đọc V1 (old SPEC.md), hãy quên nó — code thực tế
> là một system phức tạp hơn nhiều. Document này phản ánh chính xác những gì code làm.

## 1. Tổng quan

**Name**: Enterprise Local RAG Stack v3.0
**Stack**: Hybrid GraphRAG — Vector + Knowledge Graph + Community Summaries
**Target**: Apple Silicon Mac (M-series, 16GB+ RAM), 100% local, no cloud
**LLM**: Ollama `qwen3.5:4b` (Metal GPU) + `bge-m3` embedding
**Entity NER**: GLiNER `urchade/gliner_multi-v2.1` (168M, zero-shot)
**Benchmarks**: 30-query Vietnamese eval, 62.9% avg doc_recall, 0% HTTP errors

---

## 2. Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│  Nginx (port 80)                                                │
│  ├── RAG API (http://localhost:8800)     FastAPI + uvloop       │
│  └── Langfuse (http://localhost:3000)    Observability           │
└───────────┬─────────────────────────────────────────────────────┘
            │
            ├──────────────┐
            ▼              ▼
┌──────────────────┐  ┌────────────────────────────────────────┐
│ Ollama (host)    │  │ Docker Containers                       │
│ Metal GPU        │  │ rag-qdrant    Qdrant 1.13              │
│ • qwen3.5:4b     │  │ rag-neo4j     Neo4j 5.26 + APOC        │
│ • bge-m3         │  │ rag-redis     Redis 7                   │
│ • GLiNER model   │  └────────────────────────────────────────┘
└──────────────────┘
```

---

## 3. Pipeline — Ingestion (Document Indexing)

```
File bytes (PDF/DOCX/XLSX/CSV/MD/TXT/HTML)
    │
    ▼
format_router (route_and_chunk)
    │
    ▼
hierarchical_chunker
  adaptive threshold 0.40–0.70
  sentence-aware splitting
    │
    ▼
pii_mask (consistent placeholders cho PII)
    │
    ▼
consistency_simulation
  5 views: dense, paraphrase, question, summary, keywords
  self-consistency scoring per view
    │
    ▼
entity_voting (3 LLM passes → vote on entities/rels)
    │
    ├─► Qdrant: 5 named vectors + sparse BM25 + payload
    │    Collection: enterprise_kb (1024-dim, 6 named vectors)
    │    Named vectors: dense, paraphrase, question, summary, keywords, bm25
    │
    ├─► Neo4j: Chunk → CONTAINS_ENTITY → Entity → RELATES_TO → Entity
    │
    └─► [post-ingest] link_semantic_chunks (SIMILAR_TO edges)
    └─► [post-ingest] build_communities (Leiden/Louvain → Community summaries)
```

---

## 4. Pipeline — Query (Inference)

```
USER QUERY
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Query Router (query_router.py)                     │
│  Heuristic regex patterns → query type                      │
│  Types: factual | multi_hop | summarization | analytical  │
│  + out_of_domain detection (pattern matching)              │
│  No LLM call needed for routing                            │
└──────────────────────┬────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
  [out_of_domain]             [in-domain]
  → short-circuit REFUSE       → continue
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Query Understanding (query_understanding.py)        │
│  6 reformulations run in parallel via asyncio.gather:        │
│   1. rewrite     — LLM viết lại query rõ ràng hơn         │
│   2. hyde        — LLM sinh "câu trả lời giả định"         │
│   3. decompose   — LLM chia multi-hop thành sub-questions  │
│   4. step_back   — LLM trừu tượng hóa câu hỏi             │
│   5. keywords    — LLM trích xuất keywords                 │
│   6. intent      — LLM phân loại: factual|analytical|      │
│                    summarization|comparison                 │
│  Config: query_reformulations=3–6 (throttle via config)  │
│  Timeout: query_understanding_timeout=10s (default)        │
│  ALL via ollama_helper.ollama_chat (Ollama native)         │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Multi-Path Retrieval (retrieval_v2.py)             │
│  Intent → strategy map (INTENT_STRATEGY dict):              │
│                                                             │
│  factual:      views=[dense, graph_aware, keywords]           │
│                graph=False, community=False                  │
│  analytical:   views=[dense, graph_aware, summary, question] │
│                graph=True(2 hops), community=True            │
│  summarization:views=[summary, graph_aware]                 │
│                graph=False, community=True(1 hop)           │
│  comparison:   views=[dense, graph_aware, question]          │
│                graph=True(2 hops), community=False          │
│                                                             │
│  Retrieval paths:                                           │
│   1. dense (5 named vectors × N reformulations)             │
│   2. paraphrase view                                        │
│   3. question view                                          │
│   4. summary view                                           │
│   5. keywords view                                          │
│   6. sparse:bm25 (BM25 via Qdrant sparse vector)           │
│   7. graph (Neo4j entity co-occurrence)                    │
│   8. community (GraphRAG community summaries)               │
│   9. entity_pivot (query→GLiNER→entities→Neo4j→chunks)     │
│                                                             │
│  All paths run in parallel via asyncio.gather               │
│  Entity extractor: GLiNER multi-v2.1 (zero-shot NER)       │
│  Temporal filter: detect_temporal_intent → Cypher filter    │
│  Domain tagger: tag_query (domain_reward scoring)            │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Weighted RRF Fusion (retrieval_v2.py)              │
│                                                             │
│  RRF_score = path_weight × consistency_factor ×             │
│               level_factor × domain_reward                  │
│               / (k + rank)                                  │
│                                                             │
│  path_weight by reformulation kind:                         │
│    hyde=1.3 > rewrite/decompose=1.1 > original=1.0         │
│    > keywords=0.9 > step_back=0.8                          │
│    entity_pivot=1.5 (high-precision bridge path)            │
│                                                             │
│  Factors:                                                   │
│    consistency_factor: score≥0.85 → 1.2, ≥0.60 → 1.0, else 0.8
│    level_factor: section=1.1, paragraph=1.0,                │
│                  sentence=0.8, document=0.7                │
│    domain_reward: cosine(chunk_domain_vec, query_domain_vec) │
│                                                             │
│  Output: top 50 fused candidates                           │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: OOD Detection (ood_detector.py)                   │
│  detect_ood_mixed() — 2 signals combined:                   │
│                                                             │
│  Signal 1: top_score (BGE-M3 cosine)                       │
│    < 0.50 → OOD, ≥ 0.60 → in-domain                      │
│    0.50–0.60 → marginal (use signal 2)                     │
│                                                             │
│  Signal 2: keyword_overlap_ratio                            │
│    < 30% overlap → OOD                                     │
│                                                             │
│  Decision: OOD if (low_score + no_overlap)                  │
│  Confidence: 0.75–0.95                                    │
│                                                             │
│  IF OOD:                                                    │
│   → Thử standard retrieval fallback → ReAct                │
│   → Nếu vẫn low score → REFUSE                            │
│  IF in-domain: continue to generation                       │
└──────────────────────┬────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
  [factual/simple]              [multi_hop/complex]
  → Standard path               → ReAct Loop
  (fast, 1 LLM call)           (6 steps max)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 6: ReAct Loop (react_loop.py)                        │
│  Only for: multi_hop | summarization | analytical          │
│  max_steps = 6                                             │
│                                                             │
│  Actions (9 total):                                          │
│   search_entity      → Neo4j exact/fuzzy match            │
│   expand_relation    → 1-hop RELATES_TO (typed via rel_type property)│
│   retrieve_chunks    → Neo4j: CONTAINS_ENTITY → chunks     │
│   graph_aware_search→ Qdrant: GAEA refined embeddings      │
│   expand_community  → Neo4j: IN_COMMUNITY → community summaries│
│   count_entities     → Neo4j: count by type (aggregation)  │
│   verify_fact       → KG cross-check a claim before answering│
│   rerank            → rerank_stages stage 2               │
│   FINISH            → synthesis (chunks ≥ 4 required)      │
│                                                             │
│  FINISH protection: blocked if chunks_collected < 4        │
│  → forces graph_aware_search before allowing FINISH        │
│  Output: answer + full trace + sources + latency breakdown │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 7: 3-Stage Reranking (rerank_stages.py)              │
│                                                             │
│  Input: 50 candidates                                       │
│                                                             │
│  Stage 1: Cross-encoder (BAAI/bge-reranker-v2-m3)         │
│           Query + doc pairs → relevance score               │
│           Output: top 20 (lazy-load, optional)              │
│           Fallback: sorted by existing score if unavailable  │
│                                                             │
│  Stage 2: Summary-view semantic match                       │
│           Re-embed query → compare with summary view        │
│           Output: top 10                                     │
│                                                             │
│  Stage 3: LLM judge (top 5 candidates)                     │
│           Single LLM call picks best                        │
│           Output: top 5 reranked                            │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 8: Validation Gates (validation.py)                   │
│  3 gates run in parallel:                                  │
│                                                             │
│  Gate 1: Hallucination (hallucination_gate)                │
│    → extract_atomic_claims (LLM)                           │
│    → verify_claim per claim (LLM, concurrent=4)            │
│    → grounded_ratio = (YES + 0.5×PARTIAL) / total         │
│    → PASS if ratio ≥ 0.70 (configurable, enterprise ≥ 0.90)│
│                                                             │
│  Gate 2: Entity (entity_gate) — 3-tier fuzzy matching    │
│    → extract Title-Cased entities from answer (regex)       │
│    → Tier 1: exact lowercase match against Neo4j           │
│    → Tier 2: Levenshtein similarity ≥ 0.80 (same type)     │
│    → Tier 3: substring containment match                  │
│    → PASS if invalid ≤ 2 (configurable)                     │
│    → Records fuzzy_matched variants for debugging          │
│                                                             │
│  Gate 3: Citation (citation_gate)                          │
│    → find [chunk_id] markers in answer                     │
│    → PASS if cited_sentences / total ≥ 0.40               │
│    → Skip if refusal answer detected                       │
│                                                             │
│  Combined: PASS = all 3 gates pass                         │
│  On fail: Layer 10.4 self-correction loop:               │
│    → Attempt 0: corrective regeneration (stricter prompt)   │
│    → Attempt 1+: broader retrieval + regenerate           │
│    → Attempt N+1: refuse with refusal_message              │
│  Max retries configurable (default: 0 = no correction)    │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
    ANSWER (with citations [chunk_id])
```

---

## 5. Knowledge Graph Schema

### Neo4j Nodes & Edges

```
(Chunk)
  id: str
  text: str
  source: str
  format: str
  chunk_level: sentence|paragraph|section|document
  consistency_score: float
  tenant_id: str

(Entity)
  name: str (sanitized: alphanumeric + underscore)
  type: PERSON|ORGANIZATION|LOCATION|EVENT|PRODUCT|CONCEPT|TECHNOLOGY|OTHER
  description: str
  confidence: float
  tenant_id: str

(Community)
  id: str (comm_{tenant}_L{level}_{cluster_id}_{uuid})
  level: int
  summary: str (LLM-generated)
  member_count: int
  summary_vote_count: int (3-pass voting)
  generated_at: datetime
  tenant_id: str

(Document)
  id: str
  source: str
  tenant_id: str
```

### Neo4j Edges

```
(Chunk)-[:CONTAINS_ENTITY]->(Entity)
(Entity)-[:RELATES_TO {rel_type, confidence, description, vote_count}]->(Entity)
(Entity)-[:ALIAS_OF]->(Entity)       -- canonicalization: variant → canonical form
(Entity)-[:IN_COMMUNITY {level}]->(Community)
(Community)-[:SUB_COMMUNITY_OF]->(Community)
(Chunk)-[:SIMILAR_TO]->(Chunk)
(Chunk)-[:FROM_DOCUMENT]->(Document)
```

Note: `rel_type` property stores the relationship type (USES, PROPOSED_BY, CITES, etc.)
extracted by LLM. The edge label remains `RELATES_TO` (Neo4j constraint).
`ALIAS_OF` edges are created by Layer 2.2 canonicalization when Levenshtein similarity
between entity names >= 0.85.

### Entity Voting (3-pass)

```
Chunk text → extract_entities (LLM) → 3 times
→ vote: entity name (normalized lowercase)
→ store most descriptive description per name
→ vote: relationship (source→target)
→ store relationship with highest confidence
```

### Community Detection

```
1. fetch_entity_graph (Neo4j)
   → RELATES_TO edges if exist
   → OR co-occurrence edges (shared chunks → weight = shared_chunks/10)

2. cluster_leiden (igraph, preferred)
   → Louvain fallback (networkx)
   → resolution=1.0, seed=42
   → multi-level hierarchical

3. generate_consistent_summary (per cluster)
   → 3 LLM passes with different temperatures (0.2, 0.4, 0.6)
   → LLM judge picks best
   → write :Community nodes + :IN_COMMUNITY edges
```

---

## 6. Qdrant Collection Schema

```
Collection: enterprise_kb
  Vector config: 1024-dim, Cosine distance
  6 named vectors:
    dense       — standard embedding
    paraphrase  — paraphrase phrasing
    question    — question-formulated
    summary     — section-level summary
    keywords    — keyword-focused
    bm25        — sparse vector (BM25-style)

  Per point payload:
    chunk_id, text, source, format, chunk_level,
    consistency_score, access_level, doc_id,
    domain_distribution (dict), domain_primary (str),
    tenant_id, page_num, sheet_name, thread_id
```

---

## 7. Retrieval — Chi tiết từng path

| Path | Input | Storage | Khi nào dùng |
|---|---|---|---|
| `dense` | original query embed | Qdrant `dense` | Always |
| `paraphrase` | rewrite embed | Qdrant `paraphrase` | Always |
| `question` | question-formulated embed | Qdrant `question` | analytical, comparison |
| `summary` | summary embed | Qdrant `summary` | summarization intent |
| `keywords` | keyword embed | Qdrant `keywords` | Always |
| `sparse:bm25` | sparse indices/values | Qdrant `bm25` | Code IDs, proper nouns |
| `entity_pivot` | GLiNER entities | Neo4j `CONTAINS_ENTITY` | KG bridge |
| `graph` | query embed | Neo4j co-occurrence | analytical, comparison |
| `community` | community embed | Neo4j `Community.summary` | summarization |

### Weighted RRF Formula

```python
def weighted_rrf(paths, k=60, final_top_k=50, query_domain=None, domain_scale=0.3):
    for path_key, results in paths.items():
        path_weight = reformulation_weight(kind)
        for rank, c in enumerate(results, 1):
            contribution = (
                path_weight
                * consistency_factor(c.consistency_score)
                * level_factor(c.chunk_level)
                * domain_reward(chunk_domain, query_domain, scale=domain_scale)
                / (k + rank)
            )
            fused[chunk_id].rrf_score += contribution
            fused[chunk_id].matched_paths.append(path_key)
```

---

## 8. OOD Detection — Chi tiết

```python
def detect_ood_mixed(candidates, query) -> dict:
    # Signal 1: retrieval score
    top_score = max(c.score for c in candidates)
    top3_avg = avg(top3 scores)

    # Signal 2: keyword overlap
    query_terms = extract_terms(query)  # 3+ chars, no stopwords
    doc_text = " ".join(c.text for c in candidates[:5])
    overlap_ratio = matched_terms / total_terms

    # Decision matrix
    if top_score < 0.50 and overlap_ratio < 0.30:
        return is_ood=True, confidence=0.90
    elif top_score < 0.50 and overlap_ratio >= 0.30:
        return is_ood=False, confidence=0.75  # weak match
    elif top_score < 0.60 and overlap_ratio < 0.30:
        return is_ood=True, confidence=0.75
    else:
        return is_ood=False, confidence=0.90
```

---

## 9. Configuration

```bash
# LLM
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen3.5:4b
OLLAMA_EMBED_MODEL=bge-m3
OLLAMA_EMBED_URL=http://host.docker.internal:11434

# Database
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=enterprise_kb
NEO4J_URL=bolt://neo4j:7687
REDIS_URL=redis://redis:6379/0

# Retrieval
RETRIEVAL_TOP_K=8
RETRIEVAL_VECTOR_TOP_K=20
RETRIEVAL_GRAPH_TOP_K=20
RRF_K=60

# Query understanding
QUERY_REFORMULATIONS=3      # 1-6 reformulations
QUERY_UNDERSTANDING_TIMEOUT=10.0

# Validation gates
MIN_GROUNDED_RATIO=0.70
MIN_CITATION_RATIO=0.40
MAX_INVALID_ENTITIES=3

# Community
COMMUNITY_ENABLED=false     # true = nightly rebuild
```

---

## 10. File Structure (src/services/)

```
src/services/
├── ollama_helper.py         # Ollama native wrapper (bypass Qwen3 think:false)
├── embedding.py             # BGE-M3 embed_single / embed_batch
├── config.py               # Settings from environment variables
├── clients.py              # Global client holders + SemanticCache
│
├── # ── Ingestion ──
├── format_router.py         # PDF/DOCX/XLSX/CSV/MD/TXT/HTML parsing
├── chunk_quality.py        # Quality filtering post-chunking
├── pii_mask.py             # PII placeholders (consistent per doc)
├── consistency.py           # 5-view self-consistency simulation
├── entity_extractor.py     # GLiNER wrapper (zero-shot NER)
├── kg.py                   # Neo4j: extract + upsert entities/rels
├── vector_v2.py            # Qdrant V2: 6 named vectors + sparse
├── ingestion_v2.py         # Ingestion orchestrator
│
├── # ── Query Pipeline ──
├── query_router.py         # Heuristic query type classifier
├── query_understanding.py  # 6 reformulations + intent (LLM calls)
├── ood_detector.py         # OOD detection: score + keyword overlap
├── temporal_entities.py    # Temporal intent detection → Cypher filter
├── domain_tagger.py        # Domain tagging (8 axes) + reward scoring
│
├── # ── Retrieval ──
├── retrieval_v2.py          # Multi-path retrieval orchestrator
├── vector_v2.py            # Qdrant search (single view + multi-view RRF)
├── rerank_stages.py        # 3-stage reranking pipeline
├── rerank_l2r.py          # L2R (learn-to-rank) — optional
├── hefr_retrieval.py       # Hierarchical fine-grained retrieval
├── graph_embeddings.py     # GAEA refined embeddings
├── cross_doc.py            # Cross-document entity linking
│
├── # ── Reasoning ──
├── react_loop.py           # ReAct loop (9 actions, max 6 steps)
├── community.py           # Leiden/Louvain + Community summaries + incremental update
│
├── # ── Generation ──
├── validation.py           # 3 validation gates (hallucination/entity/citation)
└── models.py              # Pydantic request/response schemas
```

---

## 11. API Endpoints (api/routes_v3.py)

| Endpoint | Method | Description |
|---|---|---|
| `/api/v3/health` | GET | Liveness |
| `/api/v3/health/deep` | GET | Full dependency check |
| `/api/v3/chat` | POST | Main RAG chat endpoint |
| `/api/v3/ingest/upload` | POST | Document upload + indexing |
| `/api/v3/ingest/status/{job_id}` | GET | Ingestion job status |
| `/api/v3/search` | POST | Direct retrieval (no generation) |
| `/api/v3/tenants` | GET/POST | Tenant management |
| `/api/v3/tenants/{id}/stats` | GET | Tenant statistics |
| `/api/v3/cache/clear` | POST | Clear semantic cache |
| `/metrics` | GET | Prometheus metrics |
| `/v1/models` | GET | Available models |

---

## 12. Kết quả đánh giá (30 queries, eval tenant)

### Thực tế

| Metric | Result | Notes |
|---|---|---|
| `doc_recall` avg | **62.9%** | Retrieval miss ~37% docs cần thiết |
| `doc_recall` factual | **100%** | Simple queries hoạt động tốt |
| `doc_recall` multi-hop | **44%** | ReAct chỉ thu thập 6-8 chunks — không đủ |
| `doc_recall` comparison | **67%** | Retrieval breadth cho multi-doc không đủ |
| `kw_hit` avg | **~30%** | Keyword match thấp — paraphrase quá xa |
| `semantic_hit` avg | **0%** | Eval threshold 0.65 quá cao cho BGE-M3 |
| `refused` rate | **13.3%** | 4/30 queries refused |
| `refused_sai` | **2/30** | "Microsoft Research", "paper của Traag" — corpus có data |
| `ood_recall` | **100%** | Không false-positive trên OOD queries |
| `http_errors` | **0%** | Ổn định |
| `latency p50` | **82s** | Chậm: query understanding overhead + LLM calls |
| `latency p95` | **~150s** | Multi-hop với ReAct loop |

### Phân tích root cause

**Refuse sai (2 queries):**
- "Paper nào của Traag và cộng sự?" → routed factual → standard retrieval fail → refused
- "Microsoft Research có công trình gì?" → routed factual → miss entity → refused
- Root cause: pattern `paper nào` match multi-hop nhưng query_router không catch

**Multi-hop recall thấp (44%):**
- ReAct max_steps=6, nhưng mỗi action chỉ fetch 15 chunks
- Tổng: 6 actions × 15 chunks = 90 potential, nhưng deduplicate qua `seen_chunk_ids`
- Thực tế: 6-8 unique chunks sau deduplication
- Multi-hop cần: 3-5 docs × 3-5 chunks = 9-25 chunks → không đủ

**Latency 82s:**
- query_understanding: 3-6 LLM calls × 10-15s/call = 30-90s
- ReAct: 6 steps × (LLM call 10-15s + action 2-5s) = 72-120s
- Standard path: query_understanding + 1 generation LLM = 40-60s

---

## 13. Multi-Tenant Isolation

| Store | Method |
|---|---|
| Qdrant | `tenant_id` in every point payload + filter on query |
| Neo4j | `tenant_id` property on every node + filter on all Cypher |
| Redis | Key prefix: `rag:{tenant_id}:cache:*` |
| API | API key scoped to tenant, validated per request |

---

## 14. Observability

### Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `rag_requests_total` | Counter | Total API requests |
| `rag_requests_errors_total` | Counter | Total errors |
| `rag_cache_hits_total` | Counter | Cache hits |
| `rag_chunks_indexed_total` | Counter | Chunks indexed |
| `rag_entities_extracted_total` | Counter | Entities extracted |
| `rag_request_latency_seconds` | Histogram | Latency by endpoint |
| `rag_retrieval_latency_seconds` | Histogram | Retrieval latency |
| `rag_generation_latency_seconds` | Histogram | Generation latency |

### Langfuse Tracing

Traces for:
- Query embedding
- Query understanding (6 reformulations)
- Multi-path retrieval (all 9 paths)
- RRF fusion
- ReAct loop (each step traced)
- Reranking stages
- LLM generation
- Validation gates
- Document ingestion

---

## 15. Known Issues & Trade-offs

| Issue | Impact | Workaround |
|---|---|---|
| Misrouting: pattern `paper nào` not matched | 2 refuse-sai | Need additional multi-hop patterns |
| Multi-hop recall 44% | Bỏ sót > half docs | Tăng chunks_examined hoặc dùng community summaries |
| Latency 82s p50 | User experience kém | Cache query understanding, skip reformulations cho simple |
| `semantic_hit=0%` | Eval metric không reflect quality thực | Threshold eval 0.65 quá cao cho BGE-M3 |
| Cross-encoder reranker optional | Stage 1 có thể skip | Falls back to sorted by score |
| Community summaries need nightly rebuild | Không real-time | `community_enabled` flag + cron job |
| Entity voting 3-pass tốn 3× LLM | Ingestion chậm | Chỉ cho chunks có `entity_voting_enabled=true` |

---

## 16. So sánh V1 vs V3

| Aspect | V1 (old SPEC) | V3 (actual) |
|---|---|---|
| Retrieval paths | 2 (vector + graph) | 9 paths |
| ReRank | Semantic reranker | 3-stage: cross-encoder + semantic + LLM judge |
| Query reformulations | None | 6 reformulations (configurable) |
| Entity extraction | LLM (1-pass) | GLiNER + 3-pass voting |
| OOD detection | None | Score + keyword overlap |
| Validation | None | 3 gates: hallucination + entity + citation |
| Community summaries | None | Leiden/Louvain + LLM summaries |
| Domain tagging | None | 8-axis domain classification + reward |
| BM25 | None | Sparse vector trong Qdrant |
| Temporal filtering | None | Temporal intent → Cypher filter |
| Multi-view embeddings | None | 5 named vectors (dense/paraphrase/question/summary/keywords) |

---

*Last updated: 2026-05-14 | Code version: V3 (git: main)*
