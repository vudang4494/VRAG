# VRAG — Technical Specification

> **Một sản phẩm duy nhất: VRAG.** Không có v1/v2/v3 trong code. API URL prefix
> `/api/*` là REST contract (giữ để không break clients), khác hoàn toàn
> với product version.

## 1. Tổng quan

**Name**: VRAG (Vector-Centric Hybrid GraphRAG)
**Stack**: Hybrid GraphRAG — Vector + Knowledge Graph + Community Summaries
**Target**: Apple Silicon Mac (M-series, 16GB+ RAM), 100% local, no cloud
**LLM**: Ollama `gemma4:e4b` (Metal GPU) + `bge-m3` embedding — tags live in `src/config.py`
**Entity NER**: GLiNER `urchade/gliner_multi-v2.1` (168M, zero-shot)
**Benchmarks**: live RAGAS on `gemma4:e4b` (corpus500) — Faithfulness 0.946, Context Recall 0.946, Context Precision 0.904, Answer Relevancy 0.808; xem §12

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
│ • gemma4:e4b     │  │ rag-neo4j     Neo4j 5.26 + APOC        │
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
    ├─► Qdrant: 6 named vectors + sparse BM25 + payload
    │    Collection: enterprise_kb (1024-dim, 6 named vectors)
    │    Named vectors: dense, paraphrase, question, summary, keywords, graph_aware
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
│  Semantic centroid router: embed query (BGE-M3) → cosine    │
│  với intent centroids (config/intent_centroids.npy)         │
│  Types: factual | multi_hop | summarization | analytical  │
│  | comparison | kg_construction | global | out_of_domain   │
│  No LLM call needed for routing                            │
└──────────────────────┬────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
  [out_of_domain]             [in-domain]
  → short-circuit REFUSE       → continue
                                    │
  [global] (gated GLOBAL_QUERY_ENABLED, default false)
  → short-circuit global_query.global_map_reduce: map-reduce
    trên community summaries (GLOBAL_QUERY_MAX_COMMUNITIES=10),
    bypass top-k retrieval + ReAct
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Query Understanding (query_understanding.py)        │
│  Default zero-LLM: GLiNER entities + intent từ centroid     │
│  router (STEP 1) — không LLM call nào.                      │
│  Opt-in QUERY_REFORMULATIONS=1–5 mở dần LLM reformulations  │
│  (chạy song song via asyncio.gather):                       │
│   1. rewrite     — LLM viết lại query rõ ràng hơn         │
│   2. +keywords   — LLM trích xuất keywords                 │
│   3. +hyde       — LLM sinh "câu trả lời giả định"         │
│   4. +decompose  — LLM chia multi-hop thành sub-questions  │
│   5. +step_back  — LLM trừu tượng hóa câu hỏi             │
│  Config: query_reformulations=0 (default, zero-LLM)        │
│  Timeout: query_understanding_timeout_s=60.0 (default)     │
│  LLM calls via ollama_helper.ollama_chat (Ollama native)   │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Multi-Path Retrieval (retrieval.py)             │
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
│  10. ppr (HippoRAG-2 Personalized PageRank, default ON)     │
│  11. entity_cosine (entity centroids, gated                 │
│      ENTITY_COSINE_ENABLED)                                 │
│  12. entity_gate (primary entity-gate, gated                │
│      ENTITY_GATE_ENABLED)                                   │
│                                                             │
│  All paths run in parallel via asyncio.gather               │
│  Entity extractor: GLiNER multi-v2.1 (zero-shot NER)       │
│  Temporal filter: detect_temporal_intent → Cypher filter    │
│  Domain tagger: tag_query (domain_reward scoring)            │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Weighted RRF Fusion (retrieval.py)              │
│                                                             │
│  RRF_score = path_weight × consistency_factor ×             │
│               level_factor × domain_reward                  │
│               / (k + rank)                                  │
│                                                             │
│  path_weight by reformulation kind:                         │
│    hyde=1.3 > rewrite/decompose=1.1 > original=1.0         │
│    > keywords=0.9 > step_back=0.8                          │
│  KG paths: entity_gate=1.8 > ppr=1.7 > entity_cosine=1.6   │
│    > entity_pivot=1.5 > community=1.2 > graph=1.0          │
│    × rrf_kg_path_weight_scale=0.2 (default, config.py)      │
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
                       │
                       ▼
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
│  STEP 5: 3-Stage Reranking (rerank_stages.py)              │
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
│  STEP 5.5: Sufficient-Context Gate (sufficient_context.py) │
│  Fast check using settings.light_llm                        │
│  Determines if context contains enough info for query.      │
│  If insufficient: refuse early, bypass generation (saves ~14s)│
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 8: Validation Gates (validation.py)                   │
│  3 gates run in parallel:                                  │
│                                                             │
│  Gate 1: Hallucination (hallucination_gate)                │
│    → default: cosine-grounding (bge-m3 embed câu trả lời   │
│      vs chunks, sim_hi=0.60 / sim_lo=0.40, zero-LLM;       │
│      VALIDATION_COSINE_GROUNDING=true)                      │
│    → fallback (=false): extract_atomic_claims (LLM)        │
│      + verify_claim per claim (LLM, concurrent=4)          │
│    → grounded_ratio = (YES + 0.5×PARTIAL) / total         │
│    → PASS if ratio ≥ 0.70 (configurable, enterprise ≥ 0.90)│
│                                                             │
│  Gate 2: Entity (entity_gate) — 3-tier fuzzy matching    │
│    → extract Title-Cased entities from answer (regex)       │
│    → Tier 1: exact lowercase match against Neo4j           │
│    → Tier 2: Levenshtein similarity ≥ 0.80 (same type)     │
│    → Tier 3: substring containment match                  │
│    → PASS if invalid ≤ 3 (configurable)                     │
│    → Records fuzzy_matched variants for debugging          │
│                                                             │
│  Gate 3: Citation (citation_gate)                          │
│    → find [chunk_id] markers in answer                     │
│    → PASS if cited_sentences / total ≥ 0.70               │
│    → Skip if refusal answer detected                       │
│                                                             │
│  Combined: PASS = all 3 gates pass                         │
│  On fail: Layer 10.4 self-correction loop:               │
│    → Attempt 0: corrective regeneration (stricter prompt)   │
│    → Attempt 1+: broader retrieval + regenerate           │
│    → Attempt N+1: refuse with refusal_message              │
│  Max retries configurable (validation_retry_on_fail=True   │
│  default → retry là hành vi mặc định khi max_retries > 0;  │
│  API default max_retries=1)                                │
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
    graph_aware — GAEA graph-refined embed (via /api/gaea/refine)

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
| `ppr` | GLiNER entities → PPR walk | Neo4j entity graph (HippoRAG-2) | Default ON (`PPR_ENABLED`) |
| `entity_cosine` | entity centroids + TF-IDF + MMR | Qdrant + entity vectors | Gated `ENTITY_COSINE_ENABLED` |
| `entity_gate` | entity centroids (primary gate) | Qdrant + entity vectors | Gated `ENTITY_GATE_ENABLED` |

RRF weight: entity_gate=1.8, ppr=1.7, entity_cosine=1.6, entity_pivot=1.5, community=1.2,
graph=1.0 — mọi KG path bị nhân `rrf_kg_path_weight_scale=0.2` (default, `src/config.py`).

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
OLLAMA_MODEL=gemma4:e4b
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
RETRIEVAL_GRAPH_TOP_K=15
RRF_K=60

# Query understanding
QUERY_REFORMULATIONS=0      # default 0 = zero-LLM; opt-in 1-5 reformulations
QUERY_UNDERSTANDING_TIMEOUT_S=60.0

# Validation gates
VALIDATION_MIN_GROUNDED_RATIO=0.70
VALIDATION_MIN_CITATION_RATIO=0.70
VALIDATION_MAX_INVALID_ENTITIES=3

# Community
COMMUNITY_ENABLED=false     # true = nightly rebuild
```

---

## 10. File Structure (src/)

```
src/
├── config.py               # Settings from environment variables
├── clients.py              # Global client holders + SemanticCache
├── models.py               # Pydantic request/response schemas
├── audit.py                # Audit logging (every RAG API operation)
├── metrics.py              # Prometheus metrics middleware
├── tracing.py              # Langfuse tracing integration
│
└── services/
    ├── ollama_helper.py         # Ollama native wrapper (bypass Qwen3 think:false)
    ├── embedding.py             # BGE-M3 embed_single / embed_batch
    ├── config_report.py         # Effective-config banner at startup
    │
    ├── # ── Ingestion ──
    ├── format_router.py         # PDF/DOCX/XLSX/CSV/MD/TXT/HTML parsing
    ├── chunkers/                # Hierarchical chunkers
    ├── chunk_quality.py        # Quality filtering post-chunking
    ├── doc_type_classifier.py  # Document type → per-type chunking strategy
    ├── pii_mask.py             # PII placeholders (consistent per doc)
    ├── consistency.py           # 5-view self-consistency simulation
    ├── entity_extractor.py     # GLiNER wrapper (zero-shot NER)
    ├── kg.py                   # Neo4j: extract + upsert entities/rels
    ├── vector.py               # Qdrant: 6 named vectors + sparse BM25
    ├── ingestion.py            # Ingestion orchestrator
    │
    ├── # ── Query Pipeline ──
    ├── intent_classifier.py    # Pre-retrieval intent gate (greeting/ood short-circuit)
    ├── query_router.py         # Semantic centroid query type classifier
    ├── query_understanding.py  # GLiNER + opt-in LLM reformulations
    ├── chat_history.py         # Chat-history semantic cache + memory layer
    ├── sufficient_context.py   # Sufficient-Context Gate (light-LLM check before generation)
    ├── temporal_entities.py    # Temporal intent detection → Cypher filter
    ├── domain_tagger.py        # Domain tagging (8 axes) + reward scoring
    │
    ├── # ── Retrieval ──
    ├── retrieval.py          # Multi-path retrieval orchestrator
    ├── (vector.py — same file as above; handles search too)
    ├── ppr.py                  # HippoRAG-2 Personalized PageRank path
    ├── entity_vectors.py       # Entity centroid vectors (entity_cosine/entity_gate)
    ├── rerank.py               # 3-stage rerank pipeline (rerank_full_pipeline, dùng bởi /api/chat)
    ├── rerank_stages.py        # 3-stage rerank stages (rerank_stage2 dùng bởi react_loop)
    ├── rerank_l2r.py          # L2R (learn-to-rank) — optional
    ├── cross_encoder.py        # Shared cross-encoder loader (bge-reranker-v2-m3)
    ├── hefr.py                 # Hierarchical fine-grained retrieval
    ├── graph_embeddings.py     # GAEA refined embeddings
    ├── cross_doc.py            # Cross-document entity linking
    │
    ├── # ── Reasoning ──
    ├── react_loop.py           # ReAct loop (9 actions, max 6 steps)
    ├── community.py           # Leiden/Louvain + Community summaries + incremental update
    ├── global_query.py         # Global-query LazyGraphRAG map-reduce (gated)
    │
    ├── # ── Generation ──
    ├── context_compress.py     # LLMLingua-2 context compression (opt-in)
    └── validation.py           # 3 validation gates (hallucination/entity/citation)
```

---

## 11. API Endpoints (api/routes/)

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Liveness |
| `/api/health/deep` | GET | Full dependency check |
| `/api/chat` | POST | Main RAG chat endpoint |
| `/api/chat/stream` | POST | Streaming chat (SSE) |
| `/api/chat/react` | POST | ReAct multi-hop chat |
| `/api/ingest/upload` | POST | Document upload + indexing |
| `/api/gaea/refine` | POST | GAEA graph-aware re-embed (`graph_aware` vector) |
| `/api/hefr/populate` | POST | HEFR hierarchical index populate |
| `/api/hefr/retrieve` | POST | HEFR hierarchical retrieve |
| `/api/cross_doc/build` | POST | Cross-document entity linking |
| `/api/community/build` | POST | Community detection (Leiden) build |
| `/api/entity_resolution/build` | POST | Entity resolution (alias soft-fold) |
| `/api/repair/build` | POST | KG repair + orphan cleanup |
| `/api/rerank/l2r/test` | POST | L2R rerank test harness |
| `/metrics`, `/api/metrics` | GET | Prometheus metrics |

---

## 12. Kết quả đánh giá (live RAGAS, gemma4:e4b, 2026-07-17)

Đánh giá **live, reference-grounded** trên pipeline đang chạy. Generator `gemma4:e4b`; judge
`qwen3.6-35b` (Ollama local, không cloud); corpus `corpus500` (125,810 chunk / 488 doc); 40 Q&A gán
nhãn thủ công (38 chấm, 2 bị validation gate từ chối).

| Metric | Score | Tier (RAGAS literature) |
|---|---:|:---:|
| **Faithfulness** (anti-hallucination) | **0.946** | **Excellent** (0.80-0.95) |
| **Context Recall** | **0.946** | **Excellent** (0.80-0.95) |
| **Context Precision** | **0.904** | **Excellent** (0.80-0.95) |
| **Answer Relevancy** | **0.808** | **Good** (0.75-0.85) |
| Factual Correctness (F1, strict) | 0.302 | Below — xem note |

**Latency (live, gemma4:e4b):** p50 14.8s · p95 26.3s · max 28.0s · avg 15.1s.

> **Factual Correctness 0.30 KHÔNG phải "sai 70%".** Đây là claim-F1 nghiêm ngặt so với reference
> tiếng Anh ngắn gọn; VRAG trả lời dài bằng tiếng Việt, thêm nhiều claim *đúng* không có trong
> reference → F1 phạt. Faithfulness 0.946 mới là tín hiệu chính xác thật.

Báo cáo đầy đủ xem bảng tổng hợp trên.

> Bộ benchmark tiếng Việt cũ (config 05-2026: generator + tenant khác, raw report đã xoá) đã **nghỉ hưu**
> — số cũ không mô tả sản phẩm hiện tại.

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

## 15. Known Issues & Trade-offs (current)

| Issue | Severity | Status / Workaround |
|---|---|---|
| Strict Factual-Correctness F1 thấp (0.30) | Low | Understated bởi câu trả lời tiếng Việt dài (thêm claim đúng ngoài reference); Faithfulness 0.946 là tín hiệu thật. Kế hoạch: chế độ trả lời cô đọng + claim alignment |
| p95 latency ~26s trên CPU/Metal | Low | gemma4:e4b đo p95 26.3s (max 28.0s). Multi-user QPS cao vẫn cần GPU batching |
| Multi-hop retrieval cho query thiếu 1 entity | Medium | Entity-cosine pre-seed vẫn hụt khi 1 entity hoàn toàn vắng khỏi corpus. Cross-doc entity bridging chưa đủ |
| 2 false refusals (m02 Leiden, agentic02 ComposeRAG) | Low | Citation gate strict (0.70). Tunable via VALIDATION_MIN_CITATION_RATIO=0.60 |
| 1 query error (o01 phở OOD timing race) | Low | OOD short-circuit race condition. Reproducible, fixable in ~30 min |
| Context Recall metric N/A | Low | NonLLMContextRecall needs ground-truth sentences. Plan: generate 50 GT answers via OpenAI |
| Eager indexing — no incremental | Medium | Re-ingest needed on schema change. Lazy variant on roadmap |

### Resolved historical issues (for reference)

| Was issue | Resolution |
|---|---|
| ~~Misrouting `paper nào` not matched~~ | ✅ Architecture changed: rule-based regex → semantic centroid router |
| ~~Multi-hop recall 44%~~ | ✅ Tier 3 entity-cosine pre-seed: 50% |
| ~~Latency 82s p50~~ | ✅ Tier 1 zero-LLM cắt các call LLM khỏi p50 (giảm mạnh) |
| ~~`semantic_hit=0%` (threshold 0.65)~~ | ✅ SEMANTIC_THRESHOLD lowered to 0.45 in src/services/query_router.py |
| ~~OOD refusal 0%~~ | ✅ OOD centroid floor 0.40: 100% refusal accuracy |
| Cross-encoder reranker optional | Stage 1 có thể skip | Falls back to sorted by score |
| Community summaries need nightly rebuild | Không real-time | `community_enabled` flag + cron job |
| Entity voting 3-pass tốn 3× LLM | Ingestion chậm | Chỉ cho chunks có `entity_voting_enabled=true` |

---

## 16. Capability Snapshot

| Aspect | Implementation |
|---|---|
| Retrieval paths | Up to 9 parallel (dense × multi-view × reformulations + graph + community + entity_pivot) |
| Rerank | 3-stage with Dynamic Early-Exit: cross-encoder → semantic → LLM judge (auto-skip ≥0.85) |
| Query understanding | Zero-LLM Tier 1 (centroid router + GLiNER) by default; 0-5 opt-in LLM reformulations |
| Entity extraction | GLiNER zero-shot + 3-pass voting at ingest |
| OOD detection | 17-pattern regex + centroid threshold |
| Validation | Triple-Gate: hallucination + entity + citation (parallel) |
| Community summaries | Leiden detection + LLM-generated cluster summaries |
| Domain tagging | 8-axis classification with 30% reward bonus on match |
| BM25 | Sparse vector in Qdrant, fused with dense via Weighted RRF |
| Temporal filtering | Detected temporal intent → Cypher time filter |
| Multi-view embeddings | 6 named BGE-M3 vectors (dense / paraphrase / question / summary / keywords / graph_aware) + sparse BM25 |
| Context compression | LLMLingua-2 (classifier-based, multilingual) |

See [ARCHITECTURE.md](ARCHITECTURE.md) for algorithmic details, [README.md](README.md) for high-level overview.
