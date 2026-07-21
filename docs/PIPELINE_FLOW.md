# VRAG Pipeline — Flow Tường Minh

## Tổng quan kiến trúc

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 0: Pre-RAG (< 200ms, zero LLM call)                   │
│  ├─ Intent classification (greeting/ood/question/follow_up)   │
│  ├─ Chat-history semantic cache (Redis)                       │
│  └─ Query router (centroid embed cosine) → should_use_react   │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─ [greeting] ──→ short-circuit: return greeting text
    ├─ [ood]      ──→ short-circuit: return refusal
    └─ [question] ──→ proceed
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: Route Decision                                       │
│  ├─ classify_query() ─→ query_type (factual/comparison/multi_hop/etc) │
│  └─ should_use_react() ─→ True/False                          │
│                                                              │
│  if use_react → ReAct workflow loop (skip layers 2-7)          │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: Query Understanding (1-2 LLM call, ~8s)             │
│  ├─ understand_query()                                          │
│  │    ├─ rewrite: paraphrase query (LLM)                       │
│  │    └─ keywords: extract keywords (LLM)                      │
│  │                                                              │
│  │   [SPEC: skip for short+entity queries — saves ~8s]        │
│  │                                                              │
│  └─ entity extraction: GLiNER (LLM-free, ~200ms)              │
│       → pre_extracted_entities cho downstream                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: Multi-Path Retrieval                                │
│  ├─ [PHASE 1] Primary entity-gate (Cross-doc scope discovery)   │
│  │                                                              │
│  │   ┌────────────────────────────────────────────────────────┐  │
│  │   │  ENTITY GATE (Cross-Doc)                              │  │
│  │   │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  │
│  │   │  Input: query_vec (dense embed)                       │  │
│  │   │  Output: scope = [chunk_id_1, chunk_id_2, ...]       │  │
│  │   │           + primary_chunks (path candidates)           │  │
│  │   │           + selected_entities [(entity_name, score)]  │  │
│  │   │                                                        │  │
│  │   │  ① Dense seed: top-200 chunks via vector search       │  │
│  │   │     Qdrant query: dense view, tenant filter, limit=200 │  │
│  │   │     → returns: [chunk_id, chunk_id, ...]              │  │
│  │   │                                                        │  │
│  │   │  ② Discover entities in seed chunks                   │  │
│  │   │     Cypher: CONTAINS_ENTITY for each chunk_id         │  │
│  │   │     → returns: [entity_1, entity_2, ...] (≤200)      │  │
│  │   │                                                        │  │
│  │   │  ③ Score entities: cosine(query_vec, entity_centroid) │  │
│  │   │     × TF-IDF(entity_doc_count)  ← L1 anti-hub         │  │
│  │   │     where entity_centroid = mean(chunk_vecs)           │  │
│  │   │                                                        │  │
│  │   │  ④ MMR diversify: top-50 entities (λ=0.6)            │  │
│  │   │                                                        │  │
│  │   │  ⑤ Pull ALL chunks containing selected entities        │  │
│  │   │     → NO scope clamp (cross-doc discovery)            │  │
│  │   │     → returns: [chunk_id, text, score, matched_ents]  │  │
│  │   │                                                        │  │
│  │   │  Score floor: if best < 0.20 → OOD signal            │  │
│  │   └────────────────────────────────────────────────────────┘  │
│  │                                                              │
│  ├─ [PHASE 1] Parallel: Vector search × reformulations × views  │
│  │   Views: dense, graph_aware, keywords, question, summary     │
│  │   Reformulations: original, rewrite, hyde, step_back, etc   │
│  │   Per path: Qdrant named-vector search (limit=30)           │
│  │                                                              │
│  ├─ [PHASE 1] Graph path (if use_graph=True)                  │
│  │   KG co-occurrence retrieval                                │
│  │                                                              │
│  ├─ [PHASE 1] Community path (if use_community=True)          │
│  │   Embed community summaries → cosine with query_vec          │
│  │                                                              │
│  ├─ [PHASE 2] Entity Pivot (within entity_gate scope)          │
│  │   Cypher: CONTAINS_ENTITY + ALIAS_OF + normalized match     │
│  │   → chunk_id + entity_match_count + text                  │
│  │                                                              │
│  ├─ [PHASE 2] Entity Cosine (within scope)                    │
│  │   ┌────────────────────────────────────────────────────────┐  │
│  │   │  ENTITY COSINE (Cross-Doc, scoped)                    │  │
│  │   │  ① L5: scope entities by top-N chunks (≤200)          │  │
│  │   │  ② Score: cosine × TF-IDF × MMR diversity             │  │
│  │   │  ③ Return chunks containing selected entities          │  │
│  │   └────────────────────────────────────────────────────────┘  │
│  │                                                              │
│  ├─ [PHASE 2.1] PPR (HippoRAG 2, multi-hop)                   │
│  │   ┌────────────────────────────────────────────────────────┐  │
│  │   │  PPR (Personalized PageRank)                           │  │
│  │   │  Input: query_entities (from GLiNER)                   │  │
│  │   │  Output: [chunk_id, ppr_score, matched_entity]         │  │
│  │   │                                                      │  │
│  │   │  ① Load entity graph (Neo4j RELATES_TO or co-occur)   │  │
│  │   │     → DiGraph(nodes=entities, edges=relations)        │  │
│  │   │     → cached 10min per tenant                         │  │
│  │   │                                                      │  │
│  │   │  ② Build personalization vector                       │  │
│  │   │     seeds = query_entities ∩ graph.nodes               │  │
│  │   │     personalization[seed] = 1/len(seeds)               │  │
│  │   │                                                      │  │
│  │   │  ③ Run PageRank (α=0.5, max_iter=50, tol=1e-6)      │  │
│  │   │     nx.pagerank(graph, personalization=...)            │  │
│  │   │     → ranked_entities: [(entity, score), ...]         │  │
│  │   │                                                      │  │
│  │   │  ④ Map to chunks via CONTAINS_ENTITY                  │  │
│  │   │     chunk_score = max(PPR_score of its entities)      │  │
│  │   │     → returns: [chunk_id, text, score, entity]       │  │
│  │   └────────────────────────────────────────────────────────┘  │
│  │                                                              │
│  ├─ Fragment filter: drop chunks < 80 chars                     │
│  │   + drop lone figure-caption / section-heading patterns     │
│  └─────────────────────────────────────────────────────────────┘  │
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4: Weighted RRF Fusion                                  │
│  RRF_score = (path_weight × consistency × level × domain)      │
│              / (60 + rank)                                      │
│                                                              │
│  path_weights:                                                 │
│    entity_gate:  1.8  ← HIGHEST (most precise signal)         │
│    ppr:         1.7  ← HippoRAG 2 multi-hop                  │
│    entity_cosine:1.6  ← scoped cross-doc entity               │
│    hyde:        1.3  ← hypothetical doc captures intent         │
│    community:   1.2  ← global context                         │
│    rewrite:     1.1  ← paraphrase                             │
│    original:    1.0  ← baseline                              │
│    keywords:    0.9  ← less reliable alone                   │
│    step_back:   0.8  ← abstract, may be too general           │
│                                                              │
│  multipliers:                                                 │
│    consistency:  1.2 (≥0.85) / 1.0 (≥0.60) / 0.8 (<0.60)    │
│    level:       paragraph=1.0, sentence=0.8, section=1.1       │
│    domain:       1.0 + cosine(chunk_domain, query_domain)×0.3  │
│                                                              │
│  Output: [chunk_id_1, chunk_id_2, ..., chunk_id_50]          │
│          sorted by fused RRF score                             │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 5: Reranking                                            │
│  ├─ Stage 1: format/level normalize (min-max per format)       │
│  ├─ Stage 2: Cross-encoder (bge-reranker-v2-m3)               │
│  │    input: [query, chunk_text] → score                      │
│  │    top-50 → top-8                                          │
│  │    early-exit if score ≥ 0.85                              │
│  └─ Stage 3: LLM judge (optional, currently OFF)               │
│       → pairwise preference between top candidates              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 6: Context Compression (conditional)                     │
│  if len(context) > 5000 chars:                                  │
│    LLMLingua-2 compresses at rate=0.4 (keeps 40% tokens)      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 7: Sufficient-Context Gate (Fast check before generation)│
│  ├─ fast LLM call (settings.light_llm)                          │
│  └─ checks if context is sufficient to answer query            │
│       if insufficient → short-circuit: return refusal           │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 8: Generation                                           │
│  ├─ Outline (if enabled): 300 tokens                          │
│  ├─ Drafts (default 1): DRAFT_PROMPT, temperature=0.0        │
│  │    [SPEC: multiple drafts → judge → best]                   │
│  └─ Refine (if enabled): REFINE_PROMPT, temperature=0.0       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 9: Validation Gates (3 LLM calls, sequential)           │
│  ├─ Gate 1: grounded_ratio ≥ 0.70                             │
│  │    "Is each claim in the answer grounded in context?"      │
│  │                                                              │
│  ├─ Gate 2: citation_ratio ≥ 0.70                              │
│  │    "Does each sentence end with [chunk_id]?"                │
│  │    [SPEC: skip if grounded_ratio < 0.7 — TODO]             │
│  │                                                              │
│  └─ Gate 3: entity_gate (Title-Cased regex)                   │
│       "Are entities in answer from retrieved chunks?"           │
│                                                              │
│  Failure actions:                                              │
│    attempt=0: corrective regeneration (stricter prompt)        │
│    attempt=1: broaden retrieval (top_k increases)             │
│    attempt=2+: refuse                                         │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
  Answer + Sources + Citations + Latency Breakdown
```

---

## Cross-Doc Entity Gate — Chi tiết

Cross-doc là cơ chế **khám phá phạm vi document** trước khi retrieval. Thay vì tìm chunk trực tiếp, nó tìm **entity** trước rồi mở rộng ra toàn bộ document chứa entity đó.

### Tại sao cần?

```
Query: "Leiden algorithm khác gì Louvain?"

Problem:
  - Leiden và Louvain có thể nằm ở 2 paper khác nhau
  - Vector search đơn thuần (dense) KHÔNG cross document
  - "Leiden" search → chunks trong paper có "Leiden" ✓
  - "Louvain" search → chunks trong paper có "Louvain" ✓
  - Nhưng không có chunk nào chứa CẢ 2 cùng lúc

Solution (Cross-Doc):
  ① Entity extraction: "Leiden", "Louvain"
  ② Tìm chunks chứa "Leiden" → Paper_A
  ③ Tìm chunks chứa "Louvain" → Paper_B  
  ④ Entity centroid: trung bình vector của ALL chunks trong Paper_A/B
  ⑤ Cosine(query_vec, entity_centroid) × TF-IDF
  ⑥ → chunks CÓ entity = cả 2 paper được discover
```

### Entity centroid computation

```
Entity centroid = mean(all chunk vectors containing this entity)

Chuẩn hóa: centroid / ||centroid||  (L2 normalize)

Example:
  Entity "GraphRAG" xuất hiện trong:
    chunk_1 → vec_1 = [0.1, 0.2, ...]
    chunk_3 → vec_3 = [0.15, 0.18, ...]
    chunk_7 → vec_7 = [0.12, 0.21, ...]
  
  centroid = mean([vec_1, vec_3, vec_7])
  normalized_centroid = centroid / ||centroid||
  
  cosine(query_vec, normalized_centroid) = high  ←  "GraphRAG" là semantic concept mạnh
```

### L1 Anti-Supernova Guard (TF-IDF)

```
Problem:
  Entity "RAG" xuất hiện trong 95% chunks (hub entity)
  Entity "HippoRAG" xuất hiện trong 3 chunks (rare)
  
  cosine(query, "RAG") > cosine(query, "HippoRAG") luôn luôn
  → "RAG" luôn rank cao nhất = noise

Solution (TF-IDF weight):
  TF-IDF("RAG")     = log((N+1)/(df+1)) = log((1000+1)/(950+1)) ≈ 0.05  ← tiny
  TF-IDF("HippoRAG")= log((1000+1)/(3+1))   ≈ 5.6   ← boosted
  
  weighted_score = cosine × TF-IDF
  → Rare entities được boost, hub entities được penalize
```

---

## Chunk ID — Chi tiết

### Chunk ID Format

```
chunk_id = "{doc_id}::{format}::{chunk_level}::{index}"

Example:
  "doc_4c4d233a73493cac::pdf::paragraph::0"
  "doc_a1b2c3d4e5f6::markdown::section::2"

Components:
  doc_id:      SHA-1 hash của file content (8-16 ký tự hex)
  format:      pdf | docx | txt | xlsx | markdown | html | chat
  chunk_level: document | section | paragraph | sentence
  index:       0-based position trong document
```

### Chunk ID được sử dụng ở đâu

```
1. INGEST
   └─ Ingestion pipeline tạo chunk_id khi chunking
      → được lưu vào Qdrant payload
      → được lưu vào Neo4j (Chunk node)

2. RETRIEVAL
   ├─ Qdrant vector search → trả về chunk_id + text + score
   ├─ Neo4j Cypher: CONTAINS_ENTITY(chunk_id) → lấy related chunks
   ├─ Entity gate: entity centroid → chunk_ids_scope
   ├─ PPR: ranked entities → top chunks via CONTAINS_ENTITY
   └─ RRF fusion: score aggregation by chunk_id

3. RERANK
   └─ stage2 cross-encoder nhận [chunk_id, chunk_text, score]

4. GENERATION
   ├─ format_context: gộp chunk_id + text thành prompt context
   └─ Citation: LLM generate answer WITH [chunk_id] markers

5. VALIDATION
   ├─ grounded_ratio: LLM check claim → source [chunk_id]
   ├─ citation_ratio: verify [chunk_id] tags exist
   └─ entity_gate: Title-Cased regex → verify entity ∈ chunk_id sources

6. CACHE
   └─ Chat history: query_embed + answer → stored by session_id
```

### Chunk ID → Payload mapping

```
Qdrant payload schema:
{
  "chunk_id": "doc_xxx::pdf::paragraph::0",
  "doc_id": "doc_xxx",
  "text": "The Leiden algorithm was proposed in 2020...",
  "source": "/data/papers/rag_survey.pdf",
  "format": "pdf",
  "chunk_level": "paragraph",
  "consistency_score": 0.92,
  "domain_distribution": {"AI": 0.8, "ML": 0.2},
  "domain_primary": "AI",
  "tenant_id": "corpus500",
  "author": "Nguyen Van A",
  "parent_chunk_id": "doc_xxx::pdf::section::1",  ← link to parent
}

Neo4j Chunk node:
  (c:Chunk {id: "doc_xxx::pdf::paragraph::0"})
    -[:FROM_DOCUMENT]-> (d:Document)
    -[:CONTAINS_ENTITY]-> (e:Entity)
    -[:NEXT_CHUNK]-> (c2:Chunk)  ← linked list
```

---

## Retrieval Paths — So sánh

| Path | Input | Storage | Strength | Weakness |
|------|-------|---------|----------|----------|
| **entity_gate** | query_vec | Qdrant + Neo4j | Cross-doc discovery, scope = ALL docs | Slower (centroid compute) |
| **dense view** | query_vec | Qdrant | Fast, general | No cross-doc |
| **entity_pivot** | query_entities | Neo4j only | Precise entity match | Requires entity in query |
| **entity_cosine** | query_vec + scope | Qdrant + Neo4j | Scoped diversity | Limited to scope |
| **ppr** | query_entities | Neo4j only | Multi-hop, relational | No vector semantic |
| **community** | query_vec | Neo4j only | Global context | Low granularity |
| **graph** | query_vec | Neo4j only | Co-occurrence signal | Weak alone |

---

## Intent → Strategy Mapping

| Intent | Views | use_graph | use_pivot | use_PPR | Key signal |
|--------|-------|-----------|-----------|---------|------------|
| factual | dense, graph_aware, keywords | ❌ | ❌ | ❌ | Vector dense |
| analytical | dense, graph_aware, summary, question | ✅ | ✅ | ✅ | Graph + summary |
| comparison | dense, graph_aware, question | ✅ | ✅ | ❌ | Graph |
| multi_hop | dense, graph_aware, question | ✅ | ✅ | ✅ | PPR |
| summarization | summary, graph_aware, dense | ❌ | ❌ | ❌ | Community |
| kg_construction | dense, keywords, summary | ✅ | ✅ | ✅ | All KG paths |

---

## Các điểm nghẽn (Bottlenecks)

### 1. Query Understanding (~8s)
Chiếm 49% tổng latency. SPEC đã mark là skip cho short+entity nhưng chưa implement.

### 2. Entity Gate centroid computation (first call ~200-500ms)
- Lần đầu chạm entity mới: phải fetch chunks + compute centroid
- Lần sau: <5ms cache hit
- Cache: 5000 entries max, TTL 30 phút

### 3. PPR graph load (~500-2000ms)
- Lần đầu: load toàn bộ entity graph từ Neo4j
- Lần sau: <1ms cache hit (10 phút TTL)
- Neo4j query có thể chậm nếu graph lớn (199K edges)

### 4. Validation gates (sequential, ~10s)
- Gate 1 và Gate 2 chạy song song được nhưng đang sequential
- SPEC đã mark parallelize nhưng chưa implement

### 5. Cross-encoder rerank (~2-5s)
- Mỗi candidate = 1 forward pass bge-reranker
- 50 candidates = ~2-5s
- Early-exit khi top score > 0.85 giúp tiết kiệm
