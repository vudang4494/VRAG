# VRAG — Production Pipeline (G state)

State sau commit `bcc8983`. Đã verify trên 53-q eval: doc_recall 0.706, validation 0.947; latency p50 14.8s (đo 2026-07-17, xem §Bench).

## High-level flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│  USER QUERY                                                              │
│      │                                                                   │
│      ▼                                                                   │
│  Layer 1: Pre-RAG (< 200ms)                                              │
│      ├─ Intent classifier (heuristic + LLM fallback ambiguous)           │
│      ├─ chat_history cache lookup (semantic 0.80 threshold)              │
│      ├─ query_router.classify_query (centroid embed cosine)              │
│      ├─ should_use_react (strict gate: rules on len + entities + kw)     │
│      └─ global_map_reduce khi query_type=global + GLOBAL_QUERY_ENABLED=1 │
│                                                                          │
│  ┌─────────────────────────┬──────────────────────────────────┐         │
│  ▼ standard path           ▼ react path                       │         │
│                                                                          │
│  Layer 2: Query Understanding     Layer 2: Workflow Pre-seed            │
│      ├─ rewrite (LLM)                ├─ entity_cosine_search             │
│      └─ keywords (LLM)               └─ PPR pre-seed                     │
│        ⚠ skip if short+entity        no LLM in planning                 │
│                                                                          │
│  Layer 3: Retrieval (parallel)    Layer 3: Workflow Loop                │
│      ├─ multi_path_retrieve          select_workflow(query_type)        │
│      │   ├─ entity_gate primary      → search_entity                    │
│      │   ├─ vector × 5 views         → expand_relation                  │
│      │   ├─ entity_pivot             → retrieve_chunks                  │
│      │   ├─ PPR (HippoRAG 2)         → graph_aware_search               │
│      │   └─ community (optional)     → rerank                           │
│      ├─ weighted RRF fusion          → FINISH                           │
│      └─ fragment filter              ↑ all rule-based                   │
│                                                                          │
│  Layer 4: OOD Detection                                                  │
│      └─ score floor 0.50 + keyword overlap → refuse                     │
│                                                                          │
│  Layer 5: Rerank (bge-reranker-v2-m3)                                   │
│      └─ cross-encoder top-50 → top-8, early-exit 0.85                   │
│                                                                          │
│  Layer 6: Context Compression (conditional)                              │
│      └─ LLMLingua-2 if len(ctx) > 5000 chars                            │
│                                                                          │
│  Layer 7: Generation                                                     │
│      └─ gemma4:e4b max_tokens=1024                                       │
│                                                                          │
│  Layer 8: Validation Gates (parallel)                                    │
│      ├─ grounded_ratio ≥ 0.70 (cosine-grounding mặc định, LLM fallback)  │
│      ├─ invalid_entities ≤ 3 (entity check qua Neo4j)                    │
│      └─ citation_ratio ≥ 0.70 (per-sentence [chunk_id])                  │
│                                                                          │
│  Layer 9: Response                                                       │
│      ├─ stream answer + citations                                        │
│      └─ store to chat_history                                            │
└──────────────────────────────────────────────────────────────────────────┘
```

## Stage-by-stage breakdown

### Layer 1 — Pre-RAG (< 200ms, no LLM)

| Stage | Time | Module | Rule |
|---|--:|---|---|
| Intent classifier | <1ms | `intent_classifier.py` | greeting/question/follow_up heuristic |
| chat_history cache | 100ms | `chat_history.py` | embedding gemma 768d, threshold 0.80 |
| classify_query | 50ms | `query_router.py:_load_centroids` | 5 intent centroids cosine |
| strict_gate (ReAct trigger) | <1ms | `query_router.should_use_react` | len≥60 OR entities≥2 OR keyword match |

**Output**: route decision (standard | react | refuse).

### Layer 2 — Path-specific preparation

**Standard**: query_understanding (8s LLM, skip when short+entity_present)
**ReAct**: pre-seed entity_cosine + PPR (~650ms, no LLM)

### Layer 3 — Retrieval

**Standard — multi_path_retrieve** (`retrieval.py`):
- entity_gate primary (cross-doc, top-50 entity by cosine+TFIDF, MMR λ=0.6)
- 5 dense views (dense, paraphrase, question, summary, keywords) × N reformulations
- entity_pivot via Cypher CONTAINS_ENTITY
- PPR (HippoRAG 2 with co-occurrence fallback)
- community (optional)
- Weighted RRF fusion: entity_gate(1.8) > ppr(1.7) > entity_cosine(1.6) > entity_pivot(1.5) > hyde(1.3) > rewrite(1.1) > original(1.0)
- Fragment filter: drop chunks <80 chars

**ReAct — workflow loop** (`react_loop.py`):
Per-intent workflow from `WORKFLOWS` dict. Each step is `(action, args_builder, skip_predicate)`. No LLM in planning.

| Intent | Workflow |
|---|---|
| factual | search_entity → retrieve_chunks → rerank → FINISH |
| comparison | search_entity → expand_relation → retrieve_chunks → graph_aware_search → rerank → FINISH |
| multi_hop | expand_relation → retrieve_chunks → graph_aware_search → rerank → FINISH |
| analytical | search_entity → expand_relation → graph_aware_search → retrieve_chunks → rerank → FINISH |
| kg_construction | count_entities → expand_community → retrieve_chunks → rerank → FINISH |

### Layer 4 — Rerank

`rerank.py` + `rerank_stages.py`:
- Stage 1: format/level normalize
- Stage 2: bge-reranker-v2-m3 cross-encoder
- Stage 3: optional LLM judge (currently OFF)
- Early-exit at score 0.85

### Layer 5 — Context Compression (conditional)

`context_compress.py`:
- LLMLingua-2 multilingual-meetingbank model
- Rate 0.4 (60% reduction)
- Skip when len(ctx) < 5000 chars

### Layer 6 — Sufficient-Context Gate

`sufficient_context.py`:
- fast LLM call (settings.light_llm)
- evaluates if retrieved context contains enough info to answer query
- if NO -> returns early refusal (saves ~14s of generation)

### Layer 7 — Generation

`ollama_helper.py` → gemma4:e4b:
- max_tokens=1024 (current); **TODO: trim to 512**
- temperature=0.3 for synthesis
- format=text (json mode used selectively for classifier)

### Layer 8 — Validation gates

`validation.py` — 3 gate:
- grounded_ratio ≥ 0.70: cosine-grounding mặc định (LLM claim-extract chỉ là fallback)
- invalid_entities ≤ 3: entity trong answer verify qua Neo4j
- citation_ratio ≥ 0.70: every sentence ends with `[chunk_id]` tag
- **TODO: skip citation if grounded<0.7 (already failed)**

### Layer 9 — Response + cache

- Stream answer token-by-token via SSE
- Store (query, answer, sources) to chat_history Qdrant collection
- TTL semantic cache: 2h

## Configuration matrix (current)

| Env var | Default | Role |
|---|---|---|
| `OLLAMA_MODEL` | gemma4:e4b | Heavy LLM (synthesis, react decisions when LLM mode) |
| `LIGHT_LLM` | gemma4:e4b | Light LLM (query understanding, validation) |
| `OLLAMA_EMBED_MODEL` | bge-m3 | Chunk + query embedding |
| `ENTITY_EXTRACTOR_PROVIDER` | gliner | GLiNER multi v2.1 NER |
| `ENTITY_GATE_ENABLED` | 0 | Stage 3 primary cross-doc cosine |
| `ENTITY_GATE_TOP_K_ENTITIES` | 50 | Diverse entity pool size |
| `PPR_ENABLED` | 1 | HippoRAG 2 random walk |
| `PPR_ALPHA` | 0.5 | PPR teleport probability |
| `CONTEXT_COMPRESSION_ENABLED` | 0 | LLMLingua-2 |
| `CONTEXT_COMPRESSION_RATE` | 0.4 | Keep 40% of tokens |
| `CONSISTENCY_VIEWS_ENABLED` | 1 | 4 LLM views per chunk (ingest cost) |
| `DOC_CONTEXT_PREFIX_ENABLED` | 1 | Anthropic-style doc summary prefix |
| `REACT_WORKFLOW` | 1 | Per-intent state machine planner |
| `REACT_STRICT_GATE` | 1 | Narrow ReAct trigger via rules |
| `REACT_RULE_BASED` | 1 | Fallback if WORKFLOW=0 |
| `GENERATION_MAX_TOKENS` | 1024 | Synthesis output cap |
| `VALIDATION_ENABLED` | 1 | Triple-gate (grounded + citation + entity) |
| `PII_MASK_ENABLED` | 1 | Mask PII at ingest |
| `PII_LLM_NER_ENABLED` | 0 | Use LLM for PII NER (slow) |

## Bench (live RAGAS, gemma4:e4b, 2026-07-17)

Live reference-grounded RAGAS trên pipeline hiện tại (corpus500, 40 Q&A, judge qwen3.6-35b local):

```
  Faithfulness            0.946   Excellent
  Context Recall          0.946   Excellent
  Context Precision       0.904   Excellent
  Answer Relevancy        0.808   Good
  Factual Correctness F1  0.302   claim-F1 nghiêm ngặt (understated bởi câu trả lời VN dài)
  latency p50 / p95 / max 14.8s / 26.3s / 28.0s
```

Báo cáo đầy đủ xem bảng tổng hợp trên.
Bộ benchmark tiếng Việt cũ (config 05-2026, generator + tenant khác, đã xoá) đã nghỉ hưu.

## Known regressions (follow-up backlog)

1. **agentic_rag**: workflow chưa định nghĩa cho intent này; fallback `factual` không có expand_community. Fix: thêm WORKFLOWS["agentic_rag"] entry.
2. **multi_hop**: strict_gate có thể đang block PPR pre-seed cho 1-2 query. Fix: ensure pre-seed luôn chạy cho `multi_hop` intent regardless of strict_gate.

## Hard limits (physics)

- **gemma4:e4b @ M4 Metal**: LLM synthesis là thành phần latency chính; đo được end-to-end p50 14.8s / p95 26.3s. Không thể nhanh hơn nhiều trừ khi đổi model/hardware.
- **bge-m3 embed**: ~50ms/call. 5 view × 50ms = 250ms paralleled. OK.
- **GLiNER NER**: ~200ms/call sequential (Semaphore=1 to avoid glibc TLS crash). Bottleneck at ingest, not query.
- **Qdrant HNSW search**: ~10ms/view. OK.

## Roadmap

### Phase 1 (immediate, ~2h work) — diminishing-returns sweep
Target: giảm p50 (hiện 14.8s), recall ±0

1. Skip query_understanding for short+entity queries (~4s saved)
2. Conditional context_compression (~5.5s saved)
3. Parallel validation gates (~5s saved)
4. Skip citation when grounded<0.7 (~5s on fails)
5. GENERATION_MAX_TOKENS 1024→512 (~10s on outliers)

### Phase 2 (1-2 days) — fix regressions
6. WORKFLOWS["agentic_rag"] with expand_community early
7. WORKFLOWS["multi_hop"] with forced PPR pre-seed
8. Re-ingest with CONSISTENCY_VIEWS_ENABLED=1 (overnight) for +2-3pp recall
9. EmbeddingGemma ablation vs bge-m3 (1.5h)

### Phase 3 (week+) — scale beyond M4
10. LiteLLM Router → multi-Mac fan-out
11. GPU + vLLM (50× batched LLM throughput)
12. Distributed ingest for 10K+ PDF corpora

### Phase 4 (research, when GPU lands)
13. Fine-tune gemma4:e4b with LLM-JEPA on (query, gold_chunks) pairs
14. Continuous learning loop (Langfuse → DSPy auto-optimization)

## Module ownership (canonical files, no v1/v2/v3 legacy)

| Concern | Module |
|---|---|
| Ingest pipeline | `src/services/ingestion.py` |
| PDF chunking | `src/services/chunkers/pdf_chunker.py` |
| Sentence splitting + tiny merge | `src/services/chunkers/base.py`, `semantic_chunker.py` |
| Consistency views + doc_context | `src/services/consistency.py`, `ingestion.py:3.5` |
| Vector store + named vectors | `src/services/vector.py` |
| Multi-path retrieval | `src/services/retrieval.py` |
| Entity-gate (cross-doc cosine) | `src/services/entity_vectors.py` |
| PPR (HippoRAG 2) | `src/services/ppr.py` |
| ReAct loop + workflow | `src/services/react_loop.py` |
| Query routing + strict gate | `src/services/query_router.py` |
| Reranking | `src/services/rerank.py`, `rerank_stages.py` |
| Validation gates | `src/services/validation.py` |
| Sufficient-Context Gate | `src/services/sufficient_context.py` |
| LLM helper | `src/services/ollama_helper.py` |
| Chat endpoint | `api/routes/_chat.py` |
| ReAct endpoint | `api/routes/_react.py` |
| Re-ingest tool | `scripts/ingest_corpus.py` (recursive, resumable) |

## Operational checklist (before any push origin)

- [ ] Test E2E eval recall ≥ 0.70
- [ ] Latency p50 ≤ 20s
- [ ] Validation pass ≥ 0.90
- [ ] Refusal correct ≥ 0.95
- [ ] No category regression > 15pp
- [ ] No container restart in 4h smoke
- [ ] Backup Qdrant + Neo4j (`make backup`)
- [ ] All canonical service modules pass `ruff check`
- [ ] CHANGELOG entry committed
