# VRAG — Architecture Reference

This document is the **deep technical reference** for VRAG: data structures, algorithms, design rationale, and the trade-offs that shape every decision. For a faster overview see [README.md](README.md); for component-level config and API contracts see [SPEC.md](SPEC.md).

---

## 1. Design Philosophy

VRAG optimizes for **answer correctness** over throughput. Concretely, that means three things:

1. **Refuse when uncertain.** The Triple-Gate validator can vetoes a generated answer. Producing a refusal is preferable to a hallucination.
2. **Defer to deterministic ops.** Wherever a centroid match, regex, or graph traversal can replace an LLM call, it does. LLM calls are the bottleneck on CPU; eliminating them is the single biggest latency lever.
3. **Compose retrieval signals.** Dense vectors, sparse BM25, graph entity-pivot, and community summaries each have failure modes. Multi-path retrieval + Weighted RRF fusion exploits their complementarity.

These choices come at a cost: latency on CPU is high (100-400s p95 for Qwen 3.5 9B). Production deployments need GPU. We accept that trade because the alternative — fast but ungrounded — is unacceptable for the enterprise/regulated use cases VRAG targets.

---

## 2. Data Model

### 2.1 Qdrant collection `enterprise_kb`

| Field | Type | Notes |
|---|---|---|
| `id` | u64 (hash of `chunk_id`) | Deterministic via `to_int_id` |
| Named vector: `dense` | 1024-d, cosine | BGE-M3 embedding of raw text |
| Named vector: `paraphrase` | 1024-d, cosine | BGE-M3 of LLM paraphrase |
| Named vector: `question` | 1024-d, cosine | BGE-M3 of LLM-generated questions |
| Named vector: `summary` | 1024-d, cosine | BGE-M3 of LLM-generated summary |
| Named vector: `keywords` | 1024-d, cosine | BGE-M3 of extracted keywords |
| Sparse vector: `bm25` | u32 → f32 | Tokenized BM25 indices/values |
| Payload: `tenant_id` | str (indexed) | Multi-tenant filter |
| Payload: `chunk_id` | str (indexed) | Stable identifier `<doc_id>::<level>::<idx>` |
| Payload: `doc_id`, `source`, `text`, `format`, `chunk_level` | various | Display + filtering |
| Payload: `consistency_score` | float | Self-similarity across views |
| Payload: `access_level`, `department`, `created_at` | various | RBAC + temporal |

Missing views fall back to `dense` so the schema is always complete. Upsert in batches of 50 with `wait=False`.

### 2.2 Neo4j graph

```
(Document {doc_id, tenant_id, source, created_at})
   ↓ FROM_DOCUMENT
(Chunk {id, text, tenant_id, format, chunk_level, consistency_score})
   ↓ CONTAINS_ENTITY
(Entity {name, type, tenant_id, confidence})
   ↓ ALIAS_OF              ← Tier-2 canonicalization
(Entity)
   ↔ RELATED_TO {description}    ← Multi-pass voted relations
(Entity)
(Community {id, tenant_id, summary, member_count})
   ↓ MEMBER_OF
(Entity)
```

Indices:
- `Chunk(tenant_id)`, `Chunk(id)` — required for fast filter
- `Entity(name, tenant_id)` composite — entity lookup
- `(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)` traversed via tenant filter

### 2.3 Redis semantic cache

Key: `vrag:cache:{tenant}:{hash(query_embed)}`
Value: full chat response JSON (TTL configurable)
Hit logic: cosine ≥ 0.95 between current embed and key embed.

---

## 3. Offline Ingestion Pipeline

```
file → parse → chunk → embed (5 views + sparse) → entity_extract (GLiNER)
                                                   ↓
                              ┌────────────────────┴────────────────────┐
                              ↓                                          ↓
                         Qdrant.upsert                        canonicalize + Neo4j
                                                                ↓
                                                       (optional) cross_doc + community
```

### 3.1 Multi-format parse
- PDF: `pypdf` for text-only, `docling` for table-aware (opt-in)
- DOCX: `python-docx` + `mammoth` for inline images
- XLSX: `openpyxl` → row/cell concatenation

### 3.2 Hierarchical chunking

`src/services/chunkers/multi_signal_chunker.py` segments by:
- Heading boundaries (Markdown / DOCX styles)
- Sentence boundaries (regex + Vietnamese-aware splitter)
- List bullet detection
- Max-char caps per level (`section_max_chars`, `paragraph_max_chars`)

Two levels enabled by default: `paragraph` + `section`. Sentence-level optional for fine-grained QA.

### 3.3 Multi-view embedding

For each chunk, BGE-M3 produces:
1. `dense` — embedding of raw chunk text
2. `paraphrase` — LLM-rewrites the chunk; embed the rewrite
3. `question` — LLM generates "what questions does this answer?"; embed those
4. `summary` — LLM-generated single-paragraph summary; embed it
5. `keywords` — LLM extracts 5-15 keywords; embed concatenation
6. `bm25` — tokenize + BM25 indices/values

When `CONSISTENCY_VIEWS_ENABLED=0`, only `dense` is computed; others fall back to dense. This bypasses ~5 LLM calls per chunk for fast ingest.

The `consistency_score` is cos-similarity between `dense` and `paraphrase`; chunks where the LLM paraphrase drifts from the original (low consistency) get a downstream score penalty.

### 3.4 Entity extraction (GLiNER)

`urchade/gliner_multi-v2.1` — 168M-param zero-shot NER, runs on CPU, multilingual including Vietnamese. Operates on entity type labels: PERSON, ORGANIZATION, LOCATION, PRODUCT, CONCEPT, TECHNOLOGY, EVENT, OTHER.

Why GLiNER vs LLM-based NER:
- ~50× faster (250ms vs 12s for typical chunk)
- Deterministic output
- No API quota
- Multilingual native

### 3.5 Hybrid 3-Tier Canonicalization

Aims to keep the graph dense and aliased — preventing the common GraphRAG failure mode where "Tim Cook", "Timothy Cook", "Apple CEO" become 3 unconnected nodes.

| Tier | Method | Status |
|---|---|---|
| 1 (Exact) | Name + type match in KG → reuse canonical | ✅ Implemented |
| 2 (Lexical) | Levenshtein ratio ≥ 0.85 + same type → `ALIAS_OF` | ✅ Implemented (`difflib.SequenceMatcher`) |
| 3 (Semantic) | Vector dot product on entity embeddings → `ALIAS_OF` | 📋 Roadmap |
| 4 (Hard merge) | `apoc.refactor.mergeNodes` for confirmed duplicates | 📋 Roadmap |

The roadmap items are the largest known gap from the [original design plan](.claude/internal-docs/caitien.md).

### 3.6 Relation extraction (voted)

For each chunk, the LLM is prompted N times (`ENTITY_VOTE_PASSES=3`) to extract relations. A relation must appear in ≥ `ENTITY_VOTE_MIN=2` passes to be persisted. Filters out one-off hallucinations.

### 3.7 Community detection

`leidenalg` on the entity-entity graph weighted by co-occurrence count. Fallback to networkx Louvain if `python-igraph` unavailable.

For each detected community (≥3 members), the LLM generates a 2-3 sentence summary. Summaries are stored as `Community.summary` and embedded into Qdrant under a synthetic "community" chunk_id — allowing community-level retrieval at query time.

---

## 4. Online Query Pipeline (4 Tiers)

Reference implementation: `api/routes/_chat.py` → `multi_path_retrieve` → `rerank_full_pipeline` → `validate_answer`.

### 4.1 Tier 1 — Zero-LLM Pre-processing

`src/services/query_understanding.py::understand_query`

Runs in parallel via `asyncio.gather`:
1. **BGE-M3 embed** of query (handled downstream in `multi_path_retrieve`, but the spec considers it a Tier-1 task)
2. **Semantic Router** — `classify_query` does cosine match against 5 precomputed intent centroids (`config/intent_centroids.npy`). <1ms after the embed completes.
3. **GLiNER entity extraction** — `extract_entities_fast`. ~250ms hot.

If `QUERY_REFORMULATIONS > 0`, additional LLM tasks are added in priority order: `rewrite` → `keywords` → `hyde` → `decompose` → `step_back`. Default is **0** — no LLM at Tier 1.

#### Router design

5 intent centroids computed from 15 anchor queries each:
- `factual` — direct lookup ("X là gì", "How many parameters")
- `analytical` — reasoning ("Tại sao", "What are trade-offs")
- `comparison` — A vs B
- `multi_hop` — cross-doc reasoning, relationship chains
- `kg_construction` — schema/pipeline questions

A 17-pattern regex pre-filter catches out-of-domain queries (weather, sports, recipes) before the centroid match.

Threshold (`_SEMANTIC_THRESHOLD = 0.45`): if best centroid score below this, fall back to `factual`.

### 4.2 Tier 2 — Vector-Driven Hard Limit

`multi_path_retrieve` runs vector + graph + community paths in parallel, then runs entity-pivot in a **second phase** so the entity-pivot Cypher can scope itself to the chunks already found by dense search.

```
Phase 1 (parallel):
  - For each reformulation × view: Qdrant search
  - Graph path: BGE-M3 query embed → match Entity by name → expand 2-hop
  - Community path: query embed → match Community.summary embedding

Phase 2 (sequential after Phase 1):
  - Entity-pivot path:
      pre_extracted_entities = understanding.entities  (from GLiNER, Tier 1)
      scope = collect_chunk_ids_from_paths(phase1_paths)[:100]
      Cypher: MATCH (c)-[:CONTAINS_ENTITY]->(e {name IN $entities})
              WHERE c.id IN $scope   ← THE HARD LIMIT
```

Why the hard limit matters: without it, querying about a popular entity ("AI", "GraphRAG") triggers Cypher traversal across 10k+ chunks, blowing up latency and memory. The Qdrant top-100 supplies a semantic prior that bounds the search.

### 4.3 Tier 3 — Fusion + Compression

#### 4.3.1 Score-Weighted RRF

Standard RRF: `1 / (k + rank)`. VRAG enhances with:
- Per-path weight (e.g. `entity_pivot=1.5` because KG-validated matches are highest precision)
- Per-reformulation weight (e.g. `hyde=1.3` because hypothetical doc embeddings often match better than raw queries)
- Domain reward bonus (`+0.3 × domain_match`)

Final formula per candidate `c`:
```
score(c) = Σ_{path p} (path_weight[p] × reform_weight[p] × (1 / (60 + rank_p(c))))
         × (1 + 0.3 × is_domain_match(c, query_domain))
```

#### 4.3.2 Score normalization

Before RRF, each candidate's raw score is normalized per format group (PDF / XLSX / DOCX) to z-score-like distribution. This prevents PDF chunks (which tend to have higher absolute scores due to longer text) from dominating XLSX chunks.

#### 4.3.3 3-stage rerank with Dynamic Early-Exit

`src/services/rerank.py::rerank_full_pipeline`

```
top-30 from RRF
  ↓
Stage 1: Cross-encoder (BAAI/bge-reranker-v2-m3)
  → top-20 with stage1_score
  ↓
  [Early-Exit gate] if avg(top-5 stage1_score) >= 0.85:
                       skip Stage 3, fall through to final
  ↓
Stage 2: Re-embed query, cosine vs candidate's `summary` view
  → top-10 with stage2_score
  ↓
Stage 3: (skipped if early-exit) LLM judge per-candidate, 0.0-1.0
  → top-5 with stage3_score
  ↓
final_score = 0.4 × s1 + 0.3 × s2 + 0.3 × s3   (s3 falls back to s2 if skipped)
  ↓
rerank_l2r: feature-engineered LTR re-ranks top-5 using entity match, chunk level,
            recency, retrieval path provenance, etc. (LambdaMART when model present)
```

#### 4.3.4 LLMLingua-2 Context Compression

`src/services/context_compress.py`

After rerank produces final top-5, `format_context` assembles them into a single string. LLMLingua-2 compresses this string:

```python
PromptCompressor("microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
  .compress_prompt(context, rate=0.4, force_tokens=["[", "]", ":", "**"])
```

Why LLMLingua-2 over LLMLingua-1:
- Classifier-based (token-level binary keep/drop) instead of perplexity-based — no small LLM needed
- Multilingual XLM-RoBERTa base — works on Vietnamese
- ~5× faster on CPU

`force_tokens` preserves citation brackets `[chunk_id]` and entity boundaries so downstream citation gate can still detect them.

Typical compression: 800 tokens → 320 tokens (ratio 0.40). Downstream LLM generation latency drops ~70%; validation drops ~88%.

### 4.4 Tier 4 — Generation + Validation

#### 4.4.1 Single LLM call

`api/routes/_prompts.py::DRAFT_PROMPT` includes "Text Smoothing" instructions: use **bold** for keywords, bullet points for lists, preserve English technical terms.

If `GENERATION_DRAFTS > 1`, multiple drafts run in parallel at increasing temperatures (0.2, 0.35, 0.5...). If `GENERATION_JUDGE_ENABLED=1`, a separate LLM call picks the best.

If `GENERATION_REFINE_ENABLED=1`, a second LLM pass smooths the draft using `REFINE_PROMPT`.

Streaming endpoint `/api/v3/chat/stream` uses `ollama_chat_stream` to yield tokens via SSE.

#### 4.4.2 Triple-Gate Validation (parallel)

`src/services/validation.py::validate_answer` runs three gates via `asyncio.gather`:

| Gate | Method | Pass threshold |
|---|---|---|
| Hallucination | Extract claims → LLM verifies each against retrieved context | `grounded_ratio ≥ 0.80` |
| Entity | Extract entities from answer → check existence in Neo4j | `invalid_entities ≤ 2` |
| Citation | Count sentences ending with `[chunk_id]` | `citation_ratio ≥ 0.70` (refusals exempt) |

If any gate fails, the answer is rejected. On `max_retries > 0`, the system regenerates with stricter prompt (`correct_and_regenerate` uses a more conservative prompt + tighter top-k); otherwise it returns the configured refusal message.

This is the single most distinctive feature vs other open-source RAG systems. It guarantees that user-facing answers either cite retrieved context faithfully or explicitly refuse.

#### 4.4.3 ReAct path (multi-hop intents)

`src/services/react_loop.py::run_react`

When `classify_query` returns one of `analytical | comparison | multi_hop | kg_construction`, the chat handler can route to ReAct instead of standard generation:

```
loop (max_steps=6):
  observation: current retrieved chunks + reasoning so far
  thought + action: LLM chooses next tool
  tools:
    - graph_aware_search(query): GAEA-refined vector search
    - entity_pivot(entities): jump from query entities to chunks
    - community_lookup(topic): retrieve community summary
    - finalize(): commit to answer
```

ReAct tools share the same Qdrant/Neo4j clients; the loop accumulates chunks for the final answer. Steps are bounded by `react_max_steps` (default 6) and a token budget.

---

## 5. The Retrieval Path Catalogue

Below is the full set of paths `multi_path_retrieve` can spawn. Which paths fire depends on the intent strategy (`INTENT_STRATEGY` in `retrieval.py`) and entity availability:

| Path key | What it does | When it fires |
|---|---|---|
| `original:dense` | Raw query → `dense` view | Always |
| `original:graph_aware` | Raw query → `graph_aware` view (GAEA-refined) | For intents in views list |
| `original:summary` | Raw query → `summary` view | Analytical, summarization |
| `original:question` | Raw query → `question` view | Analytical, comparison, multi_hop |
| `original:keywords` | Raw query → `keywords` view | Factual, kg_construction |
| `rewrite:*` | LLM rewrite → all views | If `QUERY_REFORMULATIONS ≥ 1` |
| `hyde:*` | Hypothetical doc embed → all views | If `QUERY_REFORMULATIONS ≥ 3` |
| `step_back:*` | Abstracted query → all views | If `QUERY_REFORMULATIONS ≥ 5` |
| `decompose:*` | Sub-query × views | If multi-hop detected |
| `graph` | KG traversal from matched entities | If intent.use_graph |
| `community` | Community summary embedding match | If `COMMUNITY_ENABLED` |
| `entity_pivot` | `CONTAINS_ENTITY` Cypher confined to top-100 chunks | If entities detected (orthogonal to intent) |

At maximum (`QUERY_REFORMULATIONS=5`, all enabled), this can be 30+ parallel paths. The default config (`QUERY_REFORMULATIONS=0`) runs 3-7 paths, optimal for CPU.

---

## 6. Failure Modes and Defenses

| Failure | Defense |
|---|---|
| Supernode traversal explosion | Tier 2 Hard Limit (Qdrant scope → Cypher) |
| LLM hallucination | Triple-Gate Validation |
| Entity name fragmentation | 3-Tier Canonicalization (Tier 2 semantic still on roadmap) |
| Stale community summaries | `/api/v3/community/build` re-runs Leiden; summaries regenerated for changed communities |
| Cold cache 50s p95 | Bundled GLiNER + LLMLingua model in image; Qdrant filter indices warmed at boot |
| Tenant data leak | Multi-tenant filter on every Qdrant + Cypher query; verified at `multi_path_retrieve` boundary |
| Cross-encoder OOM | Stage 1 disabled by default (`RERANK_STAGE1_ENABLED=0`) when running in 1GB container |
| Long context blowing token limit | LLMLingua-2 rate=0.4; max_tokens caps; final_top_k=5 |
| Q&A on noisy / low-consistency chunks | `consistency_score` factor in `final_score` formula |

---

## 7. Observability

### 7.1 Langfuse traces

Every chat request creates a trace with spans:
- `query_understanding` (with reformulation outputs)
- `retrieval` per-path (with retrieved chunk IDs + scores)
- `rerank` per-stage
- `context_compression` (token counts before/after)
- `generation` (input prompt + output, fully captured)
- `validation` per-gate (claim verdicts, missing entities, citation tally)

### 7.2 Prometheus metrics

Exposed at `/metrics`:
- `vrag_chat_total{tenant,intent,refused}` counter
- `vrag_chat_latency_seconds{stage}` histogram
- `vrag_cache_hits_total` / `vrag_cache_misses_total`
- `vrag_validation_grounded_ratio` histogram
- `vrag_ingest_chunks_total{tenant,format}` counter
- `vrag_qdrant_search_latency_seconds` histogram

Grafana dashboard at `grafana/dashboards/` (auto-provisioned).

### 7.3 Latency breakdown

Every chat response includes `latency_breakdown_ms` per stage so users can profile their own queries:

```json
{
  "total_ms": 113763.7,
  "generation_attempt0_ms": 53718.3,
  "validation_attempt0_ms": 27971.4,
  "refinement_attempt0_ms": 16357.6,
  "query_understanding_ms": 12394.4,
  "entity_extraction_ms": 9203.1,
  "context_compression_attempt0_ms": 8083.0,
  "rerank_attempt0_ms": 1718.7,
  "retrieval_attempt0_ms": 493.5,
  "ood_detection_ms": 0.3
}
```

---

## 8. Performance Notes

CPU baseline (M-series Docker, Qwen 3.5 9B, all defaults):
- Cold p95: ~110-150s (LLM gen + validation dominant)
- Hot p95: ~80-110s
- Retrieval alone: <1s
- Tier 1 zero-LLM: <500ms hot

GPU projection (single RTX 4090 / A100 / M4 Max Metal):
- Cold p95: ~10-20s (10× LLM speedup)
- Hot p95: ~5-10s
- Throughput: 50-100× higher (gen no longer single-threaded CPU-bound)

The pipeline is **GPU-ready**; nothing in the code path is CPU-dependent. The bottleneck is purely the LLM inference.

---

## 9. Cross-Document Linking

`/api/v3/cross_doc/build` runs `link_chunks_cross_doc`:

1. Sample chunks from KG (`Chunk` nodes)
2. For each, fetch its `dense` vector from Qdrant
3. Query Qdrant for top-K similar chunks in *other* documents (filter `doc_id != self.doc_id`)
4. Write `SIMILAR_TO` edges in Neo4j with cosine score as weight

This creates a cross-document similarity graph that supports queries like "find documents similar to this passage" without re-running expensive embedding searches.

---

## 10. Domain-Aware Reward (Phase 8)

`tag_query` and `tag_chunk` classify both queries and chunks into 5 domain tags (technical, business, legal, medical, general) via keyword heuristics. The RRF fusion adds a 30% bonus to chunks whose primary domain matches the query domain.

This is a soft signal — domain mismatch doesn't disqualify a chunk, just slightly downweights it.

---

## 11. Quality vs Latency Knobs Cheat-Sheet

| Goal | Set |
|---|---|
| Maximum quality, no latency budget | `QUERY_REFORMULATIONS=5 RERANK_STAGE1_ENABLED=1 RERANK_STAGE3_ENABLED=1 COMMUNITY_ENABLED=1 GENERATION_DRAFTS=3 GENERATION_REFINE_ENABLED=1` |
| Balanced (default) | `QUERY_REFORMULATIONS=0 CONTEXT_COMPRESSION_ENABLED=1 RERANK_STAGE1_ENABLED=0 GENERATION_DRAFTS=1 GENERATION_REFINE_ENABLED=1` |
| Maximum speed, accept quality drop | `QUERY_REFORMULATIONS=0 CONTEXT_COMPRESSION_ENABLED=1 RERANK_STAGE1_ENABLED=0 RERANK_STAGE3_ENABLED=0 GENERATION_DRAFTS=1 GENERATION_REFINE_ENABLED=0 VALIDATION_ENABLED=0` |
| Stress-test ingestion | `CONSISTENCY_VIEWS_ENABLED=0 ENTITY_RELATIONS_ENABLED=0 PII_LLM_NER_ENABLED=0` |

---

## 12. References

VRAG draws on (in roughly the order of intellectual debt):

- **BGE-M3** (Chen et al., BAAI) — multi-functionality (dense+sparse+ColBERT) multilingual embedding
- **GraphRAG** (Edge et al., Microsoft) — community summaries via Leiden + LLM
- **HippoRAG** (Gutiérrez et al.) — entity-centric KG retrieval, multi-hop
- **LightRAG** — graph-augmented retrieval with low overhead
- **RAPTOR** — hierarchical clustering of summaries
- **GLiNER** (Zaratiana et al.) — zero-shot NER via span classification
- **LLMLingua-2** (Pan et al., Microsoft) — classifier-based prompt compression
- **Self-RAG** (Asai et al.) — reflection tokens; we adopt the spirit (Triple-Gate) without the tokens
- **HyDE** (Gao et al.) — hypothetical document embedding for query reformulation
- **ReAct** (Yao et al.) — reasoning + acting loop for tool-using agents
- **LambdaMART** — used in rerank_l2r as the learning-to-rank model

Full bibliography of the ~50 papers reviewed during design is in `data/eval/` (gitignored — see arxiv IDs in source comments).
