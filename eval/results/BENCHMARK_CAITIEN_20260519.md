# Benchmark Report — caitien.md Optimization với vi_benchmark_v2 (42 queries)

**Date:** 2026-05-19
**Corpus:** 51 Academic Papers on RAG Systems
**Benchmark:** vi_benchmark_v2 (42 queries)
**Tenant:** rag51
**Model:** qwen3.5:9b (Ollama, Apple Silicon Metal)
**Total Runtime:** 54.3 minutes

---

## Executive Summary

| Dimension | Before (PRODUCT_EVAL_FULL) | After (caitien) | Change |
|-----------|---------------------------|-----------------|--------|
| **System Reliability** | 100% (42/42) | 100% (42/42) | — |
| **Avg Doc Recall** | 57.9% | 20.2% | **-37.7pp (regression)** |
| **Avg Keyword Hit** | 43.9% | 28.1% | **-15.8pp (regression)** |
| **Refusal Accuracy** | 85.7% | 38.1% | **-47.6pp (regression)** |
| **p50 Latency** | 209s | 56.6s | **-73% (improved)** |
| **p95 Latency** | 253s | 150.2s | **-41% (improved)** |

**Overall: QUALITY REGRESSION** — Latency improved dramatically (209s → 57s) but doc recall dropped from 58% to 20%. 26/42 queries returned empty drafts.

---

## 1. Aggregate Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Success Rate | 42/42 (100%) | >95% | Pass |
| Avg Doc Recall | 20.2% | >70% | **Fail (severe)** |
| Avg Keyword Hit | 28.1% | >60% | **Fail** |
| Refusal Accuracy | 38.1% | >90% | **Fail (severe)** |
| p50 Latency | 56.6s | <60s | **Pass** |
| p95 Latency | 150.2s | <120s | Near-fail |
| Total Time | 54.3 min | — | — |

---

## 2. Root Cause Analysis

### Critical: Generation Prompt Changes Breaking Draft Generation

All 26 refusals return `refusal_reason: "no_drafts"` — meaning the LLM draft generation returned empty.
Root cause: the updated `DRAFT_PROMPT` and `REFINE_PROMPT` in `api/routes/_prompts.py` (Step 5 of caitien.md) introduced complex structured formatting requirements that cause `qwen3.5:9b` to fail generation or timeout.

Changes made:
- **DRAFT_PROMPT**: Changed from simple 10-line directive to structured "quality expert" prompt with bold, bullets, natural flow, English terminology preservation requirements
- **REFINE_PROMPT**: Added 6-sentence smoothing requirements with bullet points and formatting

The model likely hits the generation token limit or produces output that gets filtered somewhere, resulting in empty drafts.

### Why Latency Improved (73% reduction)

| Change | Latency Impact |
|--------|---------------|
| Semantic router: centroid dot-product vs LLM intent classification | -3-10s per query |
| GLiNER entity extraction (~100ms) vs LLM extraction | -3-10s per query |
| Fewer LLM reformulation calls | -variable |
| Intent routing more aggressive (factual → ReAct) | +variable |

### Why Keyword Hit Dropped

Most queries were refused (no answer = 0% keyword hit). Of the 16 non-refused queries, keyword hit was reasonable but mass refusals dragged the average down significantly.

---

## 3. Per-Category Breakdown

| Category | N | Non-Refused | Recall | KW Hit | p50 Lat | Assessment |
|----------|---|-------------|--------|--------|---------|------------|
| comparison | 5 | 3 | 75% | 58% | 93s | Good (not refused) |
| factual_local | 8 | 3 | 67% | 53% | 122s | Mixed (5 refused) |
| analytical | 5 | 0 | N/A | N/A | 56s | All refused |
| kg_construction | 3 | 2 | 75% | 40% | 85s | Better than before |
| reranking | 2 | 1 | 100% | 80% | 86s | Good |
| entity_pivot | 3 | 0 | N/A | N/A | 45s | All refused |
| multi_hop | 4 | 2 | 0% | 60% | 110s | Refused despite context |
| summarization | 3 | 1 | 0% | 60% | 81s | Mixed |
| vietnamese_complex | 4 | 0 | N/A | N/A | 49s | All refused |
| out_of_domain | 3 | 0 | N/A | N/A | 40s | Correctly refused |
| agentic_rag | 2 | 1 | 0% | 60% | 82s | Mixed |

---

## 4. What Worked (Quality Non-Refused Queries)

| q_id | Category | Recall | KW Hit | Notes |
|------|----------|--------|--------|-------|
| c02 | comparison | 100% | 40% | Self-RAG vs standard RAG — correct |
| c04 | comparison | 100% | 75% | BiXSE vs InfoNCE — best kw_hit ratio |
| c05 | comparison | 100% | 60% | KET-RAG vs GraphRAG — correct |
| f01 | factual_local | 100% | 40% | LightRAG dual-level — correct |
| f06 | factual_local | 100% | 20% | ColBERT late interaction — correct |
| f08 | factual_local | 100% | 100% | iText2KG — best overall kw_hit |
| kg02 | kg_construction | 50% | 60% | Partial recall |
| kg03 | kg_construction | 100% | 20% | AutoSchemaKG — correct |
| rerank01 | reranking | 100% | 80% | SetEncoder — best overall |
| m03 | multi_hop | 0% | 60% | Correct docs but low recall |
| m04 | multi_hop | 0% | 75% | Correct docs but low recall |

---

## 5. Issues Found

### Critical: Generation Prompts Breaking Draft Output

The Step 5 prompts added too many constraints causing the model to produce empty output. Must revert to simpler prompts.

### High: Semantic Router Misclassifying Factual Queries as Analytical/ReAct

`factual` queries like `f01`, `f03`, `f06`, `f08` are being routed to ReAct (`react=True`) when they should use standard retrieval. The semantic centroid matching threshold (0.30) is too permissive.

### High: INTENT_STRATEGY Missing Entries

Unknown benchmark categories (`vietnamese_complex`, `agentic_rag`) fall back to `factual` strategy which disables `entity_pivot`, `graph`, and `community` paths.

---

## 6. Recommendations

### Priority 1: Revert Generation Prompts to Original

Revert `DRAFT_PROMPT` and `REFINE_PROMPT` in `api/routes/_prompts.py` to the simpler original versions. The text smoothing requirements should be added incrementally, not all at once.

### Priority 2: Raise Semantic Router Threshold

Increase `_SEMANTIC_THRESHOLD` in `query_router.py` from 0.30 to 0.45.

### Priority 3: Add Fallback INTENT_STRATEGY

Add catch-all entry for unknown intents with entity_pivot and graph enabled.

---

## 7. What Was Successfully Implemented from caitien.md

| Step | Status | Notes |
|------|--------|-------|
| Step 1: Semantic Router (intent centroids) | Implemented | Working, but threshold too low |
| Step 2: GLiNER entity extraction | Implemented | Fast, no LLM calls |
| Step 3: Async parallel orchestration | Partial | Already existed; extended for multi_hop/kg_construction |
| Step 4: Fallback & Tuning | Not implemented | Missing threshold fallback logic |
| Step 5: Generation Smoothing | Implemented (broken) | Prompt too complex, broke generation |
| Docker fix (OLLAMA_MODEL) | Fixed | Was `qwen3.5:4b`, now `qwen3.5:9b` |

---

## Appendix: Configuration Used

- **Consistency Views:** True (5 views)
- **Community Enabled:** True
- **Validation:** False
- **OOD Detection:** False (disabled in benchmark)
- **Query Reformulations:** 3
- **Model:** qwen3.5:9b
- **Intent Matching:** Semantic (BGE-M3 centroid dot-product, threshold=0.30)
- **Retrieval V2:** Enabled

---

*Report generated from benchmark_v2_caitien.json*
