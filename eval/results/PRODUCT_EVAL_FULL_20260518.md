# Product Evaluation Report — Hybrid GraphRAG System

**Date:** 2026-05-18
**Corpus:** 51 Academic Papers on RAG Systems
**Benchmark:** vi_benchmark_v2 (42 queries)
**Tenant:** rag51
**Model:** qwen3.5:9b (Ollama, Apple Silicon Metal)
**Total Runtime:** 80.5 minutes

---

## Executive Summary

| Dimension | Status | Score | Notes |
|-----------|--------|-------|-------|
| **System Reliability** | Excellent | 100% | 42/42 queries completed, 0 errors |
| **Document Recall** | Good | 58% | Mixed — strong on factual/reranking, weak on multi-hop/summarization |
| **Keyword Hit** | Moderate | 44% | ReAct queries hit more keywords; factual queries miss terms |
| **Refusal Accuracy** | Good | 86% | OOD detection works (3/3 refused) but pattern leakage in answers |
| **Latency** | Slow | p50=209s | Target <60s; entity_pivot path is the bottleneck |
| **Query Routing** | Needs Work | — | Heavy misclassification; ReAct overuse, entity_pivot underuse |
| **Multi-hop Reasoning** | Weak | 25% recall | Correct keywords but wrong documents retrieved |

**Overall Grade: C+ (65%)** — System is functional but latency and routing need urgent optimization.

---

## 1. Aggregate Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Success Rate | 42/42 (100%) | >95% | Pass |
| Avg Doc Recall | 57.9% | >70% | Fail |
| Avg Keyword Hit | 43.9% | >60% | Fail |
| Refusal Accuracy | 85.7% | >90% | Near-pass |
| p50 Latency | 209s | <60s | Fail (3.5x over) |
| p95 Latency | 253s | <120s | Fail (2.1x over) |
| Total Time | 80.5 min | — | — |

### Trend vs Previous Benchmarks

| Metric | new10 (10q) | v2 (42q, today) | Change |
|--------|-------------|-----------------|--------|
| Doc Recall | ~60% | 57.9% | -2% |
| Keyword Hit | 24.5% | 43.9% | +19% (improved) |
| p50 Latency | 187s | 209s | +22s (slower) |
| Success Rate | 100% | 100% | stable |

---

## 2. Per-Category Breakdown

| Category | N | Recall | KW Hit | Ref Acc | p50 Lat | Assessment |
|----------|---|--------|--------|--------|---------|------------|
| comparison | 5 | 60% | 57% | 100% | 33.8s | Good |
| factual_local | 8 | 62.5% | 25% | 100% | 164.6s | Slow |
| kg_construction | 3 | 61.1% | 33.3% | 66.7% | 172.5s | Routing issue |
| reranking | 2 | 100% | 50% | 100% | 234s | Good quality |
| analytical | 5 | 53.3% | 56% | 100% | 155.7s | Good kw_hit |
| entity_pivot | 3 | 66.7% | 21.7% | 66.7% | 38.2s | Wrong routing |
| multi_hop | 4 | 25% | 58.8% | 75% | 32.8s | Misclassified |
| summarization | 3 | 27.8% | 20% | 100% | 158.9s | Weak recall |
| vietnamese_complex | 4 | 50% | 35% | 100% | 208.8s | Slow + weak |
| out_of_domain | 3 | 100% | 100% | 0% | 15.6s | Refusal broken |
| agentic_rag | 2 | 50% | 40% | 100% | 381s | Slowest category |

### Key Observations by Category

**Strengths:**
- `comparison` (5q): Best balance — 60% recall, 57% kw_hit, fast p50=34s. ReAct handles cross-entity comparison well.
- `factual_local` (8q): 62.5% recall, 100% refusal accuracy. Solid for single-entity lookups.
- `reranking` (2q): 100% recall, 50% kw_hit. Strong performance on technical reranking queries.

**Weaknesses:**
- `multi_hop` (4q): Only 25% recall despite 59% kw_hit. Keywords are in the answer but wrong/too few docs retrieved.
- `summarization` (3q): 28% recall, 20% kw_hit. Cross-document aggregation is not working.
- `out_of_domain` (3q): Refusal accuracy = 0% — ALL 3 were refused BUT the metric shows 0% because `expect_refusal` was not set to true in the benchmark dataset (the benchmark incorrectly expected them to be answered). The system DID refuse correctly (3/3), so OOD detection works.
- `entity_pivot` (3q): 67% recall but 67% refusal accuracy — m02 (Leiden vs Louvain) was incorrectly refused.
- `vietnamese_complex` (4q): 50% recall, 35% kw_hit, p50=209s. High latency + low quality.

---

## 3. Query Type Breakdown

| Query Type | N | Recall | KW Hit | ReAct Used | Assessment |
|------------|---|--------|--------|-----------|------------|
| analytical | 4 | 75% | 49% | 4/4 (100%) | Best kw_hit with ReAct |
| factual | 13 | 68% | 42% | 0/13 (0%) | Solid baseline |
| multi_hop | 13 | 46% | 48% | 13/13 (100%) | Correct routing, weak recall |
| entity_pivot | 10 | 55% | 34% | 0/10 (0%) | Should use ReAct but doesn't |
| summarization | 1 | 0% | 40% | 1/1 (100%) | Too few samples |

### Critical Routing Issue

The `entity_pivot` category is severely misrouted:
- 10 queries classified as `entity_pivot` by the benchmark
- **0 of them used ReAct** (should use ReAct for entity-crossing lookups)
- 8/10 were routed to `factual` or `entity_pivot` (standard retrieval)
- This explains why entity_pivot has low recall and low kw_hit

Meanwhile, `multi_hop` (13 queries) correctly uses ReAct 13/13 times, but still has only 46% recall. This suggests ReAct is being used but the graph traversal is not finding the right entities.

---

## 4. Per-Query Deep Dive

### Best Performers

| q_id | Category | Recall | KW Hit | Latency | Notes |
|------|----------|--------|--------|---------|-------|
| f01 | factual_local | 100% | 20% | 165s | LightRAG dual-level retrieval — correct doc, keyword mismatch |
| f07 | factual_local | 100% | 50% | 219s | E5 embedding — 50% kw_hit is decent |
| f08 | factual_local | 100% | 50% | 183s | iText2KG 4 modules — correct retrieval |
| c04 | comparison | 100% | 75% | 46s | BiXSE vs InfoNCE — best kw_hit ratio (75%) |
| m02 | multi_hop | 100% | 40% | 26s | Leiden vs Louvain — correctly answered but was refused |
| v01 | vietnamese_complex | 100% | 80% | 209s | Vietnamese query on embeddings — best overall kw_hit |
| kg03 | kg_construction | 100% | 20% | 34s | AutoSchemaKG — correct doc, low kw_hit |
| rerank01 | reranking | 100% | 80% | 64s | SetEncoder — 80% kw_hit, fast for ReAct |
| rerank02 | reranking | 100% | 20% | 234s | PERank — correct doc, slow |

### Worst Performers

| q_id | Category | Recall | KW Hit | Latency | Issue |
|------|----------|--------|--------|---------|-------|
| f02 | factual_local | 0% | 40% | 30s | RAPTOR tree — doc not retrieved, routed to summarization |
| f03 | factual_local | 0% | 20% | 35s | HippoRAG — doc not retrieved, routed to multi_hop |
| f06 | factual_local | 0% | 20% | 22s | ColBERT late interaction — doc not retrieved, routed to analytical |
| m01 | multi_hop | 0% | 60% | 37s | LightRAG vs HippoRAG — doc retrieval failed |
| m03 | multi_hop | 0% | 60% | 33s | RAPTOR vs GraphRAG — doc retrieval failed |
| m04 | multi_hop | 0% | 75% | 25s | HyDE — doc retrieval failed despite high kw_hit |
| c01 | comparison | 0% | 80% | 29s | ColBERT vs BGE-M3 — doc not retrieved |
| c03 | comparison | 0% | 50% | 34s | GraphRAG vs RAG — doc not retrieved |
| s03 | summarization | 0% | 20% | 47s | FanOutQA vs MINTQA — doc not retrieved |
| e01 | entity_pivot | 0% | 0% | 4s | Cross-doc entity — refused, 4s fast |
| v02 | vietnamese_complex | 0% | 0% | 61s | RAG evaluation — doc not retrieved |
| v04 | vietnamese_complex | 0% | 0% | 226s | Best RAG for 50+ papers — doc not retrieved |
| agentic01 | agentic_rag | 0% | 60% | 381s | MAO-ARAG — slowest query, doc not retrieved |

### Pattern: Document Not Retrieved

15 queries have doc_recall = 0%. These are the highest-priority issues. The system retrieves chunks but the `source` metadata does not match expected doc names. Root causes:

1. **Benchmark vs actual doc names**: The benchmark expects doc names like "LightRAG", "RAPTOR", "HippoRAG" but the actual PDFs might be named differently in the corpus (e.g., "LightRAG.pdf" vs just "LightRAG" in metadata).
2. **Entity-centric queries fail**: Queries about specific algorithms (RAPTOR, HippoRAG, FanOutQA) retrieve no matching sources.
3. **Cross-doc comparison fails**: Queries comparing 2+ papers (LightRAG vs HippoRAG, RAPTOR vs GraphRAG) retrieve neither.

### Pattern: High Latency + Low Quality

Queries taking >200s but producing poor results:
- `agentic01`: 381s, 0% recall
- `rerank02`: 234s, 100% recall, 20% kw_hit
- `v04`: 226s, 0% recall
- `kg02`: 252s, 50% recall, 60% kw_hit
- `f07`: 219s, 100% recall, 50% kw_hit
- `v03`: 193s, 100% recall, 60% kw_hit

The pattern is clear: slow queries are dominated by the entity_pivot and factual paths, which trigger multiple LLM calls for query understanding.

---

## 5. Latency Analysis

### By Query Type

| Query Type | p50 Latency | Notes |
|------------|-------------|-------|
| out_of_domain | 15.6s | Fast short-circuit |
| multi_hop | 32.8s | ReAct fast but recall is poor |
| comparison | 33.8s | Best latency-to-quality ratio |
| entity_pivot | 38.2s | Good latency, wrong routing |
| reranking | 64s | ReAct + reranking |
| analytical | 155.7s | Mixed |
| kg_construction | 172.5s | Entity pivot routing |
| factual_local | 164.6s | Slow for factual |
| summarization | 158.9s | Cross-doc aggregation |
| vietnamese_complex | 208.8s | Slowest standard path |
| agentic_rag | 381s | Slowest overall |

### Latency Breakdown by Query Path

**Fast path (ReAct, <60s):** multi_hop, comparison, out_of_domain
- Average: ~25-40s
- ReAct loop is fast when it works

**Slow path (entity_pivot/factual, >150s):**
- Average: 155-380s
- The entity_pivot path triggers multi-LLM-call query understanding
- Slowest: agentic_rag at 381s — combination of factual routing + long generation

### Root Cause of Latency

The dominant factor is the **query understanding overhead**:
1. `query_understanding.py` runs 3-6 reformulation LLM calls
2. Each call to `ollama_chat` takes 3-15s on qwen3.5:9b
3. Entity-pivot queries run the full reformulation pipeline (6 calls)
4. Results are then used for retrieval + generation

**The query-type-aware reformulation** (`query_reformulations_minimal`) helpsfactual queries (reduces to 2 reformulations) but not enough.

---

## 6. Refusal Analysis

### Out-of-Domain Detection

| Query | Expected | Actual | Status |
|-------|----------|--------|--------|
| o01 (pho recipe) | Answer | REFUSED | Correct (OOD detected) |
| o02 (Bitcoin) | Answer | REFUSED | Correct (OOD detected) |
| o03 (US President) | Answer | REFUSED | Correct (OOD detected) |

**OOD detection is working correctly.** All 3 out-of-domain queries were refused. The `ref_acc=0%` for out_of_domain category is a benchmark bug — `expect_refusal` was not set to `true` in the dataset.

### False Refusals

| Query | Category | Refused | Should Refuse | Impact |
|-------|----------|---------|---------------|--------|
| m02 | multi_hop | YES | NO | Leiden vs Louvain — correct answer but refused |
| e01 | entity_pivot | YES | NO | Cross-doc entity — correct refusal? |
| kg02 | kg_construction | YES | NO | Wikontic vs iText2KG — correct answer but refused |

The `kg02` (Wikontic vs iText2KG) refusal is notable — the system gave a correct refusal message for a comparison that it should have answered.

---

## 7. Query Router Analysis

### Classification Accuracy vs Benchmark Expectations

| Benchmark Category | Count | Correctly Routed | Accuracy |
|--------------------|-------|-----------------|----------|
| factual_local | 8 | ~5 | 63% |
| comparison | 5 | ~2 | 40% |
| multi_hop | 4 | ~3 | 75% |
| summarization | 3 | ~0 | 0% |
| analytical | 5 | ~2 | 40% |
| entity_pivot | 3 | ~1 | 33% |
| kg_construction | 3 | ~0 | 0% |
| reranking | 2 | ~1 | 50% |
| agentic_rag | 2 | ~0 | 0% |
| out_of_domain | 3 | ~3 | 100% |
| vietnamese_complex | 4 | ~1 | 25% |

**Overall routing accuracy: ~40%** (17/42 correctly routed by benchmark's expected category)

### Misclassification Examples

**Critical misclassifications:**
- `f02` (RAPTOR): expected=factual → routed=summarization [R] — triggers ReAct unnecessarily
- `f03` (HippoRAG): expected=factual → routed=multi_hop [R] — triggers ReAct unnecessarily
- `f06` (ColBERT): expected=factual → routed=analytical [R] — triggers ReAct unnecessarily
- `s01` (RAG survey): expected=summarization → routed=entity_pivot [S] — should use ReAct
- `s02` (embeddings): expected=summarization → routed=entity_pivot [S] — should use ReAct
- `kg01` (KG pipeline): expected=kg_construction → routed=entity_pivot [S] — should use ReAct
- `agentic01` (MAO-ARAG): expected=agentic_rag → routed=factual [S] — completely missed

### Router Pattern Analysis

The rule-based router in `query_router.py` is effective for clear patterns but misses:
1. **"như thế nào" suffix** — incorrectly routes to analytical instead of factual
2. **Paper names with "vs"** — comparison not detected, routed to multi_hop
3. **KG construction phrases** — "bao gồm những bước nào" not detected
4. **Agentic RAG** — no patterns for multi-agent orchestration queries

---

## 8. ReAct Usage Analysis

| Metric | Value |
|--------|-------|
| ReAct queries | 18/42 (43%) |
| ReAct queries with 100% recall | 10/18 (56%) |
| ReAct queries with 0% recall | 8/18 (44%) |
| Non-ReAct queries with 100% recall | 14/24 (58%) |

**ReAct helps for analytical queries** (75% recall) but provides no improvement for multi_hop (25% recall). This suggests the graph traversal step in ReAct is not finding cross-document entities effectively.

### ReAct + kw_hit Correlation

ReAct queries average 49% kw_hit vs non-ReAct at 40%. ReAct generates more comprehensive answers that include technical terms.

---

## 9. Document Retrieval Analysis

### Source Matching Issue

The benchmark tracks `doc_found` by checking if the source metadata contains expected paper names. Many queries show `doc_found=0` despite high kw_hit, meaning:
- The answer contains the right keywords
- But the source metadata doesn't match the expected doc names

This could mean:
1. PDFs in the corpus are named differently from benchmark expectations
2. Source metadata extraction is inconsistent
3. The retrieval is finding relevant chunks but from unnamed sources

### Retrieval Path Performance

From the benchmark data, ReAct queries consistently take 25-50s (fast graph traversal) while non-ReAct factual queries take 150-300s (slow retrieval + generation).

---

## 10. System Architecture Assessment

### What's Working

1. **OOD detection** — 3/3 out-of-domain queries refused correctly
2. **Comparison queries** — Best quality-to-latency ratio (p50=34s, 60% recall)
3. **ReAct integration** — Correctly triggers for multi-hop/analytical, fast execution
4. **Consistency checking** — Enabled, 5 views per query
5. **Community detection** — Enabled in Neo4j
6. **No errors** — 100% success rate across 42 queries

### What Needs Fixing

1. **Document retrieval** — 15/42 queries (36%) fail to retrieve expected docs
2. **Query router** — ~40% classification accuracy; entity_pivot and kg_construction severely underrouted
3. **Latency** — p50=209s is 3.5x the target; entity_pivot path is the bottleneck
4. **Multi-hop recall** — 25% recall despite 100% ReAct usage; graph traversal not working
5. **Summarization** — 28% recall; cross-doc aggregation missing
6. **Keyword hit** — 44% average; LLM doesn't output enough technical terms
7. **Refusal accuracy** — 86%; some correct answers incorrectly refused

---

## 11. Recommendations (Prioritized)

### Priority 1: Fix Document Retrieval (Critical)

The 15 queries with 0% doc recall are the #1 priority. Investigate:
1. PDF naming mismatch between corpus and benchmark expectations
2. Source metadata extraction in chunking pipeline
3. Qdrant payload schema — verify `source` field is populated correctly

### Priority 2: Fix Query Router (High)

Current routing accuracy is ~40%. Improve `query_router.py`:
1. Add patterns for "như thế nào" → factual (currently routes to analytical)
2. Add patterns for paper comparisons ("X vs Y", "X khác Y")
3. Add patterns for kg_construction ("bao gồm những bước", "pipeline")
4. Add patterns for agentic_rag ("multi-agent", "orchestration")
5. Route entity_pivot queries to ReAct (cross-doc entity lookups need graph traversal)

### Priority 3: Reduce Latency (High)

p50=209s is unacceptable for production. Optimize:
1. **Reduce reformulation overhead**: The query understanding pipeline runs 3-6 LLM calls. Cache reformulations for repeated queries.
2. **Reduce consistency views**: 5 views adds 3-5x latency. Consider reducing to 2.
3. **Fast-path for factual**: Already done with `query_reformulations_minimal=2`, but still takes 150-220s.
4. **Enable Redis caching**: Repeated queries should hit cache.
5. **Parallelize reformulations**: Already async, but timeout=10s may cause cascading delays.

### Priority 4: Fix Multi-hop Recall (Medium)

ReAct is correctly triggered (13/13 multi_hop queries) but only 25% recall. The graph traversal step needs:
1. Verify Neo4j entity matching is working (check `matched_entities` in responses)
2. Lower entity_match_threshold if too strict
3. Add cross-document entity bridging in ReAct loop

### Priority 5: Improve Keyword Hit (Medium)

44% kw_hit is moderate. The gap between ReAct queries (49%) and non-ReAct (40%) suggests:
1. ReAct generates more comprehensive answers
2. Consider prompting LLM to output technical terms more explicitly
3. Lower semantic similarity threshold in benchmark (currently 0.45)

---

## 12. Production Readiness Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| Success rate >95% | Pass | 100% (42/42) |
| Doc recall >70% | Fail | 58% overall, 64% for factual/comparison |
| Latency <60s p50 | Fail | 209s (3.5x over) |
| OOD detection | Pass | 3/3 correctly refused |
| KG integration | Partial | Entities extracted, graph traversal weak |
| Multi-hop reasoning | Fail | 25% recall despite ReAct |
| Router accuracy | Fail | ~40% |
| Keyword accuracy | Partial | 44% (ReAct better at 49%) |

**Production Readiness: NOT READY**

The system is functional but needs significant optimization before production:
1. Latency must be reduced by 70% (209s → <60s)
2. Multi-hop recall must improve by 200% (25% → >75%)
3. Router accuracy must improve by 100% (40% → >80%)

---

## Appendix: Raw Metrics

```
avg_doc_recall:     57.9%
avg_kw_hit:         43.9%
avg_refusal_acc:    85.7%
p50_latency:        209s
p95_latency:        253s
total_time:         80.5 min
queries_total:      42
queries_ok:         42
queries_err:        0
react_queries:      18/42 (43%)
```

*Report generated from benchmark_rag51_20260518_165651.json*
*Benchmark: eval/datasets/vi_benchmark_v2.json*
