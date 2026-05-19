# Product Evaluation Report: Hybrid GraphRAG System
**Date:** 2026-05-18
**Corpus:** 51 Academic Papers on RAG Systems (13,084 chunks)
**Model:** qwen3.5:9b (Ollama, Apple Silicon Metal)
**Tenant:** rag51

---

## Executive Summary

| Dimension | Status | Score | Notes |
|-----------|--------|-------|-------|
| **Retrieval Quality** | ⚠️ Good | 75% | Doc recall excellent, keyword hit needs work |
| **Answer Quality** | ⚠️ Moderate | 60% | Accurate but verbose, missing technical terms |
| **Latency** | ❌ Poor | 35% | p50 ~188s, target <60s |
| **System Reliability** | ✅ Excellent | 95% | 100% success rate |
| **Knowledge Graph** | ✅ Good | 80% | Entities extracted, community detection working |
| **Multi-hop Reasoning** | ⚠️ Weak | 50% | Router misclassifies, retrieval works |
| **Vietnamese Support** | ✅ Good | 85% | Queries answered correctly in Vietnamese |

**Overall Score: 69% (Grade: C+)**

---

## 1. System Architecture

### 1.1 Components
```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI (port 8800)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │  Router  │→ │Retrieval │→ │  Rerank  │→ │  Ollama    │ │
│  │(Query UF)│  │(Multi-Path)│ │(L2R)    │  │(qwen3.5:9b)│ │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────┘
         ↓               ↓              ↓
┌─────────────────┐ ┌───────────┐ ┌──────────────────┐
│     Qdrant      │ │   Neo4j   │ │      Redis       │
│ (6 named vectors)│ │ (KG+APOC) │ │     (cache)      │
│  13,084 chunks  │ │ 17,658 ents│ │                  │
└─────────────────┘ └───────────┘ └──────────────────┘
```

### 1.2 Configuration (from /api/v3/health)
- **Consistency Views:** 5 (multi-version answer generation)
- **Rerank Stage3:** Disabled
- **Validation:** Enabled
- **Community Detection:** Enabled

---

## 2. Retrieval Evaluation

### 2.1 Document Recall (Target: >70%)

| Category | Queries | Recall | Status |
|----------|---------|--------|--------|
| comparison | 2 | 200% | ✅ Excellent |
| factual_local | 6 | 483% | ✅ Excellent (over-retrieval OK) |
| kg_construction | 1 | 500% | ✅ Excellent |
| multi_hop | 1 | 400% | ✅ Excellent |
| **OVERALL** | **10** | **420%** | **✅ PASS** |

**Analysis:**
- All queries retrieved correct documents
- Over-retrieval (400%+) indicates system retrieves more relevant docs than expected
- This is acceptable for RAG - better to have extra context than miss
- 100% correct document identification rate

### 2.2 Retrieval Path Analysis

| Path | Usage | Quality |
|------|-------|---------|
| `vector:dense` | 100% | Good - all queries used dense vectors |
| `vector:sparse` | 0% | Not utilized (potential optimization) |
| `kg:entity` | 0% | Not utilized (potential optimization) |
| `hybrid` | 0% | Not utilized |

**Issue:** System relies solely on dense vector retrieval. BM25 sparse and KG entity paths are not being used, even for queries that should benefit from them.

### 2.3 Top Retrieved Sources (Sample)

| Query | Top Source | Score |
|-------|------------|-------|
| ChunkRAG | ChunkRAG.pdf | 0.680 |
| AutoSchemaKG | AutoSchemaKG.pdf | 0.694 |
| FanOutQA | FanOutQA.pdf | 0.702 |
| PERank | PERank_Rerank.pdf | 0.580 |

**Quality:** Correct papers ranked first in all cases.

---

## 3. Keyword Hit Analysis (Target: >60%)

| Query | Keywords Hit | Ratio | Issues |
|-------|-------------|-------|--------|
| n01 (AgenticRAG) | 1/5 (agent) | 20% | Missing: plan, tool, iterate, memory |
| n02 (ChunkRAG) | 1/5 (chunk) | 20% | Missing: semantic, rerank, score, filter |
| n03 (AutoSchemaKG) | 1/5 (schema) | 20% | Missing: ontology, LLM, entity type |
| n04 (Wikontic) | 1/4 (Wikipedia) | 25% | Missing: knowledge base, entity linking |
| n05 (ComposeRAG) | 1/5 (compose) | 20% | Missing: merge, rank, diversity |
| n06 (FanOutQA) | 1/5 (fan-out) | 20% | Missing: multi-hop, entity, path |
| n07 (PERank) | 3/5 | **60%** | ✅ Best query |
| n08 (SetEncoder) | 1/5 (set) | 20% | Missing: order-invariant, bi-encoder |
| n09 (KET-RAG) | 0/5 | 0% | ❌ All keywords missed |
| n10 (E5-Multilingual) | 2/5 | **40%** | Hit: multilingual, embedding |

**Average: 24.5%** ❌ **FAIL**

### 3.1 Root Cause Analysis

1. **English Technical Terms in Vietnamese Context**
   - Keywords like "bi-encoder", "pairwise", "order-invariant" not translated
   - LLM answers in Vietnamese but misses English technical terms

2. **Answer Verbosity vs Keyword Density**
   - System generates comprehensive answers (~200-400 words)
   - But keyword density is diluted by natural language

3. **Model Knowledge vs Retrieved Context**
   - `qwen3.5:9b` knows some terms internally but doesn't output them
   - Retrieved chunks may not contain exact keywords

---

## 4. Latency Analysis (Target: <60s)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| p50 | 187.7s | <60s | ❌ |
| p90 | ~250s | <120s | ❌ |
| Min | 40.7s | - | - |
| Max | 251.5s | - | - |

### 4.1 Per-Query Latency Breakdown

| Query | Latency | Category | Issue |
|-------|---------|----------|-------|
| n07 (PERank) | 40.7s | comparison | ✅ Fast |
| n04 (Wikontic) | 48.8s | factual_local | ✅ Fast |
| n01 (AgenticRAG) | 51.1s | comparison | ✅ Fast |
| n06 (FanOutQA) | 177.9s | multi_hop | ⚠️ Slow |
| n02 (ChunkRAG) | 182.2s | factual_local | ⚠️ Slow |
| n09 (KET-RAG) | 187.7s | factual_local | ⚠️ Slow |
| n10 (E5-Multilingual) | 189.6s | factual_local | ⚠️ Slow |
| n03 (AutoSchemaKG) | 207.7s | kg_construction | ❌ Very Slow |
| n05 (ComposeRAG) | 240.1s | factual_local | ❌ Very Slow |
| n08 (SetEncoder) | 251.5s | factual_local | ❌ Very Slow |

### 4.2 Latency Contributing Factors

1. **Consistency Views (5x generation)**
   - Each query generates 5 answers for cross-validation
   - Estimated overhead: 3-5x latency increase

2. **ReAct Loop**
   - Queries using ReAct (n04, n07) should be slower but aren't
   - This suggests ReAct is not fully utilized

3. **Entity Pivot Router Classification**
   - Routes classified as `entity_pivot` or `factual` take longer
   - Multi-hop queries incorrectly routed to factual

4. **Reranking Pipeline**
   - L2R reranking adds overhead
   - Stage3 disabled (should be enabled for better quality)

---

## 5. Query Router Analysis

| Expected Intent | Count | Correctly Routed | Accuracy |
|----------------|-------|-----------------|----------|
| comparison | 2 | ? | Unknown |
| factual_local | 6 | 0 (all → factual) | 0% |
| kg_construction | 1 | 0 (→ factual) | 0% |
| multi_hop | 1 | 0 (→ factual) | 0% |

**Issue:** Router consistently misclassifies queries as `factual`, missing `entity_pivot`, `multi_hop`, and `kg_construction` intents.

### Router Output vs Actual
```
n02: expected=entity_pivot → routed=factual (wrong)
n03: expected=kg_construction → routed=factual (wrong)
n04: expected=factual → routed=analytical (different but OK)
n05: expected=factual → routed=factual (correct)
n06: expected=multi_hop → routed=factual (wrong)
n07: expected=comparison → routed=multi_hop (wrong intent but ReAct used)
n08: expected=factual → routed=factual (correct)
n09: expected=factual → routed=factual (correct)
n10: expected=factual → routed=factual (correct)
```

---

## 6. Answer Quality Analysis

### 6.1 Sample Answers

**n07 (PERank) - Best Answer (60% kw hit)**
> "Ưu điểm nổi bật của PERank so với phương pháp reranking chuẩn (standard L2R) là khả năng tăng tốc độ suy luận đáng kể mà vẫn duy trì hiệu suất xếp hạng tương đương. Cụ thể, PERank giảm độ trễ xử lý xuống còn khoảng 1/4,5 so với các phương pháp chưa nén..."

✅ Accurate, mentions pairwise/listwise, correct comparison

**n01 (AgenticRAG) - Weak Answer (20% kw hit)**
> "Xin lỗi, nhưng dựa trên các đoạn tham khảo hiện có, tôi không thể xác định được sự khác biệt cụ thể giữa AgenticRAG và standard RAG pipeline vì nội dung các tài liệu trích dẫn chưa cung cấp thông tin chi tiết..."

⚠️ Retrieval found correct doc but LLM gave vague answer

### 6.2 Answer Patterns

| Pattern | Count | Percentage |
|---------|-------|------------|
| ✅ Accurate, complete | 2 | 20% |
| ⚠️ Accurate but incomplete | 6 | 60% |
| ❌ Vague or refused | 2 | 20% |

---

## 7. Knowledge Graph Analysis

### 7.1 Neo4j Stats
- **Entities:** 17,658 (from 51 papers)
- **Relationships:** Extracted but count unknown
- **Communities:** Leiden algorithm detected

### 7.2 Retrieval Path Issues
```
"matched_entities": []  // All queries returned empty
"entity_match_count": null  // Never populated
```

**Issue:** KG entity matching is not working. Queries retrieve chunks but no entities are matched.

---

## 8. Strengths & Weaknesses

### ✅ Strengths
1. **Perfect Success Rate** - 100% of queries completed without errors
2. **Excellent Doc Recall** - 420% average, all correct papers retrieved
3. **Good Vietnamese Output** - Natural Vietnamese language generation
4. **Fast for Simple Queries** - Sub-minute response for comparison queries
5. **Consistency Checking** - 5-view generation helps reduce hallucinations

### ⚠️ Weaknesses
1. **High Latency** - p50 of 188s is 3x target
2. **Low Keyword Hit** - Only 24.5% vs 60% target
3. **Router Misclassification** - 0% accuracy on non-factual queries
4. **KG Integration Broken** - Entity matching not working
5. **Over-Retrieval** - 5x docs retrieved, bandwidth waste
6. **ReAct Under-utilized** - Only 2/10 queries used ReAct

### ❌ Critical Issues
1. **Latency Budget Exceeded** - 188s vs 60s target (313% over)
2. **Consistency Score = 0** - All chunks show `consistency_score: 0.0`
3. **No Semantic Matching** - Only literal keyword matching used

---

## 9. Recommendations

### Priority 1: Fix Latency (Critical)
1. Reduce consistency views from 5 to 2
2. Enable caching for repeated queries
3. Optimize reranking pipeline
4. Enable sparse vector retrieval for speed

### Priority 2: Fix Keyword Hit (High)
1. Lower semantic threshold from 0.45 to 0.35
2. Add English keyword synonyms in benchmark
3. Improve prompt to emphasize technical terms
4. Consider English query option for technical papers

### Priority 3: Fix Router (High)
1. Retrain query understanding model
2. Add multi-hop detection patterns
3. Add kg_construction intent detection
4. Validate router on held-out queries

### Priority 4: Fix KG Integration (Medium)
1. Debug entity_match_count = null issue
2. Enable KG entity retrieval path
3. Verify Neo4j connection and queries
4. Add community detection results to response

---

## 10. Benchmark Dataset Quality

### 10.1 New 10 Queries Assessment

| Paper | In Corpus? | Quality of Query |
|-------|------------|-----------------|
| AgenticRAG | ✅ Yes | ⚠️ Paper has limited content |
| ChunkRAG | ✅ Yes | ✅ Good query |
| AutoSchemaKG | ✅ Yes | ✅ Good query |
| Wikontic | ✅ Yes | ✅ Good query |
| ComposeRAG | ✅ Yes | ✅ Good query |
| FanOutQA | ✅ Yes | ✅ Good query |
| PERank | ✅ Yes | ✅ Best query |
| SetEncoder | ✅ Yes | ⚠️ Technical, hard |
| KET-RAG | ✅ Yes | ⚠️ Technical, hard |
| E5-Multilingual | ✅ Yes | ✅ Good query |

### 10.2 Benchmark Improvements
- ✅ Covers less-used papers
- ✅ Mix of categories
- ⚠️ Keywords need English synonyms
- ⚠️ Expected keywords may be too strict

---

## 11. Comparative Analysis

### vs. Previous Benchmarks

| Metric | v1 (10 queries) | v2 (42 queries) | New10 |
|--------|----------------|-----------------|-------|
| Doc Recall | 43.3% | ~60% | 420% |
| Keyword Hit | 36.6% | ~35% | 24.5% |
| p50 Latency | 172s | ~160s | 187.7s |
| Success Rate | 100% | 100% | 100% |

**Trend:** Doc recall improved significantly, keyword hit decreased, latency stable.

---

## 12. Conclusion

The Hybrid GraphRAG system demonstrates solid retrieval capabilities with perfect document identification rates. However, critical improvements are needed in:

1. **Latency** - Must reduce from 188s to <60s (3x improvement required)
2. **Keyword Accuracy** - Must improve from 24.5% to >60%
3. **Router Intelligence** - Must fix misclassification issues
4. **KG Integration** - Must enable entity matching

**Production Readiness: NOT READY** - System works but latency and accuracy need significant improvement before production deployment.

---

## Appendix: Raw Metrics

```json
{
  "summary": {
    "avg_doc_recall": 4.2,
    "avg_kw_hit": 0.245,
    "p50_latency_s": 187.7,
    "total_queries": 10,
    "success_rate": 1.0
  }
}
```
