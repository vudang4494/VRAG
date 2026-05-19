# Benchmark Report — 51 Paper Corpus

**Date**: 2026-05-20T00:06:21Z  
**Model**: qwen3.5:9b  
**Benchmark**: eval/datasets/vi_benchmark_50.json  
**Tenant**: rag51

---

## Summary

| Metric | Value |
|--------|-------|
| Total Queries | 53 |
| Successful | 53 |
| Errors | 0 |
| Avg Doc Recall | 62.6% |
| Avg Keyword Hit | 47.3% |
| Refusal Accuracy | 88.7% |
| P50 Latency | 38.6s |
| P95 Latency | 112.9s |
| Total Time | 54.1 min |

---

## Per-Category Results

| Category | N | Doc Recall | KW Hit | Ref Acc |
|----------|---|------------|--------|---------|
| agentic_rag | 4 | 75.0% | 50.0% | 100.0% |
| analytical | 6 | 61.1% | 56.7% | 100.0% |
| comparison | 6 | 66.7% | 47.5% | 100.0% |
| entity_pivot | 4 | 50.0% | 31.2% | 100.0% |
| factual_local | 10 | 90.0% | 34.0% | 100.0% |
| kg_construction | 3 | 72.2% | 33.3% | 100.0% |
| multi_hop | 5 | 40.0% | 43.0% | 60.0% |
| out_of_domain | 4 | 0.0% | 100.0% | 0.0% |
| reranking | 2 | 100.0% | 70.0% | 100.0% |
| summarization | 4 | 70.8% | 45.0% | 100.0% |
| vietnamese_complex | 5 | 50.0% | 36.0% | 100.0% |

---

## Query Results Detail

| ID | Category | Query | Recall | KW | Refused | Latency | Type |
|----|----------|-------|--------|----|---------|---------|------|
| f01 | factual_local | LightRAG sử dụng cơ chế dual-level retri... | 100% | 40% | no ✓ | 52s | comparison |
| f02 | factual_local | RAPTOR xây dựng cây tóm tắt từ dưới lên ... | 100% | 40% | no ✓ | 75s | factual |
| f03 | factual_local | HippoRAG kết hợp kiến thức từ nhiều docu... | 100% | 60% | no ✓ | 82s | factual |
| f04 | factual_local | Self-RAG sử dụng reflection tokens để là... | 100% | 20% | no ✓ | 112s | factual |
| f05 | factual_local | BGE-M3 hỗ trợ bao nhiêu ngôn ngữ và sử d... | 100% | 20% | no ✓ | 82s | factual |
| f06 | factual_local | Late interaction trong ColBERT hoạt động... | 0% | 20% | no ✓ | 74s | factual |
| f07 | factual_local | E5 embedding model được huấn luyện bằng ... | 100% | 50% | no ✓ | 100s | factual |
| f08 | factual_local | iText2KG xây dựng knowledge graph từ pap... | 100% | 50% | no ✓ | 54s | kg_construction |
| f09 | factual_local | ACORN framework cải thiện retrieval bằng... | 100% | 40% | no ✓ | 59s | analytical |
| f10 | factual_local | LongContextRAG xử lý long documents bằng... | 100% | 0% | no ✓ | 113s | factual |
| m01 | multi_hop | So sánh LightRAG và HippoRAG: cái nào tố... | 0% | 0% | YES ✗ | 24s | comparison |
| m02 | multi_hop | Thuật toán Leiden phân cụm cộng đồng khá... | 100% | 40% | YES ✗ | 32s | multi_hop |
| m03 | multi_hop | RAPTOR và GraphRAG khác nhau như thế nào... | 0% | 60% | no ✓ | 54s | comparison |
| m04 | multi_hop | HyDE tạo hypothetical document như thế n... | 0% | 75% | no ✓ | 48s | multi_hop |
| m05 | multi_hop | DocHopQA sử dụng document hopping để trả... | 100% | 40% | no ✓ | 48s | multi_hop |
| c01 | comparison | So sánh ColBERT late interaction với BGE... | 0% | 60% | no ✓ | 41s | comparison |
| c02 | comparison | Self-RAG vs standard RAG: ưu điểm của re... | 100% | 40% | no ✓ | 50s | comparison |
| c03 | comparison | GraphRAG và standard vector RAG khác nha... | 0% | 50% | no ✓ | 48s | comparison |
| c04 | comparison | BiXSE cải thiện dense retrieval bằng các... | 100% | 75% | no ✓ | 53s | comparison |
| c05 | comparison | So sánh KET-RAG với GraphRAG về indexing... | 100% | 40% | no ✓ | 53s | comparison |
| c06 | comparison | EffiR cải thiện efficiency của RAG retri... | 100% | 20% | no ✓ | 44s | comparison |
| s01 | summarization | Tóm tắt các phương pháp cải thiện hệ thố... | 33% | 20% | no ✓ | 97s | factual |
| s02 | summarization | Các phương pháp embedding hiện đại (2024... | 50% | 60% | no ✓ | 116s | factual |
| s03 | summarization | Các benchmark đánh giá multi-hop QA (Fan... | 100% | 40% | no ✓ | 50s | multi_hop |
| s04 | summarization | GeAR cải thiện multi-hop reasoning bằng ... | 100% | 60% | no ✓ | 49s | multi_hop |
| a01 | analytical | Đánh giá: chiến lược chunking nào hiệu q... | 33% | 40% | no ✓ | 44s | analytical |
| a02 | analytical | RAG systems nên dùng cross-encoder hay b... | 33% | 60% | no ✓ | 42s | analytical |
| a03 | analytical | Tại sao retrieval granularity (propositi... | 100% | 40% | no ✓ | 56s | analytical |
| a04 | analytical | Late chunking (chunk sau khi embed) khác... | 100% | 60% | no ✓ | 39s | analytical |
| a05 | analytical | Các phương pháp nào dùng LLM để cải thiệ... | 0% | 60% | no ✓ | 50s | analytical |
| a06 | analytical | MINTQA benchmark đánh giá RAG systems tr... | 100% | 80% | no ✓ | 56s | comparison |
| e01 | entity_pivot | Entity nào xuất hiện trong nhiều papers ... | 0% | 20% | no ✓ | 38s | multi_hop |
| e02 | entity_pivot | Thuật toán community detection nào được ... | 100% | 25% | no ✓ | 44s | multi_hop |
| e03 | entity_pivot | Mối quan hệ giữa ColBERT và ModernBERT t... | 100% | 40% | no ✓ | 44s | comparison |
| e04 | entity_pivot | Query decomposition giúp cải thiện multi... | 0% | 40% | no ✓ | 58s | multi_hop |
| v01 | vietnamese_complex | Embedding model nào tốt nhất cho tiếng V... | 100% | 80% | no ✓ | 36s | analytical |
| v02 | vietnamese_complex | Làm sao để đánh giá RAG system một cách ... | 0% | 0% | no ✓ | 82s | factual |
| v03 | vietnamese_complex | Mô hình nào được sử dụng để extract enti... | 50% | 40% | no ✓ | 39s | kg_construction |
| v04 | vietnamese_complex | Hệ thống RAG nào phù hợp nhất cho corpus... | 0% | 0% | no ✓ | 39s | multi_hop |
| v05 | vietnamese_complex | MedRAG benchmark đánh giá RAG systems tr... | 100% | 60% | no ✓ | 37s | comparison |
| o01 | out_of_domain | Công thức nấu phở Hà Nội ngon nhất là gì... | 0% | 100% | no ✗ | 75s | out_of_domain |
| o02 | out_of_domain | Bitcoin sẽ lên hay xuống trong tuần tới? | 0% | 100% | no ✗ | 115s | out_of_domain |
| o03 | out_of_domain | Ai là tổng thống Hoa Kỳ năm 2026? | 0% | 100% | no ✗ | 72s | factual |
| o04 | out_of_domain | Cách nào để học lập trình Python hiệu qu... | 0% | 100% | no ✗ | 80s | factual |
| kg01 | kg_construction | Knowledge Graph construction pipeline ba... | 67% | 40% | no ✓ | 37s | kg_construction |
| kg02 | kg_construction | Wikontic khác gì iText2KG trong việc xây... | 50% | 40% | no ✓ | 58s | kg_construction |
| kg03 | kg_construction | AutoSchemaKG xây dựng schema động bằng c... | 100% | 20% | no ✓ | 50s | kg_construction |
| rerank01 | reranking | SetEncoder cải thiện reranking listwise ... | 100% | 80% | no ✓ | 54s | analytical |
| rerank02 | reranking | PERank nào đạt hiệu quả cao nhất trong p... | 100% | 60% | no ✓ | 46s | analytical |
| agentic01 | agentic_rag | MAO-ARAG sử dụng multi-agent orchestrati... | 0% | 100% | no ✓ | 101s | factual |
| agentic02 | agentic_rag | ComposeRAG modular architecture gồm nhữn... | 100% | 40% | no ✓ | 98s | factual |
| agentic03 | agentic_rag | PlanRAG sử dụng planning strategy cho re... | 100% | 0% | no ✓ | 40s | comparison |
| agentic04 | agentic_rag | DeliberateThinking trong RAG hoạt động n... | 100% | 60% | no ✓ | 105s | factual |


---

## Configuration

- **Consistency Views**: True
- **Community Enabled**: True
- **Validation Enabled**: False
- **OOD Detection**: False
- **Max Retries**: 0

---
*Generated by benchmark_eval.py*
