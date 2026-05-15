<div align="center">
  <h1>Enterprise Local GraphRAG Stack v3.0</h1>
  <p><i>A production-ready Hybrid GraphRAG system that runs 100% locally on Apple Silicon.</i></p>
  <br/>
  <img src="./architecture.png" alt="Enterprise RAG Architecture" width="800" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
</div>

<br/>

## 🚀 Key Innovations (V3.0)

This system is not just another vector database wrapper. It's a highly sophisticated **Hybrid GraphRAG** architecture designed for absolute data privacy, multi-hop reasoning, and zero hallucination.

- **Hybrid 9-Path Retrieval:** Fuses Dense Vectors (5 views), Sparse BM25, and Graph-based views (Entity Pivot, Community Summaries) to bridge any vocabulary gap.
- **GLiNER + 3-Pass LLM Voting:** Uses zero-shot NER models for resource-efficient entity extraction, followed by a 3-pass consensus mechanism for ultra-precise Knowledge Graph relationships.
- **ReAct Agent for Multi-hop:** Equips LLMs with dynamic tools to traverse the Neo4j Graph dynamically, solving complex multi-hop queries that blindside static RAG pipelines.
- **Triple-Gate Validation:** Enforces 3 strict parallel validation gates (Hallucination, Entity Verification, and Citation). If it's not strictly factual, the system refuses to answer.
- **100% Local & Multi-Tenant:** Powered by Ollama on Metal GPUs. Total tenant isolation via Qdrant payloads and Neo4j node properties.

## 🧠 Core Architecture

Read the comprehensive architectural deep dive in our [Technical Wiki](./ARCHITECTURE_WIKI.md).

```text
Query → Heuristic Router → Parallel Query Understanding (6 reformulations)
      → 9-path Hybrid Retrieval (Vector + Graph + Community + BM25)
      → Weighted RRF Fusion (Heuristic Custom Tuning)
      → Mixed-Signal OOD Detection (Cosine + Lexical Overlap)
      → Standard Path OR ReAct Graph Traversal Loop
      → 3-stage Reranking (Cross-encoder → Semantic → LLM Judge)
      → Triple Validation Gates (Hallucination / Entity / Citation)
      → Answer Generation
```

## 🛠️ Quick Start

### 1. Prerequisites
- Apple Silicon Mac (M-series), 16GB+ Unified Memory
- `brew`, `docker`, `docker-compose`, `make`
- Ollama running on host

### 2. Setup Ollama
```bash
brew install ollama
ollama pull qwen3.5:4b
ollama pull bge-m3
ollama serve
```

### 3. Initialize & Start
```bash
# Generate credentials + build images
make init

# Start all services
make up

# Initialize database schemas
make init-all

# Health check
make health
```

### 4. Try It
```bash
curl -s -X POST http://localhost:8800/api/v3/chat \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: eval" \
  -d '{"query":"GraphRAG là gì?","max_retries":0}'
```

## 📊 Evaluation (V3 Benchmarks)

Based on our internal 30-query Vietnamese benchmark suite:
- **Factual doc_recall:** 100%
- **Out-of-Domain Detection:** 100% Precision (0 False Positives)
- **Refused Rate:** 13.3% (System correctly refuses rather than hallucinates)

## 🏗️ Services Overview

| Component | Stack | Port |
|---|---|---|
| **RAG API** | FastAPI + uvloop | `8800` |
| **Vector DB** | Qdrant | `6333` |
| **Knowledge Graph** | Neo4j | `7474` |
| **Semantic Cache** | Redis | `6379` |
| **Tracing** | Langfuse | `3000` |
| **Metrics** | Grafana | `3001` |

## 📄 License
This project is licensed under the **Apache License 2.0**. See [LICENSE](./LICENSE) for details.
