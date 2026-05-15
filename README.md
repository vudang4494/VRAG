<div align="center">
  <h1>Enterprise Local GraphRAG Stack v3.0</h1>
  <p><i>A production-ready Hybrid GraphRAG system that runs 100% locally on Apple Silicon.</i></p>
  <br/>
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

```mermaid
flowchart TD
    %% Nodes
    User((User / Browser))
    
    Nginx["Nginx Reverse Proxy<br/>(Rate Limiting & Routing)"]
    
    WebUI["Open WebUI<br/>(Chat Interface)"]
    Gradio["Gradio Dashboard<br/>(Visualizer & Stats)"]
    API["RAG API<br/>(FastAPI Orchestrator)"]
    
    Redis[("Redis<br/>Semantic Cache")]
    Qdrant[("Qdrant<br/>Vector Database")]
    Ollama["Ollama (Host)<br/>Metal GPU Accelerated"]
    Postgres[("PostgreSQL<br/>App State")]
    Neo4j[("Neo4j<br/>Knowledge Graph")]
    Prometheus["Prometheus + Grafana<br/>(Metrics)"]
    Langfuse["Langfuse<br/>(Tracing)"]
    
    %% Edges
    User -- "HTTP Requests" --> Nginx
    
    Nginx -- "Port 80" --> WebUI
    Nginx -- "Port 7860" --> Gradio
    Nginx -- "Port 8800" --> API
    Nginx -- "Port 3000" --> Langfuse
    
    WebUI -- "Chat API" --> API
    Gradio -- "Stats/Graph" --> API
    
    API -- "Check Cache" --> Redis
    API -- "2. Vector Search" --> Qdrant
    API -- "1. Generate Embedding" --> Ollama
    API -- "4. LLM Generation" --> Ollama
    API -- "State Tracking" --> Postgres
    API -- "3. Graph Traversal" --> Neo4j
    API -. "Metrics" .-> Prometheus
    API -. "Tracing/Logs" .-> Langfuse
    
    %% Styling
    classDef blue fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff
    classDef orange fill:#e17055,stroke:#fab1a0,stroke-width:2px,color:#fff
    classDef green fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff
    classDef purple fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    classDef pink fill:#e84393,stroke:#fd79a8,stroke-width:2px,color:#fff
    classDef gray fill:#636e72,stroke:#b2bec3,stroke-width:2px,color:#fff
    
    class User blue
    class Nginx orange
    class WebUI,Gradio,API green
    class Redis,Qdrant,Postgres,Neo4j purple
    class Ollama pink
    class Prometheus,Langfuse gray
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
