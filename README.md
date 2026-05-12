# Enterprise Local RAG Stack v2.0 🚀

A production-ready Retrieval-Augmented Generation (RAG) stack optimized for **Apple Silicon Mac**. This stack runs **100% locally**, ensuring complete data privacy while leveraging the power of Metal GPU acceleration.

## 🌟 Key Features

- **100% Local Inference**: Powered by Ollama with Qwen3.5-4B and BGE-M3 embeddings.
- **Hybrid GraphRAG**: Combines Vector Search (Qdrant) and Knowledge Graphs (Neo4j).
- **Multi-tenant Architecture**: Strict data isolation for enterprise use cases.
- **Visual Dashboard**: Gradio-based hub for Graph visualization and Chat.
- **Observability**: Built-in Langfuse tracing, Prometheus metrics, and Grafana dashboards.

---

## 🏗 System Architecture & Data Flow

Here is the high-level architecture and data flow of the Enterprise RAG Stack:

```mermaid
graph TD
    %% Define Styles
    classDef user fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff;
    classDef proxy fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:#fff;
    classDef app fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff;
    classDef db fill:#8b5cf6,stroke:#5b21b6,stroke-width:2px,color:#fff;
    classDef model fill:#ec4899,stroke:#be185d,stroke-width:2px,color:#fff;
    classDef observe fill:#64748b,stroke:#334155,stroke-width:2px,color:#fff;

    %% Nodes
    User(("User / Browser")):::user
    Nginx["Nginx Reverse Proxy<br/>(Rate Limiting & Routing)"]:::proxy
    
    WebUI["Open WebUI<br/>(Chat Interface)"]:::app
    Dashboard["Gradio Dashboard<br/>(Visualizer & Stats)"]:::app
    RAGAPI["RAG API<br/>(FastAPI Orchestrator)"]:::app
    
    Ollama["Ollama (Host)<br/>Metal GPU Accelerated"]:::model
    
    Qdrant[("Qdrant<br/>Vector Database")]:::db
    Neo4j[("Neo4j<br/>Knowledge Graph")]:::db
    Redis[("Redis<br/>Semantic Cache")]:::db
    Postgres[("PostgreSQL<br/>App State")]:::db
    
    Langfuse["Langfuse<br/>(Tracing)"]:::observe
    Prometheus["Prometheus + Grafana<br/>(Metrics)"]:::observe

    %% Flow
    User -->|HTTP Requests| Nginx
    Nginx -->|Port 80| WebUI
    Nginx -->|Port 7860| Dashboard
    Nginx -->|Port 8800| RAGAPI
    Nginx -->|Port 3000| Langfuse
    
    WebUI -->|Chat API| RAGAPI
    Dashboard -->|Stats/Graph| RAGAPI
    Dashboard -->|Cypher Query| Neo4j
    
    RAGAPI -->|Check Cache| Redis
    RAGAPI -->|1. Generate Embedding| Ollama
    RAGAPI -->|2. Vector Search| Qdrant
    RAGAPI -->|3. Graph Traversal| Neo4j
    RAGAPI -->|4. LLM Generation| Ollama
    
    RAGAPI -.->|State Tracking| Postgres
    RAGAPI -.->|Tracing/Logs| Langfuse
    RAGAPI -.->|Metrics| Prometheus
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Hardware:** Apple Silicon Mac (M-series) with 16GB+ Unified Memory.
- **Software:** `brew`, `docker`, `docker-compose`, `make`.

### 2. Setup Ollama (Host)
For maximum Metal GPU performance, Ollama runs natively on the host Mac, not in Docker.
```bash
brew install ollama
ollama pull qwen3.5:4b
ollama pull bge-m3
ollama serve  # Leave this running in a separate terminal
```

### 3. Initialize & Start Stack
Run the following commands in the project root:
```bash
# 1. Generate secure credentials (.env) & Build Docker images
make init

# 2. Start all services via Docker Compose
make up

# 3. Initialize Qdrant and Neo4j schemas
make init-all

# 4. Run full system health check
make health
```
> **Security Note:** All sensitive credentials (API Keys, Passwords) are auto-generated and stored locally in the `.env` file. This file is safely ignored by Git and will **never** be pushed to the repository. No external keys like OpenAI, Claude, or Gemini are hardcoded in the source code.

---

## 🛠 Management Commands

Use the unified `Makefile` for operations:
- `make logs`: View logs for all services.
- `make restart`: Restart the entire stack.
- `make down`: Stop all containers (data is preserved).
- `make test-all`: Run health, embedding, and RAG E2E tests.
- `make test-perf`: Run performance benchmarks.

---

## 🔗 Access Points

Once the stack is up, you can access the localized services at:

| Component | URL | Default Credentials |
|-----------|-----|-------------------|
| **Gradio Dashboard** | `http://localhost:7860` | No auth needed |
| **Open WebUI** | `http://localhost:80` | Create your first admin account |
| **Neo4j Browser** | `http://localhost:7474` | `neo4j` / (check `.env` for password) |
| **Qdrant UI** | `http://localhost:6333/dashboard` | No auth needed |
| **Langfuse** | `http://localhost:3000` | admin@localhost / (check `.env`) |
| **Grafana** | `http://localhost:3001` | `admin` / (check `.env`) |
