# Enterprise Local RAG Stack v3.0 - Architecture & Algorithms

Welcome to the Technical Wiki for the **Enterprise Local RAG Stack v3.0**. This document provides an in-depth, transparent analysis of the system's architecture, data structures, and the core algorithms that drive its enterprise-grade Retrieval-Augmented Generation capabilities.

## 1. Architectural Overview

The system is designed around a **Hybrid GraphRAG** model. It maximizes the strengths of both Vector Databases (for semantic surface-level search) and Graph Databases (for structural, relational, and multi-hop search).

**Core Modules:**
- **Vector Store:** Qdrant (Stores multi-view dense embeddings & sparse BM25 vectors).
- **Knowledge Graph:** Neo4j (Stores Entities, Relationships, and Hierarchical Communities).
- **Reasoning Engine:** ReAct Agent (Handles multi-hop logic and dynamic data fetching).
- **Isolation & Caching:** Redis (Semantic cache) & comprehensive Multi-Tenant Isolation.

---

## 2. Ingestion & Knowledge Graph Algorithms

Data ingestion goes far beyond simple text chunking. It employs multi-dimensional analysis algorithms to build a highly precise Knowledge Graph (KG) and robust vector representations.

### 2.1 Hierarchical Chunking & 5-View Consistency Simulation
- **How it works:** Documents are hierarchically parsed (Section → Paragraph → Sentence). A consistency simulation algorithm then evaluates the chunks across 5 distinct perspectives: `dense` (original), `paraphrase`, `question`, `summary`, and `keywords`.
- **Why it is used:** To enrich the metadata and vector representations of a single data point.
- **Why it is good (The Value):** Standard RAG systems index only the raw text, which often misaligns with how users ask questions. By simulating 5 views, the system pre-computes semantic bridges, making retrieval highly resilient to varied user phrasing.

### 2.2 Zero-shot NER with GLiNER
- **How it works:** Instead of relying on heavy LLMs for Named Entity Recognition (NER), the system uses GLiNER (a specialized, lightweight zero-shot NER model) to extract entities.
- **Why it is used:** LLMs are computationally expensive, slow, and prone to formatting errors (e.g., failing to output valid JSON). 
- **Why it is good (The Value):** GLiNER operates fractionally faster and requires significantly less compute (ideal for local Apple Silicon deployment) while maintaining high recall for identifying domain-specific entities.

### 2.3 3-Pass LLM Voting for Relationships
- **How it works:** Once entities are identified, an LLM evaluates the text in 3 separate passes (at varying temperatures) to determine the *Relationships* between these entities. The system then aggregates the votes and stores the relationship with the highest consensus.
- **Why it is used:** Relationship extraction is highly nuanced and prone to ambiguity. A single LLM pass often hallucinates or misinterprets directional context.
- **Why it is good (The Value):** It guarantees that the edges in the Neo4j Knowledge Graph possess extremely high Precision. A clean, accurate graph prevents the downstream reasoning agent from traversing "hallucinated" paths.

### 2.4 Leiden/Louvain Community Detection
- **How it works:** Inspired by Microsoft GraphRAG, the system applies hierarchical graph clustering algorithms (primarily Leiden, falling back to Louvain) to group closely connected entities into "Communities." An LLM then generates a global summary for each community.
- **Why it is used:** Vector databases suffer from the "Global Context Problem" (e.g., they cannot answer "Summarize the overarching theme of this 100-page document" because they only fetch fragmented chunks).
- **Why it is good (The Value):** Community summaries provide a macro-level view of the dataset. When a user asks a high-level conceptual question, the system retrieves these community nodes, offering a holistic answer that traditional RAG simply cannot achieve.

---

## 3. Query Architecture & 9-Path Hybrid Retrieval

This is the beating heart of the system, designed to handle everything from simple factual lookups to complex analytical comparisons.

### 3.1 Multi-view Semantic Search & Graph Search
- **How it works:** A single user query is reformulated 6 ways and simultaneously queried across 9 distinct paths: 5 Dense Vector views, 1 Sparse Vector (BM25), Entity Pivot (Graph), Graph Co-occurrence, and Community Summaries.
- **Why it is used:** To solve the "Vocabulary Mismatch" problem. A user might use a keyword (BM25), a conceptual idea (Dense), or ask about a specific entity relationship (Graph).
- **Why it is good (The Value):** It guarantees maximum Recall. No matter how the user frames their query, at least one of the 9 paths will likely hit the relevant context.

### 3.2 Weighted Reciprocal Rank Fusion (RRF)
- **How it works:** Instead of a naive RRF (which treats all sources equally), the system uses a custom heuristic formula:
  ```text
  RRF_score = path_weight × consistency_factor × level_factor × domain_reward / (k + rank)
  ```
- **Why it is used:** Not all retrieval paths are equally reliable. For example, a direct "Entity Pivot" match is factually stronger than a "Paraphrase Vector" match.
- **Why it is good (The Value):** By artificially boosting high-confidence paths (e.g., `Hyde=1.3`, `Entity Pivot=1.5`) and penalizing low-consistency chunks, the fusion algorithm acts as a highly intelligent filter, bubbling only the most factually dense chunks to the top.

---

## 4. Reasoning via ReAct Agent Loop

To break free from the limitations of "Static Retrieval," the system employs a dynamic Agentic loop.

### 4.1 Dynamic Graph Traversal
- **How it works:** If the Heuristic Router identifies a complex/multi-hop query (e.g., *"What products does the company Mr. A works for make?"*), it hands control to a ReAct (Reasoning and Acting) Agent. The Agent uses tools like `search_entity` and `expand_relation` to traverse the Neo4j graph step-by-step.
- **Why it is used:** Static RAG fetches documents based on semantic similarity to the *initial query*. It fails entirely if the answer requires connecting Document A to Document B via an intermediate entity.
- **Why it is good (The Value):** The Agent navigates the data exactly like a human researcher. It finds Mr. A, looks at his edges, discovers "Company X", searches Company X, and finally retrieves the "Products".
- **Trade-offs:** While it provides unparalleled reasoning capabilities, it heavily increases Latency due to sequential LLM calls. However, on a GPU-accelerated server environment, this latency becomes negligible.

---

## 5. Safety & Validation Gates

In an Enterprise environment, hallucination is unacceptable. The system is designed to "Refuse to Answer" rather than guess.

### 5.1 Mixed-Signal Out-of-Domain (OOD) Detection
- **How it works:** The system calculates a mixed signal combining **Dense Distance** (Cosine score < 0.5) and **Lexical Overlap** (Keyword match < 30%).
- **Why it is used:** Dense vectors often suffer from "Soft Hallucinations," where the DB returns the "closest" document even if it's completely unrelated to the query.
- **Why it is good (The Value):** By anchoring the semantic search with a hard Lexical (keyword) check, the system instantly blocks OOD queries, saving compute resources and preventing nonsensical answers.

### 5.2 Parallel Triple-Gate Validation
- **How it works:** Before returning an answer, the generated response must pass three strict parallel checks:
  1. **Hallucination Gate:** The LLM's response is broken into "Atomic Claims." Each claim is verified against the retrieved context. (Must pass a `>= 0.70` Grounded Ratio).
  2. **Entity Gate:** Extracted entities in the answer are cross-referenced with the Neo4j Knowledge Graph to ensure no fake entities were generated.
  3. **Citation Gate:** The system enforces strict citation markers. (Must have a Citation Ratio `>= 0.40`).
- **Why it is used:** To establish absolute trust.
- **Why it is good (The Value):** This is the gold standard for Enterprise AI. It acts as a safety net, ensuring that every output is traceable, factual, and strictly grounded in the ingested corpus.

---

## 6. Conclusion & Scalability

The **Enterprise Local RAG Stack v3.0** is an uncompromising, highly advanced implementation of Retrieval-Augmented Generation. Its heavy reliance on multi-path retrieval, dynamic agentic reasoning, and aggressive validation makes it incredibly accurate.

While the current Apple Silicon local deployment faces latency bottlenecks due to the sheer volume of parallel tasks and LLM calls, the architectural foundation is fundamentally sound. When deployed on high-performance GPU servers, the asynchronous design (parallel reformulations, parallel 9-path fetching, parallel validation gates) will truly shine, transforming it into a blazingly fast, hyper-accurate Enterprise Knowledge Engine.
