"""Configuration — all settings read from environment variables (Mac Mini M4 Optimized)."""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "qwen3.5:9b"

    # ── Embedding ───────────────────────────────────────────────────────────
    ollama_embed_model: str = "bge-m3"
    ollama_embed_url: str = "http://host.docker.internal:11434"
    embed_dimension: int = 1024
    # M4 optimized: batch 32 (larger than default 16 for throughput)
    embed_batch_size: int = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
    # M4 optimized: only 3 concurrent Ollama embedding calls (M4 can handle 3 streams)
    embed_concurrent_limit: int = int(os.environ.get("EMBED_CONCURRENT_LIMIT", "3"))

    # ── Vector DB ────────────────────────────────────────────────────────────
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "enterprise_kb"

    # ── Knowledge Graph ───────────────────────────────────────────────────────
    neo4j_url: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # ── Cache ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://redis:6379/0"
    # M4 optimized: 2h cache (longer TTL, less memory pressure)
    semantic_cache_ttl_s: int = int(os.environ.get("SEMANTIC_CACHE_TTL", "7200"))
    enable_semantic_cache: bool = os.environ.get("ENABLE_SEMANTIC_CACHE", "true").lower() != "false"

    # ── App ─────────────────────────────────────────────────────────────────
    app_env: str = "production"
    log_level: str = "INFO"
    # M4 optimized: 6 concurrent (M4 efficiency cores handle I/O well)
    max_concurrent_requests: int = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "6"))
    request_timeout_s: int = int(os.environ.get("REQUEST_TIMEOUT_S", "120"))

    # ── Retrieval tuning ────────────────────────────────────────────────────
    retrieval_top_k: int = 8
    retrieval_vector_top_k: int = 20
    retrieval_graph_top_k: int = 15  # reduced from 20 to save memory
    retrieval_path_top_k: int = 30
    rrf_k: int = 60

    # ── Multi-tenancy ────────────────────────────────────────────────────────
    multi_tenant_enabled: bool = True
    api_internal_key: str = ""
    enforce_api_key: bool = False

    # ── Reranking ────────────────────────────────────────────────────────────
    enable_reranker: bool = True
    reranker_type: str = "semantic"  # semantic is fast, uses cosine similarity
    reranker_top_k: int = 10

    # ── Dashboard ───────────────────────────────────────────────────────────
    dashboard_port: int = 7860

    # ── VRAG pipeline — Quality-first GraphRAG ──────────────────────────────
    # Feature flag kept for backward compat with /health endpoint (always True).
    pipeline_v2_enabled: bool = bool(int(os.environ.get("PIPELINE_V2_ENABLED", "1")))

    # Consistency Simulation (ingest time)
    # consistency_views_enabled=False bypasses 4 LLM view generation calls per chunk
    # — major speedup for small/slow LLMs. Single embedding from original text used.
    # Entity extractor — SEPARATE from semantic LLM (architecture decision)
    # provider options: gliner (local NER) | openai | anthropic
    entity_extractor_provider: str = os.environ.get("ENTITY_EXTRACTOR_PROVIDER", "gliner")
    entity_extractor_model: str = os.environ.get(
        "ENTITY_EXTRACTOR_MODEL", "urchade/gliner_multi-v2.1"
    )
    entity_extractor_threshold: float = float(os.environ.get("ENTITY_EXTRACTOR_THRESHOLD", "0.5"))
    entity_relations_enabled: bool = bool(int(os.environ.get("ENTITY_RELATIONS_ENABLED", "0")))

    consistency_views_enabled: bool = bool(int(os.environ.get("CONSISTENCY_VIEWS_ENABLED", "1")))
    consistency_num_views: int = 5
    consistency_low_threshold: float = 0.60
    consistency_high_threshold: float = 0.85
    entity_vote_passes: int = int(os.environ.get("ENTITY_VOTE_PASSES", "3"))
    entity_vote_min: int = 2

    # PII masking
    pii_mask_enabled: bool = bool(int(os.environ.get("PII_MASK_ENABLED", "1")))
    pii_llm_ner_enabled: bool = bool(int(os.environ.get("PII_LLM_NER_ENABLED", "1")))

    # Hierarchical chunking — comma-separated string, parsed lazily
    chunk_levels_csv: str = os.environ.get("CHUNK_LEVELS_ENABLED", "paragraph,section")

    @property
    def chunk_levels_enabled(self) -> list[str]:
        return [x.strip() for x in self.chunk_levels_csv.split(",") if x.strip()]

    section_max_chars: int = 4000
    paragraph_max_chars: int = 800
    sentence_max_chars: int = 200

    # Query understanding — VRAG Tier 1 (Zero-LLM by default)
    # Default 0 = pure zero-LLM (only GLiNER + centroid router).
    # Each unit adds 1 LLM reformulation in order:
    #   1=rewrite, 2=+keywords, 3=+hyde, 4=+decompose, 5=+step_back
    # With qwen3.5:9b each LLM call is ~10-30s — set higher only when quality matters more than latency.
    query_understanding_enabled: bool = bool(
        int(os.environ.get("QUERY_UNDERSTANDING_ENABLED", "1"))
    )
    query_reformulations: int = int(os.environ.get("QUERY_REFORMULATIONS", "0"))
    query_understanding_timeout_s: float = float(
        os.environ.get("QUERY_UNDERSTANDING_TIMEOUT_S", "60")
    )

    # Multi-path retrieval
    retrieval_views: list[str] = ["dense", "question", "summary", "keywords"]
    retrieval_use_sparse: bool = True

    # Rerank stages
    # Stage 1 cross-encoder: needs ~600MB model from HF. Disabled by default
    # to avoid OOM in 1GB rag-api container; enable explicitly when memory is
    # available (or when running outside container). Pipeline falls back to
    # stage 2 semantic match gracefully.
    rerank_stage1_enabled: bool = bool(int(os.environ.get("RERANK_STAGE1_ENABLED", "0")))
    rerank_stage1_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_stage1_top_k: int = 50
    rerank_stage2_top_k: int = 20
    # Stage 3 LLM judge: 10 LLM calls in parallel. Heavy. Disabled by default;
    # enable with env when GPU available.
    rerank_stage3_enabled: bool = bool(int(os.environ.get("RERANK_STAGE3_ENABLED", "0")))
    rerank_stage3_top_k: int = 10
    final_top_k: int = 5
    # VRAG Tier 3b: Dynamic Early-Exit threshold. If stage1 (cross-encoder)
    # avg confidence on top-N >= this, skip stage3 LLM judge. Set to 1.1 to disable.
    rerank_early_exit_threshold: float = float(
        os.environ.get("RERANK_EARLY_EXIT_THRESHOLD", "0.85")
    )

    # VRAG Tier 3c: LLMLingua-2 context compression. Reduces context tokens
    # before LLM gen for ~30-50% gen-time savings. First call downloads ~600MB.
    context_compression_enabled: bool = bool(
        int(os.environ.get("CONTEXT_COMPRESSION_ENABLED", "0"))
    )
    context_compression_rate: float = float(
        os.environ.get("CONTEXT_COMPRESSION_RATE", "0.4")
    )

    # Generation deliberation
    # Defaults tuned for qwen3.5:9b CPU/Metal speed. Enable richer modes via env
    # when on GPU. Each draft is 1 LLM call; judge = 1 more; outline = 1 more.
    generation_outline_enabled: bool = bool(int(os.environ.get("GENERATION_OUTLINE_ENABLED", "0")))
    generation_drafts: int = int(os.environ.get("GENERATION_DRAFTS", "1"))
    generation_judge_enabled: bool = bool(int(os.environ.get("GENERATION_JUDGE_ENABLED", "0")))
    generation_refine_enabled: bool = bool(int(os.environ.get("GENERATION_REFINE_ENABLED", "1")))
    generation_max_tokens: int = 2048

    # Validation gates
    validation_enabled: bool = True
    validation_min_grounded_ratio: float = 0.70
    validation_max_invalid_entities: int = 3
    validation_min_citation_ratio: float = 0.40
    validation_retry_on_fail: bool = True

    # Community summaries
    community_enabled: bool = False  # Phase 7 — bật sau khi có data đủ lớn
    community_levels: int = 3
    community_resolution: float = 1.0
    community_min_size: int = 3
    community_summary_vote_passes: int = 3

    # Refusal
    refusal_message_vi: str = (
        "Tôi không có đủ thông tin chắc chắn để trả lời câu hỏi này dựa trên tài liệu hiện có."
    )

    # OOD detection — early refusal before generation
    ood_detection_enabled: bool = bool(int(os.environ.get("OOD_DETECTION_ENABLED", "1")))
    ood_relevance_threshold: float = float(os.environ.get("OOD_RELEVANCE_THRESHOLD", "0.50"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
