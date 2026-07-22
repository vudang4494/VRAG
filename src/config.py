"""Configuration — all settings read from environment variables (Mac Mini M4 Optimized)."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://host.docker.internal:11434"
    # Tiered model roles — keep tasks specialised so a small model can do 70%+ of
    # calls while a slightly larger one carries final synthesis. Swap to GPU-class
    # models by changing these env vars only; no code edits required.
    #
    #   LIGHT_LLM   — query understanding, intent, validation, extraction, schema
    #                 (high call volume, low complexity)
    #   HEAVY_LLM   — final answer generation + ReAct multi-hop reasoning
    #                 (low call volume, high complexity)
    #   ollama_model — read from OLLAMA_MODEL independently; it does NOT inherit
    #                 HEAVY_LLM. It is the fallback every ollama_chat() call without
    #                 an explicit model= lands on, so a GPU upgrade must set all
    #                 three vars or most heavy calls keep hitting the old model.
    #
    # Model selection rationale (Mac M4 24GB, VN-first):
    #   - qwen2.5:3b/7b: Chinese-drift on Vietnamese prompts (training bias) — UNSAFE
    #   - llama3.2:3b: Fluent VN but factually weak — UNSAFE for technical QA
    #   - qwen3:4b: VN+facts OK, but "thinking" model leaks reasoning into content
    #     even with think=False (API-side limitation) — UNSAFE for prod chat
    #   - gemma3:4b: clean VN and no thinking mode, but superseded — see below
    #   - gemma4:e4b: clean VN, correct facts, JSON-mode works for ReAct — WINNER
    #
    # gemma4:e4b advertises capability "thinking" (`ollama show gemma4:e4b`), which
    # reads like the qwen3:4b trap above. It is not the same failure: the thinking
    # text goes to a separate `message.thinking` field and never contaminates
    # `message.content`. It is still expensive, so think=False is MANDATORY, not
    # cosmetic. Measured on this box, same prompt, same answer text:
    #     native /api/chat, think=False -> 91 tokens,  3.2s
    #     native /api/chat, think=True  -> 502 tokens, 17.6s   (5.5x, output identical)
    #     /v1/chat/completions          -> 394 tokens          (4.3x; `think` silently
    #                                                           dropped by the compat layer)
    # ollama_chat() defaults think=False, which is exactly why every LLM call must go
    # through it rather than clients.llm (see src/services/ollama_helper.py).
    #
    # GPU upgrade path: set LIGHT_LLM + HEAVY_LLM + OLLAMA_MODEL together. An absent
    # tag no longer fails silently at the first intent-classification call — startup
    # verifies every configured tag against `ollama list` (see config_report.py).
    #
    # These are plain pydantic fields on purpose. BaseSettings already maps
    # light_llm <-> LIGHT_LLM, so the os.environ.get(...) these used to carry was a
    # redundant second read — and a harmful one: the call is evaluated at import, so
    # the env value got baked into the field default itself. model_fields[...].default
    # then reported the env value rather than the literal in this file, which made
    # config drift undetectable by construction. Keep them declarative; the env still
    # overrides, and the code default stays introspectable for the startup banner.
    light_llm: str = "gemma4:e4b"
    heavy_llm: str = "gemma4:e4b"
    ollama_model: str = "gemma4:e4b"

    # ── Embedding ───────────────────────────────────────────────────────────
    ollama_embed_model: str = "bge-m3"
    ollama_embed_url: str = "http://host.docker.internal:11434"
    embed_dimension: int = 1024
    # M4 optimized: batch 32 (larger than default 16 for throughput)
    embed_batch_size: int = 32
    # M4 optimized: only 3 concurrent Ollama embedding calls (M4 can handle 3 streams)
    embed_concurrent_limit: int = 3

    # ── Vector DB ────────────────────────────────────────────────────────────
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "enterprise_kb"

    # ── Entity-gate: primary scoping via cross-doc entity cosine ──────────────
    # Replaces doc-gate (proven harmful −5pp recall on 53-q eval). Discovers
    # candidate entities via a quick dense seed, scores them cosine+TF-IDF+MMR,
    # then surfaces all chunks linked to top entities (cross-doc by design).
    entity_gate_enabled: bool = False
    entity_gate_top_k_entities: int = 50
    entity_gate_seed_chunks: int = 200
    entity_gate_score_floor: float = 0.2

    # ── Knowledge Graph ───────────────────────────────────────────────────────
    neo4j_url: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # ── Cache ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://redis:6379/0"
    # M4 optimized: 2h cache (longer TTL, less memory pressure)
    semantic_cache_ttl_s: int = Field(7200, validation_alias="SEMANTIC_CACHE_TTL")
    # The last os.environ.get() that lived inside this BaseSettings. It carried both bugs
    # the other 42 conversions removed, plus one of its own:
    #   1. evaluated at import, so the env value froze INTO the field default and drift
    #      checks compared env against itself;
    #   2. `!= "false"` meant only the literal string "false" turned it off — the obvious
    #      ENABLE_SEMANTIC_CACHE=0 (and =no, =off) silently left the cache ON, so anyone
    #      benchmarking "without cache" was measuring with it.
    # Declarative pydantic parses 0/false/no/off (and 1/true/yes/on) correctly.
    enable_semantic_cache: bool = True

    # ── App ─────────────────────────────────────────────────────────────────
    app_env: str = "production"
    log_level: str = "INFO"
    # M4 optimized: 6 concurrent (M4 efficiency cores handle I/O well)
    max_concurrent_requests: int = 6
    request_timeout_s: int = 120

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
    pipeline_enabled: bool = Field(True, validation_alias="PIPELINE_V2_ENABLED")

    # Consistency Simulation (ingest time)
    # consistency_views_enabled=False bypasses 4 LLM view generation calls per chunk
    # — major speedup for small/slow LLMs. Single embedding from original text used.
    # Entity extractor — SEPARATE from semantic LLM (architecture decision)
    # provider options: gliner (local NER) | openai | anthropic
    entity_extractor_provider: str = "gliner"
    entity_extractor_model: str = "urchade/gliner_multi-v2.1"
    # 0.5→0.6: on real chunk text the 0.5 tail was mostly low-confidence junk
    # spans. 0.6 keeps named entities (person/org/tech score high) and drops the
    # noise. Env-tunable per corpus.
    entity_extractor_threshold: float = Field(0.6, validation_alias="ENTITY_EXTRACTOR_THRESHOLD")
    # Comma-separated label override. Empty → DEFAULT_LABELS (concept/event
    # dropped as noise; see entity_extractor.py). Set e.g. "person,organization,
    # concept" to restore per corpus without touching code.
    entity_extractor_labels: str = Field("", validation_alias="ENTITY_EXTRACTOR_LABELS")
    entity_relations_enabled: bool = False
    # Legacy LLM-vote entity fallback. Runs ONLY when the GLiNER extractor is
    # unavailable. Default OFF: it does entity_vote_passes LLM calls PER CHUNK and
    # silently turned a failed GLiNER init into a multi-minute ingest hang. Enable
    # explicitly only when running without GLiNER and you accept the cost.
    entity_llm_fallback_enabled: bool = Field(False, validation_alias="ENTITY_LLM_FALLBACK")

    consistency_views_enabled: bool = True
    # Doc-context prefix (Anthropic-style contextual retrieval, doc-level):
    # 1 LLM call/doc summary → prepend "[filename: summary] " to every chunk's
    # embed input. Fixes fragment chunks (citation tails, lone definitions)
    # that otherwise embed as noise. ~3s overhead per doc.
    doc_context_prefix_enabled: bool = True
    consistency_low_threshold: float = 0.60
    consistency_high_threshold: float = 0.85
    entity_vote_passes: int = 3
    entity_vote_min: int = 2

    # PII masking. Regex pass (default ON) masks structured PII — phone, email,
    # ID/CCCD, bank account, URL, IP — instantly and deterministically.
    # LLM-NER pass adds PERSON/ORGANIZATION/ADDRESS but does an LLM call per doc
    # window; on a small local model (gemma-class, ~14 tok/s) it costs minutes per
    # doc, so it is OPT-IN (default OFF). Enable only with a fast/GPU model.
    pii_mask_enabled: bool = True
    pii_llm_ner_enabled: bool = False

    # Hierarchical chunking — comma-separated string, parsed lazily
    chunk_levels_csv: str = Field("paragraph,section", validation_alias="CHUNK_LEVELS_ENABLED")

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
    # With gemma4:e4b each LLM call is ~3-5s — set higher only when quality matters more than latency.
    query_understanding_enabled: bool = True
    query_reformulations: int = 0
    query_understanding_timeout_s: float = 60.0
    sufficient_context_gate_enabled: bool = True

    # Multi-path retrieval
    retrieval_use_sparse: bool = True
    # Drop the "question"/"keywords" views from INTENT_STRATEGY. They are dense-copies
    # by construction — consistency.py only generates paraphrase+summary, and the vector
    # store backfills question/keywords with the dense vector — so searching them (same
    # query vec, identical stored vecs) returns the exact dense ranked list and RRF
    # double-counts the dense signal against graph_aware/entity_pivot/paraphrase/summary.
    # Kept as a flag (default off) until the A/B grounded/citation eval confirms it.
    retrieval_real_views_only: bool = Field(True, validation_alias="RETRIEVAL_REAL_VIEWS_ONLY")

    # Rerank stages
    # Stage 1 cross-encoder: needs ~600MB model from HF. Disabled by default
    # to avoid OOM in 1GB rag-api container; enable explicitly when memory is
    # available (or when running outside container). Pipeline falls back to
    # stage 2 semantic match gracefully.
    rerank_stage1_enabled: bool = False
    rerank_stage1_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_stage1_top_k: int = 50
    rerank_stage2_top_k: int = 20
    # Stage 3 LLM judge: 10 LLM calls in parallel. Heavy. Disabled by default;
    # enable with env when GPU available.
    rerank_stage3_enabled: bool = False
    rerank_stage3_top_k: int = 10
    final_top_k: int = 5
    # VRAG Tier 3b: Dynamic Early-Exit threshold. If stage1 (cross-encoder)
    # avg confidence on top-N >= this, skip stage3 LLM judge. Set to 1.1 to disable.
    rerank_early_exit_threshold: float = 0.85

    # VRAG Tier 3c: LLMLingua-2 context compression. Reduces context tokens
    # before LLM gen for ~30-50% gen-time savings. First call downloads ~600MB.
    context_compression_enabled: bool = False
    context_compression_rate: float = 0.4

    # ══ KG retrieval — one flow, knobs ordered by MEASURED payoff ════════════════
    # The entity/graph paths (entity_pivot, ppr, graph)
    # are a HIGH-precision, LOW-recall signal that weighted-RRF fuses with dense/
    # sparse. What each lever actually moved, most → least:
    #   LEVER 1  rrf_kg_path_weight_scale — recall@1 ×2.3, MRR +24% at 0.2×. ON.
    #   LEVER 2  entity resolution soft-fold — small/consistent; aliases already live.
    #   LEVER 3  de-hub (edge_weighting/npmi_min/degree_penalty) — graph-health only,
    #            INERT on recall at 1.1% coverage. OFF until the graph densifies.
    #
    # Phase 2.1: HippoRAG 2 — Personalized PageRank over the entity graph. Seeds the
    # walk on query entities, propagates through RELATES_TO (or NPMI co-occurrence
    # under de-hub), returns chunks linked to high-scoring entities. Alias surface
    # forms fold to canonical at seed time. Graph cached per-tenant 10 min; ~50-150ms.
    ppr_enabled: bool = True
    ppr_alpha: float = 0.5
    # Cross-doc SuperNova de-hub.
    # When ON, PPR rebuilds the entity graph WEIGHTED by NPMI over CONTAINS_ENTITY
    # co-occurrence (marginal-frequency normalization → hub pairs like generic
    # terms / dates / geos collapse toward 0 and are pruned), replacing the
    # unweighted RELATES_TO/co-occurrence graph. Payoff is graph-health (bounds
    # hub PageRank mass as the corpus grows), NOT factual recall. Zero LLM.
    ppr_edge_weighting: bool = Field(False, validation_alias="PPR_EDGE_WEIGHTING")
    # Keep only entity pairs with NPMI strictly above this floor. NPMI ∈ [-1,1];
    # hub pairs fall below 0, specific co-mentions score high. 0.0 = keep only
    # positively-associated pairs (drops the hair-ball edges).
    ppr_npmi_min: float = Field(0.0, validation_alias="PPR_NPMI_MIN")
    # Degree-penalty exponent γ in score(e)=ppr(e)/(1+deg(e))^γ, applied at ranking.
    # Fights hubs siphoning walk mass. 0 = off; 0.5 is a gentle start. Applies
    # independently of ppr_edge_weighting so each lever can be ablated.
    ppr_degree_penalty: float = Field(0.0, validation_alias="PPR_DEGREE_PENALTY")
    # LEVER 1 (the big one) — RRF weight multiplier for KG paths (entity_pivot / ppr /
    # graph / community / entity_cosine / entity_gate). Those paths ship hand-tuned RRF
    # weights ABOVE dense=1.0 (entity_pivot=1.5, ppr=1.7 …). The corpus500 recall
    # benchmark showed that at 1.0× they
    # NET-HURT recall@5 — the metric final_top_k feeds the answer — by evicting reliable
    # dense chunks from the top-5, even while DOUBLING recall@1. Scaling to ~0.2×
    # recovers recall@5 to dense parity while KEEPING the recall@1 (×2.3) + MRR (+24%)
    # gain → KG becomes a near-free precision boost. DEFAULT 0.2 (evidence-based);
    # set 1.0 to restore legacy weights (verified net-worse on every recall metric).
    rrf_kg_path_weight_scale: float = Field(0.2, validation_alias="RRF_KG_PATH_WEIGHT_SCALE")

    # VRAG Tier 3: Entity-cosine cross-document retrieval path.
    # Lazy entity vector centroids + L1 TF-IDF + L3 MMR + L5 chunk scope.
    # First query for an entity is slow (~200-500ms cold); cached after.
    entity_cosine_enabled: bool = False
    entity_cosine_top_k_entities: int = 20
    entity_cosine_mmr_lambda: float = 0.6

    # Cross-doc entity resolution soft-fold. Embedding-confirmed near-dup
    # merge: high-precision lexical proposes (case/punct/diacritic/possessive
    # variants + acronym, e.g. 'large language model'/'LLM'), the entity centroid
    # cosine disposes — WITHOUT the context-centroid trap of merging distinct
    # entities that merely co-occur. Validated on corpus500 (dry-run): the naive
    # shared-token rule over-merged 'X University'/'Y University' etc.; removed.
    # Creates (alias)-[:ALIAS_OF]->(canon): SOFT-fold — both nodes + their
    # CONTAINS_ENTITY persist; PPR/entity_pivot collapse aliases at query time
    # (no node-count reduction — that needs a separate hard merge). Backfill/repair
    # step, OFF by default.
    entity_resolution_enabled: bool = Field(False, validation_alias="ENTITY_RESOLUTION_ENABLED")
    # Centroid cosine floor to confirm a lexically-proposed alias. High on purpose:
    # centroids are context means, so a loose bar over-merges co-occurring entities.
    entity_resolution_threshold: float = Field(0.90, validation_alias="ENTITY_RESOLUTION_THRESHOLD")
    # LLM-judge for the cosine GRAY ZONE [threshold, judge_hi): measured on bench500
    # cosine 0.90 confirmed near-string-but-different pairs (llms->MLLMs, GPT-4_1->GPT-4,
    # FlowRL->GFlowRL). cos >= judge_hi auto-accepts; in-zone pairs need a light-LLM
    # "same entity? YES/NO"; judge error = no fold (fail-closed for precision).
    # Only judge_types are judged — PERSON stays cosine-only (citation variants are
    # inherently ambiguous; an LLM would thrash on 'Wang et al' forms). OFF default.
    entity_resolution_judge_enabled: bool = Field(
        False, validation_alias="ENTITY_RESOLUTION_JUDGE_ENABLED"
    )
    entity_resolution_judge_hi: float = Field(0.92, validation_alias="ENTITY_RESOLUTION_JUDGE_HI")
    entity_resolution_judge_types_csv: str = Field(
        "technology,product,organization", validation_alias="ENTITY_RESOLUTION_JUDGE_TYPES"
    )

    # Generation deliberation
    # Defaults tuned for gemma4:e4b CPU/Metal speed. Enable richer modes via env
    # when on GPU. Each draft is 1 LLM call; judge = 1 more; outline = 1 more.
    generation_outline_enabled: bool = False
    generation_drafts: int = 1
    generation_judge_enabled: bool = False
    generation_refine_enabled: bool = False
    generation_max_tokens: int = 1024

    # Validation gates
    validation_enabled: bool = True
    validation_min_grounded_ratio: float = 0.70
    validation_max_invalid_entities: int = 3
    # Tier 2: tightened from 0.40 to 0.70 to enforce per-sentence citations strictly.
    # Refusal answers exempt (citation_gate skips them via is_refusal_answer check).
    validation_min_citation_ratio: float = 0.70
    validation_retry_on_fail: bool = True
    # Grounding gate: cosine (deterministic, bge-m3 embed of answer-sentences vs
    # retrieved chunks) vs the legacy LLM claim-extract+verify (up to 6 gemma calls,
    # ~25s/answer). Cosine is default — calibrated sim_hi=0.60/sim_lo=0.40 on the live
    # stack: grounded sentences score 0.72-0.93, hallucinated 0.30-0.35, so it tracks
    # the LLM verdict (mean |Δ|=0.04) at ~0.7s. Set false to fall back to the LLM gate.
    validation_cosine_grounding_enabled: bool = Field(
        True, validation_alias="VALIDATION_COSINE_GROUNDING"
    )
    validation_grounding_sim_hi: float = 0.60
    validation_grounding_sim_lo: float = 0.40

    # Community summaries
    community_enabled: bool = False  # Phase 7 — bật sau khi có data đủ lớn
    community_levels: int = 3
    community_resolution: float = 1.0
    community_min_size: int = 3
    community_summary_vote_passes: int = 3
    # Entity types (lowercase, csv) dropped from the community graph before clustering.
    # Measured (bench500, 2026-07-19): academic corpus = 59% PERSON entities (author names)
    # → one 9822-member blob community; excluding person,date restored thematic structure.
    # Default empty = no exclusion (backward-compatible).
    community_exclude_labels_csv: str = Field("", validation_alias="COMMUNITY_EXCLUDE_LABELS")

    # Global-query LazyGraphRAG (query-time map-reduce over communities)
    global_query_enabled: bool = Field(False, validation_alias="GLOBAL_QUERY_ENABLED")
    # Cap communities mapped per global query — latency scales with this × member-chunk fetch.
    # Corpora with large communities need a lower cap to stay under request timeouts.
    global_query_max_communities: int = Field(10, validation_alias="GLOBAL_QUERY_MAX_COMMUNITIES")

    # Refusal
    refusal_message_vi: str = (
        "Tôi không có đủ thông tin chắc chắn để trả lời câu hỏi này dựa trên tài liệu hiện có."
    )

    # OOD detection — early refusal before generation (Deprecated in favor of sufficient_context_gate)
    ood_detection_enabled: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
