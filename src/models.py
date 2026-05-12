"""Pydantic models for multi-tenant, multi-source RAG."""
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    MARKDOWN = "markdown"
    WEBPAGE = "webpage"
    NOTION = "notion"
    SLACK = "slack"
    DISCORD = "discord"
    EMAIL = "email"
    GMAIL = "gmail"
    GITHUB = "github"
    ARXIV = "arxiv"
    DATABASE = "database"
    API = "api"
    YOUTUBE = "youtube"
    YOUTRANSCRIPT = "youtube_transcript"
    CSV = "csv"
    EXCEL = "excel"
    ZENDESK = "zendesk"
    SALESFORCE = "salesforce"
    HUBSPOT = "hubspot"
    CUSTOM = "custom"


class SourceStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    DISABLED = "disabled"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"
    DELETED = "deleted"


class TenantStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class AccessLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ChunkStrategy(str, Enum):
    FIXED = "fixed"          # Fixed-size with overlap
    SENTENCE = "sentence"    # Sentence-aware splitting
    PARAGRAPH = "paragraph" # Paragraph boundaries
    SEMANTIC = "semantic"    # LLM-based semantic chunks
    HIERARCHICAL = "hierarchical"  # Hierarchical (h1/h2/h3)


# ---------------------------------------------------------------------------
# Tenant
# ---------------------------------------------------------------------------

class TenantBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=128)
    slug: str = Field(..., min_length=2, max_length=64, pattern=r"^[a-z0-9-]+$")
    description: str | None = Field(default=None, max_length=512)
    logo_url: str | None = None
    default_chunk_strategy: ChunkStrategy = ChunkStrategy.SENTENCE
    default_chunk_size: int = Field(default=512, ge=128, le=4096)
    default_chunk_overlap: int = Field(default=64, ge=0, le=512)
    retrieval_top_k: int = Field(default=8, ge=1, le=100)
    vector_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    graph_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    enable_semantic_cache: bool = Field(default=True)
    semantic_cache_ttl_s: int = Field(default=3600, ge=60, le=86400)
    max_concurrent_requests: int = Field(default=8, ge=1, le=64)
    custom_llm_model: str | None = None
    custom_embed_model: str | None = None
    metadata_: dict[str, Any] = Field(default_factory=dict, alias="metadata")


class TenantCreate(TenantBase):
    owner_email: str
    plan: str = Field(default="free", pattern=r"^(free|pro|enterprise)$")


class TenantUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=128)
    description: str | None = Field(default=None, max_length=512)
    logo_url: str | None = None
    status: TenantStatus | None = None
    default_chunk_strategy: ChunkStrategy | None = None
    default_chunk_size: int | None = Field(default=None, ge=128, le=4096)
    default_chunk_overlap: int | None = Field(default=None, ge=0, le=512)
    retrieval_top_k: int | None = Field(default=None, ge=1, le=100)
    vector_weight: float | None = Field(default=None, ge=0.0, le=5.0)
    graph_weight: float | None = Field(default=None, ge=0.0, le=5.0)
    enable_semantic_cache: bool | None = None
    semantic_cache_ttl_s: int | None = Field(default=None, ge=60, le=86400)
    max_concurrent_requests: int | None = Field(default=None, ge=1, le=64)
    custom_llm_model: str | None = None
    custom_embed_model: str | None = None


class Tenant(TenantBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    slug: str
    status: TenantStatus = TenantStatus.ACTIVE
    plan: str = "free"
    created_at: datetime
    updated_at: datetime
    source_count: int = 0
    document_count: int = 0
    chunk_count: int = 0


# ---------------------------------------------------------------------------
# User / API Key
# ---------------------------------------------------------------------------

class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=128)
    scopes: list[str] = Field(default_factory=lambda: ["chat", "ingest", "read"])
    expires_at: datetime | None = None
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10000)


class ApiKey(BaseModel):
    id: str
    tenant_id: str
    name: str
    key_hash: str
    scopes: list[str]
    rate_limit_per_minute: int
    created_at: datetime
    expires_at: datetime | None
    last_used_at: datetime | None
    is_active: bool


# ---------------------------------------------------------------------------
# Source (data source / plugin config)
# ---------------------------------------------------------------------------

class SourceBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=256)
    source_type: SourceType
    description: str | None = Field(default=None, max_length=1024)
    status: SourceStatus = SourceStatus.ACTIVE
    access_level: AccessLevel = AccessLevel.INTERNAL
    tags: list[str] = Field(default_factory=list)
    schedule_cron: str | None = Field(default=None, description="Cron expression for scheduled sync")
    is_recurring: bool = False
    crawl_depth: int = Field(default=1, ge=1, le=10)
    filters: dict[str, Any] = Field(default_factory=dict)
    custom_config: dict[str, Any] = Field(default_factory=dict)


class SourceCreate(SourceBase):
    tenant_id: str
    credentials: dict[str, str] | None = Field(default=None, exclude=True)


class SourceUpdate(BaseModel):
    name: str | None = None
    status: SourceStatus | None = None
    access_level: AccessLevel | None = None
    tags: list[str] | None = None
    schedule_cron: str | None = None
    is_recurring: bool | None = None
    crawl_depth: int | None = None
    filters: dict[str, Any] | None = None
    custom_config: dict[str, Any] | None = None


class SourceCredentials(BaseModel):
    """Credentials stored encrypted. Never returned in API responses."""
    encrypted_blob: str  # Fernet-encrypted JSON of credentials


class Source(SourceBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    tenant_id: str
    credentials_id: str | None
    last_sync_at: datetime | None
    last_sync_status: str | None
    document_count: int = 0
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class DocumentBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=512)
    access_level: AccessLevel = AccessLevel.INTERNAL
    tags: list[str] = Field(default_factory=list)
    department: str | None = Field(default=None, max_length=128)
    author: str | None = Field(default=None, max_length=256)
    created_date: datetime | None = None
    metadata_: dict[str, Any] = Field(default_factory=dict, alias="metadata")


class DocumentIngest(DocumentBase):
    source_id: str
    file_content: bytes | None = None  # uploaded file
    file_url: str | None = None       # URL for web/API sources
    chunk_strategy: ChunkStrategy | None = None
    chunk_size: int | None = Field(default=None, ge=128, le=4096)
    chunk_overlap: int | None = Field(default=None, ge=0, le=512)


class Document(DocumentBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    tenant_id: str
    source_id: str
    doc_hash: str
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    entity_count: int = 0
    relationship_count: int = 0
    file_size_bytes: int = 0
    file_type: str | None = None
    error_message: str | None = None
    indexed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[Document]
    total: int
    page: int
    page_size: int
    has_more: bool


class Chunk(BaseModel):
    id: str
    tenant_id: str
    document_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    vector_ids: dict[str, str] = Field(default_factory=dict)  # {vector_type: qdrant_id}


class Entity(BaseModel):
    id: str
    tenant_id: str
    name: str
    type: str
    description: str | None
    source_chunk_ids: list[str]
    entity_count: int = 1
    created_at: datetime


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class RetrievalFilters(BaseModel):
    source_ids: list[str] | None = None
    tags: list[str] | None = None
    departments: list[str] | None = None
    authors: list[str] | None = None
    access_levels: list[AccessLevel] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    document_ids: list[str] | None = None
    exclude_document_ids: list[str] | None = None


class RetrievalResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    retrieval_modes: list[str]
    source: str
    document_id: str
    document_title: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    entities: list[dict[str, str]] = Field(default_factory=list)
    rerank_score: float | None = None


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievalResult]
    total_results: int
    retrieval_time_ms: float
    cache_hit: bool = False


# ---------------------------------------------------------------------------
# Chat / RAG
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="qwen3.5:4b")
    messages: list[ChatMessage]
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.3
    max_tokens: Annotated[int, Field(ge=1, le=16384)] = 2048
    top_p: Annotated[float, Field(ge=0.0, le=1.0)] = 0.9
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    filters: RetrievalFilters | None = None
    include_sources: bool = True
    include_graph_context: bool = True
    system_prompt_override: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict
    sources: list[RetrievalResult] | None = None
    reasoning: str | None = None


# ---------------------------------------------------------------------------
# Ingestion Job
# ---------------------------------------------------------------------------

class IngestJobResponse(BaseModel):
    job_id: str
    tenant_id: str
    source_id: str | None
    document_id: str
    status: DocumentStatus
    total_chunks: int = 0
    indexed_chunks: int = 0
    failed_chunks: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    error_message: str | None = None
    started_at: datetime
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Statistics / Metrics
# ---------------------------------------------------------------------------

class TenantStats(BaseModel):
    tenant_id: str
    total_sources: int
    active_sources: int
    total_documents: int
    documents_by_status: dict[str, int]
    total_chunks: int
    total_entities: int
    total_relationships: int
    total_api_calls: int
    cache_hit_rate: float
    avg_retrieval_time_ms: float
    avg_generation_time_ms: float
    storage_used_mb: float


class SystemStats(BaseModel):
    total_tenants: int
    active_tenants: int
    total_documents: int
    total_chunks: int
    total_entities: int
    total_api_calls_today: int
    system_health: str
    uptime_seconds: float


class HealthResponse(BaseModel):
    """Health check response format."""
    status: str
    checks: dict[str, Any]

class ServiceCheck(BaseModel):
    status: str
    detail: str | None = None
    models: list[str] | None = None
    collections: int | None = None

class IngestResponse(BaseModel):
    status: str
    filename: str
    doc_hash: str
    chunks_indexed: int
    entities_extracted: int
    relationships_extracted: int
    failed_chunks: int

class ModelList(BaseModel):
    data: list[dict[str, str]]


# ============================================================================
# Pipeline V2 — Quality-first GraphRAG
# ============================================================================


class ChunkLevel(str, Enum):
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SECTION = "section"
    DOCUMENT = "document"


class ViewType(str, Enum):
    DENSE = "dense"
    PARAPHRASE = "paraphrase"
    QUESTION = "question"
    SUMMARY = "summary"
    KEYWORDS = "keywords"


class QueryIntent(str, Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"


class RelationType(str, Enum):
    IS_A = "IS_A"
    PART_OF = "PART_OF"
    WORKS_FOR = "WORKS_FOR"
    LOCATED_IN = "LOCATED_IN"
    CAUSES = "CAUSES"
    OWNS = "OWNS"
    MEMBER_OF = "MEMBER_OF"
    PRODUCES = "PRODUCES"
    PRECEDES = "PRECEDES"
    OTHER = "OTHER"


class ChunkV2(BaseModel):
    """V2 chunk with hierarchical, multi-view, consistency-scored shape."""
    id: str
    tenant_id: str
    document_id: str
    text: str
    chunk_index: int
    chunk_level: ChunkLevel
    parent_chunk_id: str | None = None
    start_char: int = 0
    end_char: int = 0
    consistency_score: float = 0.0
    views: dict[str, str] = Field(default_factory=dict)
    view_embeddings: dict[str, list[float]] = Field(default_factory=dict)
    format: str = "unknown"
    page_num: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    sheet_name: str | None = None
    column_names: list[str] | None = None
    row_range: tuple[int, int] | None = None
    speaker: str | None = None
    timestamp: datetime | None = None
    thread_id: str | None = None
    pii_mask_map_id: str | None = None
    access_level: AccessLevel = AccessLevel.INTERNAL


class CommunityNode(BaseModel):
    id: str
    tenant_id: str
    level: int = Field(ge=0, le=5)
    summary: str
    member_count: int
    member_entity_names: list[str] = Field(default_factory=list)
    parent_community_id: str | None = None
    generated_at: datetime
    summary_vote_count: int = 1


class QueryReformulation(BaseModel):
    kind: Literal["original", "rewrite", "decompose", "hyde", "step_back", "keywords"]
    text: str
    weight: float = 1.0


class QueryUnderstanding(BaseModel):
    original: str
    reformulations: list[QueryReformulation]
    intent: QueryIntent
    intent_confidence: float = 0.0
    is_multi_hop: bool = False


class RetrievalCandidate(BaseModel):
    chunk_id: str
    text: str
    source: str
    score: float
    retrieval_path: str
    consistency_score: float = 0.7
    chunk_level: ChunkLevel = ChunkLevel.PARAGRAPH
    metadata: dict[str, Any] = Field(default_factory=dict)


class RerankedCandidate(BaseModel):
    chunk_id: str
    text: str
    source: str
    final_score: float
    stage1_score: float = 0.0
    stage2_score: float = 0.0
    stage3_score: float = 0.0
    judge_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    passed: bool
    grounded_ratio: float = 0.0
    invalid_entities: list[str] = Field(default_factory=list)
    citation_ratio: float = 0.0
    failure_reason: str | None = None
    confidence: float = 0.0


class ChatCompletionV3Response(BaseModel):
    id: str
    created: int
    model: str
    answer: str
    sources: list[RerankedCandidate]
    intent: QueryIntent
    confidence: float
    validation: ValidationResult
    latency_breakdown_ms: dict[str, float]
    trace_id: str | None = None
    refused: bool = False
    refusal_reason: str | None = None

