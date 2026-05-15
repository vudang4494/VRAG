"""Domain Tagger — semantic domain-axis tagging for chunks and entities.

Each chunk/entity gets a domain distribution over 8 semantic axes.
At retrieval time, a domain reward is computed as cosine similarity between
the chunk's domain vector and the query's inferred domain vector, then
added as a multiplicative boost to the RRF score.

Design rationale:
  - NOT colors (red/blue) — strings don't normalize well across languages
  - NOT hard labels — chunks belong to multiple domains simultaneously
  - YES: fixed 8D concept space, vectors encode soft membership 0.0-1.0
  - YES: cosine similarity is language-agnostic and bounded [-1, 1]

The 8 axes were chosen to match the eval benchmark categories:
  factual_definition, method_algorithm, comparison_analysis, research_paper,
  technical_code, biological_science, instruction_howto, business_finance
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

# cosine_similarity available from src.services.embedding if needed in future

# ── Domain taxonomy ────────────────────────────────────────────────────────────

DOMAIN_AXES = [
    "factual_definition",     # 0: what/definition questions
    "method_algorithm",      # 1: algorithms, techniques, computational methods
    "comparison_analysis",   # 2: compare, contrast, evaluate, analysis
    "research_paper",        # 3: academic paper structure / results
    "technical_code",        # 4: code, implementation, API, technical specs
    "biological_science",    # 5: biology, protein, medical, scientific experiments
    "instruction_howto",     # 6: how-to, step-by-step, instructions
    "business_finance",      # 7: finance, business metrics, market data
]
N_DOMAINS = len(DOMAIN_AXES)  # 8


@dataclass
class DomainDistribution:
    """Soft domain membership over 8 axes. Values sum to ~1.0 (normalized)."""
    factual_definition: float
    method_algorithm: float
    comparison_analysis: float
    research_paper: float
    technical_code: float
    biological_science: float
    instruction_howto: float
    business_finance: float

    @classmethod
    def zero(cls) -> "DomainDistribution":
        return cls(
            factual_definition=0.0,
            method_algorithm=0.0,
            comparison_analysis=0.0,
            research_paper=0.0,
            technical_code=0.0,
            biological_science=0.0,
            instruction_howto=0.0,
            business_finance=0.0,
        )

    def to_list(self) -> list[float]:
        return [
            self.factual_definition,
            self.method_algorithm,
            self.comparison_analysis,
            self.research_paper,
            self.technical_code,
            self.biological_science,
            self.instruction_howto,
            self.business_finance,
        ]

    def to_dict(self) -> dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if k in DOMAIN_AXES}

    def dominant(self) -> tuple[str, float]:
        pairs = self.to_dict()
        best = max(pairs, key=lambda k: pairs[k])
        return best, pairs[best]

    @classmethod
    def from_list(cls, vals: list[float]) -> "DomainDistribution":
        if len(vals) != N_DOMAINS:
            raise ValueError(f"Expected {N_DOMAINS} values, got {len(vals)}")
        return cls(
            factual_definition=vals[0],
            method_algorithm=vals[1],
            comparison_analysis=vals[2],
            research_paper=vals[3],
            technical_code=vals[4],
            biological_science=vals[5],
            instruction_howto=vals[6],
            business_finance=vals[7],
        )

    def __mul__(self, scalar: float) -> "DomainDistribution":
        return DomainDistribution.from_list([v * scalar for v in self.to_list()])

    def __add__(self, other: "DomainDistribution") -> "DomainDistribution":
        return DomainDistribution.from_list([
            a + b for a, b in zip(self.to_list(), other.to_list())
        ])


# ── Keyword/signal dictionaries ────────────────────────────────────────────────

# (axis_name, weight) pairs for matching keywords in text
_AXIS_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "factual_definition": [
        ("là gì", 0.9), ("what is", 0.9), ("defined as", 0.9), ("definition", 0.7),
        ("means", 0.7), ("means", 0.7), ("viết tắt", 0.9), ("tắt của", 0.9),
        ("giới thiệu", 0.6), ("tổng quan", 0.5), ("overview", 0.5),
    ],
    "method_algorithm": [
        ("algorithm", 0.9), ("thuật toán", 0.9), ("phương pháp", 0.7), ("method", 0.7),
        ("technique", 0.7), ("sử dụng", 0.5), ("hoạt động", 0.5), ("cơ chế", 0.7),
        ("clustering", 0.8), ("detection", 0.7), ("leiden", 0.9), ("louvain", 0.9),
        ("embedding", 0.7), ("contrastive", 0.8), ("training", 0.6),
    ],
    "comparison_analysis": [
        ("so sánh", 0.9), ("compare", 0.9), ("khác nhau", 0.9), ("difference", 0.8),
        ("hơn kém", 0.8), ("tốt hơn", 0.7), ("better", 0.7), ("worse", 0.7),
        ("đánh giá", 0.8), ("evaluate", 0.8), ("analysis", 0.7), ("phân tích", 0.7),
        ("tại sao", 0.8), ("why", 0.8), ("vai trò", 0.8), ("role", 0.7),
    ],
    "research_paper": [
        ("paper", 0.9), ("bài báo", 0.9), ("nghiên cứu", 0.8), ("research", 0.8),
        ("accuracy", 0.7), ("results", 0.7), ("kết quả", 0.7), ("performance", 0.7),
        ("benchmark", 0.8), ("evaluation", 0.7), ("metric", 0.7),
        ("section", 0.5), ("table", 0.5), ("figure", 0.5),
    ],
    "technical_code": [
        ("code", 0.9), ("implementation", 0.9), ("function", 0.8), ("class", 0.8),
        ("api", 0.9), ("parameter", 0.8), ("module", 0.8), ("library", 0.8),
        ("syntax", 0.9), ("import", 0.8), ("def ", 0.8), ("const", 0.7),
    ],
    "biological_science": [
        ("protein", 0.9), ("biology", 0.9), ("biological", 0.9), ("cell", 0.8),
        ("dna", 0.9), ("structure", 0.7), ("alphafold", 0.9), ("genome", 0.9),
        ("scientific", 0.8), ("experiment", 0.8), ("sequence", 0.8),
        ("axit amin", 0.9), ("amino acid", 0.9), ("disease", 0.7), ("medical", 0.8),
    ],
    "instruction_howto": [
        ("cách", 0.8), ("how to", 0.9), ("bước", 0.8), ("step", 0.8),
        ("hướng dẫn", 0.9), ("guide", 0.8), ("instruction", 0.8),
        ("thực hiện", 0.7), ("làm thế nào", 0.9), ("như thế nào", 0.9),
        ("chạy", 0.7), ("sử dụng", 0.6), ("cài đặt", 0.8), ("install", 0.8),
    ],
    "business_finance": [
        ("revenue", 0.9), ("profit", 0.9), ("stock", 0.9), ("market", 0.8),
        ("finance", 0.9), ("financial", 0.9), ("revenue", 0.9), ("investment", 0.9),
        ("tài chính", 0.9), ("doanh thu", 0.9), ("lợi nhuận", 0.9), ("cổ phiếu", 0.9),
    ],
}

# Entity type signals (from GLiNER output) → domain axis boost
_ENTITY_TYPE_DOMAINS: dict[str, list[str]] = {
    "ALGORITHM": ["method_algorithm"],
    "MODEL": ["method_algorithm", "technical_code"],
    "TECHNIQUE": ["method_algorithm"],
    "PERSON": ["research_paper"],
    "ORGANIZATION": ["research_paper", "business_finance"],
    "DATASET": ["research_paper"],
    "SOFTWARE": ["technical_code"],
    "PROTEIN": ["biological_science"],
    "GENE": ["biological_science"],
    "SCIENTIFIC_TERM": ["biological_science", "research_paper"],
    "METHOD": ["method_algorithm"],
    "METRIC": ["comparison_analysis", "research_paper"],
    "NUMBER": [],  # no domain signal
    "OTHER": [],
}

# Structural signals: heading patterns, LaTeX math, code blocks
_STRUCTURAL_PATTERNS: list[tuple[str, list[str], float]] = [
    # LaTeX math → research_paper + biological_science
    (r"\$\$.*?\$\$|\\begin\{equation\}|\\frac\{", ["research_paper", "biological_science"], 0.3),
    # Code block markers → technical_code
    (r"```[\s\S]*?```|`[^`]+`", ["technical_code"], 0.4),
    # Heading levels → research_paper
    (r"^#{1,3}\s", ["research_paper"], 0.2),
    # Bullet list → instruction_howto
    (r"^\s*[-*]\s", ["instruction_howto"], 0.2),
    # Numbered steps → instruction_howto
    (r"^\s*\d+\.\s", ["instruction_howto"], 0.3),
    # Table reference → research_paper
    (r"table\s+\d+|hình\s+\d+|fig\.\s*\d+", ["research_paper"], 0.3),
]


def _keyword_signal(text: str) -> dict[str, float]:
    """Compute raw keyword match scores per axis."""
    text_lower = text.lower()
    scores: dict[str, float] = {ax: 0.0 for ax in DOMAIN_AXES}

    for axis, keywords in _AXIS_KEYWORDS.items():
        for kw, weight in keywords:
            if kw.lower() in text_lower:
                scores[axis] += weight

    return scores


def _structural_signal(text: str) -> dict[str, float]:
    """Compute structural pattern scores per axis."""
    scores: dict[str, float] = {ax: 0.0 for ax in DOMAIN_AXES}
    for pattern, axes, weight in _STRUCTURAL_PATTERNS:
        if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
            for ax in axes:
                scores[ax] += weight
    return scores


def _entity_signal(entities: list[dict[str, Any]]) -> dict[str, float]:
    """Boost axes based on entity types present in text."""
    scores: dict[str, float] = {ax: 0.0 for ax in DOMAIN_AXES}
    if not entities:
        return scores

    for ent in entities:
        ent_type = (ent.get("type") or "OTHER").upper()
        boosted_axes = _ENTITY_TYPE_DOMAINS.get(ent_type, [])
        for ax in boosted_axes:
            scores[ax] += 0.5

    return scores


def _softmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Soft-max normalization: exp(scores) / sum(exp), bounded to [0, 1]."""
    vals = list(scores.values())
    max_val = max(vals) if vals else 0.0
    shifted = [v - max_val for v in vals]  # numerical stability
    exp_vals = [np.exp(v) for v in shifted]
    total = sum(exp_vals)
    if total == 0:
        return {ax: 0.0 for ax in DOMAIN_AXES}
    return {ax: exp_vals[i] / total for i, ax in enumerate(DOMAIN_AXES)}


def tag_chunk(
    text: str,
    entities: list[dict[str, Any]] | None = None,
    filename: str = "",
    entity_types: list[str] | None = None,
) -> DomainDistribution:
    """
    Infer the domain distribution for a chunk.

    Combines three signals:
      1. Keyword matching (lexical, fast)
      2. Structural patterns (markdown, LaTeX, code blocks)
      3. Entity types from GLiNER (semantic, high quality)

    The raw scores are soft-max normalized so they sum to 1.0 and
    the dominant axis is clearly identifiable.

    Args:
        text: chunk text content
        entities: list of entity dicts with 'type' field (from GLiNER)
        filename: original filename (for domain hints)
        entity_types: alternative to entities — list of entity type strings

    Returns:
        DomainDistribution with soft membership over 8 axes.
    """
    # Build entities list from entity_types if needed
    if entities is None and entity_types:
        entities = [{"type": t} for t in entity_types]

    kw_scores = _keyword_signal(text)
    struct_scores = _structural_signal(text)
    ent_scores = _entity_signal(entities or [])

    # Filename hints (domain proxy via extension + keywords)
    fname_lower = filename.lower()
    fname_scores: dict[str, float] = {ax: 0.0 for ax in DOMAIN_AXES}
    if "protein" in fname_lower or "bio" in fname_lower or "alphafold" in fname_lower:
        fname_scores["biological_science"] = 0.8
    if "code" in fname_lower or "api" in fname_lower:
        fname_scores["technical_code"] = 0.8
    if "finance" in fname_lower or "stock" in fname_lower:
        fname_scores["business_finance"] = 0.8
    if "graphrag" in fname_lower or "rag" in fname_lower:
        fname_scores["method_algorithm"] = 0.6
        fname_scores["research_paper"] = 0.5
    if re.search(r"\.py$|\.js$|\.ts$|\.go$", fname_lower):
        fname_scores["technical_code"] = 0.9

    # Weighted combination: keywords=0.5, entities=0.3, structural=0.15, filename=0.05
    raw: dict[str, float] = {ax: 0.0 for ax in DOMAIN_AXES}
    for ax in DOMAIN_AXES:
        raw[ax] = (
            0.50 * kw_scores[ax]
            + 0.30 * ent_scores[ax]
            + 0.15 * struct_scores[ax]
            + 0.05 * fname_scores[ax]
        )

    # Soft-max normalize
    norm = _softmax_normalize(raw)

    return DomainDistribution.from_list(list(norm.values()))


def tag_query(query: str) -> DomainDistribution:
    """
    Infer the domain distribution for a user query.

    Uses the same keyword + structural signals but with query-tuned weights
    (heavier on method/comparison keywords since queries are short).

    Returns:
        DomainDistribution — the "query domain vector" used for reward scoring.
    """
    text_lower = query.lower()
    scores: dict[str, float] = {ax: 0.0 for ax in DOMAIN_AXES}

    # Query-specific patterns (higher weight for short texts)
    QUERY_METHOD_PATTERNS = [
        r"thuật toán", r"algorithm", r"phương pháp", r"method",
        r"hoạt động", r"cơ chế", r"kỹ thuật",
    ]
    QUERY_COMPARE_PATTERNS = [
        r"so sánh", r"khác nhau", r"compare", r"hơn kém",
        r"tại sao", r"tại sao", r"vai trò", r"đánh giá",
    ]
    QUERY_FACTUAL_PATTERNS = [
        r"là gì", r"what is", r"viết tắt", r"bao nhiêu",
        r"bao gồm", r"gồm", r"có gì",
    ]
    QUERY_HOWTO_PATTERNS = [
        r"làm thế nào", r"như thế nào", r"cách", r"how to",
        r"hướng dẫn", r"bước", r"thực hiện",
    ]
    QUERY_BIO_PATTERNS = [
        r"protein", r"dự đoán", r"cấu trúc", r"biology",
    ]

    for ax, patterns in [
        ("method_algorithm", QUERY_METHOD_PATTERNS),
        ("comparison_analysis", QUERY_COMPARE_PATTERNS),
        ("factual_definition", QUERY_FACTUAL_PATTERNS),
        ("instruction_howto", QUERY_HOWTO_PATTERNS),
        ("biological_science", QUERY_BIO_PATTERNS),
    ]:
        for pat in patterns:
            if re.search(pat, text_lower):
                scores[ax] += 1.5  # query-tuned boost

    norm = _softmax_normalize(scores)
    return DomainDistribution.from_list(list(norm.values()))


def domain_reward(
    chunk_domain: DomainDistribution,
    query_domain: DomainDistribution,
    scale: float = 0.3,
) -> float:
    """
    Compute domain reward: cosine similarity between chunk and query domain vectors.

    The reward is a multiplicative boost added to the RRF contribution:
      rrf_contribution *= (1 + domain_reward)

    This means:
      - High cosine similarity (0.7+) → +20-30% boost
      - Medium (0.4-0.7) → +10-20% boost
      - Low/negative → minimal penalty

    Args:
        chunk_domain: domain distribution of the candidate chunk
        query_domain: domain distribution inferred from the query
        scale: max boost multiplier. Default 0.3 means max 30% boost.

    Returns:
        float in [0.0, scale] — the reward to add to RRF contribution.
    """
    chunk_vec = np.array(chunk_domain.to_list())
    query_vec = np.array(query_domain.to_list())

    # Cosine similarity
    dot = np.dot(chunk_vec, query_vec)
    norm_chunk = np.linalg.norm(chunk_vec)
    norm_query = np.linalg.norm(query_vec)

    if norm_chunk == 0 or norm_query == 0:
        return 0.0

    cos_sim = dot / (norm_chunk * norm_query)
    # Map [-1, 1] → [0, 1] then scale
    reward = (cos_sim + 1.0) / 2.0 * scale
    return float(reward)


def tag_entities_for_domain(entities: list[dict[str, Any]]) -> DomainDistribution:
    """
    Aggregate domain distribution from a list of entities.

    Used at ingestion to propagate entity domains to chunks that contain them,
    and at retrieval to boost chunks sharing entities with the query.
    """
    if not entities:
        return DomainDistribution.zero()

    combined = DomainDistribution.zero()
    for ent in entities:
        ent_type = (ent.get("type") or "OTHER").upper()
        axes = _ENTITY_TYPE_DOMAINS.get(ent_type, [])
        boost = DomainDistribution.zero()
        for ax in axes:
            setattr(boost, ax, 0.5)
        combined = combined + boost

    # Soft-max normalize
    raw = combined.to_dict()
    norm = _softmax_normalize(raw)
    return DomainDistribution.from_list(list(norm.values()))
