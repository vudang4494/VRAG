"""Query type classifier for intelligent routing between retrieval strategies.

## Algorithm: Rule-Based Priority Classifier

1. **Out-of-domain guard** (highest priority): if query matches any `_OUT_OF_DOMAIN_PATTERNS` regex → immediate refusal
2. **Multi-hop** (high priority): if matches any `_MULTI_HOP_PATTERNS` regex → ReAct loop (cross-doc reasoning)
3. **Summarization**: if matches `_SUMMARIZATION_PATTERNS` → ReAct loop (aggregate)
4. **Analytical** (lowest): if matches `_ANALYTICAL_PATTERNS` → ReAct loop (causal/why questions)
5. **Default**: factual → standard retrieval (single-doc, fast)

Why rule-based instead of LLM-as-classifier:
  - Zero latency (no LLM call per query)
  - Deterministic, reproducible
  - Easy to audit which pattern triggered which route
  - Trade-off: hard to cover all Vietnamese/English surface forms

Trade-offs per approach:
  Rule-based (current): Fast, auditable, requires pattern maintenance
  LLM classifier (Phase X): More coverage, but +200-500ms latency + cost
  Trained classifier (Phase X): Best accuracy, needs labeled training data

Priority order is CRITICAL: patterns are checked in exact sequence above.
More specific patterns should appear before general ones within each group.

Usage:
    from src.services.query_router import classify_query, should_use_react

    query_type = classify_query("GraphRAG là gì?")
    use_react = should_use_react(query_type)
"""

from __future__ import annotations

import re

_MULTI_HOP_PATTERNS = [
    # Explicit cross-doc signals
    r"so sánh",
    r"khác nhau ra sao",
    r"giống và khác",
    r"mối liên hệ",
    r"cả \w+ và \w+",
    # Cross-doc reasoning signals
    r"(?i)và \w+ đều",
    r"(?i)giữa \w+ và \w+",
    r"(?i)\w+ vs\.?\s*\w+",
    r"(?i)\w+ versus \w+",
    # Comparison question patterns (critical: "X khác Y ở điểm nào?" is multi-hop)
    r"khác.*ở điểm nào",
    r"ở đâu khác",
    r"như thế nào.*khác",
    r"khác nhau",
    # Multi-step reasoning
    r"(?i)cả hai",
    r"(?i)trước tiên.*sau đó",
    r"(?i)vì.*nên",
    # Aggregate / summary
    r"tóm tắt",
    r"tổng hợp",
    r"tổng quan",
    r"liệt kê",
    r"(?i)các kỹ thuật.*cải thiện",
    r"(?i)đóng góp chính",
    r"(?i)phương pháp huấn luyện",
    # "Which paper/doc mentions X" — needs graph walk
    r"paper nào",
    r"bài báo nào",
    r"doc.*nào",
    r"document.*nào",
    r"(?i)(graphrag|knowledge graph|leiden|community detection|đồ thị tri thức)",
    # Explicit multi-hop question structures (X và Y cùng/tất cả đều)
    r"(?i)(X và Y|[A-ZÀ-Ỹ]\w+ và [A-ZÀ-Ỹ]\w+) (đều|dùng|cùng|liên quan)",
    # General multi-hop signals (more permissive)
    r"(?i)(đều|mọi người|cùng) (dùng|sử dụng|có liên quan)",
    # Pattern: "X và Y" — two entities in same sentence often means cross-doc
    r"\w+\s+\w+\s+và\s+\w+",
    # "hoạt động thông qua" — operational/mechanism questions
    r"hoạt động thông qua",
    r"thông qua việc",
    # "sự khác biệt" or "khác biệt" alone
    r"sự khác biệt",
    # "cái gì" after mentioning two things
    r"gì$",
    # "cái nào" — comparison/choice questions → multi-hop ReAct
    r"cái nào",
    r"nào.*hơn",
    r"nào.*tốt hơn",
    r"nào.*hiệu quả hơn",
    r"nào.*đạt",
    r"đánh giá.*nào",
    # "đều là" — multiple things with shared property
    r"đều là",
    # "liên quan" — relationship questions
    r"liên quan",
    # "công trình" — research papers
    r"công trình",
    r"tác giả",
]

_SUMMARIZATION_PATTERNS = [
    r"tóm tắt",
    r"tổng hợp",
    r"tổng quan",
    r"liệt kê",
    r"kể",
    r"mô tả các",
    r"(?i)what are the main",
    r"(?i)overview of",
]

_ANALYTICAL_PATTERNS = [
    r"tại sao",
    r"vì sao",
    r"tại sao",
    r"bằng cách nào",
    r"như thế nào",
    r"hoạt động thế nào",
    r"có vai trò gì",
    r"(?i)why does",
    r"(?i)how does",
    r"(?i)what is the role",
    r"(?i)what causes",
]

_OUT_OF_DOMAIN_PATTERNS = [
    # Real-world queries not in academic corpus
    r"thời tiết",
    r"bitcoin",
    r"giá .*hôm nay",
    r"nấu (phở|canh|bún)",
    r"tin tức",
    r"(?i)news today",
    r"(?i)weather",
    r"(?i)stock price",
    r"(?i)cook (pho|soup|recipe)",
    r"(?i)news",
    # Explicit non-RAG topics
    r"bóng đá",
    r"(?i)sport",
    r"(?i)football",
    r"(?i)game",
]


def classify_query(query: str) -> str:
    """
    Classify query type using lightweight heuristics.
    Returns one of: factual | multi_hop | summarization | analytical | out_of_domain
    """
    q = query.strip()

    # Check out-of-domain first (highest priority)
    for pat in _OUT_OF_DOMAIN_PATTERNS:
        if re.search(pat, q):
            return "out_of_domain"

    # Check multi-hop (cross-doc reasoning)
    for pat in _MULTI_HOP_PATTERNS:
        if re.search(pat, q):
            return "multi_hop"

    # Check summarization
    for pat in _SUMMARIZATION_PATTERNS:
        if re.search(pat, q):
            return "summarization"

    # Check analytical (why/how)
    for pat in _ANALYTICAL_PATTERNS:
        if re.search(pat, q):
            return "analytical"

    # Default: factual (single-doc lookup, definition, entity query)
    return "factual"


def should_use_react(query_type: str) -> bool:
    """
    Decide whether to route to ReAct loop based on query type.

    ReAct is beneficial for:
      - multi_hop: cross-doc reasoning
      - summarization: aggregate across docs
      - analytical: causal/why questions benefit from step-by-step reasoning

    ReAct is NOT needed for:
      - factual: single-doc lookup, direct answer
      - out_of_domain: short-circuit early
    """
    return query_type in ("multi_hop", "summarization", "analytical")


def describe_routing(query_type: str, use_react: bool) -> str:
    """Human-readable description of routing decision."""
    if use_react:
        return f"ReAct loop ({query_type} query — cross-doc reasoning)"
    return f"Standard retrieval ({query_type} query — single-doc lookup)"
