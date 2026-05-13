"""Query type classifier for intelligent routing between retrieval strategies.

Simple heuristic classifier (no LLM call needed) categorizes queries into:
  - factual: single-entity, single-doc, definition, lookup
  - multi_hop: cross-doc, comparison, requires reasoning over multiple sources
  - summarization: aggregate, overview, "tổng hợp", "tóm tắt"
  - analytical: why/how causal questions
  - out_of_domain: clearly not in corpus

These categories drive which retrieval pipeline is used:
  - factual → standard retrieval (fast, low-refusal)
  - multi_hop / summarization / analytical → ReAct loop (better cross-doc reasoning)
  - out_of_domain → short-circuit with refusal

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
    # "Which paper/doc mentions X" — needs graph walk
    r"paper nào",
    r"bài báo nào",
    r"doc.*nào",
    r"document.*nào",
    # Definition-style questions about RAG/graph/AI concepts — require KG context
    r"(?i)\w+(là|gì|cái gì|định nghĩa)",  # "X là gì", "X là cái gì"
    # Any query explicitly about GraphRAG, knowledge graph, community — by nature multi-hop
    r"(?i)(graphrag|knowledge graph|leiden|community detection|đồ thị tri thức)",
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
