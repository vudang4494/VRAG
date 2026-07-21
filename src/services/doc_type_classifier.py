"""Phase 5b — Document type classifier + per-type chunking strategy.

Different doc types need different chunking config:
  research_paper:  respect section headings, ~800 chars
  report_financial: respect tables, ~600 chars, preserve number blocks
  chat_log:        respect turns, ~300 chars, preserve QA pairs
  legal_contract:  respect clause numbers, ~1200 chars, atomic clauses
  code:            respect function/class boundaries, ~500 chars
  generic:         default semantic chunking
"""

from __future__ import annotations

import re

# Type-specific chunking configs
TYPE_STRATEGIES: dict[str, dict] = {
    "research_paper": {
        "respect": ["section_heading", "abstract", "conclusion", "equation"],
        "chunk_size": 800,
        "preserve_atomic": ["abstract", "table_caption"],
        "boundary_threshold": 0.55,
    },
    "report_financial": {
        "respect": ["section", "table", "number_block"],
        "chunk_size": 600,
        "preserve_atomic": ["table", "number_block"],
        "boundary_threshold": 0.50,
    },
    "chat_log": {
        "respect": ["turn", "thread_id", "speaker_change"],
        "chunk_size": 300,
        "preserve_atomic": ["qa_pair"],
        "boundary_threshold": 0.40,
    },
    "legal_contract": {
        "respect": ["clause_number", "section"],
        "chunk_size": 1200,
        "preserve_atomic": ["clause"],
        "boundary_threshold": 0.70,
    },
    "code": {
        "respect": ["function_def", "class_def", "comment_block"],
        "chunk_size": 500,
        "preserve_atomic": ["function_body"],
        "boundary_threshold": 0.60,
    },
    "generic": {
        "respect": [],
        "chunk_size": 800,
        "preserve_atomic": [],
        "boundary_threshold": 0.55,
    },
}


# Heuristic regex signatures for doc type detection
_TYPE_SIGNATURES: dict[str, list[re.Pattern]] = {
    "research_paper": [
        re.compile(r"^#\s*Abstract", re.IGNORECASE | re.MULTILINE),
        re.compile(r"##\s*\d+\s*Introduction", re.IGNORECASE | re.MULTILINE),
        re.compile(r"\barXiv:\d{4}\.\d{4,5}", re.IGNORECASE),
        re.compile(r"\bReferences\s*\[\d+\]", re.IGNORECASE),
    ],
    "report_financial": [
        re.compile(r"\b(doanh thu|revenue|EBITDA|gross profit|net income)\b", re.IGNORECASE),
        re.compile(r"\b(Q[1-4]\s+\d{4}|quý\s+[1-4])\b", re.IGNORECASE),
        re.compile(r"\d+\.\d+\s+(tỷ|triệu|million|billion)", re.IGNORECASE),
    ],
    "chat_log": [
        re.compile(r'"role"\s*:\s*"(user|assistant|system)"', re.IGNORECASE),
        re.compile(r"^\[\d{2}:\d{2}\]", re.MULTILINE),
        re.compile(r"^@\w+:", re.MULTILINE),
    ],
    "legal_contract": [
        re.compile(r"\b(Điều|Article|Section)\s+\d+", re.IGNORECASE),
        re.compile(r"\b(bên A|bên B|party A|party B|whereas|hereby)\b", re.IGNORECASE),
        re.compile(r"\b(hợp đồng|contract|agreement)\s+số", re.IGNORECASE),
    ],
    "code": [
        re.compile(r"^(def|class|function|public|private)\s+\w+", re.MULTILINE),
        re.compile(r"^```(python|javascript|java|cpp|go|rust)", re.MULTILINE),
        re.compile(r"^(import|from|require|use)\s+", re.MULTILINE),
    ],
}


def classify_doc_type(text: str, filename: str = "") -> str:
    """Classify document type using regex heuristics + filename hints.

    Returns one of: research_paper, report_financial, chat_log, legal_contract,
    code, generic.
    """
    # Filename hints
    fn_lower = filename.lower()
    if any(s in fn_lower for s in ["contract", "agreement", "hợp đồng", "legal"]):
        return "legal_contract"
    if any(s in fn_lower for s in ["chat", "conversation", "log", "messages"]):
        return "chat_log"
    if any(s in fn_lower for s in ["financial", "earnings", "quarterly", "doanh-thu"]):
        return "report_financial"
    if fn_lower.endswith((".py", ".js", ".ts", ".java", ".cpp", ".go", ".rs")):
        return "code"

    # Content heuristics — vote by sig hits
    snippet = text[:3000]
    scores: dict[str, int] = dict.fromkeys(_TYPE_SIGNATURES, 0)
    for doc_type, patterns in _TYPE_SIGNATURES.items():
        for pat in patterns:
            if pat.search(snippet):
                scores[doc_type] += 1

    best = max(scores, key=lambda k: scores[k])
    if scores[best] >= 2:  # need ≥2 signals to commit
        return best
    return "generic"


def get_strategy(doc_type: str) -> dict:
    """Get chunking strategy for a doc type."""
    return TYPE_STRATEGIES.get(doc_type, TYPE_STRATEGIES["generic"])
