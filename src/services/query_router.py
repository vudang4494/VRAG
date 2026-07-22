"""Query type classifier for intelligent routing between retrieval strategies.

## Algorithm: Semantic Matcher (BGE-M3 Centroid) + OOD Guard

1. **Out-of-domain guard** (regex, highest priority): fast short-circuit, no embedding needed
2. **Semantic intent matching** (centroid dot-product): embed query via BGE-M3, compute
   cosine similarity with intent centroids, pick the highest-scoring intent

Why semantic instead of rule-based:
  - Zero maintenance of regex patterns
  - Handles unseen surface forms, typos, code-mixed queries
  - Fast: dot product on 1024-dim vector < 1ms
  - Centroid-based is robust to query phrasing variations

Usage:
    from src.services.query_router import classify_query, should_use_react

    query_type = classify_query("GraphRAG là gì?")
    use_react = should_use_react(query_type)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).parent.parent.parent  # repo root (src/services/.. → src → repo)
_CENTROIDS: dict[str, np.ndarray] | None = None
_SEMANTIC_THRESHOLD: float = 0.45
# Tier 1 fix: queries scoring below this on the best centroid are classified
# as out_of_domain (none of the 5 intent centroids match closely). Catches
# OOD queries that the 17-pattern regex misses (politics, programming, generic).
_OOD_CENTROID_FLOOR: float = 0.40

# ── OOD patterns — regex only, no embedding needed ──────────────────────────────

_OOD_PATTERNS = [
    # Real-world queries not in academic corpus
    r"thời tiết",
    r"bitcoin",
    r"giá .*hôm nay",
    r"nấu (phở|canh|bún)",
    r"tin tức",
    r"news today",
    r"weather",
    r"stock price",
    r"cook (pho|soup|recipe)",
    r"news",
    r"bóng đá",
    r"sport",
    r"football",
    r"game",
    r"làm bánh",
    r"tập gym",
    r"mua sắm",
    r"du lịch",
]

# ── Global/thematic patterns — corpus-wide questions (LazyGraphRAG map-reduce) ──

_GLOBAL_PATTERNS = [
    r"chủ đề (chính|nào|gì|lớn)",
    r"(các|những) chủ đề",
    r"xuyên (suốt )?(corpus|tài liệu|kho)",
    r"toàn bộ (tài liệu|corpus|kho tài liệu)",
    r"tổng (thể|quan|hợp)",
    r"xu hướng (chung|chính|nổi bật)",
    r"(điểm|nội dung) chung",
    r"khái quát",
    r"bức tranh (chung|tổng)",
    r"main (theme|topic)",
    r"(across|throughout) (the )?(corpus|document|dataset|collection)",
    r"overall (theme|trend|topic|picture)",
    r"high[- ]level (overview|summary)",
    r"what are the .*(theme|topic|trend)",
]


def _load_centroids() -> dict[str, np.ndarray]:
    global _CENTROIDS
    if _CENTROIDS is not None:
        return _CENTROIDS
    path = ROOT / "config" / "intent_centroids.npy"
    if not path.exists():
        logger.warning(f"intent_centroids.npy not found at {path}, using rule-based fallback")
        _CENTROIDS = {}
        return _CENTROIDS
    data = np.load(path, allow_pickle=True).item()
    for intent, vec in data.items():
        data[intent] = np.asarray(vec, dtype=np.float32)
    _CENTROIDS = data
    logger.info(f"Loaded {len(_CENTROIDS)} intent centroids from {path}")
    return _CENTROIDS


def _embed_query_sync(query: str, embed_url: str, embed_model: str) -> np.ndarray | None:
    """Embed a query via Ollama /api/embeddings — a BLOCKING call by design.

    This is the synchronous inner of classify_query. It blocks the calling thread, so
    every async caller must invoke classify_query via `asyncio.to_thread(...)` to keep it
    off the event loop. (A previous comment here claimed it self-threaded; it did not —
    it called the inner function inline, stalling the loop on every request.)
    """
    try:
        import httpx

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{embed_url}/api/embeddings",
                json={"model": embed_model, "prompt": query, "keep_alive": -1},
            )
            resp.raise_for_status()
            result = resp.json()
        return np.asarray(result["embedding"], dtype=np.float32)
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


_OOD_REGEX = re.compile("|".join(_OOD_PATTERNS), re.IGNORECASE)
_GLOBAL_REGEX = re.compile("|".join(_GLOBAL_PATTERNS), re.IGNORECASE)


def _match_ood(query: str) -> bool:
    return bool(_OOD_REGEX.search(query.strip()))


def _match_global(query: str) -> bool:
    return bool(_GLOBAL_REGEX.search(query.strip()))


def _global_enabled() -> bool:
    """Global-query branch is gated OFF by default. classify_query only emits
    'global' when GLOBAL_QUERY_ENABLED is on, so flag-off routing stays byte-identical."""
    try:
        from src.config import get_settings

        return bool(get_settings().global_query_enabled)
    except Exception:
        return False


def classify_query(query: str, embed_url: str | None = None, embed_model: str | None = None) -> str:
    """
    Classify query type using semantic centroid matching.

    Returns one of: factual | analytical | comparison | multi_hop | kg_construction
    | global | out_of_domain

    If embed_url/embed_model are None, reads from settings (so the call works inside
    a container where Ollama is at host.docker.internal, not localhost).
    """
    if _match_ood(query):
        return "out_of_domain"

    # Global/thematic → LazyGraphRAG map-reduce branch (gated OFF by default so
    # flag-off routing is byte-identical). Checked before centroid: corpus-wide.
    if _global_enabled() and _match_global(query):
        return "global"

    centroids = _load_centroids()
    if not centroids:
        # Fallback: simple keyword heuristics
        return _fallback_rule_based(query)

    if embed_url is None or embed_model is None:
        try:
            from src.config import get_settings

            s = get_settings()
            embed_url = embed_url or s.ollama_embed_url
            embed_model = embed_model or s.ollama_embed_model
        except Exception:
            embed_url = embed_url or "http://localhost:11434"
            embed_model = embed_model or "bge-m3"

    vec = _embed_query_sync(query, embed_url, embed_model)
    if vec is None:
        return _fallback_rule_based(query)

    best_intent = "factual"
    best_score = -1.0
    for intent, centroid in centroids.items():
        # Both are unit-normalized → dot product = cosine similarity
        score = float(np.dot(vec, centroid))
        if score > best_score:
            best_intent, best_score = intent, score

    # Tier 1 fix: very low centroid score means none of our 5 intents matches —
    # this is out-of-domain. Classify accordingly so OOD refusal path triggers.
    if best_score < _OOD_CENTROID_FLOOR:
        logger.info(
            f"OOD centroid floor: best={best_score:.3f} < {_OOD_CENTROID_FLOOR} "
            f"→ out_of_domain (query: {query[:60]!r})"
        )
        return "out_of_domain"

    if best_score < _SEMANTIC_THRESHOLD:
        logger.debug(
            f"Low centroid score {best_score:.3f} for query: {query[:60]}, defaulting to factual"
        )
        return "factual"

    return best_intent


def _fallback_rule_based(query: str) -> str:
    """Simple keyword fallback when centroid matching is unavailable."""
    q = query.lower()

    if any(kw in q for kw in ["tóm tắt", "tổng hợp", "tổng quan", "liệt kê"]):
        return "summarization"
    if any(kw in q for kw in ["tại sao", "vì sao", "bằng cách nào", "như thế nào"]):
        return "analytical"
    if any(kw in q for kw in ["so sánh", "khác nhau", "giống nhau", "ưu điểm", "nhược điểm"]):
        return "comparison"
    if any(kw in q for kw in ["mối liên hệ", "ảnh hưởng qua lại", "cả a và b"]):
        return "multi_hop"
    if any(kw in q for kw in ["knowledge graph", "schema", "ontology", "xây dựng đồ thị"]):
        return "kg_construction"

    return "factual"


# ── ReAct gating ───────────────────────────────────────────────────────────────

_REACT_INTENTS = {"analytical", "comparison", "multi_hop", "kg_construction"}

# Rule-based ReAct gating signals. ReAct trigger is expensive (6 LLM decisions
# per query); restrict to queries that ACTUALLY benefit from multi-step.
_MULTI_DOC_KEYWORDS = (
    "so sánh",
    "khác",
    "khác nhau",
    "khác gì",
    "vs ",
    " vs.",
    "hơn ",
    "cả ",
    "vừa ",
    "compare",
    "difference",
    "khác biệt",
)
_MULTI_HOP_KEYWORDS = (
    "mối liên hệ",
    "liên quan đến",
    "ảnh hưởng",
    "dẫn đến",
    "qua lại",
    "thông qua",
    "bằng cách nào",
    "tại sao",
    "vì sao",
)
_REASONING_KEYWORDS = (
    "phân tích",
    "đánh giá",
    "tổng hợp",
    "lý do",
    "nguyên nhân",
    "ưu điểm",
    "nhược điểm",
    "trade-off",
    "trade off",
)


def _count_query_entities(query: str) -> int:
    """Cheap regex heuristic — count likely entity mentions (proper nouns,
    capitalized tokens, acronyms). No GLiNER call. Used as rule-based proxy
    for whether the query is multi-entity (needs ReAct) or single-entity
    (factual, no ReAct needed)."""
    if not query:
        return 0
    tokens = re.findall(r"\b[A-Z][A-Za-z0-9\-]{1,30}\b", query)
    # Filter out common Vietnamese sentence-starters that aren't entities.
    blacklist = {
        "Tôi",
        "Bạn",
        "Cái",
        "Khi",
        "Tại",
        "Trong",
        "Với",
        "Cho",
        "Nếu",
        "Vì",
        "Nó",
        "Họ",
        "Đây",
        "Đó",
        "Vậy",
        "Hãy",
        "Có",
        "Không",
        "Là",
    }
    return sum(1 for t in tokens if t not in blacklist)


def should_use_react(query_type: str, query: str = "") -> bool:
    """Decide whether to route to ReAct loop.

    Two-tier gate (no LLM):
      1. Intent must be in _REACT_INTENTS (semantic centroid).
      2. AT LEAST ONE rule-based signal must fire:
         - query length >= 60 chars (complex enough to need multi-step), OR
         - >= 2 entity-like tokens (multi-doc reasoning candidate), OR
         - matches one of the multi-doc / multi-hop / reasoning keyword sets

    This prevents ReAct from triggering on simple single-entity factual
    queries that got mis-classified as `comparison` by the centroid
    (e.g. "LightRAG dual-level là gì?" → centroid says comparison but
    it's actually factual single-doc lookup).

    Set REACT_STRICT_GATE=0 in env to bypass tier-2 (legacy mode, ReAct
    on every intent match).
    """
    if query_type not in _REACT_INTENTS:
        return False

    import os as _os

    if not bool(int(_os.environ.get("REACT_STRICT_GATE", "1"))):
        return True

    q = (query or "").lower()
    if len(query or "") >= 60:
        return True
    if _count_query_entities(query or "") >= 2:
        return True
    if any(kw in q for kw in _MULTI_DOC_KEYWORDS):
        return True
    if any(kw in q for kw in _MULTI_HOP_KEYWORDS):
        return True
    if any(kw in q for kw in _REASONING_KEYWORDS):
        return True
    return False


def describe_routing(query_type: str, use_react: bool) -> str:
    """Human-readable description of routing decision."""
    if use_react:
        return f"ReAct loop ({query_type} query — multi-step reasoning)"
    return f"Standard retrieval ({query_type} query — single-doc lookup)"
