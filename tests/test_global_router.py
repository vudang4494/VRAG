"""Phase 1 — global-query intent routing (LazyGraphRAG).

Verifies: _match_global regex separates thematic/corpus-wide questions from
factual ones; classify_query only emits "global" when GLOBAL_QUERY_ENABLED is on
(flag-off routing stays byte-identical); "global" never triggers the ReAct loop.
Pure/regex + monkeypatch — no network, no stack.
"""

import os

import pytest

from src.config import get_settings
from src.services import query_router as qr

POSITIVE_GLOBAL = [
    "Các chủ đề chính của corpus là gì?",
    "Những chủ đề lớn xuyên suốt tài liệu?",
    "Tổng thể các tài liệu nói về điều gì?",
    "Xu hướng chung của toàn bộ tài liệu?",
    "Khái quát nội dung kho tài liệu",
    "Bức tranh chung của corpus?",
    "Điểm chung giữa các tài liệu là gì?",
    "Chủ đề nào nổi bật nhất?",
    "Tổng hợp các chủ đề chính",
    "Toàn bộ corpus tập trung vào gì?",
    "What are the main themes across the corpus?",
    "Give a high-level overview of the dataset",
    "Overall trends throughout the documents",
    "What are the main topics in the collection?",
]

NEGATIVE_GLOBAL = [
    "RRF fusion là gì?",
    "GraphRAG cải thiện multi-hop thế nào?",
    "So sánh BM25 và dense retrieval",
    "Doanh thu Disney quý 2 là bao nhiêu?",
    "Định nghĩa entity resolution",
    "Cayman Islands nằm ở đâu?",
    "Liệt kê các loại embedding",
    "Neo4j lưu gì?",
    "Ollama chạy model nào?",
    "bge-m3 có bao nhiêu chiều?",
    "So sánh HippoRAG và LightRAG",
    "Công thức tính RRF score",
    "What is a named vector?",
    "How does reranking work?",
]


def _set_flag(val: str | None) -> None:
    if val is None:
        os.environ.pop("GLOBAL_QUERY_ENABLED", None)
    else:
        os.environ["GLOBAL_QUERY_ENABLED"] = val
    get_settings.cache_clear()


@pytest.mark.parametrize("q", POSITIVE_GLOBAL)
def test_match_global_positive(q):
    assert qr._match_global(q) is True, q


@pytest.mark.parametrize("q", NEGATIVE_GLOBAL)
def test_match_global_negative(q):
    assert qr._match_global(q) is False, q


def test_classify_query_emits_global_when_enabled():
    _set_flag("1")
    try:
        assert qr._global_enabled() is True
        for q in POSITIVE_GLOBAL:
            # thematic short-circuits before any embedding call → no network
            assert qr.classify_query(q) == "global", q
    finally:
        _set_flag(None)


def test_classify_query_no_global_when_disabled(monkeypatch):
    _set_flag("0")
    # force the rule-based fallback so no embedding network call is made
    monkeypatch.setattr(qr, "_load_centroids", lambda: {})
    try:
        assert qr._global_enabled() is False
        for q in POSITIVE_GLOBAL:
            assert qr.classify_query(q) != "global", q
    finally:
        _set_flag(None)


def test_global_never_routes_to_react():
    assert qr.should_use_react("global", "các chủ đề chính của corpus?") is False
