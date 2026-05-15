"""Tests cho Pipeline V2 modules.

These are mostly UNIT tests — không cần stack chạy (trừ một số e2e marked).
Chạy bằng: make test-v2  hoặc  pytest tests/test_pipeline_v2.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


# ════════════════════════════════════════════════════════════════════════════
# Chunkers
# ════════════════════════════════════════════════════════════════════════════


def test_base_chunker_split_sentences_vietnamese():
    from src.services.chunkers.base import BaseChunker

    text = "Đây là câu thứ nhất. Đây là câu thứ hai! Đây là câu thứ ba?"
    sentences = BaseChunker.split_sentences(text)
    assert len(sentences) == 3
    assert "câu thứ nhất" in sentences[0]
    assert "câu thứ ba" in sentences[2]


def test_base_chunker_pack_units():
    from src.services.chunkers.base import BaseChunker

    units = ["abc", "def", "ghi", "jkl"]
    packed = BaseChunker.pack_units(units, max_chars=10, joiner=" ")
    assert all(len(p) <= 10 for p in packed)
    assert " ".join(packed).replace(" ", "") == "abcdefghijkl"


@pytest.mark.asyncio
async def test_semantic_chunker_basic_text():
    from src.services.chunkers.semantic_chunker import SemanticChunker

    chunker = SemanticChunker(emit_levels=("paragraph",))
    text = "Câu một. Câu hai. " * 50
    units = await chunker.chunk(text, filename="test.txt")
    assert len(units) >= 1
    assert all(u.chunk_level == "paragraph" for u in units)


@pytest.mark.asyncio
async def test_xlsx_chunker_csv():
    from src.services.chunkers.xlsx_chunker import XlsxChunker

    csv = b"STT,Khach hang,Doanh thu\n1,ABC,500\n2,XYZ,300\n3,DEF,200\n"
    chunker = XlsxChunker(rows_per_chunk=2, emit_levels=("paragraph",))
    units = await chunker.chunk(csv, filename="test.csv")
    assert len(units) >= 1
    assert "STT" in units[0].text
    assert units[0].metadata.get("sheet_name") == "Sheet1"


@pytest.mark.asyncio
async def test_chat_chunker_json():
    from src.services.chunkers.chat_chunker import ChatChunker

    msgs = json.dumps(
        [
            {
                "role": "user",
                "content": "Doanh thu Q3?",
                "thread_id": "t1",
                "timestamp": "2024-10-01",
            },
            {
                "role": "assistant",
                "content": "500 tỷ.",
                "thread_id": "t1",
                "timestamp": "2024-10-01",
            },
            {"role": "user", "content": "Q4?", "thread_id": "t1", "timestamp": "2024-10-02"},
        ]
    ).encode("utf-8")
    chunker = ChatChunker(qa_pair_window=1, emit_levels=("paragraph",))
    units = await chunker.chunk(msgs, filename="test.json")
    assert len(units) >= 1
    assert units[0].metadata.get("thread_id") == "t1"


# ════════════════════════════════════════════════════════════════════════════
# Format Router
# ════════════════════════════════════════════════════════════════════════════


def test_format_router_pdf_extension():
    from src.services.format_router import detect_format

    assert detect_format("report.pdf") == "pdf"
    assert detect_format("data.xlsx") == "xlsx"
    assert detect_format("notes.md") == "md"
    assert detect_format("chat.jsonl") == "chat"
    assert detect_format("email.eml") == "email"
    assert detect_format("unknown.xyz") == "txt"


def test_format_router_pdf_magic_bytes():
    from src.services.format_router import detect_format

    pdf_header = b"%PDF-1.7\n..."
    assert detect_format("noext", pdf_header) == "pdf"


# ════════════════════════════════════════════════════════════════════════════
# PII Mask
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pii_mask_regex_email_phone():
    from src.services.pii_mask import mask_pii, MaskMap

    text = "Email: nguyen.a@company.com, SĐT: 0912345678"
    masked, _ = await mask_pii(text, llm=None, use_llm_ner=False)
    assert "nguyen.a@company.com" not in masked
    assert "0912345678" not in masked
    assert "<EMAIL_" in masked
    assert "<PHONE_" in masked


@pytest.mark.asyncio
async def test_pii_mask_consistent_placeholder():
    from src.services.pii_mask import MaskMap

    mm = MaskMap()
    p1 = mm.add("Nguyễn Văn A", "PERSON")
    p2 = mm.add("Nguyễn Văn A", "PERSON")  # same value → same placeholder
    assert p1 == p2
    p3 = mm.add("Trần Thị B", "PERSON")
    assert p3 != p1


# ════════════════════════════════════════════════════════════════════════════
# Consistency Simulation
# ════════════════════════════════════════════════════════════════════════════


def test_consistency_score_perfect():
    from src.services.consistency import consistency_score

    vec = [1.0] + [0.0] * 1023
    score = consistency_score({"v1": vec, "v2": vec, "v3": vec})
    assert score >= 0.99


def test_consistency_score_orthogonal():
    from src.services.consistency import consistency_score

    v1 = [1.0] + [0.0] * 1023
    v2 = [0.0, 1.0] + [0.0] * 1022
    score = consistency_score({"v1": v1, "v2": v2})
    assert score < 0.1


def test_consistency_boost_thresholds():
    from src.services.consistency import consistency_boost, classify_consistency

    assert consistency_boost(0.9) == 1.2
    assert consistency_boost(0.7) == 1.0
    assert consistency_boost(0.4) == 0.8
    assert classify_consistency(0.9) == "high"
    assert classify_consistency(0.7) == "normal"
    assert classify_consistency(0.4) == "low"


# ════════════════════════════════════════════════════════════════════════════
# Validation Gates
# ════════════════════════════════════════════════════════════════════════════


def test_citation_gate_passes_when_cited():
    from src.services.validation import citation_gate

    answer = "Doanh thu Q3 đạt 500 tỷ [doc_abc::para::1]. Lợi nhuận 80 tỷ [doc_abc::para::2]."
    result = citation_gate(answer, min_ratio=0.5)
    assert result["passed"]
    assert result["citation_ratio"] >= 0.5


def test_citation_gate_fails_when_uncited():
    from src.services.validation import citation_gate

    answer = "Doanh thu Q3 đạt 500 tỷ. Lợi nhuận 80 tỷ. Tăng 25%."
    result = citation_gate(answer, min_ratio=0.7)
    assert not result["passed"]


# ════════════════════════════════════════════════════════════════════════════
# Vector V2
# ════════════════════════════════════════════════════════════════════════════


def test_to_int_id_deterministic():
    from src.services.vector_v2 import to_int_id

    assert to_int_id("doc_abc::para::1") == to_int_id("doc_abc::para::1")
    assert to_int_id("doc_abc::para::1") != to_int_id("doc_abc::para::2")


def test_normalize_scores_by_format():
    from src.services.vector_v2 import normalize_scores_by_format

    cands = [
        {"chunk_id": "1", "format": "pdf", "score": 0.9},
        {"chunk_id": "2", "format": "pdf", "score": 0.7},
        {"chunk_id": "3", "format": "xlsx", "score": 0.5},
        {"chunk_id": "4", "format": "xlsx", "score": 0.3},
    ]
    out = normalize_scores_by_format(cands)
    pdf_scores = [c["score_normalized"] for c in out if c["format"] == "pdf"]
    xlsx_scores = [c["score_normalized"] for c in out if c["format"] == "xlsx"]
    # Mean of normalized scores per group should be ~0
    assert abs(sum(pdf_scores) / len(pdf_scores)) < 0.01
    assert abs(sum(xlsx_scores) / len(xlsx_scores)) < 0.01


def test_level_factor():
    from src.services.vector_v2 import level_factor

    assert level_factor("section") > level_factor("paragraph") > level_factor("sentence")
    assert level_factor("document") < level_factor("paragraph")


def test_consistency_factor():
    from src.services.vector_v2 import consistency_factor

    assert consistency_factor(0.9) == 1.2
    assert consistency_factor(0.7) == 1.0
    assert consistency_factor(0.4) == 0.8


# ════════════════════════════════════════════════════════════════════════════
# Retrieval V2 — RRF
# ════════════════════════════════════════════════════════════════════════════


def test_weighted_rrf_fusion():
    from src.services.retrieval_v2 import weighted_rrf

    paths = {
        "original:dense": [
            {
                "chunk_id": "c1",
                "score": 0.9,
                "consistency_score": 0.85,
                "chunk_level": "paragraph",
                "retrieval_path": "vector:dense",
                "text": "...",
                "source": "x",
            },
            {
                "chunk_id": "c2",
                "score": 0.7,
                "consistency_score": 0.7,
                "chunk_level": "paragraph",
                "retrieval_path": "vector:dense",
                "text": "...",
                "source": "x",
            },
        ],
        "hyde:dense": [
            {
                "chunk_id": "c1",
                "score": 0.85,
                "consistency_score": 0.85,
                "chunk_level": "paragraph",
                "retrieval_path": "vector:dense",
                "text": "...",
                "source": "x",
            },
            {
                "chunk_id": "c3",
                "score": 0.65,
                "consistency_score": 0.5,
                "chunk_level": "sentence",
                "retrieval_path": "vector:dense",
                "text": "...",
                "source": "x",
            },
        ],
    }
    fused = weighted_rrf(paths, k=60, final_top_k=10)
    # c1 appears in both paths → should rank first
    assert fused[0]["chunk_id"] == "c1"
    # c3 has low consistency AND sentence level → ranks lower
    assert fused[-1]["chunk_id"] == "c3" if len(fused) >= 3 else True


# ════════════════════════════════════════════════════════════════════════════
# Query Understanding (mocked LLM)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_classify_intent_factual_keyword():
    from src.services.query_understanding import classify_intent

    mock_llm = MagicMock()
    mock_llm.chat = MagicMock()
    mock_llm.chat.completions = MagicMock()

    async def mock_create(**kwargs):
        msg = MagicMock()
        msg.message.content = "factual"
        return MagicMock(choices=[msg])

    mock_llm.chat.completions.create = AsyncMock(side_effect=mock_create)
    intent = await classify_intent("Doanh thu Q3 là bao nhiêu?", mock_llm)
    assert intent == "factual"


# ════════════════════════════════════════════════════════════════════════════
# Models V2
# ════════════════════════════════════════════════════════════════════════════


def test_chunk_level_enum():
    from src.models import ChunkLevel

    assert ChunkLevel.PARAGRAPH.value == "paragraph"
    assert ChunkLevel.SECTION.value == "section"


def test_query_intent_enum():
    from src.models import QueryIntent

    assert QueryIntent.FACTUAL.value == "factual"
    assert QueryIntent.SUMMARIZATION.value == "summarization"


def test_view_type_enum():
    from src.models import ViewType

    assert {v.value for v in ViewType} == {"dense", "paraphrase", "question", "summary", "keywords"}
