"""Unit tests for the KG-path RRF weight scale (rrf_kg_path_weight_scale).

The N=839 corpus500 recall benchmark showed the KG paths' >dense RRF weights
net-hurt recall@5 at 1.0×; scaling them to ~0.2× recovers recall@5 while keeping
the recall@1/MRR gain. This locks in that ONLY the KG paths scale — the dense /
sparse reformulation kinds must stay fixed.
"""

from __future__ import annotations

import os

from src.config import get_settings
from src.services.retrieval import reformulation_weight

_KG = {
    "entity_pivot": 1.5,
    "graph": 1.0,
    "community": 1.2,
    "entity_cosine": 1.6,
    "ppr": 1.7,
    "entity_gate": 1.8,
}
_NON_KG = {
    "original": 1.0,
    "rewrite": 1.1,
    "hyde": 1.3,
    "step_back": 0.8,
    "keywords": 0.9,
    "decompose": 1.1,
}


def _set_scale(v):
    if v is None:
        os.environ.pop("RRF_KG_PATH_WEIGHT_SCALE", None)
    else:
        os.environ["RRF_KG_PATH_WEIGHT_SCALE"] = str(v)
    get_settings.cache_clear()


def test_default_is_evidence_based_0_2():
    # Default is 0.2 (SUPERNOVA_BENCHMARK_20260718.md): KG paths scaled down, dense
    # untouched. Non-KG reformulation kinds must keep their legacy weights.
    _set_scale(None)
    try:
        assert get_settings().rrf_kg_path_weight_scale == 0.2
        for kind, w in _KG.items():
            assert abs(reformulation_weight(kind) - w * 0.2) < 1e-9, kind
        for kind, w in _NON_KG.items():
            assert reformulation_weight(kind) == w, kind
    finally:
        _set_scale(None)


def test_scale_1_restores_legacy_weights():
    # Explicit 1.0 restores the shipped hand-tuned weights (escape hatch).
    _set_scale(1.0)
    try:
        for kind, w in {**_KG, **_NON_KG}.items():
            assert reformulation_weight(kind) == w, kind
    finally:
        _set_scale(None)


def test_scale_applies_only_to_kg_paths():
    _set_scale(0.2)
    try:
        for kind, w in _KG.items():
            assert abs(reformulation_weight(kind) - w * 0.2) < 1e-9, kind
        for kind, w in _NON_KG.items():  # dense/sparse untouched
            assert reformulation_weight(kind) == w, kind
    finally:
        _set_scale(None)


def test_scale_zero_neutralizes_kg_paths():
    _set_scale(0.0)
    try:
        for kind in _KG:
            assert reformulation_weight(kind) == 0.0, kind
        assert reformulation_weight("original") == 1.0
    finally:
        _set_scale(None)
