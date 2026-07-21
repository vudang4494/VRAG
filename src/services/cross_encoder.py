"""Shared cross-encoder loader — single source of truth for bge-reranker-v2-m3."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    # sentence-transformers is commented out of api/requirements.txt on purpose (it
    # pulls CUDA torch, which breaks Apple Silicon), so it is absent at runtime AND
    # at type-check time — hence the ignore. This import must never execute; the real
    # one is lazy, inside the try below.
    from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]

# Lazy-loaded singleton
_CROSS_ENCODER: CrossEncoder | None | bool = None


def _load_cross_encoder(model_name: str = "BAAI/bge-reranker-v2-m3") -> CrossEncoder | None:
    """Load cross-encoder in-process. Returns None if model unavailable."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER if _CROSS_ENCODER is not True else None
    try:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER = CrossEncoder(model_name, max_length=512)
        logger.info(f"Loaded cross-encoder: {model_name}")
    except Exception as e:
        logger.warning(f"Cross-encoder not available ({e}); stage 1 will be skipped.")
        _CROSS_ENCODER = True  # sentinel: tried and failed
    return _CROSS_ENCODER if _CROSS_ENCODER is not True else None
