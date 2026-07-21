"""VRAG Tier 3c — Context Compression (LLMLingua-2).

Compress the retrieved context before passing it to the LLM generator. Reduces
generation latency by 30-50% and lowers token cost.

Uses LLMLingua-2 (classifier-based, no LLM call) — much faster on CPU than the
original LLMLingua which needs a small GPT-style model for perplexity scoring.

Model: `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` (~600MB, multilingual,
includes Vietnamese support).

Lazy-loaded. First call may take 30-60s while model downloads. Subsequent calls
~200-500ms on CPU for typical 2000-token contexts.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

_COMPRESSOR: Any | None = None
_LOAD_LOCK = asyncio.Lock()


async def _get_compressor() -> Any | None:
    global _COMPRESSOR
    if _COMPRESSOR is not None:
        return _COMPRESSOR if _COMPRESSOR is not False else None
    async with _LOAD_LOCK:
        if _COMPRESSOR is not None:
            return _COMPRESSOR if _COMPRESSOR is not False else None
        try:
            from llmlingua import PromptCompressor

            def _load():
                return PromptCompressor(
                    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                    use_llmlingua2=True,
                    device_map="cpu",
                )

            _COMPRESSOR = await asyncio.to_thread(_load)
            logger.info("LLMLingua-2 compressor loaded")
        except Exception as e:
            logger.warning(f"LLMLingua-2 unavailable, compression disabled: {e}")
            _COMPRESSOR = False  # sentinel — don't retry
            return None
    return _COMPRESSOR


async def compress_context(
    context: str,
    rate: float = 0.4,
    force_tokens: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Compress context string. Returns (compressed_text, stats).

    Args:
      context: raw context string (post-format_context).
      rate: target compression rate (0.4 = keep 40% of tokens). Plan = 0.4.
      force_tokens: tokens that must NEVER be compressed (e.g. citation markers,
        entity names). Defaults to preserving "[", "]", ":", and digits.

    Returns:
      (compressed_text, {
        "original_tokens": int,
        "compressed_tokens": int,
        "ratio": float,
        "compressed": bool,  # False if compressor unavailable
      })
    """
    if not context or not context.strip():
        return context, {"compressed": False, "reason": "empty_input"}

    compressor = await _get_compressor()
    if compressor is None:
        return context, {"compressed": False, "reason": "no_compressor"}

    # Preserve citation markers and structure
    force = force_tokens or ["[", "]", ":", "**"]

    try:
        result = await asyncio.to_thread(
            compressor.compress_prompt,
            context,
            rate=rate,
            force_tokens=force,
        )
        compressed = result.get("compressed_prompt", context)
        origin_tokens = int(result.get("origin_tokens", 0) or 0)
        compressed_tokens = int(result.get("compressed_tokens", 0) or 0)
        # LLMLingua may return ratio as str like "1.5x"; compute numeric ratio ourselves.
        numeric_ratio = compressed_tokens / origin_tokens if origin_tokens > 0 else 1.0
        return compressed, {
            "compressed": True,
            "original_tokens": origin_tokens,
            "compressed_tokens": compressed_tokens,
            "ratio": f"{numeric_ratio:.2f}",
            "ratio_numeric": numeric_ratio,
        }
    except Exception as e:
        logger.warning(f"Context compression failed, using raw: {e}")
        return context, {"compressed": False, "reason": f"error: {e}"}
