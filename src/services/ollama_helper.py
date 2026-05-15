"""Ollama native chat helper — bypasses OpenAI-compat layer.

Why this exists:
  Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`) silently drops
  the `think` parameter. For Qwen3-family models with thinking mode enabled by
  default, this causes:
    • All generated tokens go to `message.thinking` (hidden)
    • `message.content` returns empty string
    • All downstream code (validation, consistency, entity extraction) silently
      receives empty responses and refuses or degrades

Fix:
  Call Ollama native `/api/chat` directly with `think: False`. Returns content
  populated as expected. Use this helper everywhere instead of
  `clients.llm.chat.completions.create()` for Ollama models.

Usage:
    from src.services.ollama_helper import ollama_chat
    content = await ollama_chat(
        messages=[{"role": "user", "content": prompt}],
        model="qwen3.5:4b",
        max_tokens=512,
        temperature=0.2,
    )
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger


async def ollama_chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: float | None = None,
    think: bool = False,
    keep_alive: int | str = -1,
    _retries: int = 3,
) -> str:
    """Send a chat request via Ollama native /api/chat. Returns content string.

    Retries up to _retries times on connection errors with exponential backoff.
    Returns empty string on final failure (caller should handle).
    """
    import asyncio as _aio

    from src.clients import get_clients as _get_clients
    from src.config import get_settings as _get_settings

    settings = _get_settings()
    clients = _get_clients()

    last_error = ""
    for attempt in range(_retries):
        try:
            body = {
                "model": model or settings.ollama_model,
                "messages": messages,
                "stream": False,
                "think": think,
                "keep_alive": keep_alive,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            resp = await clients.http.post(
                f"{settings.ollama_base_url}/api/chat",
                json=body,
                timeout=timeout or settings.request_timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message") or {}
            content = (msg.get("content") or "").strip()
            if not content:
                content = (msg.get("thinking") or "").strip()
                if content:
                    logger.debug(
                        f"ollama_chat: empty content, used thinking fallback ({len(content)} chars)"
                    )
            return content
        except httpx.HTTPStatusError as e:
            last_error = f"HTTP {e.response.status_code}: {e}"
            logger.warning(f"ollama_chat attempt {attempt + 1}/{_retries} failed: {last_error}")
        except httpx.ConnectError as e:
            last_error = f"ConnectError: {e}"
            logger.warning(f"ollama_chat attempt {attempt + 1}/{_retries} failed: {last_error}")
        except httpx.RemoteProtocolError as e:
            last_error = f"RemoteProtocolError: {e}"
            logger.warning(f"ollama_chat attempt {attempt + 1}/{_retries} failed: {last_error}")
        except Exception as e:
            last_error = str(e)
            logger.warning(f"ollama_chat attempt {attempt + 1}/{_retries} failed: {last_error}")

        if attempt < _retries - 1:
            wait = 2**attempt
            logger.info(f"ollama_chat: retrying in {wait}s...")
            await _aio.sleep(wait)

    logger.error(f"ollama_chat: all {_retries} attempts failed. Last error: {last_error}")
    return ""


async def ollama_chat_stream(
    messages: list[dict[str, str]],
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    timeout: float | None = None,
    think: bool = False,
):
    """Stream chat tokens via Ollama native /api/chat (stream=True).

    Yields dicts: {"token": str, "done": bool, "thinking": str | None}.
    Final yield has done=True with cumulative stats.

    Note: even with think=False, some Qwen3 builds still emit a brief thinking
    block at start. Filter those out — only yield content tokens.
    """
    import json as _json

    from src.clients import get_clients
    from src.config import get_settings

    settings = get_settings()
    clients = get_clients()
    model = model or settings.ollama_model
    timeout = timeout or settings.request_timeout_s

    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": think,
        "keep_alive": -1,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        async with clients.http.stream(
            "POST",
            f"{settings.ollama_base_url}/api/chat",
            json=body,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = _json.loads(line)
                except Exception:
                    continue
                msg = chunk.get("message") or {}
                token = msg.get("content") or ""
                thinking = msg.get("thinking")
                done = bool(chunk.get("done"))
                if token or done:
                    yield {
                        "token": token,
                        "thinking": thinking,
                        "done": done,
                        "eval_count": chunk.get("eval_count"),
                        "done_reason": chunk.get("done_reason"),
                    }
                if done:
                    break
    except Exception as e:
        logger.warning(f"ollama_chat_stream failure: {e}")
        yield {"token": "", "done": True, "error": str(e)[:200]}


async def ollama_chat_with_meta(
    messages: list[dict[str, str]],
    model: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: float | None = None,
    think: bool = False,
) -> dict[str, Any]:
    """Same as ollama_chat() but returns full response dict for instrumentation.

    Returned shape:
      {
        "content": str,
        "thinking": str,
        "eval_count": int,
        "eval_duration_ms": float,
        "total_duration_ms": float,
        "done_reason": str,
        "error": str | None,
      }
    """
    from src.clients import get_clients
    from src.config import get_settings

    settings = get_settings()
    clients = get_clients()
    model = model or settings.ollama_model
    timeout = timeout or settings.request_timeout_s

    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "keep_alive": -1,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }

    try:
        resp = await clients.http.post(
            f"{settings.ollama_base_url}/api/chat",
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"content": "", "thinking": "", "error": str(e)[:200]}

    msg = data.get("message") or {}
    return {
        "content": (msg.get("content") or "").strip(),
        "thinking": (msg.get("thinking") or "").strip(),
        "eval_count": int(data.get("eval_count", 0)),
        "eval_duration_ms": data.get("eval_duration", 0) / 1e6,
        "total_duration_ms": data.get("total_duration", 0) / 1e6,
        "done_reason": data.get("done_reason"),
        "error": None,
    }
