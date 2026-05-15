"""Embedding service — optimized for Apple Silicon M4 Metal GPU."""

import asyncio

import httpx
from loguru import logger

from src.config import get_settings


async def embed_single(
    http_client: httpx.AsyncClient,
    url: str,
    model: str,
    text: str,
    timeout: float = 60.0,
) -> list[float]:
    """Embed a single text via Ollama /api/embeddings."""
    try:
        resp = await http_client.post(
            f"{url}/api/embeddings",
            json={"model": model, "prompt": text, "keep_alive": -1},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except httpx.HTTPStatusError as e:
        logger.warning(
            f"HTTP {e.response.status_code} embedding text (len={len(text)}): {e.request.url}"
        )
        raise
    except Exception as e:
        logger.warning(f"Embedding failed for text (len={len(text)}): {e}")
        raise


async def embed_batch(
    http_client: httpx.AsyncClient,
    url: str,
    model: str,
    texts: list[str],
    batch_size: int | None = None,
    timeout: float = 120.0,
) -> list[list[float]]:
    """
    Batch embed via Ollama — optimized for M4 Metal GPU.

    Improvements over v2:
    - batch_size from config (default 32, up from 16)
    - Semaphore uses config embed_concurrent_limit (default 3)
    - Slightly larger batches = fewer Ollama round-trips
    - Zero vector fallback uses actual embed_dimension from config
    """
    if not texts:
        return []

    settings = get_settings()
    batch_size = batch_size or settings.embed_batch_size
    concurrency = settings.embed_concurrent_limit

    results: list[list[float]] = []
    semaphore = asyncio.Semaphore(concurrency)
    dim = settings.embed_dimension

    async def embed_one(text: str) -> list[float]:
        async with semaphore:
            try:
                resp = await http_client.post(
                    f"{url}/api/embeddings",
                    json={"model": model, "prompt": text, "keep_alive": -1},
                    timeout=timeout,
                )
                resp.raise_for_status()
                return resp.json()["embedding"]
            except Exception as e:
                logger.warning(f"Batch embed failed: {e}")
                return [0.0] * dim

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_results = await asyncio.gather(*[embed_one(t) for t in batch])
        results.extend(batch_results)

    return results


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
