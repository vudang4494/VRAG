"""Embedding service — optimized for Apple Silicon M4 Metal GPU."""

import asyncio

import httpx
from loguru import logger

from src.config import get_settings


async def _embed_texts(
    http_client: httpx.AsyncClient,
    url: str,
    model: str,
    texts: list[str],
    timeout: float,
) -> list[list[float]]:
    """One POST to Ollama /api/embed. `input` takes an array, so N texts cost 1 round-trip.

    The older /api/embeddings takes a single `prompt` and therefore forces N round-trips
    for N texts. Both return identical vectors (verified: cosine 1.000000 on bge-m3).
    """
    resp = await http_client.post(
        f"{url}/api/embed",
        json={"model": model, "input": texts, "keep_alive": -1},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


async def embed_single(
    http_client: httpx.AsyncClient,
    url: str,
    model: str,
    text: str,
    timeout: float = 60.0,
) -> list[float]:
    """Embed a single text. Raises on failure — callers depend on that."""
    try:
        return (await _embed_texts(http_client, url, model, [text], timeout))[0]
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
    """Embed many texts, `batch_size` per HTTP request, `embed_concurrent_limit` in flight.

    Never raises: a text that cannot be embedded comes back as a zero vector of
    `embed_dimension`, so the caller's list stays index-aligned with `texts`.

    This used to send one request per text — `batch_size` only sliced the gather loop, so
    it cut neither round-trips nor concurrency (the semaphore did that), while the docstring
    claimed "fewer Ollama round-trips". Measured on bge-m3 (idle M4, median of 3): 32 texts
    took 1.22s over 32 requests vs 0.70s over 1 — 1.7x, same vectors.

    On batch failure each text is retried alone, so one bad input costs its own zero vector
    instead of zeroing the whole batch.
    """
    if not texts:
        return []

    settings = get_settings()
    batch_size = batch_size or settings.embed_batch_size
    dim = settings.embed_dimension
    semaphore = asyncio.Semaphore(settings.embed_concurrent_limit)

    async def embed_chunk(chunk: list[str]) -> list[list[float]]:
        async with semaphore:
            try:
                vectors = await _embed_texts(http_client, url, model, chunk, timeout)
            except Exception as e:
                logger.warning(f"Batch embed of {len(chunk)} texts failed ({e}); retrying singly")
            else:
                if len(vectors) == len(chunk):
                    return vectors
                logger.warning(
                    f"Ollama returned {len(vectors)} vectors for {len(chunk)} texts; retrying singly"
                )

        # Degraded path: isolate the bad input(s) rather than zeroing the whole batch.
        out: list[list[float]] = []
        for text in chunk:
            try:
                async with semaphore:
                    out.append((await _embed_texts(http_client, url, model, [text], timeout))[0])
            except Exception as e:
                logger.warning(f"Embed failed for text (len={len(text)}): {e}")
                out.append([0.0] * dim)
        return out

    chunks = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    results = await asyncio.gather(*[embed_chunk(c) for c in chunks])
    return [vector for chunk_result in results for vector in chunk_result]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
