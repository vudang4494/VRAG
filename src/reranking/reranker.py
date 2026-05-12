"""Cross-encoder reranking for improved retrieval quality."""
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger


class BaseReranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """
        Rerank candidates by relevance to query.

        Args:
            query: The user query
            candidates: List of dicts with at least 'text' and 'score' keys
            top_k: Return only top N results

        Returns:
            Candidates sorted by rerank_score (descending), with new 'rerank_score' key added.
        """
        ...


class NoOpReranker(BaseReranker):
    """Pass-through reranker that skips reranking."""

    async def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        for c in candidates:
            c["rerank_score"] = c.get("score", 0.0)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]


class OllamaReranker(BaseReranker):
    """
    Rerank using an Ollama LLM as a scoring model.

    Uses a pairwise comparison approach: for each candidate, prompts the LLM
    to score relevance from 0-10.
    """

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        base_url: str = "http://localhost:11434",
        top_k: int = 20,
        batch_size: int = 5,
    ):
        self.model = model
        self.base_url = base_url
        self.default_top_k = top_k
        self.batch_size = batch_size
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            import httpx
            limits = httpx.Limits(max_connections=10)
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                limits=limits,
                timeout=httpx.Timeout(60.0),
            )
        return self._client

    async def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        if not candidates:
            return []

        if len(candidates) == 1:
            candidates[0]["rerank_score"] = candidates[0].get("score", 1.0)
            return candidates[:top_k]

        client = await self._get_client()
        reranked: list[dict] = []

        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i : i + self.batch_size]

            tasks = []
            for c in batch:
                tasks.append(self._score_single(client, query, c["text"][:1000]))

            scores = await asyncio.gather(*tasks, return_exceptions=True)

            for c, score_or_exc in zip(batch, scores):
                if isinstance(score_or_exc, Exception):
                    logger.warning(f"Rerank scoring failed: {score_or_exc}")
                    c["rerank_score"] = c.get("score", 0.0)
                else:
                    c["rerank_score"] = score_or_exc
                reranked.append(c)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    async def _score_single(self, client: Any, query: str, text: str) -> float:
        prompt = (
            f"Query: {query}\n\n"
            f"Document: {text}\n\n"
            f"Rate the relevance of this document to the query on a scale of 0 to 10.\n"
            f"0 = completely irrelevant, 10 = perfectly relevant.\n"
            f"Respond with ONLY a number between 0 and 10. No explanation."
        )
        try:
            r = await client.post(
                "/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 8,
                    "temperature": 0.0,
                },
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"].get("content", "") or r.json()["choices"][0]["message"].get("reasoning", "")
            import re
            match = re.search(r"\d+(?:\.\d+)?", content)
            return float(match.group(0)) / 10.0 if match else 0.0
        except Exception as e:
            logger.warning(f"LLM rerank error: {e}")
            return 0.0


class SemanticReranker(BaseReranker):
    """
    Rerank using cosine similarity between query embedding and candidate text embeddings.
    No extra LLM call needed — uses the same embedding model as the vector search.
    """

    def __init__(
        self,
        embed_url: str = "http://localhost:11434",
        embed_model: str = "bge-m3",
    ):
        self.embed_url = embed_url
        self.embed_model = embed_model
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.embed_url,
                timeout=httpx.Timeout(30.0),
            )
        return self._client

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        client = await self._get_client()
        vectors: list[list[float]] = []
        semaphore = asyncio.Semaphore(4)

        async def embed_one(text: str) -> list[float]:
            async with semaphore:
                r = await client.post(
                    "/api/embeddings",
                    json={"model": self.embed_model, "prompt": text[:2000]},
                )
                r.raise_for_status()
                return r.json()["embedding"]

        import asyncio
        results = await asyncio.gather(*[embed_one(t) for t in texts], return_exceptions=True)
        for vec in results:
            if isinstance(vec, Exception):
                vectors.append([0.0] * 1024)
            else:
                vectors.append(vec)
        return vectors

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    async def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        if not candidates:
            return []

        query_vecs = await self._embed_texts([query])
        doc_vecs = await self._embed_texts([c["text"][:2000] for c in candidates])
        query_vec = query_vecs[0]

        for c, vec in zip(candidates, doc_vecs):
            c["rerank_score"] = self._cosine(query_vec, vec)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]


import asyncio
