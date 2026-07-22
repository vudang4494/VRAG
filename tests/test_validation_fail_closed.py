"""Unit tests verifying validation gate fail-closed behavior on backend exceptions."""

import pytest

from src.services.validation import grounding_gate_cosine


@pytest.mark.asyncio
async def test_grounding_gate_cosine_fails_closed_on_embed_error():
    """Verify that when the embedding backend fails, grounding_gate_cosine returns passed: False (fail-closed)."""

    class BrokenClient:
        async def post(self, *args, **kwargs):
            raise RuntimeError("Embedding service unavailable")

    result = await grounding_gate_cosine(
        answer="Meta reported $35B revenue in Q3.",
        passages=["Meta reported strong financial results."],
        http=BrokenClient(),
        embed_url="http://invalid:11434",
        embed_model="bge-m3",
    )

    assert result["passed"] is False
    assert result["grounded_ratio"] == 0.0
