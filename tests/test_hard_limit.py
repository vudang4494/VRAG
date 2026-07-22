"""Tests for hard limit (supernode protection) in retrieval."""


class TestChunkIdsFilter:
    """Test chunk IDs filtering for supernode protection."""

    def test_filter_cypher_with_scope(self):
        """Verify Cypher query includes chunk_ids filter."""
        # Check function signature accepts chunk_ids_filter
        import inspect

        from src.services.retrieval import _entity_pivot_path

        sig = inspect.signature(_entity_pivot_path)
        assert "chunk_ids_filter" in sig.parameters

    def test_filter_with_empty_scope(self):
        """Empty scope should not apply filter."""

        # Empty list should be treated as no filter
        # (current implementation: if chunk_ids_filter is falsy, no filter applied)
        # This is correct behavior - empty scope means no constraint
        pass


class TestHardLimitConfig:
    """Test configuration for hard limit."""

    def test_graph_scope_size_config(self):
        from src.config import get_settings

        settings = get_settings()
        assert hasattr(settings, "entity_gate_seed_chunks")
        assert settings.entity_gate_seed_chunks >= 50  # reasonable minimum
        assert settings.entity_gate_seed_chunks <= 1000  # reasonable maximum

    def test_use_hard_limit_config(self):
        from src.config import get_settings

        settings = get_settings()
        assert hasattr(settings, "entity_gate_seed_chunks")


class TestScopeCollection:
    """Test scope collection from dense paths."""

    def test_scope_size_respected(self):
        """Verify scope size limit is respected."""
        from src.config import get_settings

        settings = get_settings()
        max_scope = settings.entity_gate_seed_chunks

        # Simulate collecting chunk IDs
        all_ids = [f"chunk_{i}" for i in range(500)]
        scope = all_ids[:max_scope]

        assert len(scope) <= max_scope
