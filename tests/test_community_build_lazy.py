"""Phase 2 — cluster-only (lazy) community build.

summarize=False must skip ALL LLM work (no generate_consistent_summary, no
chunk fetch) and write Community membership with summary=None (invisible to the
local cosine path). summarize=True still calls the LLM summary. Fully mocked —
no Neo4j, no Ollama.
"""

from unittest.mock import AsyncMock

import pytest

from src.services import community as cm

MEMBERSHIP = {"a": 0, "b": 0, "c": 0, "d": 1, "e": 1, "f": 1, "g": 2}
ENTITIES = [{"name": n} for n in "abcdefg"]


class _FakeSession:
    async def run(self, *a, **k):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeDriver:
    def session(self):
        return _FakeSession()


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(cm, "fetch_entity_graph", AsyncMock(return_value=(ENTITIES, [])))
    monkeypatch.setattr(cm, "cluster_leiden", lambda names, edges, resolution=1.0: dict(MEMBERSHIP))
    gcs = AsyncMock(return_value=("SUMMARY", 3))
    fce = AsyncMock(return_value=[{"chunk_id": "x", "text": "t"}])
    wc = AsyncMock()
    monkeypatch.setattr(cm, "generate_consistent_summary", gcs)
    monkeypatch.setattr(cm, "fetch_chunks_for_entities", fce)
    monkeypatch.setattr(cm, "write_community", wc)
    return gcs, fce, wc


async def test_lazy_build_writes_membership_without_llm(patched):
    gcs, fce, wc = patched
    res = await cm.build_communities_for_tenant(
        _FakeDriver(), llm=None, tenant_id="t", summarize=False, min_size=3
    )
    assert gcs.await_count == 0, "lazy build must not call the summary LLM"
    assert fce.await_count == 0, "lazy build must not fetch chunks"
    assert wc.await_count == 2, "2 groups (abc, def) >= min_size; g skipped"
    assert all(c.kwargs.get("summary") is None for c in wc.await_args_list)
    assert res["communities_written"] == 2
    assert res["summaries_written"] == 0
    assert res["skipped_small"] == 1


async def test_eager_build_calls_summary_llm(patched):
    gcs, fce, wc = patched
    res = await cm.build_communities_for_tenant(
        _FakeDriver(), llm=None, tenant_id="t", summarize=True, min_size=3
    )
    assert gcs.await_count == 2, "eager build summarizes each group >= min_size"
    assert wc.await_count == 2
    assert all(c.kwargs.get("summary") == "SUMMARY" for c in wc.await_args_list)
    assert res["summaries_written"] == 2


async def test_exclude_labels_drops_types_before_clustering(patched, monkeypatch):
    ents = [{"name": n, "type": ("person" if n in "ab" else "org")} for n in "abcdefg"]
    edges = [("a", "c", 1.0), ("c", "d", 1.0), ("b", "e", 1.0)]
    monkeypatch.setattr(cm, "fetch_entity_graph", AsyncMock(return_value=(ents, edges)))
    seen: dict = {}

    def fake_cluster(names, edges, resolution=1.0):
        seen["names"] = list(names)
        seen["edges"] = list(edges)
        return dict.fromkeys(names, 0)

    monkeypatch.setattr(cm, "cluster_leiden", fake_cluster)
    res = await cm.build_communities_for_tenant(
        _FakeDriver(),
        llm=None,
        tenant_id="t",
        summarize=False,
        min_size=3,
        exclude_labels=["PERSON"],  # uppercase input must normalize
    )
    assert seen["names"] == list("cdefg"), "person entities dropped before clustering"
    assert seen["edges"] == [("c", "d", 1.0)], "edges touching dropped entities removed"
    assert res["entities_excluded"] == 2
    assert res["entities_total"] == 7, "total reports pre-filter count"


async def test_exclude_labels_empty_is_noop(patched):
    res = await cm.build_communities_for_tenant(
        _FakeDriver(), llm=None, tenant_id="t", summarize=False, min_size=3, exclude_labels=[]
    )
    assert res["entities_excluded"] == 0
    assert res["communities_written"] == 2, "same result as before the knob existed"
