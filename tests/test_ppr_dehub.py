"""Unit tests for the cross-doc SuperNova de-hub in PPR (pick #1).

Covers the two pieces with real logic risk:
  1. `_build_npmi_graph` — NPMI must down-weight/prune hub pairs (an entity that
     co-occurs with everything) while keeping specific co-mentions.
  2. `_run_pagerank` — must work with or without scipy (the pure-python fallback
     that fixed PPR silently returning [] when scipy is absent).
"""

from __future__ import annotations

import networkx as nx

from src.services import ppr


class _FakeResult:
    """Async-iterable over pre-baked Neo4j records."""

    def __init__(self, rows):
        self._rows = rows

    def __aiter__(self):
        self._it = iter(self._rows)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    async def run(self, *_args, **_kwargs):
        return _FakeResult(self._rows)


# HUB co-occurs with every chunk; A and B co-occur strongly and specifically.
_CHUNKS = [
    {"ents": ["HUB", "A", "B"]},
    {"ents": ["HUB", "A", "B"]},
    {"ents": ["HUB", "C"]},
    {"ents": ["HUB", "D"]},
    {"ents": ["HUB", "E"]},
]


async def _build(npmi_min):
    g = nx.DiGraph()
    await ppr._build_npmi_graph(g, _FakeSession(_CHUNKS), "t", lambda x: x, npmi_min)
    return g


async def test_npmi_keeps_specific_pair():
    g = await _build(0.0)
    # A–B co-occur in every appearance → NPMI = 1.0, kept and highly weighted.
    assert g.has_edge("A", "B")
    assert g["A"]["B"]["weight"] > 0.9
    # Symmetric.
    assert g.has_edge("B", "A")


async def test_npmi_prunes_hub_edges():
    g = await _build(0.0)
    # HUB appears in every context → NPMI(HUB, x) = 0 → pruned at npmi_min=0.
    for other in ("A", "B", "C", "D", "E"):
        assert not g.has_edge("HUB", other), f"hub edge HUB-{other} should be pruned"


async def test_npmi_hub_edges_are_weak_when_kept():
    # With a permissive floor the hub edges appear but carry ~0 weight, so a
    # weighted walk still ignores them.
    g = await _build(-1.0)
    assert g.has_edge("HUB", "A")
    assert g["HUB"]["A"]["weight"] < 0.05
    assert g["A"]["B"]["weight"] > g["HUB"]["A"]["weight"]


async def test_ppr_retrieve_folds_alias_seed(monkeypatch):
    """A raw alias surface form must seed its canonical node.

    Graph nodes are canonical ("Cayman Islands"); the query entity arrives as the
    alias ("CAYMAN ISLANDS"). Without seed-folding the seed misses the graph and
    PPR returns [] — the bug the corpus500 benchmark caught.
    """
    g = nx.DiGraph()
    g.add_edge("Cayman Islands", "Ministry", weight=1.0)
    g.add_edge("Ministry", "Cayman Islands", weight=1.0)
    g.graph["alias_map"] = {"CAYMAN ISLANDS": "Cayman Islands"}

    async def _fake_load(_driver, _tenant):
        return g

    captured: dict = {}

    async def _fake_e2c(_driver, ranked_entities, _tenant, _k, chunk_ids_filter=None):
        captured["top_entity"] = ranked_entities[0][0] if ranked_entities else None
        return [
            {
                "id": "c1",
                "chunk_id": "c1",
                "text": "",
                "source": "s",
                "matched_entity": ranked_entities[0][0],
                "score": 1.0,
                "retrieval_path": "ppr",
            }
        ]

    monkeypatch.setattr(ppr, "_load_entity_graph", _fake_load)
    monkeypatch.setattr(ppr, "_entities_to_chunks", _fake_e2c)

    # Alias surface form folds → canonical node is seeded → chunks returned.
    chunks = await ppr.ppr_retrieve(None, ["CAYMAN ISLANDS"], "t", top_k_chunks=5)
    assert chunks, "alias seed must fold to canonical and return chunks"
    assert captured["top_entity"] == "Cayman Islands"

    # Negative control: an entity absent from graph AND alias map yields nothing.
    captured.clear()
    empty = await ppr.ppr_retrieve(None, ["Nonexistent Entity"], "t", top_k_chunks=5)
    assert empty == []


def test_run_pagerank_without_scipy_path():
    # Force the pure-python path and confirm it returns a valid distribution.
    g = nx.DiGraph()
    g.add_edge("a", "b", weight=1.0)
    g.add_edge("b", "a", weight=1.0)
    g.add_edge("b", "c", weight=0.5)
    g.add_edge("c", "b", weight=0.5)
    original = ppr._HAS_SCIPY
    try:
        ppr._HAS_SCIPY = False
        ranked = ppr._run_pagerank(
            g,
            alpha=0.5,
            personalization={"a": 1.0, "b": 0.0, "c": 0.0},
            max_iter=50,
            tol=1e-6,
            weight="weight",
        )
    finally:
        ppr._HAS_SCIPY = original
    assert set(ranked) == {"a", "b", "c"}
    assert abs(sum(ranked.values()) - 1.0) < 1e-6
    # Seeded on "a"; "a" must not be starved.
    assert ranked["a"] > 0.0
