"""Global-query branch in /chat/stream (SSE).

Mocked (no Neo4j, no Ollama): flag ON + router says "global" → stream emits
meta(shortcut=global) + answer tokens + done; router says non-global → falls
through to the standard pipeline; flag OFF → router is never called (stream
behavior byte-identical to pre-feature).
"""

import json
from types import SimpleNamespace

import pytest

from api.routes import _chat_stream as cs


def _settings(global_on: bool):
    return SimpleNamespace(
        global_query_enabled=global_on,
        global_query_max_communities=10,
        ollama_model="m",
        light_llm="m",
        heavy_llm="m",
        ollama_embed_url="http://x",
        query_understanding_timeout_s=1.0,
    )


def _body():
    return {
        "query": "các chủ đề chính xuyên suốt tài liệu là gì?",
        "tenant_id": "t",
        "disable_intent": True,
        "disable_history_cache": True,
    }


async def _events(resp) -> list[dict]:
    out = []
    async for chunk in resp.body_iterator:
        for line in chunk.splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[len("data: ") :]))
    return out


def _patch_env(monkeypatch, global_on: bool):
    monkeypatch.setattr("src.config.get_settings", lambda: _settings(global_on))
    monkeypatch.setattr("src.clients.get_clients", lambda: SimpleNamespace(llm=None))
    monkeypatch.setattr("api.routes._chat_stream.get_settings", lambda: _settings(global_on))
    monkeypatch.setattr("api.routes._chat_stream.get_clients", lambda: SimpleNamespace(llm=None))


async def test_stream_global_shortcut(monkeypatch):
    _patch_env(monkeypatch, global_on=True)
    monkeypatch.setattr("src.services.query_router.classify_query", lambda q: "global")

    async def fake_gq(query, tenant_id, clients, settings, max_communities=20, **k):
        assert max_communities == 10, "must pass the config knob through"
        return {
            "answer": "GLOBAL ANSWER",
            "communities_used": 3,
            "communities_total": 5,
            "sources": [{"community_id": "c1", "chunk_id": "x"}],
            "no_data": False,
        }

    monkeypatch.setattr("src.services.global_query.global_map_reduce", fake_gq)

    events = await _events(await cs.chat_stream(_body()))
    meta = next(e for e in events if e["type"] == "meta")
    assert meta["shortcut"] == "global"
    assert meta["communities_used"] == 3
    assert meta["sources"] == [{"community_id": "c1", "chunk_id": "x"}]
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert answer == "GLOBAL ANSWER"
    done = next(e for e in events if e["type"] == "done")
    assert done["refused"] is False
    assert "global_map_reduce_ms" in done["latency_breakdown_ms"]


async def test_stream_non_global_falls_through(monkeypatch):
    _patch_env(monkeypatch, global_on=True)
    monkeypatch.setattr("src.services.query_router.classify_query", lambda q: "factual")

    async def gq_must_not_run(*a, **k):
        raise AssertionError("global_map_reduce must not run for non-global queries")

    monkeypatch.setattr("src.services.global_query.global_map_reduce", gq_must_not_run)

    async def sentinel(*a, **k):
        raise RuntimeError("reached-standard-pipeline")

    monkeypatch.setattr("src.services.query_understanding.understand_query", sentinel)

    events = await _events(await cs.chat_stream(_body()))
    err = next(e for e in events if e["type"] == "error")
    assert "reached-standard-pipeline" in err["error"]


async def test_stream_flag_off_never_calls_router(monkeypatch):
    _patch_env(monkeypatch, global_on=False)

    def router_must_not_run(q):
        raise AssertionError("classify_query must not be called when flag is OFF")

    monkeypatch.setattr("src.services.query_router.classify_query", router_must_not_run)

    async def sentinel(*a, **k):
        raise RuntimeError("reached-standard-pipeline")

    monkeypatch.setattr("src.services.query_understanding.understand_query", sentinel)

    events = await _events(await cs.chat_stream(_body()))
    err = next(e for e in events if e["type"] == "error")
    assert "reached-standard-pipeline" in err["error"]


async def test_stream_missing_query_still_400():
    with pytest.raises(Exception) as ei:
        await cs.chat_stream({"tenant_id": "t"})
    assert getattr(ei.value, "status_code", None) == 400
