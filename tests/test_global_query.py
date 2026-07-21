"""Phase 3 — query-time map-reduce (global_query.global_map_reduce).

Mocked (no Neo4j, no Ollama): verifies MAP gathers all communities, NO_INFO
survivors are dropped, REDUCE receives only survivors, sources track survivors,
0-community fails loud (no_data), and all-NO_INFO refuses.
"""

from types import SimpleNamespace

from src.services import global_query as gq


def _settings():
    return SimpleNamespace(
        light_llm="light",
        heavy_llm="heavy",
        generation_max_tokens=512,
        refusal_message_vi="REFUSE",
    )


def _clients():
    return SimpleNamespace(neo4j=object())


async def test_map_reduce_filters_no_info(monkeypatch):
    async def fake_comms(driver, tid, limit):
        return [("c1", ["a", "b", "c"]), ("c2", ["d", "e", "f"])]

    async def fake_chunks(driver, members, tid, limit):
        return [{"chunk_id": "x1", "text": "some text"}]

    calls = []

    async def fake_chat(messages, model, temperature=0.2, max_tokens=400, **k):
        content = messages[0]["content"]
        calls.append((model, content))
        if model == "heavy":  # reduce
            return "TỔNG HỢP TOÀN CỤC"
        if "[c2]" in content:  # map for c2 → irrelevant
            return "NO_INFO"
        return "- phát hiện cụm 1"

    monkeypatch.setattr(gq, "_fetch_communities", fake_comms)
    monkeypatch.setattr(gq, "fetch_chunks_for_entities", fake_chunks)
    monkeypatch.setattr(gq, "ollama_chat", fake_chat)

    res = await gq.global_map_reduce("q", "t", _clients(), _settings())

    assert res["no_data"] is False
    assert res["communities_total"] == 2
    assert res["communities_used"] == 1, "c2 NO_INFO must be dropped"
    assert "TỔNG HỢP" in res["answer"]
    reduce_content = next(c for m, c in calls if m == "heavy")
    assert "[c1]" in reduce_content
    assert "[c2]" not in reduce_content, "reduce must not receive the dropped cluster"
    assert res["sources"] == [{"community_id": "c1", "chunk_id": "x1"}]


async def test_no_community_fails_loud(monkeypatch):
    async def empty(driver, tid, limit):
        return []

    monkeypatch.setattr(gq, "_fetch_communities", empty)
    res = await gq.global_map_reduce("q", "t", _clients(), _settings())
    assert res["no_data"] is True
    assert res["communities_used"] == 0
    assert "build" in res["answer"].lower()


async def test_all_no_info_refuses(monkeypatch):
    async def comms(driver, tid, limit):
        return [("c1", ["a"])]

    async def fake_chunks(driver, members, tid, limit):
        return [{"chunk_id": "x", "text": "t"}]

    async def fake_chat(messages, model, **k):
        return "NO_INFO"

    monkeypatch.setattr(gq, "_fetch_communities", comms)
    monkeypatch.setattr(gq, "fetch_chunks_for_entities", fake_chunks)
    monkeypatch.setattr(gq, "ollama_chat", fake_chat)

    res = await gq.global_map_reduce("q", "t", _clients(), _settings())
    assert res["no_data"] is False
    assert res["communities_used"] == 0
    assert res["answer"] == "REFUSE"
