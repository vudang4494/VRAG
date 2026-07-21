"""LLM-judge gray-zone for entity resolution (resolve gate + audit).

Fully mocked (no Neo4j, no Ollama). Locks in:
- gray zone [threshold, judge_hi): judge NO -> no fold (fail-closed), YES -> fold;
- cos >= judge_hi: auto-accept, judge NOT called;
- audit: judged-NO edges deleted, judge error keeps the edge (fail-safe).
"""

from __future__ import annotations

import math

import numpy as np

from src.services import kg


class _FakeResult:
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
    def __init__(self, driver):
        self._driver = driver

    async def run(self, query, **params):
        self._driver.queries.append((query, params))
        if "RETURN e.name AS name" in query:  # candidate fetch (resolve)
            return _FakeResult(self._driver.candidates)
        if "RETURN a.name AS alias" in query:  # pair fetch (audit)
            return _FakeResult(self._driver.pairs)
        return _FakeResult([])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeDriver:
    def __init__(self, candidates=(), pairs=()):
        self.candidates = list(candidates)
        self.pairs = list(pairs)
        self.queries: list = []

    def session(self):
        return _FakeSession(self)

    def merges(self):
        return [q for q, _ in self.queries if "MERGE (alias)-[:ALIAS_OF]" in q]

    def deletes(self):
        return [p for q, p in self.queries if "DELETE r" in q]


def _vec(cos: float) -> np.ndarray:
    return np.array([cos, math.sqrt(max(0.0, 1 - cos * cos))])


_BASE = np.array([1.0, 0.0])


def _patch_vectors(monkeypatch, mapping):
    async def fake_vec(name, tenant_id, driver, qc, collection):
        return mapping.get(name)

    monkeypatch.setattr("src.services.entity_vectors.get_entity_vector", fake_vec)


def test_norm_eq_provable_identity():
    assert kg._norm_eq("MBPP_", "MBPP")
    assert kg._norm_eq("T-DRIVE", "T -DRIVE")
    assert kg._norm_eq("Direct Prompting", "direct prompting")
    assert kg._norm_eq("ScanNet__", "ScanNet")
    assert not kg._norm_eq("GPT-4_1", "GPT-4")  # 41 != 4 — version khác
    assert not kg._norm_eq("QLoRA", "LoRA")
    assert not kg._norm_eq("", "")


async def test_norm_eq_pair_skips_judge_and_folds(monkeypatch):
    d = _FakeDriver(
        candidates=[
            {"name": "MBPP", "type": "TECHNOLOGY", "deg": 10},
            {"name": "MBPP_", "type": "TECHNOLOGY", "deg": 5},
        ]
    )
    _patch_vectors(monkeypatch, {"MBPP": _BASE, "MBPP_": _vec(0.91)})

    async def judge_must_not_run(a, b, t, m):
        raise AssertionError("norm-eq pair must never reach the judge")

    monkeypatch.setattr(kg, "_judge_same_entity", judge_must_not_run)
    res = await kg.resolve_entity_aliases(
        d, None, "col", "t", judge_enabled=True, judge_types=["technology"]
    )
    assert res["resolved"] == 1 and res["judged"] == 0


async def test_audit_norm_eq_kept_without_judge(monkeypatch):
    d = _FakeDriver(pairs=[{"alias": "MBPP_", "etype": "TECHNOLOGY", "canon": "MBPP"}])

    async def judge_must_not_run(a, b, t, m):
        raise AssertionError("norm-eq pair must never reach the judge")

    async def vec_must_not_run(*a, **k):
        raise AssertionError("norm-eq pair needs no vectors")

    monkeypatch.setattr(kg, "_judge_same_entity", judge_must_not_run)
    monkeypatch.setattr("src.services.entity_vectors.get_entity_vector", vec_must_not_run)
    res = await kg.audit_alias_gray_zone(d, None, "col", "t", judge_types=["technology"])
    assert res["norm_kept"] == 1 and res["deleted"] == 0 and d.deletes() == []


async def test_judge_parses_yes_no_and_error(monkeypatch):
    async def yes(**k):
        return " YES"

    async def no(**k):
        return "No."

    async def boom(**k):
        raise RuntimeError("ollama down")

    monkeypatch.setattr("src.services.ollama_helper.ollama_chat", yes)
    assert await kg._judge_same_entity("a", "b", "TECHNOLOGY", None) is True
    monkeypatch.setattr("src.services.ollama_helper.ollama_chat", no)
    assert await kg._judge_same_entity("a", "b", "TECHNOLOGY", None) is False
    monkeypatch.setattr("src.services.ollama_helper.ollama_chat", boom)
    assert await kg._judge_same_entity("a", "b", "TECHNOLOGY", None) is None


async def _resolve(monkeypatch, driver, judge_verdict, cos, judge_hi=0.92):
    _patch_vectors(monkeypatch, {"LLMs": _BASE, "MLLMs": _vec(cos)})
    calls = []

    async def fake_judge(a, b, t, m):
        calls.append((a, b))
        return judge_verdict

    monkeypatch.setattr(kg, "_judge_same_entity", fake_judge)
    res = await kg.resolve_entity_aliases(
        driver,
        None,
        "col",
        "t",
        threshold=0.90,
        judge_enabled=True,
        judge_hi=judge_hi,
        judge_types=["technology"],
    )
    return res, calls


def _tech_candidates():
    return [
        {"name": "LLMs", "type": "TECHNOLOGY", "deg": 10},
        {"name": "MLLMs", "type": "TECHNOLOGY", "deg": 5},
    ]


async def test_gray_zone_judge_no_blocks_fold(monkeypatch):
    d = _FakeDriver(candidates=_tech_candidates())
    res, calls = await _resolve(monkeypatch, d, judge_verdict=False, cos=0.91)
    assert calls == [("MLLMs", "LLMs")]
    assert res["resolved"] == 0 and res["judge_rejected"] == 1
    assert d.merges() == [], "judge NO must block the ALIAS_OF write"


async def test_gray_zone_judge_yes_folds(monkeypatch):
    d = _FakeDriver(candidates=_tech_candidates())
    res, calls = await _resolve(monkeypatch, d, judge_verdict=True, cos=0.91)
    assert calls and res["resolved"] == 1 and res["judge_rejected"] == 0
    assert len(d.merges()) == 1


async def test_above_hi_skips_judge(monkeypatch):
    d = _FakeDriver(candidates=_tech_candidates())
    res, calls = await _resolve(monkeypatch, d, judge_verdict=False, cos=0.95)
    assert calls == [], "cos >= judge_hi must not call the judge"
    assert res["resolved"] == 1 and res["judged"] == 0


async def test_judge_error_fails_closed(monkeypatch):
    d = _FakeDriver(candidates=_tech_candidates())
    res, _ = await _resolve(monkeypatch, d, judge_verdict=None, cos=0.91)
    assert res["resolved"] == 0 and res["judge_rejected"] == 1
    assert d.merges() == []


async def test_audit_deletes_judged_no_keeps_errors(monkeypatch):
    pairs = [
        {"alias": "GPT-4_1", "etype": "PRODUCT", "canon": "GPT-4"},  # judge NO -> delete
        {"alias": "VLMs", "etype": "TECHNOLOGY", "canon": "VLM"},  # cos hi -> auto keep
        {"alias": "X1", "etype": "TECHNOLOGY", "canon": "X2"},  # judge error -> keep
    ]
    d = _FakeDriver(pairs=pairs)
    vectors = {
        "GPT-4_1": _vec(0.91),
        "GPT-4": _BASE,
        "VLMs": _vec(0.95),
        "VLM": _BASE,
        "X1": _vec(0.90),
        "X2": _BASE,
    }
    _patch_vectors(monkeypatch, vectors)

    async def fake_judge(a, b, t, m):
        return None if a == "X1" else False

    monkeypatch.setattr(kg, "_judge_same_entity", fake_judge)
    res = await kg.audit_alias_gray_zone(
        d, None, "col", "t", judge_hi=0.92, judge_types=["technology", "product"]
    )
    assert res["pairs"] == 3
    assert res["auto_kept"] == 1
    assert res["deleted"] == 1
    assert res["errors"] == 1
    assert [p["alias"] for p in d.deletes()] == ["GPT-4_1"], "only judged-NO edge deleted"
    assert res["removed"][0]["alias"] == "GPT-4_1"


async def test_audit_dry_run_deletes_nothing(monkeypatch):
    d = _FakeDriver(pairs=[{"alias": "A1", "etype": "TECHNOLOGY", "canon": "A2"}])
    _patch_vectors(monkeypatch, {"A1": _vec(0.90), "A2": _BASE})

    async def fake_judge(a, b, t, m):
        return False

    monkeypatch.setattr(kg, "_judge_same_entity", fake_judge)
    res = await kg.audit_alias_gray_zone(
        d, None, "col", "t", judge_types=["technology"], delete=False
    )
    assert res["deleted"] == 1 and d.deletes() == []
