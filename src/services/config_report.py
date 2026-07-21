"""Effective-config self-report: what this process IS, not what the repo claims.

Why this module exists
----------------------
A 527-document ingest run was evaluated against a configuration nobody was running.
Every layer disagreed and none of them said so:

  - src/config.py declared LLM gemma3:4b, a tag not installed on the box at all.
    Nothing failed at startup; the first intent-classification call 404'd instead.
  - .env.example shipped PII_LLM_NER_ENABLED=1 against a code default of 0, so every
    .env copied from it inherited a 62x ingest cost silently.
  - docker-compose.yml pinned its own third opinion of the same three model vars.
  - Six docs named three different LLMs; exactly one line in one doc was right.

None of that was exotic. It survived because the process never had to say out loud
what it had resolved. This module makes it say so, on every boot.

The rule this encodes: overrides are legitimate — deploys need them — but a silent
override is not. Anything that differs from the code default gets printed with a
marker, so drift shows up in the logs of every single run instead of being
excavated months later.

Safety by construction
----------------------
_GROUPS is an explicit allow-list. Secrets cannot leak here because they are never
named, rather than because a redaction regex is expected to catch them (a regex on
r"token" would also redact generation_max_tokens, which is exactly the kind of
almost-right that started all this).

Honesty about the drift check
-----------------------------
Drift = effective value != the field's declared default. That comparison is only
meaningful when the default is a literal in config.py. A field written as

    x: str = os.environ.get("X", "lit")

evaluates os.environ.get at import, so pydantic records the ENV VALUE as the field
default and the literal becomes unreachable — the check would compare the env value
against itself and report "no drift" for a field that is drifting. Rather than
quietly emit that false negative, _frozen_default_fields() re-reads config.py and
reports those fields as unknown. The set shrinks on its own as fields are converted
to declarative defaults; no hand-maintained list to fall out of date.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from loguru import logger

# Explicit allow-list of the settings that determine what a run MEANS. If two runs
# agree on every value here, their numbers are comparable; if they differ, they are
# not. Grouped for a readable banner.
_GROUPS: list[tuple[str, list[str]]] = [
    (
        "models",
        ["light_llm", "heavy_llm", "ollama_model", "ollama_embed_model", "embed_dimension"],
    ),
    (
        "retrieval",
        [
            "query_reformulations",
            "retrieval_top_k",
            "rrf_k",
            "retrieval_use_sparse",
            "retrieval_real_views_only",
            "ppr_enabled",
            "rrf_kg_path_weight_scale",
            "entity_cosine_enabled",
            "entity_gate_enabled",
            "community_enabled",
            "global_query_enabled",
            "global_query_max_communities",
        ],
    ),
    (
        "rerank",
        [
            "rerank_stage1_enabled",
            "rerank_stage3_enabled",
            "rerank_early_exit_threshold",
            "final_top_k",
        ],
    ),
    (
        "generation",
        [
            "generation_outline_enabled",
            "generation_drafts",
            "generation_judge_enabled",
            "generation_refine_enabled",
            "context_compression_enabled",
        ],
    ),
    (
        "validation",
        [
            "validation_enabled",
            "validation_min_grounded_ratio",
            "validation_max_invalid_entities",
            "validation_min_citation_ratio",
            "validation_retry_on_fail",
        ],
    ),
    (
        "ingest",
        [
            "pii_mask_enabled",
            "pii_llm_ner_enabled",
            "consistency_views_enabled",
            "doc_context_prefix_enabled",
            "entity_extractor_provider",
            "entity_vote_passes",
            "chunk_levels_csv",
        ],
    ),
    ("storage", ["qdrant_collection"]),
]

# Matches `name: T = <anything> os.environ.get(` — the pattern whose default is
# frozen at import and therefore not introspectable. See module docstring.
_ENV_GET_RE = re.compile(r"^\s{4}(\w+)\s*:\s*[^=\n]+=\s*[^\n]*os\.environ\.get\(", re.MULTILINE)


def _frozen_default_fields() -> set[str]:
    """Field names in config.py whose declared default is not trustworthy.

    Read from source rather than kept as a literal list here, so this cannot rot:
    convert a field to a declarative default and it drops out of the set with no
    edit to this module.
    """
    try:
        src = Path(__file__).resolve().parents[1] / "config.py"
        return set(_ENV_GET_RE.findall(src.read_text(encoding="utf-8")))
    except OSError as e:
        # Never let the self-report break startup — but do not pretend it worked.
        logger.warning(f"config_report: cannot read config.py to check defaults: {e}")
        return set()


def effective_config(settings: Any) -> dict[str, Any]:
    """Resolved value of every material setting, with drift against the code default.

    Returns {group: {field: {"value", "default", "drift"}}}. `drift` is None when the
    field's default is frozen (not introspectable) rather than False, so a consumer
    cannot mistake "cannot tell" for "verified identical".
    """
    frozen = _frozen_default_fields()
    fields = type(settings).model_fields
    out: dict[str, Any] = {}
    for group, names in _GROUPS:
        entries: dict[str, Any] = {}
        for name in names:
            info = fields.get(name)
            if info is None:
                # A rename landed and this list did not follow. Say so; do not skip.
                entries[name] = {"value": None, "default": None, "drift": None, "missing": True}
                continue
            value = getattr(settings, name, None)
            if name in frozen:
                entries[name] = {"value": value, "default": None, "drift": None}
            else:
                default = info.default
                entries[name] = {"value": value, "default": default, "drift": value != default}
        out[group] = entries
    return out


async def verify_models(clients: Any, settings: Any) -> dict[str, Any]:
    """Check every configured model tag actually exists in Ollama.

    This is the check whose absence let gemma3:4b sit in config.py as the default for
    all three LLM roles while not being installed: /api/health/deep already fetched
    /api/tags, but only listed them — it never asked whether the model this process
    intends to call was among them. So health reported ok and the failure surfaced
    much later, one call deep, as a 404 on an unrelated-looking request.

    Returns {"ok", "missing", "configured", "available", "error"}. Never raises:
    a self-report that can take the API down is worse than the drift it reports.
    """
    configured = {
        "light_llm": settings.light_llm,
        "heavy_llm": settings.heavy_llm,
        "ollama_model": settings.ollama_model,
        "ollama_embed_model": settings.ollama_embed_model,
    }
    try:
        resp = await clients.http.get(f"{settings.ollama_base_url}/api/tags", timeout=10.0)
        resp.raise_for_status()
        available = [m.get("name", "") for m in (resp.json().get("models") or [])]
    except Exception as e:
        return {
            "ok": None,  # unknown, NOT ok — Ollama unreachable is not a pass
            "missing": [],
            "configured": configured,
            "available": [],
            "error": str(e)[:200],
        }

    # Ollama treats `bge-m3` and `bge-m3:latest` as the same model; compare on the
    # bare name so a correct config is not reported as missing.
    bare = {a.split(":")[0] for a in available}
    missing = [
        f"{role}={tag}"
        for role, tag in configured.items()
        if tag not in available and tag.split(":")[0] not in bare
    ]
    return {
        "ok": not missing,
        "missing": missing,
        "configured": configured,
        "available": available,
        "error": None,
    }


def format_banner(settings: Any) -> list[str]:
    """Render the effective config as log lines. `*` marks a value overriding the code default."""
    cfg = effective_config(settings)
    lines = ["  ── effective config (`*` = overrides src/config.py default) ──"]
    for group, entries in cfg.items():
        parts = []
        for name, e in entries.items():
            if e.get("missing"):
                parts.append(f"{name}=<FIELD MISSING>")
                continue
            mark = "*" if e["drift"] else ""
            parts.append(f"{name}={e['value']}{mark}")
        lines.append(f"    {group:11} " + "  ".join(parts))
    frozen = _frozen_default_fields()
    shown = {n for _, names in _GROUPS for n in names}
    unknown = sorted(frozen & shown)
    if unknown:
        lines.append(
            f"    (default not introspectable for {len(unknown)} field(s) still using "
            f"os.environ.get, so `*` cannot be computed for them: {', '.join(unknown[:6])}"
            + (" ..." if len(unknown) > 6 else "")
            + ")"
        )
    return lines


async def log_startup_report(clients: Any, settings: Any) -> dict[str, Any]:
    """Log the effective-config banner and the model-existence check. Returns the check."""
    for line in format_banner(settings):
        logger.info(line)

    check = await verify_models(clients, settings)
    if check["ok"] is None:
        logger.warning(
            f"  model check SKIPPED — Ollama unreachable at {settings.ollama_base_url}: "
            f"{check['error']}"
        )
    elif check["ok"]:
        logger.info(
            f"  models present in Ollama: {', '.join(sorted(set(check['configured'].values())))}"
        )
    else:
        # Loud on purpose. This is the failure that previously waited to surface as a
        # 404 mid-request, long after startup had reported success.
        logger.error(
            f"  CONFIGURED MODEL NOT INSTALLED: {', '.join(check['missing'])}. "
            f"Available: {', '.join(check['available']) or '(none)'}. "
            f"Calls using it will fail at request time, not here. Fix: `ollama pull <tag>` "
            f"or correct LIGHT_LLM/HEAVY_LLM/OLLAMA_MODEL."
        )
    return check
