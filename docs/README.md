# VRAG Documentation

This directory contains the full technical documentation for the VRAG project.

## Documentation Index

| File | Purpose |
|------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Deep technical reference — algorithms, data structures, retrieval paths, failure modes |
| [SPEC.md](SPEC.md) | Technical specification — pipeline tiers, KG schema, Qdrant schema, config |
| [SPEC-PREGEN-GATE.md](SPEC-PREGEN-GATE.md) | Pre-generation gate spec — PASS/RECOVER/REFUSE decision before generation |
| [PIPELINE_FLOW.md](PIPELINE_FLOW.md) | End-to-end query flow, stage by stage |
| [PRODUCTION_PIPELINE.md](PRODUCTION_PIPELINE.md) | Production config, Docker, env vars, latency targets |
| [STRUCTURE.md](STRUCTURE.md) | Module map — where each part of the pipeline lives |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Developer setup, ground rules, PR template |
| [`src/config.py`](../src/config.py) | LLM/embedding model decisions and rationale — lives in the config docstring so it cannot drift from the values it explains |

## Project Root

| File | Purpose |
|------|---------|
| [README.md](../README.md) | High-level overview, quick start, benchmarks |
| [.agents/AGENTS.md](../.agents/AGENTS.md) | Agent configuration and instructions |
