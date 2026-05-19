# Contributing to VRAG

Thanks for your interest in VRAG. This guide covers the things you need to know before opening a PR or filing an issue.

---

## Ground Rules

1. **VRAG is one product.** No `v1`/`v2`/`v3` in filenames, function names, classes, modules, or branding. The only `v3` allowed is the REST URL prefix `/api/v3/` — that's an API contract version, not a product version. If you find leftover version suffixes anywhere else, please file a PR to remove them.
2. **No emojis in source code.** Comments, docstrings, log messages — text only. Emojis are fine in README, issues, and PR descriptions.
3. **LLM calls must go through `src.services.ollama_helper.ollama_chat`.** Don't call `clients.llm.chat.completions.create()` directly. We need a single point for caching, tracing, and retries.
4. **Retrieval changes go through `multi_path_retrieve`** (`src/services/retrieval.py`). Don't add parallel retrieval orchestrators.
5. **Ingestion changes go through `ingest_document`** (`src/services/ingestion.py`). Same reason.
6. **Vector upsert goes through `upsert`** in `src/services/vector.py`.
7. **Never commit `.env` or files under `.claude/`.** Both are in `.gitignore` for a reason.

---

## Development Setup

### Prerequisites
- Apple Silicon (M-series) or x86_64 Linux
- Docker + Docker Compose
- Python 3.12+ (for running scripts/tests outside container)
- [Ollama](https://ollama.ai) on host with `qwen3.5:9b` and `bge-m3` pulled
- 16GB+ RAM (32GB recommended if running cross-encoder)

### Bootstrap

```bash
# Clone
git clone https://github.com/vudang4494/VRAG.git
cd VRAG

# Pull models on host Ollama
ollama pull qwen3.5:9b
ollama pull bge-m3

# Configure
cp .env.example .env   # set passwords

# Build + start stack
docker compose up -d

# Initialize storage
make init-all
python3 scripts/build_intent_centroids.py

# Smoke test
make smoke
```

### Iterating on code

```bash
# Edit src/ or api/
# Rebuild only the api container
docker compose build rag-api && docker compose up -d --force-recreate rag-api

# Tail logs
docker logs rag-api --tail 100 -f

# Run unit tests (inside or outside container)
make test-pytest
```

---

## Project Layout

See [README.md § Project Layout](README.md#project-layout) for the full tree. Key directories:

| Directory | Purpose |
|---|---|
| `api/` | FastAPI app, routes, Dockerfile, requirements |
| `src/` | Library code; everything LLM/retrieval-related lives here |
| `src/services/` | Domain logic — retrieval, ingestion, validation, etc. |
| `src/services/chunkers/` | Chunking strategies (hierarchical, multi-signal) |
| `config/` | Generated assets (intent centroids); committed when stable |
| `scripts/` | Operational scripts (build centroids, smoke test, benchmarks) |
| `dashboard/` | Gradio UI |
| `eval/` | Benchmark datasets + reports |
| `tests/` | pytest |
| `.claude/internal-docs/` | Project-internal docs — **gitignored** |

---

## What to Work On

High-value areas (see also [README.md § Roadmap](README.md#roadmap)):

1. **Multimodal ingestion** — table extraction (we have `docling`), image embedding (ColPali / SigLIP), chart understanding. Currently text-only.
2. **ETL connectors** — Google Drive, SharePoint, Confluence, Slack, Jira. Reuse `langchain_community.document_loaders` for the adapter and wire into `ingest_document`.
3. **Canonicalization Tier 2** — semantic entity merge via embedding cosine + `apoc.refactor.mergeNodes` for hard merges. See `ARCHITECTURE.md § 3.5`.
4. **RAGAS eval harness** — make benchmarks comparable to other open-source RAG systems.
5. **GPU optimization** — currently CPU-bound for LLM gen; a GPU profile in compose + tuned `OLLAMA_NUM_PARALLEL` would help.
6. **DSPy feedback loop** — capture 👍/👎 → optimize prompts and retrieval weights.

Smaller cleanup PRs that are always welcome:
- Tighter Cypher (eliminate `valid_to` / `domain_distribution` property warnings in logs)
- Test coverage for `retrieval.py` Phase 1 + Phase 2 split
- Better Vietnamese tokenization for BM25 sparse vectors

---

## Pull Requests

### Before you open one

- Run `make lint` and `make test-pytest` locally
- Make sure your branch is up to date with `main`
- For algorithmic changes, run `scripts/benchmark_eval.py` and include the report in the PR description so we can compare before/after

### PR description template

```markdown
## Summary
- One bullet per concrete change

## Why
- The user-facing problem this solves

## Test plan
- [ ] make test-pytest passes
- [ ] make smoke passes
- [ ] [optional] benchmark vs main: ...

## Notes for reviewers
- Anything weird, anything you punted on
```

### Commit messages

Imperative present tense, scoped prefix:

```
feat(retrieval): add adaptive scope size based on entropy
fix(rerank): handle empty candidates list in stage1
docs(architecture): clarify Tier 2 hard limit rationale
refactor(kg): extract canonicalization into separate module
```

Allowed scopes: `retrieval`, `ingestion`, `rerank`, `validation`, `kg`, `community`, `react`, `api`, `infra`, `docs`, `test`, `ops`.

Never:
- Amend an existing commit (we always create new ones)
- Force push to `main`
- Skip pre-commit hooks (`--no-verify`)
- Sign with `--no-gpg-sign` without explicit ask

---

## Code Style

- **Python:** `ruff check` + `ruff format` (config in `pyproject.toml`)
- **Types:** add type hints; we're not strict-mypy yet but new code should be typed
- **Imports:** absolute (`from src.services.vector import upsert`), not relative
- **Logging:** `loguru` everywhere; INFO for state transitions, DEBUG for per-item, WARNING for recoverable issues
- **Async:** prefer `asyncio.gather` for parallel I/O; never `time.sleep` in async code
- **Comments:** only when the *why* is non-obvious. Don't narrate what the code does.

---

## Testing

```bash
# Unit + integration (pytest)
make test-pytest

# End-to-end smoke (requires running stack)
make smoke

# Vietnamese eval benchmark (requires ingested corpus)
python3 scripts/benchmark_eval.py --eval eval/datasets/vi_benchmark_v2.json --tenant your_tenant
```

Adding tests:
- Unit tests go in `tests/test_<module>.py`
- Integration tests that need the full stack go in `tests/test_e2e_*.py` and are marked `@pytest.mark.e2e`
- Don't mock LLM calls in integration tests — use a tiny model (Qwen 0.5B) via `OLLAMA_MODEL` env override

---

## Reporting Issues

Good issue reports include:

1. **VRAG version** (commit hash from `git rev-parse HEAD`)
2. **Stack state** (`make ps` output)
3. **What you did** (the API call or UI action)
4. **What you expected**
5. **What happened** (response body + relevant log lines from `docker logs rag-api`)
6. **For perf issues:** include the `latency_breakdown_ms` from the response

---

## Code of Conduct

Be kind. Disagree with ideas, not people. No discrimination, harassment, or bad-faith arguments.

---

## License

By contributing you agree your work is licensed under [Apache 2.0](LICENSE).
