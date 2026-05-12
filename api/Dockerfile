FROM --platform=linux/arm64 python:3.12-slim

LABEL maintainer="Enterprise RAG Team"
LABEL description="Enterprise RAG API — FastAPI + Hybrid GraphRAG + Ollama"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (layer cached) ────────────────────────────────────────────────
COPY api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# ── Source ────────────────────────────────────────────────────────────────────
COPY src/ ./src/
COPY api/ ./api/
COPY pyproject.toml .

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8800

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -fsS http://localhost:8800/health || exit 1

# Single worker: Ollama is the bottleneck, not FastAPI
# Async concurrency handled by semaphore in code
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8800", \
     "--workers", "1", \
     "--proxy-headers", \
     "--forwarded-allow-ips=*", \
     "--log-level", "info", \
     "--loop", "uvloop"]
