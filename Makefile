# ==============================================================================
# Enterprise RAG Stack — Makefile
# Target: Apple Silicon Mac + Ollama + Qdrant + Neo4j + Langfuse
# ==============================================================================

.PHONY: help
help:
	@echo ""
	@echo "  Enterprise RAG Stack — Available commands:"
	@echo ""
	@echo "  === SETUP ==="
	@echo "  make secrets          Generate .env with strong secrets"
	@echo "  make init            First-time: secrets + directories + build"
	@echo ""
	@echo "  === STACK LIFECYCLE ==="
	@echo "  make up              Start all services"
	@echo "  make up-core         Start core only (qdrant, neo4j, postgres, redis, ollama, rag-api)"
	@echo "  make down            Stop all (keep data)"
	@echo "  make restart         Restart all"
	@echo "  make restart-core    Restart core only"
	@echo ""
	@echo "  === LOGS ==="
	@echo "  make logs            Tail all logs"
	@echo "  make logs-ollama     Ollama logs (watch model loading)"
	@echo "  make logs-api        RAG API logs"
	@echo "  make logs-qdrant     Qdrant logs"
	@echo "  make logs-neo4j      Neo4j logs"
	@echo ""
	@echo "  === MODELS ==="
	@echo "  make models          List available Ollama models"
	@echo "  make pull-model      Pull a model (usage: make pull-model MODEL=qwen3.5:4b)"
	@echo "  make preload-models  Pre-load LLM + embedding models into Ollama"
	@echo ""
	@echo "  === DATABASE INIT ==="
	@echo "  make init-qdrant     Create Qdrant collection (enterprise_kb)"
	@echo "  make init-neo4j      Create Neo4j schema (constraints + indexes)"
	@echo "  make init-all        Init both Qdrant + Neo4j"
	@echo ""
	@echo "  === HEALTH & STATUS ==="
	@echo "  make ps              Container status"
	@echo "  make health          Full health check all services"
	@echo "  make stats           Container CPU/RAM usage"
	@echo "  make deep-health     Deep health check with details"
	@echo ""
	@echo "  === TESTING ==="
	@echo "  make test-pytest    Full pytest suite (61 tests, 5m33s)"
	@echo "  make test-health    Service + API health (17 tests)"
	@echo "  make test-models    LLM + embedding quality (15 tests)"
	@echo "  make test-rag       Ingest + retrieval + chat (18 tests)"
	@echo "  make test-perf      Performance benchmarks (11 tests)"
	@echo "  make test-llm       Smoke: LLM chat via curl"
	@echo "  make test-embed     Smoke: Embedding via curl"
	@echo "  make test-ingest    Smoke: Document ingest via curl"
	@echo "  make test-all       Quick smoke tests (embed + rag)"
	@echo ""
	@echo "  === MAINTENANCE ==="
	@echo "  make pull            Pull latest images"
	@echo "  make build           Build rag-api Docker image"
	@echo "  make clean           Stop + remove containers (keep volumes)"
	@echo "  make nuke            DESTROY everything including volumes"
	@echo "  make backup          Backup volumes to ./backups/<timestamp>/"
	@echo "  make prune           Docker system prune (dangling images, volumes)"
	@echo ""
	@echo "  === DEV ==="
	@echo "  make lint            Run ruff linter"
	@echo "  make shell-api       Open shell inside rag-api container"
	@echo "  make shell-ollama    Open shell inside ollama container"
	@echo ""
	@echo ""

# ==============================================================================
# SETUP
# ==============================================================================

.PHONY: secrets init
secrets:
	@if [ -f .env ]; then \
		echo "WARNING: .env already exists. Backup or delete first."; \
		exit 1; \
	fi
	@echo "Generating strong secrets..."
	@openssl rand -hex 16 > /dev/null 2>&1 || echo "openssl not found — using fallback"
	@echo "# Enterprise RAG Stack — Environment Variables\n\
\n\
HF_TOKEN=${HF_TOKEN:-}\n\
ADMIN_EMAIL=${ADMIN_EMAIL:-admin@localhost}\n\
\n\
POSTGRES_PASSWORD=$$(openssl rand -base64 24 2>/dev/null | tr -d '/+=' | cut -c1-24 || python3 -c 'import secrets; print(secrets.token_hex(18))')\n\
NEO4J_PASSWORD=$$(openssl rand -base64 24 2>/dev/null | tr -d '/+=' | cut -c1-24 || python3 -c 'import secrets; print(secrets.token_hex(18))')\n\
REDIS_PASSWORD=$$(openssl rand -base64 24 2>/dev/null | tr -d '/+=' | cut -c1-24 || python3 -c 'import secrets; print(secrets.token_hex(18))')\n\
\n\
QDRANT_API_KEY=$$(openssl rand -base64 24 2>/dev/null | tr -d '/+=' | cut -c1-24 || python3 -c 'import secrets; print(secrets.token_hex(18))')\n\
\n\
LANGFUSE_DB_PASSWORD=$$(openssl rand -base64 24 2>/dev/null | tr -d '/+=' | cut -c1-24 || python3 -c 'import secrets; print(secrets.token_hex(18))')\n\
CLICKHOUSE_PASSWORD=$$(openssl rand -base64 24 2>/dev/null | tr -d '/+=' | cut -c1-24 || python3 -c 'import secrets; print(secrets.token_hex(18))')\n\
LANGFUSE_NEXTAUTH_SECRET=$$(openssl rand -base64 32 2>/dev/null || python3 -c 'import secrets; print(secrets.token_hex(32))')\n\
LANGFUSE_SALT=$$(openssl rand -base64 24 2>/dev/null | tr -d '/+=' | cut -c1-24 || python3 -c 'import secrets; print(secrets.token_hex(18))')\n\
LANGFUSE_ENCRYPTION_KEY=$$(openssl rand -hex 32 2>/dev/null || python3 -c 'import secrets; print(secrets.token_hex(32))')\n\
LANGFUSE_ADMIN_PASSWORD=$$(openssl rand -base64 12 2>/dev/null | tr -d '/+=' | cut -c1-16 || python3 -c 'import secrets; print(secrets.token_hex(12))')\n\
\n\
API_INTERNAL_KEY=$$(openssl rand -hex 32 2>/dev/null || python3 -c 'import secrets; print(secrets.token_hex(32))')\n\
WEBUI_SECRET_KEY=$$(openssl rand -base64 32 2>/dev/null || python3 -c 'import secrets; print(secrets.token_hex(32))')\n\
GRAFANA_PASSWORD=$$(openssl rand -base64 12 2>/dev/null | tr -d '/+=' | cut -c1-16 || python3 -c 'import secrets; print(secrets.token_hex(12))')\n\
\n\
OLLAMA_MODEL=qwen3.5:4b\n\
LOG_LEVEL=INFO\n\
" > .env
	@echo ".env created with auto-generated secrets."
	@echo "  Edit .env to set HF_TOKEN and ADMIN_EMAIL."
	@echo ""

init: secrets
	@echo ""
	@echo "=== Building Docker images ==="
	@make build
	@echo ""
	@echo "=== Creating SSL directory ==="
	@mkdir -p ssl
	@echo ""
	@echo "Setup complete! Next: make up"
	@echo ""

# ==============================================================================
# STACK LIFECYCLE
# ==============================================================================

up:
	@if [ ! -f .env ]; then echo "ERROR: No .env found. Run 'make secrets' or 'make init' first."; exit 1; fi
	docker compose up -d
	@echo ""
	@echo "Stack starting. Watch model loading: make logs-ollama"
	@echo "Run 'make health' to check services, 'make init-all' to init DBs."

up-core:
	@if [ ! -f .env ]; then echo "ERROR: No .env found. Run 'make secrets' first."; exit 1; fi
	docker compose up -d qdrant neo4j postgres redis langfuse-db langfuse-clickhouse langfuse ollama rag-api

down:
	docker compose down

restart:
	docker compose restart

restart-core:
	docker compose restart qdrant neo4j postgres redis ollama rag-api

# ==============================================================================
# LOGS
# ==============================================================================

logs:
	docker compose logs -f --tail=50

logs-ollama:
	docker compose logs -f --tail=100 ollama

logs-api:
	docker compose logs -f --tail=200 rag-api

logs-qdrant:
	docker compose logs -f --tail=50 qdrant

logs-neo4j:
	docker compose logs -f --tail=50 neo4j

logs-langfuse:
	docker compose logs -f --tail=50 langfuse

# ==============================================================================
# MODELS
# ==============================================================================

models:
	@echo "Available Ollama models:"
	@curl -sS http://localhost:11434/api/tags | python3 -m json.tool 2>/dev/null || echo "Ollama not ready yet"

pull-model:
	@if [ -z "$(MODEL)" ]; then echo "Usage: make pull-model MODEL=qwen3.5:4b"; exit 1; fi
	docker compose exec ollama ollama pull $(MODEL)

preload-models:
	@echo "Pre-loading models into Ollama..."
	@docker compose exec -d ollama sh -c "ollama pull qwen3.5:4b && ollama pull bge-m3"
	@echo "Models pulling in background. Check: make logs-ollama"

# ==============================================================================
# DATABASE INIT
# ==============================================================================

init-qdrant:
	@echo "Initializing Qdrant collection..."
	@bash scripts/init-qdrant.sh

init-neo4j:
	@echo "Initializing Neo4j schema..."
	@NEO4J_PASSWORD=$$(grep NEO4J_PASSWORD .env 2>/dev/null | cut -d= -f2); \
	if [ -z "$$NEO4J_PASSWORD" ]; then echo "ERROR: NEO4J_PASSWORD not found in .env"; exit 1; fi; \
	cat scripts/init-neo4j.cypher | docker exec -i rag-neo4j cypher-shell -u neo4j -p "$$NEO4J_PASSWORD"

init-all: init-qdrant init-neo4j

# ==============================================================================
# HEALTH & STATUS
# ==============================================================================

ps:
	docker compose ps

stats:
	@echo "=== Container Resource Usage ==="
	@docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" 2>/dev/null || echo "Docker stats unavailable"

health:
	@echo "=== Health Check ==="
	@echo ""
	@printf "%-20s" "Ollama:" && curl -fsS http://localhost:11434/api/tags > /dev/null 2>&1 && echo "OK" || echo "FAIL"
	@printf "%-20s" "Qdrant:" && curl -fsS http://localhost:6333/healthz > /dev/null 2>&1 && echo "OK" || echo "FAIL"
	@printf "%-20s" "Neo4j:" && curl -fsS http://localhost:7474 > /dev/null 2>&1 && echo "OK" || echo "FAIL"
	@printf "%-20s" "Postgres:" && docker exec rag-postgres pg_isready -U raguser -d ragdb > /dev/null 2>&1 && echo "OK" || echo "FAIL"
	@printf "%-20s" "Redis:" && docker exec rag-redis redis-cli -a "$$(grep REDIS_PASSWORD .env | cut -d= -f2)" ping > /dev/null 2>&1 && echo "OK" || echo "FAIL"
	@printf "%-20s" "RAG API:" && curl -fsS http://localhost:8800/health > /dev/null 2>&1 && echo "OK" || echo "FAIL"
	@printf "%-20s" "Langfuse:" && curl -fsS http://localhost:3000/api/public/health > /dev/null 2>&1 && echo "OK" || echo "FAIL"
	@printf "%-20s" "Prometheus:" && curl -fsS http://localhost:9090/-/healthy > /dev/null 2>&1 && echo "OK" || echo "FAIL"
	@printf "%-20s" "Grafana:" && curl -fsS http://localhost:3001/api/health > /dev/null 2>&1 && echo "OK" || echo "FAIL"
	@echo ""

deep-health:
	@echo "=== Deep Health Check ==="
	@echo ""
	@echo "--- Ollama models ---"
	@curl -sS http://localhost:11434/api/tags 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); [print('  -', m['name']) for m in d.get('models',[])]" || echo "  No models loaded"
	@echo ""
	@echo "--- Qdrant collections ---"
	@QDRANT_KEY=$$(grep QDRANT_API_KEY .env 2>/dev/null | cut -d= -f2); \
	curl -sS http://localhost:6333/collections $$(test -n "$$QDRANT_KEY" && echo "-H api-key: $$QDRANT_KEY" | tr -d '\n') 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('  Collections:', len(d.get('result',{}).get('collections',[])))" 2>/dev/null || echo "  FAIL"
	@echo ""
	@echo "--- RAG API deep ---"
	@curl -sS http://localhost:8800/health/deep 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  FAIL"
	@echo ""

# ==============================================================================
# TESTING  (pytest-based full suite)
# ==============================================================================

PYTEST := /Users/vudang/miniconda3/bin/python -m pytest

test-health:
	@echo "Running health tests..."
	$(PYTEST) tests/test_health.py -v

test-models:
	@echo "Running model tests (LLM + embedding)..."
	$(PYTEST) tests/test_models.py -v

test-rag:
	@echo "Running RAG pipeline tests (ingest + retrieval + chat)..."
	$(PYTEST) tests/test_rag_pipeline.py -v

test-perf:
	@echo "Running performance benchmarks..."
	$(PYTEST) tests/test_performance.py -v -s

test-pytest:
	@echo "Running FULL pytest suite (61 tests)..."
	$(PYTEST) tests/ -v --tb=short

test-all: test-health test-models test-rag

test-llm:
	@echo "Testing LLM chat via Ollama (host-native)..."
	curl -sS http://localhost:11434/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model":"qwen3.5:4b","messages":[{"role":"user","content":"Xin chào, bạn là ai?"}],"max_tokens":100}' \
		| python3 -m json.tool 2>/dev/null || echo "FAILED — Ollama may still be loading models"

test-embed:
	@echo "Testing embedding..."
	curl -sS http://localhost:11434/api/embeddings \
		-H "Content-Type: application/json" \
		-d '{"model":"bge-m3","prompt":"test embedding"}' \
		| python3 -c "import json,sys; d=json.load(sys.stdin); print(f'OK — dims: {len(d.get(\"embedding\",[]))}')" 2>/dev/null || echo "FAILED"

test-ingest:
	@echo "Testing document ingest..."
	@echo "Enterprise RAG system test document. Artificial Intelligence transforms businesses with retrieval augmented generation." > /tmp/rag_test.txt && \
	curl -sS -X POST http://localhost:8800/ingest/upload \
		-F "file=@/tmp/rag_test.txt" | python3 -m json.tool 2>/dev/null; \
	rm -f /tmp/rag_test.txt

test-rag-e2e:
	@echo "End-to-end RAG test..."
	curl -sS http://localhost:8800/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model":"qwen3.5:4b","messages":[{"role":"user","content":"Enterprise AI la gi?"}],"max_tokens":200}' \
		| python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('choices',[{}])[0].get('message',{}).get('content','NO RESPONSE'))" 2>/dev/null || echo "FAILED"

# ==============================================================================
# MAINTENANCE
# ==============================================================================

pull:
	docker compose pull

build:
	docker compose build rag-api

clean:
	docker compose down --remove-orphans

nuke:
	@echo "WARNING: This deletes ALL data (Qdrant, Neo4j, Postgres, Redis)."
	@echo -n "Type 'YES' to confirm: " && read confirm && [ "$$confirm" = "YES" ] || exit 1
	docker compose down -v --remove-orphans
	@echo "All volumes destroyed."

backup:
	@TS=$$(date +%Y%m%d_%H%M%S); \
	mkdir -p backups/$$TS; \
	echo "Backing up to backups/$$TS/"; \
	for vol in qdrant_data neo4j_data postgres_data redis_data langfuse_db; do \
		docker run --rm \
			-v "$${PWD}/backups/$$TS:/backup" \
			-v "rag_$$vol:/data:ro" \
			alpine \
			tar czf "/backup/$$vol.tar.gz" -C /data . 2>/dev/null && echo "  $$vol OK" || echo "  $$vol SKIP; \
	done; \
	echo "Backup complete → backups/$$TS/"

prune:
	docker system prune -f --volumes

# ==============================================================================
# DEV
# ==============================================================================

lint:
	@python3 -m ruff check src/ 2>/dev/null || echo "Install ruff: pip install ruff"

shell-api:
	docker compose exec rag-api /bin/sh

shell-ollama:
	docker compose exec ollama /bin/sh

# ==============================================================================
# WATCH UTILITIES
# ==============================================================================

watch-ollama:
	@echo "Watching Ollama (Ctrl+C to exit)..."
	docker compose logs -f --since=0 ollama

watch-models:
	@echo "Checking if models are loaded..."
	@curl -sS http://localhost:11434/api/tags | python3 -c "import json,sys; d=json.load(sys.stdin); models=d.get('models',[]); [print('  -', m['name'], '|', m.get('size','?')[:10]) for m in models]; print(); print('Loaded:', len(models), 'models')"
