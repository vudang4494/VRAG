#!/bin/bash
# ==============================================================================
# start-rag-mini.sh — Start RAG stack on Mac Mini M4
# Usage: ./scripts/start-rag-mini.sh
# Prerequisites:
#   1. Ollama running on host:  ollama serve
#   2. Models pulled:           ollama pull gemma4:e4b && ollama pull bge-m3
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==============================================="
echo "  RAG Stack — Mac Mini M4 (24GB) Startup"
echo "==============================================="

# Check Ollama is running
echo "[1/5] Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ERROR: Ollama is not running!"
    echo "  Start it with: ollama serve"
    echo "  Or: ./scripts/start-ollama.sh"
    exit 1
fi
echo "  OK: Ollama is running"

# Check models
echo "[2/5] Checking models..."
MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; print([m['name'] for m in json.load(sys.stdin).get('models',[])])" 2>/dev/null || echo "[]")
echo "  Models: $MODELS"

# Initialize Qdrant collection
echo "[3/5] Initializing Qdrant..."
docker run --rm --network host \
    curlimages/curl:latest \
    -s http://localhost:6333/collections/enterprise_kb > /dev/null 2>&1 && echo "  OK: Collection exists" || {
    echo "  Creating optimized collection..."
    curl -s -X PUT http://localhost:6333/collections/enterprise_kb \
        -H 'Content-Type: application/json' \
        -d '{
            "vectors": {
                "size": 1024,
                "distance": "Cosine",
                "hnsw_config": {"m": 8, "ef_construct": 100},
                "quantization_config": {
                    "scalar": {"type": "int8", "quantile": 0.99, "always_ram": true}
                }
            },
            "hnsw_index": {"m": 8, "ef_construct": 100, "full_scan_threshold": 10000, "on_disk": false},
            "optimizers_config": {"indexing_threshold": 10000, "flush_interval_sec": 5}
        }' > /dev/null
    echo "  OK: Collection created"
}

# Initialize Neo4j schema
echo "[4/5] Checking Neo4j..."
sleep 2
NEO4J_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7474 || echo "000")
if [ "$NEO4J_HEALTH" = "200" ]; then
    echo "  OK: Neo4j is running"
else
    echo "  WARNING: Neo4j may not be ready (HTTP $NEO4J_HEALTH)"
fi

# Start Docker Compose
echo "[5/5] Starting Docker services..."
cd "$PROJECT_DIR"
docker compose -f docker-compose.mini.yml up -d --build

echo ""
echo "==============================================="
echo "  Services starting..."
echo ""
echo "  API:       http://localhost:8800"
echo "  Docs:      http://localhost:8800/docs"
echo "  Dashboard: http://localhost:7860"
echo "  Neo4j:     http://localhost:7474"
echo "  Qdrant:    http://localhost:6333"
echo "  Redis:     localhost:6379"
echo ""
echo "  Health:    curl http://localhost:8800/health"
echo "  Deep:      curl http://localhost:8800/health/deep"
echo "==============================================="
echo ""
echo "Waiting for API to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8800/health | grep -q "ok"; then
        echo "API is ready!"
        exit 0
    fi
    sleep 2
done
echo "Warning: API may still be starting. Check logs: docker logs rag-api"
