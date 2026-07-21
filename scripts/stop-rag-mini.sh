#!/bin/bash
# ==============================================================================
# stop-rag-mini.sh — Stop RAG stack on Mac Mini M4
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
echo "Stopping RAG services..."
docker compose -f docker-compose.mini.yml down
echo "Done."
