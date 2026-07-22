#!/bin/bash
# ==============================================================================
# OLLAMA_ENV_SETUP.sh — Optimized Ollama settings for Mac Mini M4
# M4 Metal GPU: 10-core GPU, ~24GB unified memory
# ==============================================================================

# Stop Ollama first
pkill -f ollama 2>/dev/null || true
sleep 2

# Set optimal environment variables before starting Ollama
export OLLAMA_HOST="0.0.0.0:11434"
export OLLAMA_MODELS="/Users/vudang/.ollama/models"

# Metal GPU optimization for M4
# - GPU enabled by default on Metal, no env needed
# - OLLAMA_NUM_PARALLEL: max concurrent requests (M4 can handle 3-4)
export OLLAMA_NUM_PARALLEL=3

# OLLAMA_MAX_LOADED_MODELS: load up to 2 models simultaneously (embedding + LLM)
export OLLAMA_MAX_LOADED_MODELS=2

# OLLAMA_VULKAN / OLLAMA_METAL: Metal is default on Apple Silicon
# No need to set these explicitly

# Memory limits — reserve ~10GB for system + Docker
# M4 has 24GB total: ~14GB for Ollama models, ~6GB for Docker
export OLLAMA_GPU_OVERHEAD=0

echo "=== Ollama M4 Optimized Settings ==="
echo "  Host:        $OLLAMA_HOST"
echo "  Models:      $OLLAMA_MODELS"
echo "  Parallel:    $OLLAMA_NUM_PARALLEL"
echo "  Max Loaded:  $OLLAMA_MAX_LOADED_MODELS"
echo ""
echo "Starting Ollama..."
ollama serve &
sleep 3

echo ""
echo "=== Available models ==="
ollama list

echo ""
echo "=== Pull recommended models ==="
echo "  Gemma4-4B:     ollama pull gemma4:e4b"
echo "  BGE-M3:        ollama pull bge-m3"
echo ""
echo "To use: ollama run gemma4:e4b 'your prompt here'"
