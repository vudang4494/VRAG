#!/bin/bash
# ==============================================================================
# init-qdrant.sh — Initialize Qdrant collection for Mac Mini M4
# Optimized: HNSW with m=8 (reduced from 16) + scalar quantization
# RAM: ~0.5GB for index on 100K vectors
# ==============================================================================

QDRANT_HOST="${QDRANT_HOST:-http://localhost:6333}"
COLLECTION="${COLLECTION:-enterprise_kb}"
DIM="${DIM:-1024}"

echo "Initializing Qdrant collection: $COLLECTION at $QDRANT_HOST"

# Delete existing collection
curl -s -X DELETE "$QDRANT_HOST/collections/$COLLECTION" > /dev/null 2>&1

# Create optimized collection
# - HNSW m=8 (was 16) — less RAM, similar accuracy
# - HNSW ef_construct=100 (was 200) — faster indexing
# - Scalar quantization ON — 4x RAM reduction for vectors
# - on_disk=false — keep index in RAM for speed
curl -s -X PUT "$QDRANT_HOST/collections/$COLLECTION" \
  -H 'Content-Type: application/json' \
  -d "{
    \"vectors\": {
      \"size\": $DIM,
      \"distance\": \"Cosine\",
      \"hnsw_config\": {
        \"m\": 8,
        \"ef_construct\": 100
      },
      \"quantization_config\": {
        \"scalar\": {
          \"type\": \"int8\",
          \"quantile\": 0.99,
          \"always_ram\": true
        }
      }
    },
    \"hnsw_index\": {
      \"m\": 8,
      \"ef_construct\": 100,
      \"full_scan_threshold\": 10000,
      \"on_disk\": false
    },
    \"optimizers_config\": {
      \"indexing_threshold\": 10000,
      \"memmap_threshold\": 50000,
      \"flush_interval_sec\": 5
    },
    \"params\": {
      \"max_optimize_thread_count\": 2,
      \"max_segment_size\": 50000
    }
  }"

echo ""
echo "Collection '$COLLECTION' created with optimized HNSW + scalar quantization"
echo "Estimated RAM: ~0.5GB for 100K vectors (vs 2GB without quantization)"
