#!/bin/bash
# ==============================================================================
# init-qdrant.sh — Initialize Qdrant collection for Pipeline V2 (Mac Mini M4)
#
# Schema:
#   - 5 named dense vectors: dense, paraphrase, question, summary, keywords (1024D each)
#   - 1 sparse vector: bm25
#   - Payload indexes for tenant_id, format, doc_id, access_level, chunk_level,
#     parent_chunk_id, sheet_name, thread_id, page_num, consistency_score
#
# Optimizations:
#   - HNSW m=8, ef_construct=100 (M4-tuned, less RAM)
#   - Scalar int8 quantization (4× RAM reduction)
#   - Full scan threshold 10K (for small filter sets)
#
# Compatibility:
#   - Pipeline V1 used a SINGLE 'dense' vector. V2 schema is INCOMPATIBLE.
#   - This script DROPS the existing collection. Run `make backup` first if you
#     have data to preserve.
# ==============================================================================

set -e

QDRANT_HOST="${QDRANT_HOST:-http://localhost:6333}"
COLLECTION="${COLLECTION:-enterprise_kb}"
DIM="${DIM:-1024}"
QDRANT_API_KEY_HEADER=""

if [ -n "$QDRANT_API_KEY" ]; then
  QDRANT_API_KEY_HEADER="-H api-key: $QDRANT_API_KEY"
fi

echo "[init-qdrant] Target: $QDRANT_HOST / collection $COLLECTION (dim=$DIM)"

# ── 1. Drop existing collection ────────────────────────────────────────────────
echo "[init-qdrant] Dropping existing collection (if any)..."
curl -s -X DELETE "$QDRANT_HOST/collections/$COLLECTION" $QDRANT_API_KEY_HEADER > /dev/null 2>&1 || true

# ── 2. Create collection with 5 named vectors + sparse ────────────────────────
echo "[init-qdrant] Creating collection with 5 named vectors + sparse BM25..."

VECTOR_CONFIG=$(cat <<EOF
{
  "dense": {
    "size": $DIM,
    "distance": "Cosine",
    "hnsw_config": { "m": 8, "ef_construct": 100, "on_disk": false }
  },
  "paraphrase": {
    "size": $DIM,
    "distance": "Cosine",
    "hnsw_config": { "m": 8, "ef_construct": 100, "on_disk": false }
  },
  "question": {
    "size": $DIM,
    "distance": "Cosine",
    "hnsw_config": { "m": 8, "ef_construct": 100, "on_disk": false }
  },
  "summary": {
    "size": $DIM,
    "distance": "Cosine",
    "hnsw_config": { "m": 8, "ef_construct": 100, "on_disk": false }
  },
  "keywords": {
    "size": $DIM,
    "distance": "Cosine",
    "hnsw_config": { "m": 8, "ef_construct": 100, "on_disk": false }
  }
}
EOF
)

SPARSE_CONFIG=$(cat <<EOF
{
  "bm25": {
    "index": { "on_disk": false }
  }
}
EOF
)

curl -s -X PUT "$QDRANT_HOST/collections/$COLLECTION" \
  -H 'Content-Type: application/json' \
  $QDRANT_API_KEY_HEADER \
  -d "{
    \"vectors\": $VECTOR_CONFIG,
    \"sparse_vectors\": $SPARSE_CONFIG,
    \"optimizers_config\": {
      \"indexing_threshold\": 10000,
      \"memmap_threshold\": 50000,
      \"flush_interval_sec\": 5,
      \"max_optimization_threads\": 2
    },
    \"quantization_config\": {
      \"scalar\": {
        \"type\": \"int8\",
        \"quantile\": 0.99,
        \"always_ram\": true
      }
    }
  }" > /tmp/qdrant_create.json

if grep -q '"status":"ok"' /tmp/qdrant_create.json 2>/dev/null; then
  echo "[init-qdrant] Collection created"
else
  echo "[init-qdrant] Collection create response:"
  cat /tmp/qdrant_create.json
fi

# ── 3. Create payload indexes for filter performance ──────────────────────────
echo "[init-qdrant] Creating payload indexes..."

create_index() {
  local field="$1"
  local schema="$2"
  echo "  - $field ($schema)"
  curl -s -X PUT "$QDRANT_HOST/collections/$COLLECTION/index" \
    -H 'Content-Type: application/json' \
    $QDRANT_API_KEY_HEADER \
    -d "{\"field_name\": \"$field\", \"field_schema\": \"$schema\"}" > /dev/null
}

# Always-filter fields (highest priority)
create_index "tenant_id"       "keyword"
create_index "format"          "keyword"
create_index "doc_id"          "keyword"
create_index "source"          "keyword"
create_index "access_level"    "keyword"
create_index "chunk_level"     "keyword"
create_index "parent_chunk_id" "keyword"

# Format-specific fields
create_index "sheet_name"      "keyword"
create_index "thread_id"       "keyword"
create_index "speaker"         "keyword"
create_index "page_num"        "integer"

# Quality + ranking fields
create_index "consistency_score" "float"

# RBAC + audit
create_index "department"      "keyword"
create_index "tags"            "keyword"
create_index "author"          "keyword"

# ── 4. Verify ──────────────────────────────────────────────────────────────────
echo ""
echo "[init-qdrant] Verifying collection..."
curl -s "$QDRANT_HOST/collections/$COLLECTION" $QDRANT_API_KEY_HEADER | python3 -m json.tool 2>/dev/null | head -50 || true

echo ""
echo "[init-qdrant] DONE."
echo "  Named vectors: dense, paraphrase, question, summary, keywords (each $DIM-D, Cosine, int8-quantized)"
echo "  Sparse vector: bm25"
echo "  Payload indexes: 14 fields"
echo "  Est. RAM @ 100K chunks: ~2 GB (with quantization)"
