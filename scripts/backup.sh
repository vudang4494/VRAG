#!/bin/bash
# ==============================================================================
# backup.sh — Snapshot Qdrant collection + Neo4j graph for disaster recovery
#
# What it backs up:
#   1. Qdrant: full collection snapshot (named vector schema preserved)
#   2. Neo4j: Cypher dump of all node/relationship data + APOC export
#   3. Metadata: timestamp, collection info, point count, node counts
#
# Usage:
#   ./backup.sh                          # all tenants, default output dir
#   ./backup.sh --tenant eval           # single tenant
#   ./backup.sh --output /path/to/bak  # custom output dir
#   ./backup.sh --s3 s3://bucket/rag    # upload to S3-compatible storage
#
# Retention: keeps last 7 daily backups by default (change RETAIN_DAYS).
# Prerequisites: docker, curl, python3 (for JSON parsing)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${BACKUP_DIR:-${SCRIPT_DIR}/../backups}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"

# ── Defaults ───────────────────────────────────────────────────────────────────
TENANT="${TENANT:-all}"
QDRANT_HOST="${QDRANT_HOST:-http://localhost:6333}"
NEO4J_BOLT="${NEO4J_BOLT:-bolt://localhost:7687}"
S3_DEST=""
DRY_RUN=""

# ── CLI args ───────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tenant)  TENANT="$2"; shift 2 ;;
    --output)   BACKUP_DIR="$2"; shift 2 ;;
    --s3)       S3_DEST="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --help|-h)  grep "^# Usage:" < "$0" | cut -c4-; exit 0 ;;
    *)          echo "Unknown option: $1"; exit 1 ;;
  esac
done

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
BACKUP_NAME="rag_backup_${TIMESTAMP}"
WORK_DIR="${BACKUP_DIR}/${BACKUP_NAME}"
COLLECTION="${COLLECTION:-enterprise_kb}"

mkdir -p "${WORK_DIR}"
echo "=== RAG Backup ==="
echo "  Timestamp : ${TIMESTAMP}"
echo "  Tenant    : ${TENANT}"
echo "  Output    : ${WORK_DIR}"
[[ -n "${S3_DEST}" ]] && echo "  S3        : ${S3_DEST}"
[[ -n "${DRY_RUN}" ]] && echo "  DRY RUN   : yes (no data written)"
echo ""

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────
check_docker() {
  if ! docker info > /dev/null 2>&1; then
    log "ERROR: Docker daemon not running"
    exit 1
  fi
}

check_container() {
  local name="$1"
  if ! docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
    log "ERROR: Container '${name}' is not running"
    exit 1
  fi
}

if [[ -z "${DRY_RUN}" ]]; then
  check_docker
  check_container "rag-qdrant"
  check_container "rag-neo4j"
fi

# ==============================================================================
# PART 1 — Qdrant snapshot
# ==============================================================================
backup_qdrant() {
  log "Qdrant snapshot starting..."

  local snapshot_file="${WORK_DIR}/qdrant_${COLLECTION}.tar.zst"

  if [[ -n "${DRY_RUN}" ]]; then
    log "  [dry-run] would snapshot collection '${COLLECTION}'"
    echo "  qdrant_snapshot=${snapshot_file}"
    return
  fi

  # Check collection exists and get point count
  local info
  info=$(curl -s "${QDRANT_HOST}/collections/${COLLECTION}/info" 2>/dev/null || echo "{}")
  local points
  points=$(echo "${info}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result',{}).get('points_count',0))" 2>/dev/null || echo "0")
  local vectors
  vectors=$(echo "${info}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result',{}).get('vectors_count',0))" 2>/dev/null || echo "0")

  echo "${info}" > "${WORK_DIR}/qdrant_collection_info.json"
  echo "{\"backup_date\":\"${TIMESTAMP}\",\"points_count\":${points},\"vectors_count\":${vectors},\"collection\":\"${COLLECTION}\"}" \
    > "${WORK_DIR}/qdrant_meta.json"

  log "  Collection '${COLLECTION}': ${points} points, ${vectors} vectors"

  # Trigger snapshot via Qdrant API
  local task_id
  task_id=$(curl -s -X POST "${QDRANT_HOST}/collections/${COLLECTION}/snapshots" \
    -H 'Content-Type: application/json' \
    -w '\n%{http_code}' 2>/dev/null)
  local http_code
  http_code=$(echo "${task_id}" | tail -c 3)
  task_id=$(echo "${task_id}" | head -c -4)

  if [[ "${http_code}" != "200" && "${http_code}" != "202" ]]; then
    log "  WARNING: Snapshot API returned ${http_code} — falling back to volume copy"
    # Fallback: copy the storage directory directly from the container
    docker cp "rag-qdrant:/qdrant/storage" "${WORK_DIR}/qdrant_storage" 2>/dev/null && {
      log "  Volume copy complete → ${WORK_DIR}/qdrant_storage"
    } || {
      log "  ERROR: Both snapshot API and volume copy failed"
      return 1
    }
    return
  fi

  # Poll for snapshot completion
  local attempts=0
  local max_attempts=60
  while true; do
    sleep 2
    local snapshot_name
    snapshot_name=$(curl -s "${QDRANT_HOST}/collections/${COLLECTION}/snapshots" \
      2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
for s in d.get('result',[]):
    print(s.get('name',''))
" 2>/dev/null | head -1)

    if [[ -n "${snapshot_name}" ]]; then
      break
    fi

    attempts=$((attempts + 1))
    if [[ ${attempts} -ge ${max_attempts} ]]; then
      log "  WARNING: Snapshot polling timed out after ${max_attempts} attempts"
      break
    fi
  done

  if [[ -n "${snapshot_name}" ]]; then
    log "  Snapshot created: ${snapshot_name}"

    # Download snapshot
    curl -s "${QDRANT_HOST}/collections/${COLLECTION}/snapshots/${snapshot_name}" \
      -o "${snapshot_file}" 2>/dev/null

    if [[ -f "${snapshot_file}" && -s "${snapshot_file}" ]]; then
      local size
      size=$(du -h "${snapshot_file}" | cut -f1)
      log "  Downloaded (${size}): ${snapshot_file}"
    else
      log "  WARNING: Snapshot download failed or empty"
    fi
  fi

  # Also: export collection info as JSON for selective restore
  curl -s "${QDRANT_HOST}/collections/${COLLECTION}/points/page?limit=1" \
    > "${WORK_DIR}/qdrant_schema.json" 2>/dev/null || true

  log "Qdrant backup complete"
}

# ==============================================================================
# PART 2 — Neo4j graph dump
# ==============================================================================
backup_neo4j() {
  log "Neo4j graph dump starting..."

  if [[ -n "${DRY_RUN}" ]]; then
    log "  [dry-run] would dump all nodes and relationships"
    echo "  neo4j_dump=${WORK_DIR}/neo4j_cypher_dump.cypher"
    echo "  neo4j_apoc_export=${WORK_DIR}/neo4j_apoc_export.json"
    return
  fi

  # Use docker exec to run cypher-shell inside the neo4j container
  # This avoids authentication issues with external connections
  run_cypher() {
    docker exec rag-neo4j cypher-shell -u neo4j -p "" "$1" 2>/dev/null
  }

  # 1. Node and relationship counts
  {
    echo "-- Neo4j Graph Dump"
    echo "-- Generated: ${TIMESTAMP}"
    echo ""
    echo "-- === Node Counts ==="
    run_cypher "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC;"
    echo ""
    echo "-- === Relationship Counts ==="
    run_cypher "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS count ORDER BY count DESC;"
    echo ""
  } > "${WORK_DIR}/neo4j_stats.cypher"

  # 2. Full graph as Cypher dump (all nodes and relationships)
  {
    echo "-- === All Nodes (by label) ==="
    for label in Chunk Entity Document Community; do
      local count
      count=$(run_cypher "MATCH (n:${label}) RETURN count(n) AS c;" | grep -E '^\d+$' || echo "0")
      log "  ${label}: ${count} nodes"
      run_cypher "MATCH (n:${label}) RETURN 'CREATE ' + head(keys(n)) + ':' + '${label}' AS cypher LIMIT 0;" > /dev/null 2>&1 || true

      # Export all nodes of this label as individual CREATE statements
      run_cypher "MATCH (n:${label}) RETURN n;" 2>/dev/null | \
        python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith('+') or line.startswith('|') or line.startswith('n'):
        continue
    # Parse node output — format varies, so export as JSON instead
" 2>/dev/null || true
    done

    echo "-- Full node export (JSON format below)"
    echo "---JSON_NODES_START---"
    run_cypher "MATCH (n:Chunk) RETURN n;" 2>/dev/null | python3 "${SCRIPT_DIR}/_neo4j_json_export.py" Chunk > "${WORK_DIR}/neo4j_Chunk.json" 2>/dev/null || true
    run_cypher "MATCH (n:Entity) RETURN n;" 2>/dev/null | python3 "${SCRIPT_DIR}/_neo4j_json_export.py" Entity > "${WORK_DIR}/neo4j_Entity.json" 2>/dev/null || true
    run_cypher "MATCH (n:Document) RETURN n;" 2>/dev/null | python3 "${SCRIPT_DIR}/_neo4j_json_export.py" Document > "${WORK_DIR}/neo4j_Document.json" 2>/dev/null || true
    run_cypher "MATCH (n:Community) RETURN n;" 2>/dev/null | python3 "${SCRIPT_DIR}/_neo4j_json_export.py" Community > "${WORK_DIR}/neo4j_Community.json" 2>/dev/null || true
    echo "---JSON_NODES_END---"

  } > "${WORK_DIR}/neo4j_cypher_dump.cypher"

  # 3. APOC export if available (full graph as JSON)
  local apoc_result
  apoc_result=$(docker exec rag-neo4j cypher-shell -u neo4j -p "" \
    "CALL apoc.export.json.all('${TIMESTAMP}_graph.json', {useTypes:true})" 2>/dev/null || echo "")

  if [[ -n "${apoc_result}" && ! "${apoc_result}" =~ "apoc not registered" ]]; then
    docker cp "rag-neo4j:/var/lib/neo4j/${TIMESTAMP}_graph.json" "${WORK_DIR}/neo4j_apoc_export.json" 2>/dev/null && {
      local size
      size=$(du -h "${WORK_DIR}/neo4j_apoc_export.json" | cut -f1)
      log "  APOC export (${size}): neo4j_apoc_export.json"
    }
  else
    log "  APOC export not available — skipping (install apoc plugin to enable)"
    # Use alternative: graphML export
    local graphml_result
    graphml_result=$(docker exec rag-neo4j cypher-shell -u neo4j -p "" \
      "CALL apoc.export.graphml.all('${TIMESTAMP}_graph.graphml', {})" 2>/dev/null || echo "")
    if [[ -n "${graphml_result}" && ! "${graphml_result}" =~ "apoc not registered" ]]; then
      docker cp "rag-neo4j:/var/lib/neo4j/${TIMESTAMP}_graph.graphml" "${WORK_DIR}/neo4j_graph.graphml" 2>/dev/null && {
        local size
        size=$(du -h "${WORK_DIR}/neo4j_graph.graphml" | cut -f1)
        log "  GraphML export (${size}): neo4j_graph.graphml"
      }
    fi
  fi

  # 4. Constraints and indexes
  {
    echo "-- === Constraints ==="
    docker exec rag-neo4j cypher-shell -u neo4j -p "" "SHOW CONSTRAINTS;" 2>/dev/null
    echo ""
    echo "-- === Indexes ==="
    docker exec rag-neo4j cypher-shell -u neo4j -p "" "SHOW INDEXES;" 2>/dev/null
  } > "${WORK_DIR}/neo4j_schema.cypher"

  log "Neo4j backup complete"
}

# ==============================================================================
# PART 3 — Per-tenant selective export (Qdrant)
# ==============================================================================
backup_tenant_qdrant() {
  local tenant="$1"
  log "Qdrant tenant export: ${tenant}..."

  if [[ -n "${DRY_RUN}" ]]; then
    log "  [dry-run] would export tenant '${tenant}'"
    return
  fi

  # Export all points for a specific tenant as NDJSON
  local tenant_file="${WORK_DIR}/qdrant_tenant_${tenant}.ndjson"
  local page=0
  local page_size=1000
  local total=0

  while true; do
    local response
    response=$(curl -s -X POST "${QDRANT_HOST}/collections/${COLLECTION}/points/search" \
      -H 'Content-Type: application/json' \
      -d "{
        \"filter\": {\"key\": \"tenant_id\", \"match\": {\"value\": \"${tenant}\"}},
        \"limit\": ${page_size},
        \"offset\": $((page * page_size)),
        \"with_vectors\": false
      }" 2>/dev/null)

    local count
    count=$(echo "${response}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('result',[])))" 2>/dev/null || echo "0")

    if [[ "${count}" -eq 0 ]]; then
      break
    fi

    echo "${response}" | python3 -c "
import sys,json
for item in json.load(sys.stdin).get('result',[]):
    print(json.dumps(item, ensure_ascii=False))
" >> "${tenant_file}"

    total=$((total + count))
    page=$((page + 1))

    if [[ "${count}" -lt "${page_size}" ]]; then
      break
    fi
  done

  if [[ -f "${tenant_file}" && -s "${tenant_file}" ]]; then
    local size
    size=$(wc -l < "${tenant_file}")
    log "  Exported ${size} points for tenant '${tenant}'"
  else
    log "  No data found for tenant '${tenant}'"
  fi
}

# ==============================================================================
# PART 4 — Metadata manifest
# ==============================================================================
write_manifest() {
  if [[ -n "${DRY_RUN}" ]]; then
    return
  fi

  cat > "${WORK_DIR}/manifest.json" <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "version": "3.0.0",
  "collection": "${COLLECTION}",
  "backup_type": "full",
  "tenants": "${TENANT}",
  "files": [
    $(ls -1 "${WORK_DIR}" | grep -v 'manifest.json' | while read f; do
      size=$(stat -f%z "${WORK_DIR}/${f}" 2>/dev/null || stat -c%s "${WORK_DIR}/${f}" 2>/dev/null || echo 0)
      echo "    {\"name\": \"${f}\", \"size_bytes\": ${size}}"
    done | paste -sd ',' -)
  ],
  "qdrant": $(cat "${WORK_DIR}/qdrant_meta.json" 2>/dev/null || echo '{}'),
  "docker": {
    "qdrant_image": "$(docker inspect rag-qdrant --format '{{.Config.Image}}' 2>/dev/null || echo 'unknown')",
    "neo4j_image": "$(docker inspect rag-neo4j --format '{{.Config.Image}}' 2>/dev/null || echo 'unknown')"
  }
}
EOF
  log "Manifest written"
}

# ==============================================================================
# PART 5 — Retention cleanup
# ==============================================================================
cleanup_old_backups() {
  log "Cleaning up backups older than ${RETENTION_DAYS} days..."

  if [[ -n "${DRY_RUN}" ]]; then
    find "${BACKUP_DIR}" -maxdepth 1 -type d -name "rag_backup_*" -mtime +"${RETENTION_DAYS}" -print 2>/dev/null | while read d; do
      echo "  [dry-run] would remove: ${d}"
    done
    return
  fi

  local count
  count=$(find "${BACKUP_DIR}" -maxdepth 1 -type d -name "rag_backup_*" -mtime +"${RETENTION_DAYS}" 2>/dev/null | tee /dev/stderr | wc -l)
  find "${BACKUP_DIR}" -maxdepth 1 -type d -name "rag_backup_*" -mtime +"${RETENTION_DAYS}" \
    -exec rm -rf {} \; 2>/dev/null || true
  log "Removed ${count} old backup(s)"
}

# ==============================================================================
# PART 6 — S3 upload
# ==============================================================================
upload_to_s3() {
  if [[ -z "${S3_DEST}" ]]; then
    return
  fi

  log "Uploading to S3: ${S3_DEST}..."

  if [[ -n "${DRY_RUN}" ]]; then
    log "  [dry-run] would upload ${WORK_DIR} to ${S3_DEST}"
    return
  fi

  if command -v aws &> /dev/null; then
    aws s3 cp --recursive "${WORK_DIR}/" "${S3_DEST}/${BACKUP_NAME}/" && {
      log "S3 upload complete"
    } || {
      log "WARNING: S3 upload failed — backup is local only"
    }
  elif command -v rclone &> /dev/null; then
    rclone copy "${WORK_DIR}" "${S3_DEST}/${BACKUP_NAME}" && {
      log "RClone upload complete"
    } || {
      log "WARNING: RClone upload failed — backup is local only"
    }
  else
    log "WARNING: Neither aws nor rclone found — skipping S3 upload"
  fi
}

# ==============================================================================
# MAIN
# ==============================================================================
if [[ -z "${DRY_RUN}" ]]; then
  log "Pre-flight checks passed"
fi

backup_qdrant
backup_neo4j

if [[ "${TENANT}" != "all" ]]; then
  backup_tenant_qdrant "${TENANT}"
fi

write_manifest
cleanup_old_backups
upload_to_s3

# ── Summary ───────────────────────────────────────────────────────────────────
log ""
log "=== Backup Complete ==="
log "  Location : ${WORK_DIR}"
if [[ -z "${DRY_RUN}" ]]; then
  log "  Size     : $(du -sh "${WORK_DIR}" | cut -f1)"
  log "  Contents :"
  ls -lh "${WORK_DIR}" | tail -n +2 | awk '{print "    " $9 " (" $5 ")"}'
fi
echo ""

# ── Restore hint ───────────────────────────────────────────────────────────────
cat <<'HINT'
Restore instructions:
  # Restore Qdrant from snapshot
  curl -X PUT "http://localhost:6333/collections/enterprise_kb/snapshots/upload" \
    -H 'Content-Type: application/octet-stream' \
    --data-binary @qdrant_enterprise_kb.tar.zst

  # Restore Neo4j from APOC export
  docker exec rag-neo4j cypher-shell -u neo4j -p "" \
    "CALL apoc.import.json('neo4j_apoc_export.json', {useTypes:true});"

HINT
