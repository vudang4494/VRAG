# Enterprise Local RAG Stack V3 — Claude Code Entry Point

> ⚠ **BẮT BUỘC ĐỌC NGAY KHI MỞ SESSION**: `.claude/internal-docs/TODO.md`
>
> File đó có:
> - Trạng thái 11 phase (đã làm + chưa wire)
> - Việc tiếp theo theo thứ tự ưu tiên
> - Lệnh smoke test + đường dẫn quan trọng
>
> Nếu `.claude/` không tồn tại (clone mới), TODO file sẽ không có — hỏi user.

## Tổng quan

Sản phẩm: **Hybrid GraphRAG** chạy 100% local trên Apple Silicon (M-series Docker). V3 only — V1 legacy đã xóa hoàn toàn ở commit `4d7b556`.

Stack:
- Vector DB: Qdrant 1.13 (collection `enterprise_kb`, 6 named vectors gồm `graph_aware` GAEA)
- Knowledge Graph: Neo4j 5.26 Community + APOC
- LLM: Ollama (host, Metal GPU) — `qwen3.5:4b` + `bge-m3` embedding
- Entity NER: GLiNER `urchade/gliner_multi-v2.1` (168M zero-shot)
- API: FastAPI + uvloop, port 8800, V3 only
- Cache: Redis 7
- Observability: Langfuse v3 + Prometheus + Grafana

Toàn bộ surface API live ở `/api/v3/*` — xem `api/routes_v3.py` cho 12 endpoint.

## Lệnh thường dùng

```bash
# Health + smoke
curl -s http://localhost:8800/api/v3/health

# Real chat (eval tenant đã có data)
curl -s -X POST http://localhost:8800/api/v3/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"GraphRAG là gì?","tenant_id":"eval","max_retries":0}'

# Rebuild sau khi sửa code
docker compose build rag-api && docker compose up -d --force-recreate rag-api

# Logs
docker logs rag-api --tail 50 -f

# Ablation eval
python3 scripts/ablation_eval.py --tenant eval
```

## Convention

- Tất cả V3 code import qua `src.services.*` (không có V1 modules nữa)
- Mọi LLM call **phải** qua `src.services.ollama_helper.ollama_chat` (Qwen3 thinking-mode bypass)
- Mọi retrieval **phải** qua `multi_path_retrieve` ở `src/services/retrieval_v2.py`
- Internal docs nằm `.claude/internal-docs/` (gitignored, không push)
- Phase reports nằm `.claude/internal-docs/reports/`
- Plan tổng nằm `.claude/internal-docs/MASTER_ROADMAP.md`

## ĐỪNG làm

- Đừng tạo file V1 mới (`retrieval.py`, `ingestion.py`, `vector.py` đã xóa, đừng resurrect)
- Đừng push file `.claude/` lên GitHub (gitignore đã set)
- Đừng dùng `clients.llm.chat.completions.create()` trực tiếp (Qwen3 sẽ silent fail vào `message.thinking`) — phải qua `ollama_helper`
- Đừng commit emoji vào source code (theo system convention)
- Đừng amend commit cũ — luôn tạo commit mới

## Branches

- `main` — production, sync với origin
- `backup-c204404` — backup trước force-push history rewrite (V1 era)
- `backup-pre-cleanup-20260513` — backup trước V1 cleanup commit `4d7b556`
