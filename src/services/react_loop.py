"""ReAct Loop — explicit multi-step Thought→Action→Observation reasoning.

Phase 2 novel contribution: makes LLM reasoning traceable + helps small LLM
(qwen3.5:4b) by decomposing complex queries into discrete sub-tasks.

Each step:
  1. THOUGHT: LLM decides next action based on history
  2. ACTION: execute one of the registered actions
  3. OBSERVATION: action result fed back to next thought

Action library:
  - search_entity(name)        — find entity in KG
  - expand_relation(entity)    — get related entities via RELATES_TO
  - retrieve_chunks(entities)  — chunks containing those entities
  - graph_aware_search(query)  — Qdrant search on refined (GAEA) embeddings
  - rerank(query, chunks)      — score top candidates by cross-encoder
  - FINISH                      — enough info to synthesize answer

Loop terminates at max_steps or FINISH. Final synthesis from accumulated chunks.

Why this beats vanilla LLM-as-decision-maker: small LLMs hallucinate when given
massive context. ReAct breaks task into small steps, each with focused context.
Each action result is concrete (Cypher rows, vector hits) — LLM just routes.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from loguru import logger


_THOUGHT_PROMPT = """Bạn là agent AI giúp trả lời câu hỏi qua việc khám phá knowledge graph.

Câu hỏi gốc: {query}

Các action có sẵn:
1. search_entity {{"name": "<tên entity>"}}
   → Tìm entity trong KG, trả về entity nodes + neighbors gần.
2. expand_relation {{"entity": "<tên>"}}
   → Lấy các entity related (1-hop) qua RELATES_TO.
3. retrieve_chunks {{"entities": ["<tên1>", "<tên2>"]}}
   → Chunks chứa CONTAINS_ENTITY tới những entities này.
4. graph_aware_search {{"query": "<text>"}}
   → Vector search trên refined embeddings, surface semantic-related chunks.
5. rerank {{"chunks_idxs": [0,1,2,...]}}
   → Rerank tập chunks đã thu thập theo relevance.
6. FINISH
   → Chỉ được FINISH khi ĐÃ THU THẬP ĐƯỢC ÍT NHẤT 4 CHUNKS có liên quan.
   Nếu chunks_collected < 4, phải tiếp tục bằng graph_aware_search hoặc retrieve_chunks.

Lịch sử các bước đã làm:
{history}

Số chunks đã thu thập đến giờ: {chunks_collected}

Bước tiếp theo nên là gì? Trả lời CHỈ với JSON:
{{"thought": "<lý do ngắn 1 câu>", "action": "<action_name>", "args": {{...}}}}

JSON:"""


_SYNTHESIZE_PROMPT = """Bạn là trợ lý AI nội bộ. Tổng hợp câu trả lời từ chunks đã được agent
thu thập qua nhiều bước.

Câu hỏi gốc: {query}

Lịch sử trace (cho thấy reasoning của agent):
{trace}

Các chunks tham khảo (chunk_id ở đầu):
{context}

QUY TẮC:
- Trả lời tiếng Việt, 2-5 câu
- MỖI câu kèm [chunk_id] ở cuối
- CHỈ dùng thông tin có trong chunks tham khảo
- Nếu chunks không đủ, viết "Tôi không có đủ thông tin chắc chắn"

Câu trả lời:"""


# ── Action executor ────────────────────────────────────────────────────────────


class ReActAction:
    """Holds context for action execution."""

    def __init__(self, clients: Any, settings: Any, tenant_id: str):
        self.clients = clients
        self.settings = settings
        self.tenant_id = tenant_id
        # Accumulated state across actions
        self.collected_chunks: list[dict] = []
        self.seen_chunk_ids: set[str] = set()
        self.discovered_entities: set[str] = set()

    async def search_entity(self, name: str) -> dict:
        """Find entity in KG + return nearby entities."""
        if not name or len(name.strip()) < 2:
            return {"entities": [], "message": "name too short"}

        # Try exact + fuzzy match
        cypher = """
        MATCH (e:Entity)
        WHERE (toLower(e.name) = toLower($name) OR toLower(e.name) CONTAINS toLower($name))
              AND e.tenant_id = $tid
        RETURN e.name AS name, e.type AS type, e.description AS desc, e.confidence AS conf
        LIMIT 10
        """
        async with self.clients.neo4j.session() as s:
            r = await s.run(cypher, name=name, tid=self.tenant_id)
            rows = await r.data()
        results = [
            {"name": row["name"], "type": row["type"], "description": (row.get("desc") or "")[:200], "confidence": row.get("conf")}
            for row in rows
        ]
        for r in results:
            self.discovered_entities.add(r["name"])
        return {"entities": results, "count": len(results)}

    async def expand_relation(self, entity: str) -> dict:
        """1-hop RELATES_TO from entity."""
        if not entity:
            return {"related": [], "message": "no entity provided"}
        cypher = """
        MATCH (e:Entity {tenant_id: $tid})
        WHERE toLower(e.name) = toLower($name) OR toLower(e.name) CONTAINS toLower($name)
        OPTIONAL MATCH (e)-[r:RELATES_TO]-(other:Entity)
        RETURN e.name AS source, other.name AS related, type(r) AS rel_type, r.description AS desc
        LIMIT 30
        """
        async with self.clients.neo4j.session() as s:
            r = await s.run(cypher, name=entity, tid=self.tenant_id)
            rows = await r.data()
        # Filter out None related (entity had no relations)
        related = [
            {"name": row["related"], "via": row["rel_type"], "description": (row.get("desc") or "")[:150]}
            for row in rows if row.get("related")
        ]
        for r in related:
            self.discovered_entities.add(r["name"])
        return {"source_entity": entity, "related": related, "count": len(related)}

    async def retrieve_chunks(self, entities: list[str], limit: int = 15) -> dict:
        """Chunks containing CONTAINS_ENTITY → any of given entities."""
        if not entities:
            return {"chunks_added": 0, "message": "no entities"}
        names_lower = [e.lower() for e in entities]
        cypher = """
        UNWIND $names AS qname
        MATCH (e:Entity {tenant_id: $tid})
        WHERE toLower(e.name) = qname OR toLower(e.name) CONTAINS qname
        MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e)
        WHERE c.tenant_id = $tid
        WITH c, count(DISTINCT e) AS match_count
        ORDER BY match_count DESC
        LIMIT $limit
        RETURN c.id AS chunk_id, c.text AS text, c.source AS source,
               c.format AS format, c.chunk_level AS chunk_level,
               match_count
        """
        async with self.clients.neo4j.session() as s:
            r = await s.run(cypher, names=names_lower, tid=self.tenant_id, limit=limit)
            rows = await r.data()

        added = 0
        for row in rows:
            cid = row["chunk_id"]
            if cid in self.seen_chunk_ids:
                continue
            self.seen_chunk_ids.add(cid)
            self.collected_chunks.append({
                "chunk_id": cid,
                "text": row["text"],
                "source": row["source"],
                "format": row.get("format"),
                "chunk_level": row.get("chunk_level"),
                "score": float(row["match_count"]) / max(len(entities), 1),
                "retrieval_path": "react:entity_pivot",
                "match_count": row["match_count"],
            })
            added += 1
        return {"chunks_added": added, "total_chunks": len(self.collected_chunks)}

    async def graph_aware_search(self, query: str, limit: int = 15) -> dict:
        """Vector search using GAEA-refined embeddings (limit raised from 10 → 15)."""
        if not query:
            return {"chunks_added": 0, "message": "empty query"}
        from src.services.embedding import embed_single
        from src.services.vector_v2 import build_tenant_filter, search_single_view

        try:
            q_vec = await embed_single(
                self.clients.http,
                self.settings.ollama_embed_url,
                self.settings.ollama_embed_model,
                query,
                timeout=30.0,
            )
        except Exception as e:
            return {"chunks_added": 0, "error": str(e)[:200]}

        flt = build_tenant_filter(tenant_id=self.tenant_id)
        # Try graph_aware first; fall back to dense if graph_aware not present.
        results = await search_single_view(
            self.clients.qdrant, self.settings.qdrant_collection,
            q_vec, view="graph_aware", limit=limit, filter_=flt,
        )
        if not results:
            results = await search_single_view(
                self.clients.qdrant, self.settings.qdrant_collection,
                q_vec, view="dense", limit=limit, filter_=flt,
            )

        added = 0
        for r in results:
            cid = r["chunk_id"]
            if cid in self.seen_chunk_ids:
                continue
            self.seen_chunk_ids.add(cid)
            self.collected_chunks.append({
                **r,
                "retrieval_path": "react:graph_aware",
            })
            added += 1
        return {"chunks_added": added, "total_chunks": len(self.collected_chunks)}

    async def rerank(self, query: str, top_n: int = 8) -> dict:
        """Rerank accumulated chunks by stage 2 semantic match (cheap, no LLM)."""
        if not self.collected_chunks:
            return {"reranked": 0, "message": "no chunks"}
        from src.services.rerank_stages import rerank_stage2

        ranked = await rerank_stage2(
            query, self.collected_chunks,
            self.clients.http,
            self.settings.ollama_embed_url,
            self.settings.ollama_embed_model,
            top_k=top_n,
        )
        self.collected_chunks = ranked
        return {
            "reranked": len(ranked),
            "top_scores": [round(c.get("stage2_score", 0), 3) for c in ranked[:3]],
        }


async def _decide_next_action_retry(
    query: str,
    history: list[dict],
    chunks_collected: int,
    model: str,
    retries: int = 1,
) -> dict[str, Any]:
    """Retry wrapper for _decide_next_action with limited retries."""
    import asyncio
    for attempt in range(retries + 1):
        result = await _decide_next_action(query, history, chunks_collected, model)
        action = result.get("action", "").upper()
        # Only accept FINISH if we have at least 4 chunks (raised from 2)
        if action == "FINISH" and chunks_collected < 4:
            if attempt < retries:
                continue
            # Last resort: force graph_aware_search
            result["action"] = "graph_aware_search"
            result["args"] = {"query": query}
        else:
            return result
    return result


# ── Thought decoder ────────────────────────────────────────────────────────────


async def _decide_next_action(
    query: str,
    history: list[dict],
    chunks_collected: int,
    model: str,
) -> dict[str, Any]:
    """LLM picks next action. Returns dict with 'thought', 'action', 'args'."""
    from src.services.ollama_helper import ollama_chat

    history_str = "\n".join(
        f"  Bước {i+1}: thought={h['thought'][:100]}, action={h['action']}, args={h.get('args')}, "
        f"observation={h.get('observation_summary', '')[:200]}"
        for i, h in enumerate(history)
    ) or "  (chưa có bước nào)"

    prompt = _THOUGHT_PROMPT.format(
        query=query,
        history=history_str,
        chunks_collected=chunks_collected,
    )

    raw = await ollama_chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.15,
        max_tokens=300,
    )

    # Parse JSON — retry on parse failure instead of silently FINISHing
    if not raw:
        logger.warning("ReAct: LLM returned empty, retrying step")
        return await _decide_next_action_retry(query, history, chunks_collected, model, retries=1)
    raw = re.sub(r"```(?:json)?\s*|\s*```$", "", raw).strip()
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        logger.warning("ReAct: JSON not found in LLM response, retrying step")
        return await _decide_next_action_retry(query, history, chunks_collected, model, retries=1)
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        logger.warning("ReAct: JSON decode failed, retrying step")
        return await _decide_next_action_retry(query, history, chunks_collected, model, retries=1)
    parsed.setdefault("thought", "")
    # Block FINISH if LLM chose it without enough chunks — force another action (4 chunks minimum)
    if parsed.get("action", "").upper() == "FINISH" and chunks_collected < 4:
        logger.info(f"ReAct: FINISH blocked (only {chunks_collected} chunks), forcing graph_aware_search")
        parsed["action"] = "graph_aware_search"
        parsed["args"] = {"query": query}
    else:
        parsed.setdefault("action", "graph_aware_search")  # safe default instead of FINISH
    parsed.setdefault("args", {})
    return parsed


# ── Summarize step ─────────────────────────────────────────────────────────────


def _format_context(chunks: list[dict], max_chunks: int = 5) -> str:
    lines = []
    for c in chunks[:max_chunks]:
        cid = c.get("chunk_id", "?")
        src = c.get("source", "?")
        text = (c.get("text") or "")[:1000]
        lines.append(f"[{cid}] (source: {src})\n{text}")
    return "\n\n---\n\n".join(lines)


def _format_trace(history: list[dict]) -> str:
    lines = []
    for i, h in enumerate(history):
        lines.append(
            f"Bước {i+1}: [{h['action']}] "
            f"{h['thought'][:100]} → {h.get('observation_summary','')[:100]}"
        )
    return "\n".join(lines)


async def _synthesize_answer(
    query: str,
    chunks: list[dict],
    history: list[dict],
    model: str,
    max_tokens: int = 600,
) -> str:
    from src.services.ollama_helper import ollama_chat
    if not chunks:
        return "Tôi không có đủ thông tin chắc chắn để trả lời câu hỏi này."

    prompt = _SYNTHESIZE_PROMPT.format(
        query=query,
        trace=_format_trace(history),
        context=_format_context(chunks),
    )
    return await ollama_chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.2,
        max_tokens=max_tokens,
    ) or "Tôi không có đủ thông tin chắc chắn."


def _observation_summary(action: str, result: dict) -> str:
    """Compact summary for the next thought's context."""
    if action == "search_entity":
        ents = result.get("entities", [])
        if not ents:
            return "0 entities found"
        names = [e["name"] for e in ents[:5]]
        return f"found {len(ents)} entities, top: {', '.join(names)}"
    if action == "expand_relation":
        rel = result.get("related", [])
        if not rel:
            return "no related entities"
        return f"{len(rel)} related: {', '.join(r['name'] for r in rel[:5])}"
    if action == "retrieve_chunks":
        return f"added {result.get('chunks_added',0)} new chunks (total {result.get('total_chunks',0)})"
    if action == "graph_aware_search":
        return f"added {result.get('chunks_added',0)} chunks via vector search"
    if action == "rerank":
        return f"reranked {result.get('reranked',0)}, top scores {result.get('top_scores',[])}"
    return str(result)[:200]


# ── Main loop orchestrator ────────────────────────────────────────────────────


async def react_chat(
    query: str,
    clients: Any,
    settings: Any,
    tenant_id: str = "default",
    max_steps: int = 6,   # increased from 4 — multi-hop needs more iterations for:
                            # search_entity → expand_relation → retrieve_chunks →
                            # graph_aware_search → rerank → FINISH (6 steps minimum)
) -> dict[str, Any]:
    """Run multi-step ReAct loop, return answer + full trace."""
    started = time.monotonic()
    actor = ReActAction(clients, settings, tenant_id)
    history: list[dict] = []
    step_latencies: list[dict] = []

    for step in range(max_steps):
        t0 = time.monotonic()
        thought_json = await _decide_next_action(
            query, history, len(actor.collected_chunks), settings.ollama_model,
        )
        thought_t = time.monotonic() - t0
        action_name = thought_json["action"]
        args = thought_json["args"] or {}

        if action_name == "FINISH":
            # Hard guard: never FINISH with fewer than 4 chunks collected.
            # Multi-hop / analytical queries need at least 4 chunks for good synthesis.
            # Force a final retrieval action before allowing synthesis.
            if len(actor.collected_chunks) < 4 and step < max_steps - 1:
                logger.info(
                    f"ReAct: FINISH rejected (only {len(actor.collected_chunks)} chunks), "
                    "forcing graph_aware_search before synthesis"
                )
                # Re-do the next action as graph_aware_search
                args = {"query": query}
                action_name = "graph_aware_search"
            else:
                history.append({
                    "step": step + 1,
                    "thought": thought_json["thought"],
                    "action": "FINISH",
                    "args": {},
                    "observation_summary": f"agent decided to finish with {len(actor.collected_chunks)} chunks",
                    "thought_ms": thought_t * 1000,
                    "action_ms": 0,
                })
                break

        # Execute action
        t1 = time.monotonic()
        result: dict[str, Any] = {}
        try:
            if action_name == "search_entity":
                result = await actor.search_entity(args.get("name", ""))
            elif action_name == "expand_relation":
                result = await actor.expand_relation(args.get("entity", ""))
            elif action_name == "retrieve_chunks":
                ents = args.get("entities", [])
                if isinstance(ents, str):
                    ents = [ents]
                result = await actor.retrieve_chunks(ents)
            elif action_name == "graph_aware_search":
                result = await actor.graph_aware_search(args.get("query", query))
            elif action_name == "rerank":
                result = await actor.rerank(query)
            else:
                result = {"error": f"unknown action: {action_name}"}
        except Exception as e:
            result = {"error": str(e)[:200]}
        action_t = time.monotonic() - t1

        history.append({
            "step": step + 1,
            "thought": thought_json["thought"],
            "action": action_name,
            "args": args,
            "observation_summary": _observation_summary(action_name, result),
            "thought_ms": thought_t * 1000,
            "action_ms": action_t * 1000,
        })

    # Final synthesize
    t2 = time.monotonic()
    # Ensure rerank ran at least once before synthesize
    if actor.collected_chunks and not any(h["action"] == "rerank" for h in history):
        try:
            await actor.rerank(query)
        except Exception:
            pass
    answer = await _synthesize_answer(query, actor.collected_chunks, history, settings.ollama_model)
    synth_t = time.monotonic() - t2

    total = time.monotonic() - started
    return {
        "answer": answer,
        "trace": history,
        "steps_used": len(history),
        "chunks_examined": len(actor.collected_chunks),
        "discovered_entities": list(actor.discovered_entities)[:30],
        "sources": [
            {
                "chunk_id": c.get("chunk_id"),
                "source": c.get("source"),
                "text": (c.get("text") or "")[:300],
                "score": c.get("stage2_score", c.get("score")),
            }
            for c in actor.collected_chunks[:5]
        ],
        "latency_ms": {
            "total": total * 1000,
            "synthesize": synth_t * 1000,
        },
    }
