"""ReAct Loop — explicit multi-step Thought→Action→Observation reasoning.

Phase 2 novel contribution: makes LLM reasoning traceable + helps small LLM
(gemma4:e4b) by decomposing complex queries into discrete sub-tasks.

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
   → Lấy các entity related (1-hop) qua typed RELATES_TO edges.
3. retrieve_chunks {{"entities": ["<tên1>", "<tên2>"]}}
   → Chunks chứa CONTAINS_ENTITY tới những entities này.
4. graph_aware_search {{"query": "<text>"}}
   → Vector search trên refined embeddings, surface semantic-related chunks.
5. expand_community {{"entity": "<tên>"}}
   → Tìm community membership của entity + summary. Hữu ích cho multi-hop queries.
6. entity_cosine_search {{"query": "<text>"}}
   → Cross-document entity-aware retrieval. Find top entities by vector cosine
     (with TF-IDF anti-supernova weighting), then pull chunks from those entities.
     ⭐ BEST FOR: multi-hop comparative queries (X vs Y, X khác Y) where you need
     chunks across multiple papers connected via shared/similar entities.
7. count_entities {{"entity_type": "<PERSON|ORG|LOC|..."}}
   → Đếm số entities theo type. Dùng cho aggregation queries.
8. verify_fact {{"claim": "<mệnh đề cần kiểm tra>"}}
   → Kiểm tra claim có support từ KG không. Dùng trước khi đưa vào câu trả lời.
9. rerank {{"top_n": 8}}
   → Rerank tập chunks đã thu thập theo relevance.
10. FINISH
   → Chỉ được FINISH khi ĐÃ THU THẬP ĐƯỢC ÍT NHẤT 4 CHUNKS có liên quan.
   Nếu chunks_collected < 4, phải tiếp tục bằng graph_aware_search hoặc retrieve_chunks.

CRITICAL RULE: Nếu bạn gọi search_entity hoặc expand_relation mà trả về 0 kết quả,
BẮT BUỘC phải gọi retrieve_chunks (dùng entity names từ câu hỏi gốc)
hoặc graph_aware_search (dùng câu hỏi gốc) TRƯỚC KHI ĐƯỢC PHÉP gọi FINISH.
Từ chối trả lời (FINISH khi không đủ chunks) là lựa chọn CUỐI CÙNG.

SUGGESTION: Với multi-hop queries, dùng expand_community trước để lấy global context,
sau đó search_entity + expand_relation để lấy local details.

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

Các chunks tham khảo:
{context}

QUY TẮC:
- Trả lời tiếng Việt, 3-7 câu
- Nếu câu nào có thể trích dẫn được, kèm [chunk_id] ở cuối
- Bạn được phép tổng hợp và diễn giải dựa trên ý nghĩa ngữ cảnh, miễn là không bịa đặt sự thật mới
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
        # Set by react_chat to drive workflow selection (factual / comparison /
        # multi_hop / analytical / kg_construction). Default is safe fallback.
        self.query_type: str = "factual"

    async def search_entity(self, name: str) -> dict:
        """Find entity in KG + return nearby entities. Uses e.norm_name (indexed,
        strip whitespace+underscore) so PDF-extract artifacts like 'Plan_RAG' /
        'A STUTE RAG' resolve from query 'PlanRAG' / 'AstuteRAG'."""
        if not name or len(name.strip()) < 2:
            return {"entities": [], "message": "name too short"}

        import re as _re

        name_norm = _re.sub(r"[\s_]+", "", name).lower()
        cypher = """
        MATCH (e:Entity)
        WHERE (toLower(e.name) = toLower($name)
               OR e.norm_name = $name_norm
               OR toLower(e.name) CONTAINS toLower($name))
              AND e.tenant_id = $tid
        RETURN e.name AS name, e.type AS type, e.description AS desc, e.confidence AS conf
        LIMIT 10
        """
        async with self.clients.neo4j.session() as s:
            r = await s.run(cypher, name=name, name_norm=name_norm, tid=self.tenant_id)
            rows = await r.data()
        results = [
            {
                "name": row["name"],
                "type": row["type"],
                "description": (row.get("desc") or "")[:200],
                "confidence": row.get("conf"),
            }
            for row in rows
        ]
        for r in results:
            self.discovered_entities.add(r["name"])
        return {"entities": results, "count": len(results)}

    async def expand_relation(self, entity: str) -> dict:
        """1-hop RELATES_TO from entity. Uses e.norm_name for whitespace-tolerant match."""
        if not entity:
            return {"related": [], "message": "no entity provided"}
        import re as _re

        name_norm = _re.sub(r"[\s_]+", "", entity).lower()
        cypher = """
        MATCH (e:Entity {tenant_id: $tid})
        WHERE toLower(e.name) = toLower($name)
              OR e.norm_name = $name_norm
              OR toLower(e.name) CONTAINS toLower($name)
        OPTIONAL MATCH (e)-[r:RELATES_TO]-(other:Entity)
        RETURN e.name AS source, other.name AS related,
               coalesce(r.rel_type, 'RELATES_TO') AS rel_type,
               r.description AS desc
        LIMIT 30
        """
        async with self.clients.neo4j.session() as s:
            r = await s.run(cypher, name=entity, name_norm=name_norm, tid=self.tenant_id)
            rows = await r.data()
        # Filter out None related (entity had no relations)
        related = [
            {
                "name": row["related"],
                "via": row["rel_type"],
                "description": (row.get("desc") or "")[:150],
            }
            for row in rows
            if row.get("related")
        ]
        for r in related:
            self.discovered_entities.add(r["name"])
        return {"source_entity": entity, "related": related, "count": len(related)}

    async def retrieve_chunks(self, entities: list[str], limit: int = 15) -> dict:
        """Chunks containing CONTAINS_ENTITY → any of given entities.

        Uses indexed `e.norm_name` for whitespace/underscore-tolerant match
        (rescues 'Plan_RAG' from query 'PlanRAG' etc., see backfill cypher).
        """
        if not entities:
            return {"chunks_added": 0, "message": "no entities"}
        import re as _re

        names_lower = [e.lower() for e in entities]
        names_norm = [_re.sub(r"[\s_]+", "", e).lower() for e in entities]
        cypher = """
        UNWIND range(0, size($names) - 1) AS i
        WITH $names[i] AS qname, $names_norm[i] AS qnorm
        MATCH (e:Entity {tenant_id: $tid})
        WHERE toLower(e.name) = qname
              OR e.norm_name = qnorm
              OR toLower(e.name) CONTAINS qname
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
            r = await s.run(
                cypher,
                names=names_lower,
                names_norm=names_norm,
                tid=self.tenant_id,
                limit=limit,
            )
            rows = await r.data()

        added = 0
        for row in rows:
            cid = row["chunk_id"]
            if cid in self.seen_chunk_ids:
                continue
            self.seen_chunk_ids.add(cid)
            self.collected_chunks.append(
                {
                    "chunk_id": cid,
                    "text": row["text"],
                    "source": row["source"],
                    "format": row.get("format"),
                    "chunk_level": row.get("chunk_level"),
                    "score": float(row["match_count"]) / max(len(entities), 1),
                    "retrieval_path": "react:entity_pivot",
                    "match_count": row["match_count"],
                }
            )
            added += 1
        return {"chunks_added": added, "total_chunks": len(self.collected_chunks)}

    async def graph_aware_search(self, query: str, limit: int = 15) -> dict:
        """Vector search using GAEA-refined embeddings (limit raised from 10 → 15)."""
        if not query:
            return {"chunks_added": 0, "message": "empty query"}
        from src.services.embedding import embed_single
        from src.services.vector import build_tenant_filter, search_single_view

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
            self.clients.qdrant,
            self.settings.qdrant_collection,
            q_vec,
            view="graph_aware",
            limit=limit,
            filter_=flt,
        )
        if not results:
            results = await search_single_view(
                self.clients.qdrant,
                self.settings.qdrant_collection,
                q_vec,
                view="dense",
                limit=limit,
                filter_=flt,
            )

        added = 0
        for r in results:
            cid = r["chunk_id"]
            if cid in self.seen_chunk_ids:
                continue
            self.seen_chunk_ids.add(cid)
            self.collected_chunks.append(
                {
                    **r,
                    "retrieval_path": "react:graph_aware",
                }
            )
            added += 1
        return {"chunks_added": added, "total_chunks": len(self.collected_chunks)}

    async def entity_cosine_search(self, query: str, top_k_entities: int = 15) -> dict:
        """Tier 3: cross-document entity-aware retrieval.

        Uses lazy entity centroids + L1 TF-IDF + L3 MMR + L5 sub-graph hard limit
        (scoped to chunks ALREADY collected in this ReAct session if any, else
        falls back to a fresh top-100 dense search for scope). Surfaces entities
        semantically similar to the query and pulls chunks from them — useful
        for multi-hop comparative queries.
        """
        if not query:
            return {"chunks_added": 0, "message": "empty query"}
        from src.services.embedding import embed_single
        from src.services.entity_vectors import entity_cosine_retrieve
        from src.services.vector import build_tenant_filter, search_single_view

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

        # Determine chunk_ids_scope (L5):
        # If we already collected chunks in this session, use those as scope.
        # Otherwise, run a quick dense search to establish scope.
        chunk_ids_scope: list[str]
        if self.collected_chunks:
            chunk_ids_scope = [
                c.get("chunk_id", "") for c in self.collected_chunks if c.get("chunk_id")
            ]
        else:
            flt = build_tenant_filter(tenant_id=self.tenant_id)
            seed = await search_single_view(
                self.clients.qdrant,
                self.settings.qdrant_collection,
                q_vec,
                view="dense",
                limit=100,
                filter_=flt,
            )
            chunk_ids_scope = [r["chunk_id"] for r in seed]

        if not chunk_ids_scope:
            return {"chunks_added": 0, "message": "no scope chunks for entity_cosine"}

        try:
            import numpy as np

            qv = np.asarray(q_vec, dtype=np.float32)
            qn = float(np.linalg.norm(qv))
            if qn > 0:
                qv = qv / qn
            ec_chunks, ec_entities = await entity_cosine_retrieve(
                query_vec=qv,
                chunk_ids_scope=chunk_ids_scope,
                tenant_id=self.tenant_id,
                neo4j_driver=self.clients.neo4j,
                qdrant_client=self.clients.qdrant,
                collection=self.settings.qdrant_collection,
                top_k_entities=top_k_entities,
                top_k_chunks=30,
                lambda_mmr=0.6,
            )
        except Exception as e:
            return {"chunks_added": 0, "error": f"entity_cosine failed: {str(e)[:150]}"}

        added = 0
        for r in ec_chunks:
            cid = r.get("chunk_id", "")
            if not cid or cid in self.seen_chunk_ids:
                continue
            self.seen_chunk_ids.add(cid)
            self.collected_chunks.append({**r, "retrieval_path": "react:entity_cosine"})
            added += 1
        top_entity_names = [e[0] for e in ec_entities[:5]]
        return {
            "chunks_added": added,
            "total_chunks": len(self.collected_chunks),
            "top_entities": top_entity_names,
        }

    async def ppr_search(self, entities: list[str] | None = None) -> dict:
        """HippoRAG 2 Personalized PageRank over the entity graph.

        Seeds the walk on the provided entities (or, if none, on entities
        already extracted from collected chunks). Surfaces chunks linked to
        entities reachable 2–3 hops away — the standard fix for multi-hop
        questions that single-shot retrieval misses.
        """
        from src.services.ppr import ppr_retrieve

        seed_names: list[str] = list(entities or [])
        if not seed_names:
            # Fall back to entities discovered in already-collected chunks.
            for c in self.collected_chunks:
                for e in (c.get("entities") or [])[:3]:
                    name = e.get("name") if isinstance(e, dict) else str(e)
                    if name and name not in seed_names:
                        seed_names.append(name)
            seed_names = seed_names[:10]

        if not seed_names:
            return {"chunks_added": 0, "message": "no seed entities"}

        try:
            ppr_chunks = await ppr_retrieve(
                neo4j_driver=self.clients.neo4j,
                query_entities=seed_names,
                tenant_id=self.tenant_id,
                top_k_chunks=20,
                alpha=getattr(self.settings, "ppr_alpha", 0.5),
            )
        except Exception as e:
            return {"chunks_added": 0, "error": f"ppr failed: {str(e)[:150]}"}

        added = 0
        for r in ppr_chunks:
            cid = r.get("chunk_id", "")
            if not cid or cid in self.seen_chunk_ids:
                continue
            self.seen_chunk_ids.add(cid)
            self.collected_chunks.append({**r, "retrieval_path": "react:ppr"})
            added += 1
        return {
            "chunks_added": added,
            "total_chunks": len(self.collected_chunks),
            "seeds_used": seed_names[:5],
        }

    async def rerank(self, query: str, top_n: int = 8) -> dict:
        """Rerank accumulated chunks by stage 2 semantic match (cheap, no LLM)."""
        if not self.collected_chunks:
            return {"reranked": 0, "message": "no chunks"}
        from src.services.rerank_stages import rerank_stage2

        ranked = await rerank_stage2(
            query,
            self.collected_chunks,
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

    async def expand_community(self, entity: str) -> dict:
        """Find which community an entity belongs to + return community summary.

        Enables the agent to access community-level global context in multi-hop queries.
        """
        if not entity:
            return {"communities": [], "message": "no entity provided"}
        import re as _re

        name_norm = _re.sub(r"[\s_]+", "", entity).lower()
        cypher = """
        MATCH (e:Entity {tenant_id: $tid})
        WHERE toLower(e.name) = toLower($name)
              OR e.norm_name = $name_norm
              OR toLower(e.name) CONTAINS toLower($name)
        MATCH (e)-[:IN_COMMUNITY]->(c:Community)
        RETURN c.id AS community_id, c.summary AS summary,
               c.level AS level, c.member_count AS member_count
        LIMIT 5
        """
        try:
            async with self.clients.neo4j.session() as s:
                r = await s.run(cypher, name=entity, name_norm=name_norm, tid=self.tenant_id)
                rows = await r.data()
        except Exception as e:
            logger.debug(f"expand_community failed: {e}")
            return {"communities": [], "message": str(e)}

        communities = [
            {
                "id": row["community_id"],
                "summary": (row.get("summary") or "")[:300],
                "level": row.get("level", 0),
                "member_count": row.get("member_count", 0),
            }
            for row in rows
            if row.get("community_id")
        ]
        return {"communities": communities, "count": len(communities)}

    async def count_entities(self, entity_type: str | None = None) -> dict:
        """Count entities in KG, optionally filtered by type.

        Useful for aggregation queries: "How many papers discuss X?"
        """
        try:
            async with self.clients.neo4j.session() as s:
                if entity_type:
                    cypher = """
                    MATCH (e:Entity {tenant_id: $tid})
                    WHERE e.type = $etype
                    RETURN count(e) AS total, collect(e.name)[..20] AS samples
                    """
                    r = await s.run(cypher, tid=self.tenant_id, etype=entity_type)
                else:
                    cypher = """
                    MATCH (e:Entity {tenant_id: $tid})
                    RETURN count(e) AS total
                    """
                    r = await s.run(cypher, tid=self.tenant_id)
                rows = await r.data()
        except Exception as e:
            logger.debug(f"count_entities failed: {e}")
            return {"total": 0, "samples": [], "message": str(e)}

        if not rows:
            return {"total": 0, "samples": []}
        row = rows[0]
        return {
            "total": row.get("total", 0),
            "samples": row.get("samples", [])[:10],
        }

    async def verify_fact(self, claim: str) -> dict:
        """Check if a specific claim is supported by KG entities/chunks.

        Pre-answer verification: instead of guessing, the agent can verify
        a specific claim before including it in the final answer.
        """
        if not claim or len(claim.strip()) < 5:
            return {"supported": False, "evidence": [], "message": "claim too short"}
        # Extract potential entity names from claim (simple heuristic)
        words = claim.split()
        entity_candidates = [w for w in words if w[0].isupper() and len(w) > 2]
        if not entity_candidates:
            return {"supported": False, "evidence": [], "message": "no entities in claim"}

        # Check if these entities exist in KG (norm_name-aware match)
        import re as _re

        names_top = entity_candidates[:5]
        names_norm = [_re.sub(r"[\s_]+", "", n).lower() for n in names_top]
        try:
            async with self.clients.neo4j.session() as s:
                cypher = """
                UNWIND range(0, size($names) - 1) AS i
                WITH $names[i] AS n, $names_norm[i] AS nnorm
                MATCH (e:Entity {tenant_id: $tid})
                WHERE toLower(e.name) = toLower(n)
                      OR e.norm_name = nnorm
                      OR toLower(e.name) CONTAINS toLower(n)
                RETURN e.name AS name, e.type AS type, e.description AS desc
                LIMIT 10
                """
                r = await s.run(
                    cypher,
                    names=names_top,
                    names_norm=names_norm,
                    tid=self.tenant_id,
                )
                rows = await r.data()
        except Exception as e:
            logger.debug(f"verify_fact failed: {e}")
            return {"supported": False, "evidence": [], "message": str(e)}

        if not rows:
            return {"supported": False, "evidence": [], "message": "no KG evidence"}

        evidence = [
            {
                "entity": row["name"],
                "type": row.get("type"),
                "description": (row.get("desc") or "")[:200],
            }
            for row in rows
        ]
        # If at least half of the entity candidates are found in KG, consider supported
        support_ratio = len(rows) / max(len(entity_candidates[:5]), 1)
        return {
            "supported": support_ratio >= 0.5,
            "support_ratio": round(support_ratio, 2),
            "evidence": evidence,
        }


# Note: _decide_next_action_retry was removed — its recursive retry pattern
# could loop indefinitely. _decide_next_action now has bounded internal retry.


# ── Thought decoder ────────────────────────────────────────────────────────────


# ─── Workflow engine ──────────────────────────────────────────────────────────
# Per-intent state machine. Each workflow is a finite list of (action, skip).
# - `action` is the tool name to execute (must be registered in react_chat).
# - `skip` is a pure-Python predicate ctx → bool; if True, the step is skipped.
# No LLM is consulted to choose actions or decide transitions.
# Loop-safe by construction (finite list, monotonic step pointer).
# Model-agnostic: swapping LLM doesn't change workflow execution.


def _skip_no_entities(ctx: dict) -> bool:
    return not ctx.get("entities")


def _skip_chunks_ge_8(ctx: dict) -> bool:
    return ctx.get("chunks_collected", 0) >= 8


def _skip_chunks_lt_4(ctx: dict) -> bool:
    return ctx.get("chunks_collected", 0) < 4


# Workflow shape: list[tuple[action_name, args_builder | None, skip_predicate | None]]
# args_builder = function ctx → dict (None means {}).
# skip_predicate = function ctx → bool (None means never skip).


def _args_search_entity(ctx: dict) -> dict:
    ents = ctx.get("entities") or []
    return {"name": ents[0] if ents else ""}


def _args_expand_relation(ctx: dict) -> dict:
    ents = ctx.get("entities") or []
    return {"entity": ents[0] if ents else ""}


def _args_retrieve_chunks(ctx: dict) -> dict:
    return {"entities": (ctx.get("entities") or [])[:5]}


def _args_graph_aware_search(ctx: dict) -> dict:
    return {"query": ctx.get("query", "")}


def _args_rerank(ctx: dict) -> dict:
    return {"top_n": 8}


def _args_expand_community(ctx: dict) -> dict:
    ents = ctx.get("entities") or []
    return {"entity": ents[0] if ents else ""}


def _args_count_entities(ctx: dict) -> dict:
    return {"entity_type": ctx.get("entity_type", "")}


WORKFLOWS: dict[str, list[tuple[str, Any, Any]]] = {
    "factual": [
        ("search_entity", _args_search_entity, _skip_no_entities),
        ("retrieve_chunks", _args_retrieve_chunks, None),
        ("rerank", _args_rerank, _skip_chunks_lt_4),
        ("FINISH", None, None),
    ],
    "comparison": [
        ("search_entity", _args_search_entity, _skip_no_entities),
        ("expand_relation", _args_expand_relation, _skip_no_entities),
        ("retrieve_chunks", _args_retrieve_chunks, None),
        ("graph_aware_search", _args_graph_aware_search, _skip_chunks_ge_8),
        ("rerank", _args_rerank, None),
        ("FINISH", None, None),
    ],
    "multi_hop": [
        ("expand_relation", _args_expand_relation, _skip_no_entities),
        ("retrieve_chunks", _args_retrieve_chunks, None),
        ("graph_aware_search", _args_graph_aware_search, _skip_chunks_ge_8),
        ("rerank", _args_rerank, None),
        ("FINISH", None, None),
    ],
    "analytical": [
        ("search_entity", _args_search_entity, _skip_no_entities),
        ("expand_relation", _args_expand_relation, _skip_no_entities),
        ("graph_aware_search", _args_graph_aware_search, None),
        ("retrieve_chunks", _args_retrieve_chunks, None),
        ("rerank", _args_rerank, None),
        ("FINISH", None, None),
    ],
    "kg_construction": [
        ("count_entities", _args_count_entities, None),
        ("expand_community", _args_expand_community, _skip_no_entities),
        ("retrieve_chunks", _args_retrieve_chunks, None),
        ("rerank", _args_rerank, None),
        ("FINISH", None, None),
    ],
}


def select_workflow(query_type: str) -> list[tuple[str, Any, Any]]:
    """Pure rule-based workflow selection. Defaults to `factual`."""
    return WORKFLOWS.get(query_type, WORKFLOWS["factual"])


def next_workflow_step(
    workflow: list[tuple[str, Any, Any]],
    step_index: int,
    ctx: dict,
) -> tuple[int, dict[str, Any]]:
    """Advance the workflow pointer past any skip-true steps, then return the
    next executable action. Returns (next_step_index, action_dict).

    Action dict shape: {"thought": str, "action": str, "args": dict}.
    Out-of-bounds step_index ⇒ FINISH (safety net).
    """
    while step_index < len(workflow):
        action_name, args_builder, skip = workflow[step_index]
        if skip is not None and skip(ctx):
            step_index += 1
            continue
        args = args_builder(ctx) if args_builder else {}
        return step_index + 1, {
            "thought": f"workflow[{step_index}]: {action_name}",
            "action": action_name,
            "args": args,
        }
    return step_index, {"thought": "workflow exhausted", "action": "FINISH", "args": {}}


def plan_next_react_action(
    query: str,
    history: list[dict],
    chunks_collected: int,
    seed_entities: list[str] | None = None,
) -> dict[str, Any]:
    """Legacy single-workflow rule planner. Kept for back-compat callers.
    Production path uses select_workflow + next_workflow_step instead."""
    seen = {h.get("action") for h in history}
    if chunks_collected >= 10 and "rerank" in seen:
        return {"thought": "rule: rich pre-seed, finish", "action": "FINISH", "args": {}}
    if seed_entities and "search_entity" not in seen:
        return {
            "thought": "rule: anchor",
            "action": "search_entity",
            "args": {"name": seed_entities[0]},
        }
    if "search_entity" in seen and "expand_relation" not in seen and seed_entities:
        return {
            "thought": "rule: expand",
            "action": "expand_relation",
            "args": {"entity": seed_entities[0]},
        }
    if "retrieve_chunks" not in seen:
        ents = seed_entities[:5] if seed_entities else []
        return {
            "thought": "rule: retrieve",
            "action": "retrieve_chunks",
            "args": {"entities": ents},
        }
    if "graph_aware_search" not in seen and chunks_collected < 6:
        return {"thought": "rule: dense", "action": "graph_aware_search", "args": {"query": query}}
    if "rerank" not in seen and chunks_collected >= 4:
        return {"thought": "rule: rerank", "action": "rerank", "args": {"top_n": 8}}
    if chunks_collected >= 4:
        return {"thought": "rule: finish", "action": "FINISH", "args": {}}
    return {
        "thought": "rule: final sweep",
        "action": "graph_aware_search",
        "args": {"query": query},
    }


async def _decide_next_action(
    query: str,
    history: list[dict],
    chunks_collected: int,
    model: str,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """LLM picks next action. Returns dict with 'thought', 'action', 'args'.

    Bounded internal retry: up to `max_attempts` LLM calls before giving up
    and returning a safe-default graph_aware_search. Previously this used
    recursive retry which had no exit condition, causing infinite loops
    when the LLM consistently returned non-JSON output.
    """
    from src.services.ollama_helper import ollama_chat

    history_str = (
        "\n".join(
            f"  Bước {i + 1}: thought={h['thought'][:100]}, action={h['action']}, args={h.get('args')}, "
            f"observation={h.get('observation_summary', '')[:200]}"
            for i, h in enumerate(history)
        )
        or "  (chưa có bước nào)"
    )

    prompt = _THOUGHT_PROMPT.format(
        query=query,
        history=history_str,
        chunks_collected=chunks_collected,
    )

    last_failure_reason = "no attempts made"
    for attempt in range(1, max_attempts + 1):
        raw = await ollama_chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.15,
            max_tokens=300,
        )

        if not raw:
            last_failure_reason = "empty response"
            logger.warning(f"ReAct: LLM returned empty (attempt {attempt}/{max_attempts})")
            continue
        raw_clean = re.sub(r"```(?:json)?\s*|\s*```$", "", raw).strip()
        match = re.search(r"\{[\s\S]*\}", raw_clean)
        if not match:
            last_failure_reason = "no JSON in response"
            logger.warning(f"ReAct: no JSON in response (attempt {attempt}/{max_attempts})")
            continue
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            last_failure_reason = "JSON decode failed"
            logger.warning(f"ReAct: JSON decode failed (attempt {attempt}/{max_attempts})")
            continue

        # Successful parse → enforce action policy
        parsed.setdefault("thought", "")
        action = parsed.get("action", "").upper()
        if action == "FINISH" and chunks_collected < 4:
            if attempt < max_attempts:
                # Inject hint into prompt and retry
                logger.info(f"ReAct: FINISH blocked (chunks={chunks_collected}, attempt {attempt})")
                continue
            # Last attempt: force safe default
            parsed["action"] = "graph_aware_search"
            parsed["args"] = {"query": query}
        else:
            parsed.setdefault("action", "graph_aware_search")
        parsed.setdefault("args", {})
        return parsed

    # All attempts exhausted — return safe default rather than infinite loop
    logger.error(
        f"ReAct: all {max_attempts} attempts failed ({last_failure_reason}) — "
        f"falling back to graph_aware_search"
    )
    return {
        "thought": f"(LLM kept returning invalid output: {last_failure_reason})",
        "action": "graph_aware_search",
        "args": {"query": query},
    }


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
            f"Bước {i + 1}: [{h['action']}] "
            f"{h['thought'][:100]} → {h.get('observation_summary', '')[:100]}"
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
    return (
        await ollama_chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        or "Tôi không có đủ thông tin chắc chắn."
    )


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
        return f"added {result.get('chunks_added', 0)} new chunks (total {result.get('total_chunks', 0)})"
    if action == "graph_aware_search":
        return f"added {result.get('chunks_added', 0)} chunks via vector search"
    if action == "entity_cosine_search":
        top_ents = result.get("top_entities", [])
        return (
            f"added {result.get('chunks_added', 0)} chunks via entity-cosine "
            f"(top entities: {', '.join(top_ents) if top_ents else 'none'})"
        )
    if action == "ppr_search":
        seeds = result.get("seeds_used", [])
        return (
            f"added {result.get('chunks_added', 0)} chunks via PPR walk "
            f"(seeds: {', '.join(seeds) if seeds else 'none'})"
        )
    if action == "rerank":
        return f"reranked {result.get('reranked', 0)}, top scores {result.get('top_scores', [])}"
    return str(result)[:200]


# ── Main loop orchestrator ────────────────────────────────────────────────────


async def react_chat(
    query: str,
    clients: Any,
    settings: Any,
    tenant_id: str = "default",
    max_steps: int = 6,
    query_type: str = "factual",
) -> dict[str, Any]:
    """Run multi-step ReAct loop, return answer + full trace.
    query_type selects the per-intent workflow when REACT_WORKFLOW=1."""
    started = time.monotonic()
    actor = ReActAction(clients, settings, tenant_id)
    actor.query_type = query_type
    history: list[dict] = []
    step_latencies: list[dict] = []

    # Tier 3: pre-seed with entity_cosine_search when the feature is enabled.
    # Most multi-hop / comparative queries benefit from cross-doc entity-aware
    # retrieval before the LLM starts reasoning. Pre-seeding eliminates the
    # LLM's freedom to skip the strongest tool. The result chunks become the
    # starting context for subsequent ReAct steps.
    if getattr(settings, "entity_cosine_enabled", False):
        t_seed = time.monotonic()
        try:
            seed_result = await actor.entity_cosine_search(query, top_k_entities=20)
            history.append(
                {
                    "step": 0,
                    "thought": "(pre-seed) Tier 3: cross-doc entity-cosine retrieval",
                    "action": "entity_cosine_search",
                    "args": {"query": query, "top_k_entities": 20},
                    "observation_summary": _observation_summary(
                        "entity_cosine_search", seed_result
                    ),
                    "thought_ms": 0,
                    "action_ms": (time.monotonic() - t_seed) * 1000,
                }
            )
            logger.info(
                f"ReAct pre-seed entity_cosine_search: {seed_result.get('chunks_added', 0)} chunks, "
                f"top entities = {seed_result.get('top_entities', [])[:5]}"
            )
        except Exception as e:
            logger.warning(f"ReAct pre-seed entity_cosine_search failed: {e}")

    # Phase 2.1: pre-seed with PPR when query entities are available. PPR is
    # the strongest signal for multi-hop questions and the LLM cannot skip it
    # if we seed it first. Cheap (~50-150ms once the graph is cached).
    if getattr(settings, "ppr_enabled", False) and clients.neo4j is not None:
        # Extract query entities via GLiNER if available.
        seed_entities: list[str] = []
        entity_extractor = getattr(clients, "entity_extractor", None)
        if entity_extractor is not None:
            try:
                ents, _ = await entity_extractor.extract(query)
                seed_entities = [e.name for e in ents if e.name and len(e.name) >= 2][:10]
            except Exception as e:
                logger.debug(f"ReAct PPR pre-seed entity extract failed: {e}")
        if seed_entities:
            t_seed = time.monotonic()
            try:
                ppr_result = await actor.ppr_search(seed_entities)
                history.append(
                    {
                        "step": len(history),
                        "thought": "(pre-seed) Phase 2.1: PPR multi-hop walk",
                        "action": "ppr_search",
                        "args": {"entities": seed_entities[:5]},
                        "observation_summary": _observation_summary("ppr_search", ppr_result),
                        "thought_ms": 0,
                        "action_ms": (time.monotonic() - t_seed) * 1000,
                    }
                )
                logger.info(
                    f"ReAct pre-seed ppr_search: {ppr_result.get('chunks_added', 0)} chunks, "
                    f"seeds = {ppr_result.get('seeds_used', [])[:5]}"
                )
            except Exception as e:
                logger.warning(f"ReAct pre-seed ppr_search failed: {e}")

    # Planner selection (no LLM in decision path when REACT_WORKFLOW=1):
    #   REACT_WORKFLOW=1  → per-intent workflow state machine (best, default)
    #   REACT_RULE_BASED=1 → single rule playbook (legacy rule-based)
    #   Both 0           → LLM-driven _decide_next_action (legacy)
    # Workflow wins because each query type follows its own optimized
    # sequence (factual ≠ multi_hop). Saves ~30s/query, model-agnostic,
    # impossible to loop (finite list).
    import os as _os

    use_workflow = bool(int(_os.environ.get("REACT_WORKFLOW", "1")))
    use_rule_based = bool(int(_os.environ.get("REACT_RULE_BASED", "1")))

    # Extract seed entities once for both rule-based and workflow modes.
    _seed_entities: list[str] = []
    if use_workflow or use_rule_based:
        entity_extractor = getattr(clients, "entity_extractor", None)
        if entity_extractor is not None:
            try:
                _ents, _ = await entity_extractor.extract(query)
                _seed_entities = [e.name for e in _ents if e.name and len(e.name) >= 2][:10]
            except Exception as e:
                logger.debug(f"planner entity extract failed: {e}")

    # Workflow setup: select per-intent workflow once. Step pointer advances
    # past skip-true steps deterministically; no LLM in the loop.
    _workflow: list[tuple[str, Any, Any]] = []
    _wf_step_index: int = 0
    if use_workflow:
        query_type = getattr(actor, "query_type", None) or "factual"
        _workflow = select_workflow(query_type)
        logger.info(f"ReAct workflow[{query_type}]: {[step[0] for step in _workflow]}")

    for step in range(max_steps):
        t0 = time.monotonic()
        if use_workflow:
            ctx = {
                "query": query,
                "entities": _seed_entities,
                "chunks_collected": len(actor.collected_chunks),
            }
            _wf_step_index, thought_json = next_workflow_step(_workflow, _wf_step_index, ctx)
        elif use_rule_based:
            thought_json = plan_next_react_action(
                query,
                history,
                len(actor.collected_chunks),
                seed_entities=_seed_entities,
            )
        else:
            thought_json = await _decide_next_action(
                query,
                history,
                len(actor.collected_chunks),
                settings.ollama_model,
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
                history.append(
                    {
                        "step": step + 1,
                        "thought": thought_json["thought"],
                        "action": "FINISH",
                        "args": {},
                        "observation_summary": f"agent decided to finish with {len(actor.collected_chunks)} chunks",
                        "thought_ms": thought_t * 1000,
                        "action_ms": 0,
                    }
                )
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
            elif action_name == "entity_cosine_search":
                result = await actor.entity_cosine_search(
                    args.get("query", query),
                    top_k_entities=int(args.get("top_k_entities", 15) or 15),
                )
            elif action_name == "ppr_search":
                result = await actor.ppr_search(args.get("entities") or None)
            elif action_name == "expand_community":
                result = await actor.expand_community(args.get("entity", ""))
            elif action_name == "count_entities":
                result = await actor.count_entities(args.get("entity_type"))
            elif action_name == "verify_fact":
                result = await actor.verify_fact(args.get("claim", ""))
            elif action_name == "rerank":
                result = await actor.rerank(query, top_n=args.get("top_n", 8))
            else:
                result = {"error": f"unknown action: {action_name}"}
        except Exception as e:
            result = {"error": str(e)[:200]}
        action_t = time.monotonic() - t1

        history.append(
            {
                "step": step + 1,
                "thought": thought_json["thought"],
                "action": action_name,
                "args": args,
                "observation_summary": _observation_summary(action_name, result),
                "thought_ms": thought_t * 1000,
                "action_ms": action_t * 1000,
            }
        )

    # Final synthesize
    t2 = time.monotonic()
    # Ensure rerank ran at least once before synthesize
    if actor.collected_chunks and not any(h["action"] == "rerank" for h in history):
        try:
            await actor.rerank(query)
        except Exception as e:
            logger.debug(f"Rerank before synthesis failed: {e}")
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
