"""Entity extractor — separate from semantic LLM.

Default backend: GLiNER (specialized NER model, ~168M params, runs on CPU).
Pluggable design: can swap to OpenAI/Anthropic API later by changing config.

Purpose: extract structured entities + types from raw chunk text. This is the
"bridge" between unstructured vector embeddings and structured graph queries.
Quality of entity extraction directly determines GraphRAG capabilities.
"""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from loguru import logger

# ── Entity types we extract (tailored for enterprise + research) ──────────────
DEFAULT_LABELS = [
    "person",
    "organization",
    "location",
    "product",
    "technology",
    "concept",
    "event",
    "date",
]

# Map GLiNER lowercase labels → our canonical UPPERCASE types
_TYPE_MAP = {
    "person": "PERSON",
    "organization": "ORGANIZATION",
    "location": "LOCATION",
    "product": "PRODUCT",
    "technology": "TECHNOLOGY",
    "concept": "CONCEPT",
    "event": "EVENT",
    "date": "DATE",
}


@dataclass
class ExtractedEntity:
    name: str
    type: str
    description: str = ""
    confidence: float = 1.0


@dataclass
class ExtractedRelation:
    source: str
    target: str
    description: str = ""
    type: str = "RELATES_TO"
    confidence: float = 1.0


class BaseEntityExtractor(ABC):
    @abstractmethod
    async def extract(self, text: str) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        """Return (entities, relations) from a single chunk text."""

    async def extract_batch(
        self, texts: list[str], concurrent_limit: int = 4
    ) -> list[tuple[list[ExtractedEntity], list[ExtractedRelation]]]:
        sem = asyncio.Semaphore(concurrent_limit)

        async def _one(t: str):
            async with sem:
                return await self.extract(t)

        return await asyncio.gather(*[_one(t) for t in texts])


# ────────────────────────────────────────────────────────────────────────────
# GLiNER backend — local NER, fast, no LLM needed
# ────────────────────────────────────────────────────────────────────────────


class GLiNERExtractor(BaseEntityExtractor):
    """
    Local NER via GLiNER (Zaratiana et al., 2023).

    Pros: fast (~50-200ms/chunk on CPU), 100% local, multilingual (incl. Vietnamese).
    Cons: NER only — does NOT extract relationships. Pair with LLM for relations
          (see CombinedExtractor below) or skip relations entirely.
    """

    def __init__(
        self,
        model_name: str = "urchade/gliner_multi-v2.1",
        labels: list[str] | None = None,
        threshold: float = 0.5,
        max_chars: int = 2500,
    ):
        self.model_name = model_name
        self.labels = labels or DEFAULT_LABELS
        self.threshold = threshold
        self.max_chars = max_chars
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        # Disable progress bars to avoid `tqdm._lock` AttributeError in async context.
        import os as _os

        _os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        _os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        _os.environ.setdefault("TQDM_DISABLE", "1")
        try:
            from gliner import GLiNER

            self._model = GLiNER.from_pretrained(self.model_name)
            logger.info(f"GLiNER loaded: {self.model_name}")
        except ImportError:
            logger.error("gliner package not installed. Run: pip install gliner")
            raise
        except Exception as e:
            logger.error(f"GLiNER model load failed ({self.model_name}): {e}")
            raise

    async def extract(self, text: str) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        if not text or len(text.strip()) < 10:
            return ([], [])

        snippet = text[: self.max_chars]
        try:
            # GLiNER is sync — run in thread pool
            raw = await asyncio.to_thread(self._sync_extract, snippet)
        except Exception as e:
            logger.warning(f"GLiNER extract failed: {e}")
            return ([], [])

        # Dedup by lowercased name; keep highest-confidence
        best: dict[str, ExtractedEntity] = {}
        for item in raw:
            name = (item.get("text") or "").strip()
            lbl = (item.get("label") or "").lower()
            score = float(item.get("score", 0.0))
            if len(name) < 2 or score < self.threshold:
                continue
            key = name.lower()
            etype = _TYPE_MAP.get(lbl, lbl.upper() or "OTHER")
            if key not in best or best[key].confidence < score:
                best[key] = ExtractedEntity(name=name, type=etype, confidence=score)

        return (list(best.values()), [])  # no relations from GLiNER

    def _sync_extract(self, text: str) -> list[dict]:
        self._load()
        return self._model.predict_entities(text, self.labels, threshold=self.threshold)


# ────────────────────────────────────────────────────────────────────────────
# Relation extractor — small LLM call, only on already-extracted entities
# ────────────────────────────────────────────────────────────────────────────

_REL_PROMPT = """Given the entities below extracted from a text, identify direct relationships between them.

Entities: {entities}

Text:
{text}

Output JSON ONLY (no explanation):
{{"relations": [{{"source": "<entity_name>", "target": "<entity_name>", "type": "USES|PROPOSED_BY|CITES|IS_A|PART_OF|WORKS_FOR|LOCATED_IN|OTHER", "description": "<short>"}}]}}

Only include relations between entities listed above. Use exact entity names. If no clear relations exist, return {{"relations": []}}.

JSON:"""


class LLMRelationExtractor:
    """Lightweight LLM call to find relations BETWEEN already-extracted entities.

    Much more reliable than full entity+relation extraction because:
    - Entities are pre-supplied (no NER hallucination)
    - LLM only needs to pick pairs and types
    - Shorter prompt, smaller output
    """

    def __init__(self, llm: Any, model: str, max_chars: int = 2500):
        self.llm = llm
        self.model = model
        self.max_chars = max_chars

    async def extract_relations(
        self, text: str, entities: list[ExtractedEntity]
    ) -> list[ExtractedRelation]:
        import json as _json

        if len(entities) < 2 or not text.strip():
            return []

        entity_str = ", ".join(
            f'"{e.name}"' for e in entities[:20]
        )  # cap at 20 to keep prompt small
        snippet = text[: self.max_chars]
        prompt = _REL_PROMPT.format(entities=entity_str, text=snippet)

        from src.services.ollama_helper import ollama_chat

        try:
            raw = await ollama_chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
                max_tokens=400,
            )
            if not raw:
                return []
            raw = re.sub(r"```(?:json)?\s*|\s*```$", "", raw).strip()
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                raw = match.group(0)
            data = _json.loads(raw)
        except Exception as e:
            logger.debug(f"Relation extraction failed: {e}")
            return []

        # Validate: only keep relations whose source+target are in entity list
        entity_names = {e.name.lower() for e in entities}
        out: list[ExtractedRelation] = []
        for rel in data.get("relations", []):
            src = (rel.get("source") or "").strip()
            tgt = (rel.get("target") or "").strip()
            if src.lower() in entity_names and tgt.lower() in entity_names and src != tgt:
                out.append(
                    ExtractedRelation(
                        source=src,
                        target=tgt,
                        description=(rel.get("description") or "")[:300],
                        type=rel.get("type", "RELATES_TO"),
                        confidence=0.8,
                    )
                )
        return out


# ────────────────────────────────────────────────────────────────────────────
# Combined: NER via GLiNER + Relations via small LLM call
# ────────────────────────────────────────────────────────────────────────────


class CombinedExtractor(BaseEntityExtractor):
    """GLiNER for entities + (optional) small LLM for relations."""

    def __init__(
        self,
        ner: BaseEntityExtractor,
        rel: LLMRelationExtractor | None = None,
        extract_relations: bool = False,
    ):
        self.ner = ner
        self.rel = rel
        self.extract_relations = extract_relations

    async def extract(self, text: str) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        ents, _ = await self.ner.extract(text)
        if not self.extract_relations or not self.rel or len(ents) < 2:
            return (ents, [])
        rels = await self.rel.extract_relations(text, ents)
        return (ents, rels)


# ────────────────────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────────────────────


def create_entity_extractor(
    provider: str = "gliner",
    model: str = "urchade/gliner_multi-v2.1",
    labels: list[str] | None = None,
    threshold: float = 0.5,
    llm_for_relations: Any = None,
    relation_model: str = "qwen3.5:9b",
    extract_relations: bool = False,
) -> BaseEntityExtractor:
    """
    Build an entity extractor based on config.

    Args:
        provider: "gliner" | "openai" | "anthropic" (only gliner implemented now)
        model: model name for the chosen provider
        labels: entity types to extract
        threshold: confidence threshold for NER hits
        llm_for_relations: optional LLM client for relation extraction pass
        relation_model: model name for relation LLM
        extract_relations: if True, run LLM relation pass after NER
    """
    if provider == "gliner":
        ner = GLiNERExtractor(model_name=model, labels=labels, threshold=threshold)
        if extract_relations and llm_for_relations:
            rel = LLMRelationExtractor(llm_for_relations, relation_model)
            return CombinedExtractor(ner=ner, rel=rel, extract_relations=True)
        return CombinedExtractor(ner=ner, rel=None, extract_relations=False)
    # Future: add OpenAI, Anthropic providers here
    raise ValueError(f"Unknown entity extractor provider: {provider}")
