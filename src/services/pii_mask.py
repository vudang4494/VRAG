"""PII Masker — regex + LLM-NER hybrid với consistent placeholders.

Mục tiêu: thay PII (tên người, CMND, SĐT, email, số tài khoản) bằng placeholder
ổn định (cùng entity → cùng placeholder). Mapping lưu ở payload để unmask sau.
"""

import asyncio
import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# Regex patterns — Việt Nam centric
_PHONE_VN = re.compile(r"(?<!\d)(?:\+?84|0)(?:3[2-9]|5[6-9]|7[06-9]|8[1-9]|9[0-46-9])\d{7}(?!\d)")
_EMAIL = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
_CMND_CCCD = re.compile(r"(?<!\d)\d{9}(?:\d{3})?(?!\d)")
_BANK_ACCOUNT = re.compile(r"(?<!\d)\d{8,14}(?!\d)")
_DATE_VN = re.compile(r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b")
_URL = re.compile(r"https?://[^\s<>\"]+")
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


@dataclass
class MaskMap:
    """Mapping placeholder → original value, plus reverse lookup."""

    id: str = field(default_factory=lambda: f"mask_{uuid.uuid4().hex[:12]}")
    forward: dict[str, str] = field(default_factory=dict)  # original → placeholder
    reverse: dict[str, str] = field(default_factory=dict)  # placeholder → original

    def add(self, original: str, kind: str) -> str:
        if original in self.forward:
            return self.forward[original]
        short_hash = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
        placeholder = f"<{kind}_{short_hash}>"
        self.forward[original] = placeholder
        self.reverse[placeholder] = original
        return placeholder

    def unmask(self, text: str) -> str:
        result = text
        for placeholder, original in self.reverse.items():
            result = result.replace(placeholder, original)
        return result


def _mask_regex(text: str, mask_map: MaskMap) -> str:
    """Apply regex-based masking pass."""

    def replace_with(kind: str):
        def _r(m: re.Match) -> str:
            return mask_map.add(m.group(0), kind)

        return _r

    text = _EMAIL.sub(replace_with("EMAIL"), text)
    text = _PHONE_VN.sub(replace_with("PHONE"), text)
    text = _URL.sub(replace_with("URL"), text)
    text = _IP.sub(replace_with("IP"), text)
    text = _BANK_ACCOUNT.sub(replace_with("ACCOUNT"), text)
    text = _CMND_CCCD.sub(replace_with("ID_NUMBER"), text)
    return text


_LLM_NER_PROMPT = """Bạn là chuyên gia phát hiện thông tin nhạy cảm.
Trích xuất từ văn bản dưới đây các thực thể thuộc loại:
- PERSON: tên người (họ và tên đầy đủ hoặc tên riêng có dấu hiệu là người)
- ORGANIZATION: tên công ty, tổ chức cụ thể (không phải tên chung)
- ADDRESS: địa chỉ cụ thể (số nhà, đường, quận)

Trả về CHỈ JSON, không giải thích:
{{"entities": [{{"text": "...", "type": "PERSON|ORGANIZATION|ADDRESS"}}]}}

Văn bản:
{text}
"""


async def _detect_entities(
    text: str, llm: Any, model: str = "gemma4:e4b", max_chars: int = 3000
) -> list[tuple[str, str]]:
    """One LLM-NER call → list of (entity_text, entity_type). Detection only, no masking."""
    import json

    from src.services.ollama_helper import ollama_chat

    snippet = text[:max_chars]
    try:
        raw = await ollama_chat(
            messages=[{"role": "user", "content": _LLM_NER_PROMPT.format(text=snippet)}],
            model=model,
            temperature=0.1,
            max_tokens=256,  # bound generation cost — entity lists are short
        )
        if not raw:
            return []
        raw = re.sub(r"```(?:json)?\s*|\s*```$", "", raw).strip()
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            raw = match.group(0)
        if not raw or raw == "{}":
            return []
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    except Exception as e:
        logger.debug(f"LLM-NER failed (PII mask skipped LLM pass): {e}")
        return []

    out: list[tuple[str, str]] = []
    for ent in data.get("entities", []):
        ent_text = (ent.get("text") or "").strip()
        ent_type = (ent.get("type") or "OTHER").upper()
        if len(ent_text) >= 2 and ent_type in ("PERSON", "ORGANIZATION", "ADDRESS"):
            out.append((ent_text, ent_type))
    return out


def _apply_entities(text: str, mask_map: MaskMap) -> str:
    """Replace every known LLM-NER entity in `mask_map` within `text` (longest first)."""
    for original, placeholder in sorted(
        mask_map.forward.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        text = re.compile(r"(?<!\w)" + re.escape(original) + r"(?!\w)").sub(placeholder, text)
    return text


async def _mask_llm_ner(
    text: str,
    mask_map: MaskMap,
    llm: Any,
    model: str = "gemma4:e4b",
    max_chars: int = 3000,
) -> str:
    """Use LLM to detect PERSON / ORGANIZATION / ADDRESS, then mask (single-text path)."""
    for ent_text, ent_type in await _detect_entities(text, llm, model, max_chars):
        mask_map.add(ent_text, ent_type)
    return _apply_entities(text, mask_map)


async def mask_pii(
    text: str,
    llm: Any = None,
    model: str = "gemma4:e4b",
    use_llm_ner: bool = True,
    existing_map: MaskMap | None = None,
) -> tuple[str, MaskMap]:
    """
    Mask PII với consistent placeholders.

    Trả về (masked_text, mask_map). Pass `existing_map` để giữ consistency
    qua nhiều chunk cùng document.
    """
    mask_map = existing_map or MaskMap()

    # Pass 1: regex (fast, deterministic)
    masked = _mask_regex(text, mask_map)

    # Pass 2: LLM-NER (slower, catches names/orgs)
    if use_llm_ner and llm is not None:
        masked = await _mask_llm_ner(masked, mask_map, llm, model)

    return masked, mask_map


async def mask_chunks(
    chunks: list[dict],
    llm: Any = None,
    model: str = "gemma4:e4b",
    use_llm_ner: bool = True,
    ner_window: int = 3000,
    ner_concurrency: int = 2,
) -> tuple[list[dict], MaskMap]:
    """
    Mask một list chunk dùng chung 1 MaskMap để placeholder consistent qua các
    chunk của cùng document.

    Perf: LLM-NER chạy ở **doc-level** trên các cửa sổ text (concurrent, có
    overlap), KHÔNG phải 1 LLM call/chunk. Với tài liệu N chunk, số LLM call giảm
    từ N xuống ~len(text)/ner_window — thường nhanh hơn ~5-10× mà vẫn giữ
    consistency (1 shared MaskMap). Regex thì áp per-chunk (nhanh, deterministic).
    """
    shared_map = MaskMap()

    # Pass 1 (LLM-NER): detect entities once at doc level, populate the shared map.
    if use_llm_ner and llm is not None:
        full = "\n\n".join((c.get("text") or "") for c in chunks)
        if full.strip():
            overlap = 400  # catch entities split across window boundaries
            step = max(ner_window - overlap, 1)
            windows = [full[i : i + ner_window] for i in range(0, len(full), step)]
            sem = asyncio.Semaphore(max(1, ner_concurrency))

            async def _ner(w: str) -> list[tuple[str, str]]:
                async with sem:
                    return await _detect_entities(w, llm, model, ner_window)

            for res in await asyncio.gather(*[_ner(w) for w in windows], return_exceptions=True):
                if isinstance(res, BaseException):
                    continue
                for ent_text, ent_type in res:
                    shared_map.add(ent_text, ent_type)  # sync + idempotent → consistent

    # Pass 2: per chunk = regex masking + apply the doc-level entity map.
    masked_chunks = []
    for chunk in chunks:
        text = _mask_regex(chunk.get("text", ""), shared_map)
        text = _apply_entities(text, shared_map)
        masked_chunks.append({**chunk, "text": text, "pii_mask_map_id": shared_map.id})
    return masked_chunks, shared_map
