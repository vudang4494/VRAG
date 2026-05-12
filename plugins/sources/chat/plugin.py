"""Chat / Email source plugin.

Supports:
  - JSON array of {role, content, name?, timestamp?, thread_id?}
  - JSONL (one message per line)
  - .eml files (single or multipart)

Actual chunking happens in ingestion_v2.py via ChatChunker.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from loguru import logger

from plugins.base import (
    BaseSourcePlugin,
    ParsedDocument,
    PluginCapability,
    PluginConfig,
)


class ChatSourcePlugin(BaseSourcePlugin):
    name: ClassVar[str] = "chat"
    version: ClassVar[str] = "1.0.0"
    capabilities: ClassVar[list[PluginCapability]] = [
        PluginCapability.INGEST_FILE,
        PluginCapability.INGEST_STREAM,
    ]
    supported_types: ClassVar[list[str]] = ["json", "jsonl", "eml", "msg", "chat"]

    def __init__(self, config: PluginConfig | None = None, credentials: Any = None):
        self.config = config or PluginConfig(raw={})
        self.credentials = credentials

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "name": self.name, "version": self.version}

    async def fetch(self, url_or_path: str, **kwargs) -> ParsedDocument:
        content: bytes | None = kwargs.get("content")
        filename = kwargs.get("filename") or Path(url_or_path).name

        if content is None and url_or_path:
            try:
                with open(url_or_path, "rb") as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"chat plugin: cannot read {url_or_path}: {e}")
                raise

        if not content:
            raise ValueError("chat plugin: no content provided")

        ext = (Path(filename).suffix or "").lower().lstrip(".")
        is_eml = ext in ("eml", "msg")

        if is_eml:
            preview, meta = self._preview_eml(content)
            file_type = "email"
        else:
            preview, meta = self._preview_chat(content)
            file_type = "chat"

        return ParsedDocument(
            title=filename,
            content=preview,
            raw_content=content,
            url=url_or_path,
            file_type=file_type,
            file_size_bytes=len(content),
            created_date=datetime.utcnow(),
            metadata={**meta, "filename": filename},
        )

    def _preview_chat(self, content: bytes) -> tuple[str, dict[str, Any]]:
        text = content.decode("utf-8", errors="replace").strip()
        messages = []
        if text.startswith("["):
            try:
                messages = json.loads(text)
            except Exception:
                pass
        if not messages:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                    if isinstance(m, dict):
                        messages.append(m)
                except Exception:
                    continue

        if not messages:
            return "(empty or unparseable chat)", {"message_count": 0}

        threads = {}
        for m in messages:
            tid = m.get("thread_id") or m.get("conversation_id") or "default"
            threads.setdefault(tid, []).append(m)

        preview_parts = [f"# Chat Log ({len(messages)} messages, {len(threads)} threads)\n"]
        for tid, msgs in list(threads.items())[:3]:
            preview_parts.append(f"\n## Thread: {tid} ({len(msgs)} messages)\n")
            for m in msgs[:5]:
                role = m.get("role") or m.get("from") or "unknown"
                content_text = (m.get("content") or m.get("text") or "")[:200]
                ts = m.get("timestamp", "")
                preview_parts.append(f"- **{role}** ({ts}): {content_text}")
            if len(msgs) > 5:
                preview_parts.append(f"  ... ({len(msgs) - 5} more)")

        return "\n".join(preview_parts), {
            "message_count": len(messages),
            "thread_count": len(threads),
            "thread_ids": list(threads.keys())[:50],
        }

    def _preview_eml(self, content: bytes) -> tuple[str, dict[str, Any]]:
        try:
            from email import policy
            from email.parser import BytesParser
            msg = BytesParser(policy=policy.default).parsebytes(content)
        except Exception as e:
            logger.warning(f"Email parse failed: {e}")
            return f"(unable to parse email: {e})", {}

        subject = msg.get("subject", "")
        sender = msg.get("from", "")
        to = msg.get("to", "")
        date = msg.get("date", "")
        message_id = msg.get("message-id", "")

        body_parts: list[str] = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body_parts.append(part.get_content() or "")
                    except Exception:
                        continue
        else:
            try:
                body_parts.append(msg.get_content() or "")
            except Exception:
                pass

        body = "\n\n".join(b for b in body_parts if b).strip()

        preview = (
            f"**Subject**: {subject}\n"
            f"**From**: {sender}\n"
            f"**To**: {to}\n"
            f"**Date**: {date}\n\n"
            f"{body[:1500]}"
        )
        return preview, {
            "message_id": message_id,
            "subject": subject,
            "from": sender,
            "to": to,
            "date": date,
            "body_length": len(body),
            "is_multipart": msg.is_multipart(),
        }
