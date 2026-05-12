"""Chat / Email chunker — turn-aware, thread-aware.

Input format (chat):
- JSON array of {role, content, name?, timestamp?, thread_id?}
- JSONL với mỗi line như trên
- Email .eml file (single message hoặc multipart)

Output: ChunkUnit per turn (paragraph level) + thread summary (section level).
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from loguru import logger

from src.services.chunkers.base import BaseChunker, ChunkUnit


class ChatChunker(BaseChunker):
    name = "chat"

    def __init__(self, qa_pair_window: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.qa_pair_window = qa_pair_window

    async def chunk(self, content: bytes | str, filename: str = "") -> list[ChunkUnit]:
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = content

        is_email = filename.lower().endswith((".eml", ".msg"))
        if is_email:
            messages = self._parse_eml(text)
        else:
            messages = self._parse_json_or_jsonl(text)

        if not messages:
            return []

        # Group by thread_id
        threads: dict[str, list[dict]] = {}
        for m in messages:
            tid = str(m.get("thread_id") or m.get("id") or "default")
            threads.setdefault(tid, []).append(m)

        units: list[ChunkUnit] = []
        idx = 0
        for thread_idx, (tid, msgs) in enumerate(threads.items()):
            msgs.sort(key=lambda m: m.get("timestamp") or "")
            section_parent = None
            if "section" in self.emit_levels and msgs:
                snippet = self._format_thread_summary(msgs)
                units.append(ChunkUnit(
                    text=snippet,
                    chunk_index=idx,
                    chunk_level="section",
                    parent_index=None,
                    metadata={
                        "thread_id": tid,
                        "message_count": len(msgs),
                        "first_ts": str(msgs[0].get("timestamp", "")),
                        "last_ts": str(msgs[-1].get("timestamp", "")),
                        "filename": filename,
                        "format": "email" if is_email else "chat",
                    },
                ))
                section_parent = idx
                idx += 1

            # Pair Q+A or sliding window
            for i in range(0, len(msgs), self.qa_pair_window + 1):
                window = msgs[i : i + self.qa_pair_window + 1]
                if not window:
                    continue
                turn_text = self._format_window(window)
                speakers = list({m.get("role") or m.get("from") or "unknown" for m in window})
                units.append(ChunkUnit(
                    text=turn_text,
                    chunk_index=idx,
                    chunk_level="paragraph",
                    parent_index=section_parent,
                    metadata={
                        "thread_id": tid,
                        "speakers": speakers,
                        "timestamps": [str(m.get("timestamp", "")) for m in window],
                        "in_reply_to": window[0].get("in_reply_to"),
                        "filename": filename,
                        "format": "email" if is_email else "chat",
                    },
                ))
                idx += 1
        return units

    def _parse_json_or_jsonl(self, text: str) -> list[dict]:
        text = text.strip()
        if not text:
            return []
        # Try JSON array first
        if text.startswith("["):
            try:
                arr = json.loads(text)
                return [self._normalize_msg(m) for m in arr if isinstance(m, dict)]
            except Exception:
                pass
        # Fallback JSONL
        out: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
                if isinstance(m, dict):
                    out.append(self._normalize_msg(m))
            except Exception:
                continue
        return out

    def _parse_eml(self, text: str) -> list[dict]:
        try:
            from email import policy
            from email.parser import Parser
            msg = Parser(policy=policy.default).parsestr(text)
        except Exception as e:
            logger.warning(f"Email parse failed: {e}")
            return []

        body_parts: list[str] = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body_parts.append(part.get_content() or "")
        else:
            body_parts.append(msg.get_content() or "")

        body = "\n\n".join(b for b in body_parts if b).strip()
        return [{
            "role": msg.get("from", "unknown"),
            "content": body,
            "timestamp": msg.get("date", ""),
            "thread_id": msg.get("message-id") or msg.get("references", "default"),
            "in_reply_to": msg.get("in-reply-to"),
            "subject": msg.get("subject", ""),
            "to": msg.get("to", ""),
        }]

    @staticmethod
    def _normalize_msg(m: dict) -> dict:
        return {
            "role": m.get("role") or m.get("from") or m.get("speaker") or "user",
            "content": m.get("content") or m.get("text") or m.get("message") or "",
            "timestamp": m.get("timestamp") or m.get("ts") or m.get("date") or "",
            "thread_id": m.get("thread_id") or m.get("conversation_id") or "default",
            "in_reply_to": m.get("in_reply_to"),
        }

    @staticmethod
    def _format_window(window: list[dict]) -> str:
        lines = []
        for m in window:
            ts = m.get("timestamp", "")
            ts_str = f"[{ts}] " if ts else ""
            lines.append(f"{ts_str}{m['role']}: {m['content']}")
        return "\n".join(lines)

    @staticmethod
    def _format_thread_summary(msgs: list[dict]) -> str:
        first = msgs[0]
        last = msgs[-1]
        sample = msgs[:3]
        head = "\n".join(f"{m['role']}: {(m['content'] or '')[:200]}" for m in sample)
        return (
            f"Thread {first.get('thread_id', '?')} với {len(msgs)} tin nhắn từ "
            f"{first.get('timestamp', '?')} đến {last.get('timestamp', '?')}.\n"
            f"Bắt đầu:\n{head}"
        )
