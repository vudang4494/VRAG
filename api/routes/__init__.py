"""api/routes — modular route handlers.

Structure:
  _prompts.py     — all 4 LLM prompt templates
  _utils.py       — format_context, llm_complete, shared helpers
  _health.py      — /health, /health/deep
  _ingest.py      — /ingest/upload
  _chat.py        — /chat
  _chat_stream.py — /chat/stream
  _react.py       — /chat/react
  _admin.py       — /gaea/refine, /hefr/populate, /hefr/retrieve,
                    /cross_doc/build, /community/build, /rerank/l2r/test

Usage:
  from api.routes import router   # includes ALL endpoints
  app.include_router(router, prefix="/api")
"""

from fastapi import APIRouter

from api.routes._admin import router as _admin_router
from api.routes._chat import router as _chat_router
from api.routes._chat_stream import router as _chat_stream_router
from api.routes._health import router as _health_router
from api.routes._ingest import router as _ingest_router
from api.routes._react import router as _react_router

router = APIRouter()

router.include_router(_health_router)
router.include_router(_ingest_router)
router.include_router(_chat_router)
router.include_router(_chat_stream_router)
router.include_router(_react_router)
router.include_router(_admin_router)

__all__ = ["router"]
