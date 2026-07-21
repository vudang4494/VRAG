"""Ingest endpoint — /ingest/upload."""

import asyncio
import contextlib

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from loguru import logger

from src.services.ingestion import ingest_document

router = APIRouter()


@router.post("/ingest/upload", tags=["ingest"])
async def ingest_upload(
    request: Request,
    file: UploadFile = File(...),
    filename: str | None = Form(default=None),
    tenant_id: str = Form(default="default"),
    access_level: str = Form(default="INTERNAL"),
    department: str | None = Form(default=None),
    author: str | None = Form(default=None),
):
    """Ingest a single document through VRAG pipeline.

    Ingestion is a long, expensive call. FastAPI/uvicorn does NOT cancel the
    handler when the client disconnects, so a client that gives up (or a timed-out
    curl) would otherwise leave the pipeline churning server-side as an orphan.
    We run ingest as a task and poll for disconnect, cancelling it if the client
    goes away. Ingestion is idempotent (deterministic doc/chunk ids), so an
    aborted run is safely overwritten by a re-ingest.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) > 200 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (>200MB)")

    fname = filename or file.filename or "upload"
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    clients = clients_get()

    task = asyncio.ensure_future(
        ingest_document(
            content=content,
            filename=fname,
            clients=clients,
            tenant_id=tenant_id,
            access_level=access_level,
            department=department,
            author=author,
        )
    )
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=2.0)
            if task in done:
                return task.result()
            if await request.is_disconnected():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                logger.warning(f"[ingest] client disconnected — aborted ingest of {fname}")
                raise HTTPException(status_code=499, detail="client disconnected; ingest aborted")
    except asyncio.CancelledError:
        # The request itself was cancelled — make sure the ingest task dies too.
        task.cancel()
        raise
