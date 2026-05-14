"""Ingest endpoint — /ingest/upload."""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.services.ingestion_v2 import ingest_document_v2

router = APIRouter()


@router.post("/ingest/upload", tags=["v3"])
async def ingest_upload_v3(
    file: UploadFile = File(...),
    filename: str | None = Form(default=None),
    tenant_id: str = Form(default="default"),
    access_level: str = Form(default="INTERNAL"),
    department: str | None = Form(default=None),
    author: str | None = Form(default=None),
):
    """Ingest a single document through Pipeline V2."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) > 200 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (>200MB)")

    fname = filename or file.filename or "upload"
    clients_get = __import__("src.clients", fromlist=["get_clients"]).get_clients
    clients = clients_get()
    result = await ingest_document_v2(
        content=content,
        filename=fname,
        clients=clients,
        tenant_id=tenant_id,
        access_level=access_level,
        department=department,
        author=author,
    )
    return result
