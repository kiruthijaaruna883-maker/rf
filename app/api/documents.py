"""Document management and ingestion routes for Regulatory Affairs Assistant."""

from datetime import datetime
import logging
from pathlib import Path
import re
import shutil
from typing import Annotated, Optional
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.database.models import Document, User
from app.database.session import get_db
from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingError, embed_chunks
from app.services.text_extraction import extract_document_text
from app.services.vector_storage import store_document_chunks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

# Base directory for uploads: data/documents/ relative to project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "documents"

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class DocumentResponse(BaseModel):
    """Schema for document metadata returned by API."""

    id: int
    filename: str
    file_type: str
    file_path: str
    uploaded_by: Optional[int] = None
    created_at: datetime
    chunks_count: Optional[int] = None
    message: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a regulatory document",
)
def upload_document(
    file: Annotated[UploadFile, File(description="Regulatory document file (.pdf, .docx, .txt)")],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> DocumentResponse:
    """Upload and ingest a raw regulatory document (.pdf, .docx, or .txt).

    Validates file extension, sanitizes filename, prevents path traversal,
    saves the raw file under data/documents/, extracts text, splits into chunks,
    generates dense vector embeddings, and stores document and chunks in PostgreSQL.
    """
    # 1. Validate file presence and filename
    if not file or not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file was supplied for upload",
        )

    # 2. Extract and validate file extension
    raw_filename = Path(file.filename).name.strip()
    ext = Path(raw_filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed types: .pdf, .docx, .txt",
        )

    file_type = ext.lstrip(".")

    # 3. Sanitize filename to prevent path traversal and unsafe characters
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_filename).lstrip(".")
    if not safe_name:
        safe_name = f"document{ext}"

    # 4. Generate unique filename to prevent overwrites
    unique_filename = f"{uuid.uuid4().hex}_{safe_name}"

    # Ensure upload directory exists
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination_path = (UPLOAD_DIR / unique_filename).resolve()

    # Verify destination is strictly within UPLOAD_DIR (path traversal guard)
    if not destination_path.is_relative_to(UPLOAD_DIR.resolve()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or unsafe filename",
        )

    # 5. Save raw uploaded file to disk
    try:
        with destination_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file to storage",
        )
    finally:
        file.file.close()

    # 6. Extract and normalize document text
    try:
        extracted_text = extract_document_text(destination_path, file_type)
    except Exception as exc:
        logger.error("Text extraction failed for %s: %s", destination_path, exc)
        if destination_path.exists():
            destination_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to extract text from document: {exc}",
        )

    if not extracted_text or not extracted_text.strip():
        if destination_path.exists():
            destination_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document contains no extractable text or content for ingestion.",
        )

    # 7. Chunk document text
    try:
        chunks = chunk_text(extracted_text)
    except Exception as exc:
        logger.error("Chunking failed for %s: %s", destination_path, exc)
        if destination_path.exists():
            destination_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to chunk document text: {exc}",
        )

    if not chunks:
        if destination_path.exists():
            destination_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document produced no chunks for ingestion.",
        )

    # 8. Generate dense vector embeddings
    try:
        embeddings = embed_chunks(chunks)
    except Exception as exc:
        logger.error("Embedding generation failed for %s: %s", destination_path, exc)
        if destination_path.exists():
            destination_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Embedding service failed: {exc}",
        )

    # 9. Record document metadata in PostgreSQL
    stored_path = str(destination_path)
    db_filename = raw_filename[:255]

    try:
        new_document = Document(
            filename=db_filename,
            file_type=file_type,
            file_path=stored_path,
            uploaded_by=current_user.id,
        )
        db.add(new_document)
        db.commit()
        db.refresh(new_document)
    except Exception as exc:
        db.rollback()
        logger.error("Database failure recording document metadata: %s", exc)
        if destination_path.exists():
            destination_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record document metadata in database",
        )

    # 10. Store document chunks and embeddings in pgvector
    try:
        stored_chunks = store_document_chunks(
            db=db,
            document_id=new_document.id,
            chunks=chunks,
            embeddings=embeddings,
        )
    except Exception as exc:
        logger.error("Database failure storing document chunks: %s", exc)
        try:
            db.delete(new_document)
            db.commit()
        except Exception:
            db.rollback()
        if destination_path.exists():
            destination_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store document chunks and embeddings: {exc}",
        )

    return DocumentResponse(
        id=new_document.id,
        filename=new_document.filename,
        file_type=new_document.file_type,
        file_path=new_document.file_path,
        uploaded_by=new_document.uploaded_by,
        created_at=new_document.created_at,
        chunks_count=len(stored_chunks),
        message=f"Document successfully ingested into RAG vector store ({len(stored_chunks)} chunks).",
    )
