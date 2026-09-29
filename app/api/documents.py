"""Document management and ingestion routes for Regulatory Affairs Assistant."""

from datetime import datetime
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
    saves the raw file under data/documents/, and records document metadata.
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

    # 6. Record metadata in the database
    stored_path = str(destination_path)
    # Also ensure original filename fits within DB column length (255)
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
    except Exception:
        db.rollback()
        # Clean up written file if database insertion fails
        if destination_path.exists():
            try:
                destination_path.unlink()
            except OSError:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record document metadata in database",
        )

    return new_document
