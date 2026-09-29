"""Services package for Regulatory Affairs Assistant."""

from app.services.chunking import (
    Chunk,
    chunk_text,
)
from app.services.embeddings import (
    EmbeddingAPIError,
    EmbeddingConfigError,
    EmbeddingError,
    embed_chunks,
    embed_text,
    embed_texts,
)
from app.services.text_extraction import (
    extract_document_text,
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_txt,
    normalize_text,
)

__all__ = [
    "Chunk",
    "EmbeddingAPIError",
    "EmbeddingConfigError",
    "EmbeddingError",
    "chunk_text",
    "embed_chunks",
    "embed_text",
    "embed_texts",
    "extract_document_text",
    "extract_text_from_docx",
    "extract_text_from_pdf",
    "extract_text_from_txt",
    "normalize_text",
]
