"""Vector storage service for Regulatory Affairs Assistant.

Persists document chunks and their dense vector embeddings into PostgreSQL
using SQLAlchemy and pgvector.
"""

from typing import List, Sequence
from sqlalchemy.orm import Session

from app.database.models import Document, DocumentChunk
from app.services.chunking import Chunk

EXPECTED_EMBEDDING_DIMENSION: int = 768


def store_document_chunks(
    db: Session,
    document_id: int,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
) -> List[DocumentChunk]:
    """Persist document chunks and their corresponding vector embeddings in PostgreSQL.

    Executes in a single atomic transaction without modifying or deleting pre-existing chunks.

    Args:
        db: Active SQLAlchemy database session.
        document_id: Existing parent Document ID.
        chunks: Sequence of Chunk instances to persist.
        embeddings: Sequence of 768-dimensional float vectors corresponding to chunks.

    Returns:
        List of persisted DocumentChunk instances in the same order as supplied chunks.

    Raises:
        ValueError: If inputs fail validation (nonexistent document_id, empty sequences,
            mismatched lengths, invalid chunk indexes, non-768 dimensions, or non-numeric values).
        RuntimeError: If database persistence fails.
    """
    # 1. Validate document_id
    if not isinstance(document_id, int) or isinstance(document_id, bool) or document_id <= 0:
        raise ValueError(f"document_id must be a positive integer, got: {document_id}")

    parent_doc = db.get(Document, document_id)
    if parent_doc is None:
        raise ValueError(f"Document with ID {document_id} does not exist.")

    # 2. Validate chunks and embeddings presence and lengths
    if not chunks:
        raise ValueError("chunks sequence must not be empty.")
    if not embeddings:
        raise ValueError("embeddings sequence must not be empty.")
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk and embedding count mismatch: received {len(chunks)} chunks and {len(embeddings)} embeddings."
        )

    # 3. Validate chunk types, indexes, and embedding dimensions
    seen_chunk_indexes = set()
    records: List[DocumentChunk] = []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        if not isinstance(chunk, Chunk):
            raise ValueError(f"Element at index {i} in chunks is not a Chunk instance, got {type(chunk).__name__}")

        if not isinstance(chunk.chunk_index, int) or isinstance(chunk.chunk_index, bool) or chunk.chunk_index < 0:
            raise ValueError(f"chunk_index at index {i} must be a non-negative integer, got: {chunk.chunk_index}")

        if chunk.chunk_index in seen_chunk_indexes:
            raise ValueError(f"Duplicate chunk_index found in batch: {chunk.chunk_index}")
        seen_chunk_indexes.add(chunk.chunk_index)

        # Validate embedding
        if len(emb) != EXPECTED_EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding at index {i} has invalid dimension: expected {EXPECTED_EMBEDDING_DIMENSION}, got {len(emb)}."
            )

        clean_embedding: List[float] = []
        for v in emb:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"Non-numeric value detected in embedding at index {i}.")
            clean_embedding.append(float(v))

        # Build DocumentChunk instance
        chunk_record = DocumentChunk(
            document_id=document_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            char_count=chunk.char_count,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            metadata_json=chunk.metadata if chunk.metadata is not None else {},
            embedding=clean_embedding,
        )
        records.append(chunk_record)

    # 4. Atomic persistence
    try:
        db.add_all(records)
        db.commit()
        for rec in records:
            db.refresh(rec)
    except Exception as exc:
        db.rollback()
        raise RuntimeError(f"Failed to persist document chunks to database: {exc.__class__.__name__}") from exc

    return records
