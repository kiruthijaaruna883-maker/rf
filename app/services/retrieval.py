"""Similarity retrieval service for Regulatory Affairs Assistant.

Provides cosine-distance vector similarity search against document chunks
stored in PostgreSQL using pgvector, returning structured retrieval results.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import Document, DocumentChunk
from app.services.embeddings import embed_text

EXPECTED_EMBEDDING_DIMENSION: int = 1536


@dataclass
class ChunkSearchResult:
    """Structured result representing a retrieved document chunk and similarity metrics."""

    chunk_id: int
    document_id: int
    chunk_index: int
    content: str
    distance: float
    similarity: float
    metadata_json: Optional[Dict[str, Any]] = None
    filename: Optional[str] = None


def search_similar_chunks(
    db: Session,
    query_vector: Sequence[float],
    top_k: int = 5,
    document_id: Optional[int] = None,
    document_ids: Optional[Sequence[int]] = None,
    max_distance: Optional[float] = None,
) -> List[ChunkSearchResult]:
    """Execute low-level vector similarity search against stored document chunks.

    Performs approximate nearest neighbor search using pgvector cosine distance (<=>).
    This function is strictly read-only and does not mutate database state.

    Args:
        db: Active SQLAlchemy database session.
        query_vector: Sequence of 1536 numeric float values representing the query embedding.
        top_k: Maximum number of matching chunks to return (default 5).
        document_id: Optional single document ID filter.
        document_ids: Optional sequence of document IDs filter.
        max_distance: Optional maximum cosine distance cutoff.

    Returns:
        List of ChunkSearchResult instances ordered by distance ascending.
        Returns empty list if no chunks match or if table is empty.

    Raises:
        ValueError: If query_vector, top_k, max_distance, or document filters are invalid.
    """
    # 1. Validate query_vector
    if query_vector is None or not isinstance(query_vector, (list, tuple)):
        raise ValueError("query_vector must be a sequence of floats.")

    if len(query_vector) != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"query_vector has invalid dimension: expected {EXPECTED_EMBEDDING_DIMENSION}, got {len(query_vector)}."
        )

    clean_vector: List[float] = []
    for i, val in enumerate(query_vector):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"Non-numeric value detected in query_vector at index {i}: {val!r}")
        clean_vector.append(float(val))

    # 2. Validate top_k
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got: {top_k}")

    # 3. Validate document_id and document_ids filters
    if document_id is not None and document_ids is not None:
        raise ValueError("Ambiguous document filtering: cannot specify both document_id and document_ids.")

    validated_doc_id: Optional[int] = None
    if document_id is not None:
        if not isinstance(document_id, int) or isinstance(document_id, bool) or document_id <= 0:
            raise ValueError(f"document_id must be a positive integer, got: {document_id}")
        validated_doc_id = document_id

    validated_doc_ids: Optional[List[int]] = None
    if document_ids is not None:
        if not isinstance(document_ids, (list, tuple, set)) or len(document_ids) == 0:
            raise ValueError("document_ids must be a non-empty sequence of positive integers.")
        clean_ids: List[int] = []
        for d in document_ids:
            if not isinstance(d, int) or isinstance(d, bool) or d <= 0:
                raise ValueError(f"Each element in document_ids must be a positive integer, got: {d}")
            clean_ids.append(d)
        validated_doc_ids = clean_ids

    # 4. Validate max_distance
    if max_distance is not None:
        if not isinstance(max_distance, (int, float)) or isinstance(max_distance, bool):
            raise ValueError(f"max_distance must be a numeric value, got: {max_distance!r}")
        if max_distance < 0.0:
            raise ValueError(f"max_distance cannot be negative, got: {max_distance}")

    # 5. Build SQLAlchemy query using pgvector cosine distance (<=>)
    distance_expr = DocumentChunk.embedding.cosine_distance(clean_vector).label("distance")

    stmt = (
        select(DocumentChunk, Document.filename, distance_expr)
        .join(Document, DocumentChunk.document_id == Document.id, isouter=True)
    )

    if validated_doc_id is not None:
        stmt = stmt.where(DocumentChunk.document_id == validated_doc_id)
    elif validated_doc_ids is not None:
        stmt = stmt.where(DocumentChunk.document_id.in_(validated_doc_ids))

    if max_distance is not None:
        stmt = stmt.where(distance_expr <= float(max_distance))

    stmt = stmt.order_by(distance_expr.asc()).limit(top_k)

    # 6. Execute query (strictly read-only)
    rows = db.execute(stmt).all()

    results: List[ChunkSearchResult] = []
    for chunk, filename, distance in rows:
        dist_val = float(distance)
        sim_val = 1.0 - dist_val
        results.append(
            ChunkSearchResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                distance=dist_val,
                similarity=sim_val,
                metadata_json=chunk.metadata_json,
                filename=filename,
            )
        )

    return results


def retrieve_chunks(
    db: Session,
    query: str,
    top_k: int = 5,
    document_id: Optional[int] = None,
    document_ids: Optional[Sequence[int]] = None,
    max_distance: Optional[float] = None,
    client: Optional[httpx.Client] = None,
) -> List[ChunkSearchResult]:
    """Retrieve relevant document chunks for a natural language query.

    Generates a dense vector embedding using embed_text, then performs
    cosine-distance similarity search in PostgreSQL via search_similar_chunks.

    Args:
        db: Active SQLAlchemy database session.
        query: Non-empty natural language query string.
        top_k: Maximum number of matching chunks to return (default 5).
        document_id: Optional single document ID filter.
        document_ids: Optional sequence of document IDs filter.
        max_distance: Optional maximum cosine distance cutoff.
        client: Optional httpx.Client passed to embed_text (for testing/mocking).

    Returns:
        List of ChunkSearchResult instances ordered by distance ascending.

    Raises:
        ValueError: If query is empty or whitespace-only, or if arguments fail validation.
        EmbeddingError: If embedding generation fails.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query string cannot be empty or whitespace-only.")

    query_vector = embed_text(query, client=client)

    return search_similar_chunks(
        db=db,
        query_vector=query_vector,
        top_k=top_k,
        document_id=document_id,
        document_ids=document_ids,
        max_distance=max_distance,
    )
