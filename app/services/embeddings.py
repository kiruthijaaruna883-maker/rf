"""Embedding generation service for Regulatory Affairs Assistant.

Provides single-text and batch embedding generation using httpx against
Ollama's local embedding API (/api/embed).
"""

from typing import Any, Dict, List, Optional, Sequence
import httpx

from app.config import Settings, get_settings
from app.services.chunking import Chunk

DEFAULT_EMBEDDING_TIMEOUT: float = 30.0
EMBEDDING_BATCH_SIZE = 10


class EmbeddingError(Exception):
    """Base exception raised for embedding generation errors."""

    pass


class EmbeddingConfigError(EmbeddingError):
    """Raised when embedding configuration or credentials are missing or invalid."""

    pass


class EmbeddingAPIError(EmbeddingError):
    """Raised when the embedding API returns an error response."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _resolve_base_url(base_url: Optional[str]) -> str:
    """Resolve the Ollama base URL, falling back to default if placeholder or empty."""
    if not base_url or not base_url.strip() or base_url.strip().startswith("your-") or "placeholder" in base_url.lower():
        return "http://127.0.0.1:11434"
    return base_url.strip().rstrip("/")


def _resolve_embedding_model(model: Optional[str]) -> str:
    """Resolve the embedding model name, falling back to nomic-embed-text:latest if placeholder or empty."""
    if not model or not model.strip() or model.strip().startswith("your-") or "placeholder" in model.lower():
        return "nomic-embed-text:latest"
    return model.strip()


def _parse_embeddings_response(
    response_data: Any,
    expected_count: int,
) -> List[List[float]]:
    """Safely parse and validate the JSON response from Ollama's /api/embed endpoint.

    Args:
        response_data: Parsed JSON response.
        expected_count: Number of inputs sent in the request.

    Returns:
        List of float vectors matching the exact input order.

    Raises:
        EmbeddingError: If structure, count, or value validation fails.
    """
    if not isinstance(response_data, dict):
        raise EmbeddingError("Malformed response: expected JSON object from embedding provider.")

    embeddings = response_data.get("embeddings")
    if embeddings is None or not isinstance(embeddings, list):
        raise EmbeddingError("Malformed response: missing 'embeddings' array in embedding response.")

    if len(embeddings) != expected_count:
        raise EmbeddingError(
            f"Embedding count mismatch: expected {expected_count} embeddings, got {len(embeddings)}."
        )

    clean_embeddings: List[List[float]] = []

    for idx, emb in enumerate(embeddings):
        if emb is None or not isinstance(emb, list) or len(emb) == 0:
            raise EmbeddingError(f"Malformed response: missing or empty embedding vector at index {idx}.")

        clean_vector: List[float] = []
        for val in emb:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise EmbeddingError(
                    f"Invalid embedding value: non-numeric element detected at index {idx}."
                )
            clean_vector.append(float(val))

        clean_embeddings.append(clean_vector)

    return clean_embeddings


def _send_embedding_request(
    endpoint_url: str,
    payload: Dict[str, Any],
    expected_count: int,
    client: Optional[httpx.Client] = None,
    timeout: float = DEFAULT_EMBEDDING_TIMEOUT,
) -> List[List[float]]:
    """Send an HTTP POST request to Ollama's /api/embed endpoint and return parsed vectors.

    Args:
        endpoint_url: Ollama embedding endpoint URL.
        payload: JSON payload to send (model, input).
        expected_count: Expected number of returned embeddings.
        client: Optional httpx.Client for dependency injection (mocking/testing).
        timeout: Request timeout in seconds.

    Returns:
        List of dense vector embeddings.

    Raises:
        EmbeddingAPIError: For HTTP 4xx/5xx responses.
        EmbeddingError: For network, timeout, or parsing failures.
    """
    headers = {
        "Content-Type": "application/json",
    }

    try:
        if client is not None:
            response = client.post(endpoint_url, json=payload, headers=headers, timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as internal_client:
                response = internal_client.post(endpoint_url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise EmbeddingError("Embedding API request timed out.") from exc
    except httpx.ConnectError as exc:
        raise EmbeddingError("Failed to connect to embedding API provider.") from exc
    except httpx.RequestError as exc:
        raise EmbeddingError(f"Network error during embedding request: {exc.__class__.__name__}.") from exc

    if response.status_code != 200:
        raise EmbeddingAPIError(
            f"Embedding API error with HTTP status {response.status_code}: {response.text}",
            status_code=response.status_code,
        )

    try:
        response_json = response.json()
    except Exception as exc:
        raise EmbeddingError("Malformed response: failed to parse JSON from embedding provider.") from exc

    return _parse_embeddings_response(response_json, expected_count=expected_count)


def embed_text(
    text: str,
    client: Optional[httpx.Client] = None,
    settings: Optional[Settings] = None,
) -> List[float]:
    """Generate a dense vector embedding for a single text string using Ollama.

    Args:
        text: Non-empty text string to embed (e.g. search query).
        client: Optional httpx.Client for dependency injection (testing/mocking).
        settings: Optional Settings instance. If omitted, loaded from get_settings().

    Returns:
        A list of floats representing the dense vector embedding.

    Raises:
        ValueError: If text is empty or whitespace-only.
        EmbeddingError: If the API call fails or response is invalid.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text cannot be empty or whitespace-only.")

    cfg = settings or get_settings()
    base_url = _resolve_base_url(getattr(cfg, "OLLAMA_BASE_URL", None))
    model = _resolve_embedding_model(getattr(cfg, "EMBEDDING_MODEL", None))
    endpoint_url = f"{base_url}/api/embed"

    payload = {
        "model": model,
        "input": text,
    }

    embeddings = _send_embedding_request(
        endpoint_url=endpoint_url,
        payload=payload,
        expected_count=1,
        client=client,
    )
    return embeddings[0]


def embed_texts(
    texts: List[str],
    client: Optional[httpx.Client] = None,
    settings: Optional[Settings] = None,
) -> List[List[float]]:
    """Generate dense vector embeddings for a batch of text strings using Ollama.

    Preserves 1-to-1 input ordering.

    Args:
        texts: Non-empty list of non-empty text strings to embed.
        client: Optional httpx.Client for dependency injection (testing/mocking).
        settings: Optional Settings instance. If omitted, loaded from get_settings().

    Returns:
        List of float lists representing dense vector embeddings in input order.

    Raises:
        ValueError: If texts is empty or contains empty/whitespace-only items.
        EmbeddingError: If the API call fails or response is invalid.
    """
    if not isinstance(texts, list) or len(texts) == 0:
        raise ValueError("Input texts list cannot be empty.")

    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            raise ValueError(f"Input text at index {i} cannot be empty or whitespace-only.")

    cfg = settings or get_settings()
    base_url = _resolve_base_url(getattr(cfg, "OLLAMA_BASE_URL", None))
    model = _resolve_embedding_model(getattr(cfg, "EMBEDDING_MODEL", None))
    endpoint_url = f"{base_url}/api/embed"

    payload = {
        "model": model,
        "input": texts,
    }

    return _send_embedding_request(
        endpoint_url=endpoint_url,
        payload=payload,
        expected_count=len(texts),
        client=client,
    )


def embed_chunks(
    chunks: Sequence[Chunk],
    client: Optional[httpx.Client] = None,
    settings: Optional[Settings] = None,
) -> List[List[float]]:
    """Generate dense vector embeddings for a sequence of Chunk objects using Ollama.

    Extracts chunk.content, preserves chunk order, and does NOT modify Chunk objects.
    Does not store anything in PostgreSQL or modify database models.

    Args:
        chunks: Sequence of Chunk objects to embed.
        client: Optional httpx.Client for dependency injection (testing/mocking).
        settings: Optional Settings instance. If omitted, loaded from get_settings().

    Returns:
        List of float lists representing dense vector embeddings in chunk order.

    Raises:
        ValueError: If chunks is empty or any chunk has empty content.
        EmbeddingError: If the API call fails or response is invalid.
    """
    if not isinstance(chunks, (list, tuple)) or len(chunks) == 0:
        raise ValueError("Chunks sequence cannot be empty.")

    contents: List[str] = []
    for i, chunk in enumerate(chunks):
        if not hasattr(chunk, "content") or not isinstance(chunk.content, str) or not chunk.content.strip():
            raise ValueError(f"Chunk at index {i} has empty or invalid content.")
        contents.append(chunk.content)

    all_embeddings: List[List[float]] = []
    for start in range(0, len(contents), EMBEDDING_BATCH_SIZE):
        batch = contents[start : start + EMBEDDING_BATCH_SIZE]
        batch_embeddings = embed_texts(batch, client=client, settings=settings)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
