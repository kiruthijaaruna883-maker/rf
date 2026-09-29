"""Embedding generation service for Regulatory Affairs Assistant.

Provides single-text and batch embedding generation using httpx against
OpenAI-compatible embedding endpoints without requiring the OpenAI Python SDK.
"""

from typing import Any, Dict, List, Optional, Sequence
import httpx

from app.config import Settings, get_settings
from app.services.chunking import Chunk

DEFAULT_EMBEDDING_TIMEOUT: float = 30.0
OPENAI_EMBEDDINGS_URL: str = "https://api.openai.com/v1/embeddings"


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


def _validate_api_key(api_key: Optional[str]) -> str:
    """Validate that the API key is present and not a template placeholder.

    Args:
        api_key: The configured API key.

    Returns:
        The validated API key string.

    Raises:
        EmbeddingConfigError: If the API key is missing or is a placeholder.
    """
    if not api_key or not api_key.strip():
        raise EmbeddingConfigError(
            "OpenAI API key is not configured. Please set OPENAI_API_KEY in the environment or .env file."
        )

    cleaned = api_key.strip()
    if cleaned.startswith("your-") or "placeholder" in cleaned.lower():
        raise EmbeddingConfigError(
            "OpenAI API key is set to a placeholder value. A valid API key is required for live embedding generation."
        )

    return cleaned


def _get_embedding_model(model: Optional[str]) -> str:
    """Resolve the embedding model name, falling back to text-embedding-3-small if placeholder.

    Args:
        model: Configured model string.

    Returns:
        Resolved model identifier.
    """
    if not model or not model.strip() or model.strip().startswith("your-") or "placeholder" in model.lower():
        return "text-embedding-3-small"
    return model.strip()


def _parse_embeddings_response(
    response_data: Any,
    expected_count: int,
) -> List[List[float]]:
    """Safely parse and validate the JSON response from an embeddings endpoint.

    Ensures ordering matches input indices even if provider returns data out of order.

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

    data = response_data.get("data")
    if data is None or not isinstance(data, list):
        raise EmbeddingError("Malformed response: missing 'data' array in embedding response.")

    if len(data) != expected_count:
        raise EmbeddingError(
            f"Embedding count mismatch: expected {expected_count} embeddings, got {len(data)}."
        )

    ordered_embeddings: List[Optional[List[float]]] = [None] * expected_count

    for item in data:
        if not isinstance(item, dict):
            raise EmbeddingError("Malformed response: each item in 'data' must be an object.")

        idx = item.get("index")
        if idx is None or not isinstance(idx, int) or isinstance(idx, bool) or idx < 0 or idx >= expected_count:
            raise EmbeddingError(f"Malformed response: invalid or out-of-range index '{idx}'.")

        embedding = item.get("embedding")
        if embedding is None or not isinstance(embedding, list) or len(embedding) == 0:
            raise EmbeddingError(f"Malformed response: missing or empty 'embedding' vector at index {idx}.")

        # Validate numeric contents (disallow bools which are subclasses of int)
        clean_vector: List[float] = []
        for val in embedding:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise EmbeddingError(
                    f"Invalid embedding value: non-numeric element detected at index {idx}."
                )
            clean_vector.append(float(val))

        if ordered_embeddings[idx] is not None:
            raise EmbeddingError(f"Malformed response: duplicate index '{idx}' returned in data.")

        ordered_embeddings[idx] = clean_vector

    for i, emb in enumerate(ordered_embeddings):
        if emb is None:
            raise EmbeddingError(f"Malformed response: missing embedding for input index {i}.")

    return [emb for emb in ordered_embeddings if emb is not None]


def _send_embedding_request(
    payload: Dict[str, Any],
    expected_count: int,
    api_key: str,
    client: Optional[httpx.Client] = None,
    timeout: float = DEFAULT_EMBEDDING_TIMEOUT,
) -> List[List[float]]:
    """Send an HTTP POST request to the embeddings endpoint and return parsed vectors.

    Never exposes API keys or Authorization headers in raised exceptions.

    Args:
        payload: JSON payload to send.
        expected_count: Expected number of returned embeddings.
        api_key: Validated API key.
        client: Optional httpx.Client for dependency injection (mocking/testing).
        timeout: Request timeout in seconds.

    Returns:
        List of dense vector embeddings.

    Raises:
        EmbeddingAPIError: For HTTP 4xx/5xx responses.
        EmbeddingError: For network, timeout, or parsing failures.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        if client is not None:
            response = client.post(OPENAI_EMBEDDINGS_URL, json=payload, headers=headers, timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as internal_client:
                response = internal_client.post(OPENAI_EMBEDDINGS_URL, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise EmbeddingError("Embedding API request timed out.") from exc
    except httpx.ConnectError as exc:
        raise EmbeddingError("Failed to connect to embedding API provider.") from exc
    except httpx.RequestError as exc:
        raise EmbeddingError(f"Network error during embedding request: {exc.__class__.__name__}.") from exc

    if response.status_code == 401:
        raise EmbeddingAPIError("Authentication failed: invalid or unauthorized OpenAI API key.", status_code=401)
    elif response.status_code == 429:
        raise EmbeddingAPIError("Rate limit exceeded from embedding provider.", status_code=429)
    elif 400 <= response.status_code < 500:
        raise EmbeddingAPIError(
            f"Embedding API client error with HTTP status {response.status_code}.",
            status_code=response.status_code,
        )
    elif response.status_code >= 500:
        raise EmbeddingAPIError(
            f"Embedding API server error with HTTP status {response.status_code}.",
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
    """Generate a dense vector embedding for a single text string.

    Args:
        text: Non-empty text string to embed (e.g. search query).
        client: Optional httpx.Client for dependency injection (testing/mocking).
        settings: Optional Settings instance. If omitted, loaded from get_settings().

    Returns:
        A list of floats representing the dense vector embedding.

    Raises:
        ValueError: If text is empty or whitespace-only.
        EmbeddingConfigError: If OpenAI API key is missing or placeholder.
        EmbeddingError: If the API call fails or response is invalid.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text cannot be empty or whitespace-only.")

    cfg = settings or get_settings()
    api_key = _validate_api_key(cfg.OPENAI_API_KEY)
    model = _get_embedding_model(cfg.EMBEDDING_MODEL)

    payload = {
        "input": text,
        "model": model,
    }

    embeddings = _send_embedding_request(
        payload=payload,
        expected_count=1,
        api_key=api_key,
        client=client,
    )
    return embeddings[0]


def embed_texts(
    texts: List[str],
    client: Optional[httpx.Client] = None,
    settings: Optional[Settings] = None,
) -> List[List[float]]:
    """Generate dense vector embeddings for a batch of text strings.

    Preserves 1-to-1 input ordering.

    Args:
        texts: Non-empty list of non-empty text strings to embed.
        client: Optional httpx.Client for dependency injection (testing/mocking).
        settings: Optional Settings instance. If omitted, loaded from get_settings().

    Returns:
        List of float lists representing dense vector embeddings in input order.

    Raises:
        ValueError: If texts is empty or contains empty/whitespace-only items.
        EmbeddingConfigError: If OpenAI API key is missing or placeholder.
        EmbeddingError: If the API call fails or response is invalid.
    """
    if not isinstance(texts, list) or len(texts) == 0:
        raise ValueError("Input texts list cannot be empty.")

    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            raise ValueError(f"Input text at index {i} cannot be empty or whitespace-only.")

    cfg = settings or get_settings()
    api_key = _validate_api_key(cfg.OPENAI_API_KEY)
    model = _get_embedding_model(cfg.EMBEDDING_MODEL)

    payload = {
        "input": texts,
        "model": model,
    }

    return _send_embedding_request(
        payload=payload,
        expected_count=len(texts),
        api_key=api_key,
        client=client,
    )


def embed_chunks(
    chunks: Sequence[Chunk],
    client: Optional[httpx.Client] = None,
    settings: Optional[Settings] = None,
) -> List[List[float]]:
    """Generate dense vector embeddings for a sequence of Chunk objects.

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
        EmbeddingConfigError: If OpenAI API key is missing or placeholder.
        EmbeddingError: If the API call fails or response is invalid.
    """
    if not isinstance(chunks, (list, tuple)) or len(chunks) == 0:
        raise ValueError("Chunks sequence cannot be empty.")

    contents: List[str] = []
    for i, chunk in enumerate(chunks):
        if not hasattr(chunk, "content") or not isinstance(chunk.content, str) or not chunk.content.strip():
            raise ValueError(f"Chunk at index {i} has empty or invalid content.")
        contents.append(chunk.content)

    return embed_texts(contents, client=client, settings=settings)
