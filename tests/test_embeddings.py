"""Unit tests for embedding generation service with mocked HTTP layer."""

from unittest.mock import MagicMock
import unittest
import httpx

from app.config import Settings
from app.services.chunking import Chunk
from app.services.embeddings import (
    EmbeddingAPIError,
    EmbeddingConfigError,
    EmbeddingError,
    embed_chunks,
    embed_text,
    embed_texts,
)

MOCK_SETTINGS = Settings(
    OPENAI_API_KEY="sk-valid-mock-key-1234567890",
    EMBEDDING_MODEL="text-embedding-3-small",
    JWT_SECRET_KEY="test-jwt-secret",
)


def make_mock_embedding_response(vectors: list, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response matching the OpenAI embeddings endpoint format."""
    data = []
    for idx, vec in enumerate(vectors):
        data.append({
            "object": "embedding",
            "index": idx,
            "embedding": vec,
        })
    return httpx.Response(
        status_code=status_code,
        json={"object": "list", "data": data, "model": "text-embedding-3-small"},
        request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
    )


class TestEmbeddings(unittest.TestCase):
    def test_embed_text_success(self):
        dummy_vector = [0.05] * 1536
        mock_response = make_mock_embedding_response([dummy_vector])

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        text = "Regulatory compliance guideline."
        result = embed_text(text, client=mock_client, settings=MOCK_SETTINGS)

        self.assertEqual(len(result), 1536)
        self.assertEqual(result, dummy_vector)
        # Verify request construction
        mock_client.post.assert_called_once()
        _, kwargs = mock_client.post.call_args
        self.assertEqual(kwargs["json"]["input"], text)
        self.assertEqual(kwargs["json"]["model"], "text-embedding-3-small")
        self.assertIn("Authorization", kwargs["headers"])
        self.assertEqual(kwargs["headers"]["Authorization"], f"Bearer {MOCK_SETTINGS.OPENAI_API_KEY}")

    def test_embed_texts_batch_order_preservation(self):
        v0 = [0.1] * 1536
        v1 = [0.2] * 1536
        v2 = [0.3] * 1536
        # Intentionally order the API response out-of-order to test index restoration
        data = [
            {"object": "embedding", "index": 2, "embedding": v2},
            {"object": "embedding", "index": 0, "embedding": v0},
            {"object": "embedding", "index": 1, "embedding": v1},
        ]
        mock_response = httpx.Response(
            status_code=200,
            json={"object": "list", "data": data, "model": "text-embedding-3-small"},
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
        )
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        texts = ["Text A", "Text B", "Text C"]
        results = embed_texts(texts, client=mock_client, settings=MOCK_SETTINGS)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], v0)
        self.assertEqual(results[1], v1)
        self.assertEqual(results[2], v2)

    def test_embed_chunks_success(self):
        v0 = [0.11] * 1536
        v1 = [0.22] * 1536
        mock_response = make_mock_embedding_response([v0, v1])

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        chunks = [
            Chunk(chunk_index=0, content="Chunk zero text", char_count=15, start_char=0, end_char=15),
            Chunk(chunk_index=1, content="Chunk one text", char_count=14, start_char=16, end_char=30),
        ]
        results = embed_chunks(chunks, client=mock_client, settings=MOCK_SETTINGS)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], v0)
        self.assertEqual(results[1], v1)
        # Verify chunks were not mutated
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].content, "Chunk zero text")

    def test_validation_empty_inputs(self):
        mock_client = MagicMock(spec=httpx.Client)
        with self.assertRaises(ValueError):
            embed_text("", client=mock_client, settings=MOCK_SETTINGS)
        with self.assertRaises(ValueError):
            embed_text("   \n\t  ", client=mock_client, settings=MOCK_SETTINGS)
        with self.assertRaises(ValueError):
            embed_texts([], client=mock_client, settings=MOCK_SETTINGS)
        with self.assertRaises(ValueError):
            embed_texts(["Valid", ""], client=mock_client, settings=MOCK_SETTINGS)
        with self.assertRaises(ValueError):
            embed_chunks([], client=mock_client, settings=MOCK_SETTINGS)

    def test_missing_or_placeholder_api_key(self):
        bad_settings = [
            Settings(OPENAI_API_KEY=None, JWT_SECRET_KEY="sec"),
            Settings(OPENAI_API_KEY="", JWT_SECRET_KEY="sec"),
            Settings(OPENAI_API_KEY="your-openai-api-key", JWT_SECRET_KEY="sec"),
            Settings(OPENAI_API_KEY="placeholder_key_here", JWT_SECRET_KEY="sec"),
        ]
        for cfg in bad_settings:
            with self.assertRaises(EmbeddingConfigError):
                embed_text("Sample", settings=cfg)

    def test_http_api_status_errors(self):
        mock_client = MagicMock(spec=httpx.Client)
        # 401 Unauthorized
        mock_client.post.return_value = httpx.Response(
            401,
            json={"error": {"message": "Invalid API key"}},
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
        )
        with self.assertRaises(EmbeddingAPIError) as ctx401:
            embed_text("Sample", client=mock_client, settings=MOCK_SETTINGS)
        self.assertEqual(ctx401.exception.status_code, 401)

        # 429 Rate limit
        mock_client.post.return_value = httpx.Response(
            429,
            json={"error": {"message": "Rate limit reached"}},
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
        )
        with self.assertRaises(EmbeddingAPIError) as ctx429:
            embed_text("Sample", client=mock_client, settings=MOCK_SETTINGS)
        self.assertEqual(ctx429.exception.status_code, 429)

        # 500 Server error
        mock_client.post.return_value = httpx.Response(
            500,
            json={"error": {"message": "Internal error"}},
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
        )
        with self.assertRaises(EmbeddingAPIError) as ctx500:
            embed_text("Sample", client=mock_client, settings=MOCK_SETTINGS)
        self.assertEqual(ctx500.exception.status_code, 500)

    def test_network_and_timeout_errors(self):
        mock_client = MagicMock(spec=httpx.Client)
        # Timeout
        mock_client.post.side_effect = httpx.TimeoutException("Connection timed out")
        with self.assertRaises(EmbeddingError) as ctx_timeout:
            embed_text("Sample", client=mock_client, settings=MOCK_SETTINGS)
        self.assertIn("timed out", str(ctx_timeout.exception).lower())

        # Connect error
        mock_client.post.side_effect = httpx.ConnectError("Could not resolve host")
        with self.assertRaises(EmbeddingError) as ctx_connect:
            embed_text("Sample", client=mock_client, settings=MOCK_SETTINGS)
        self.assertIn("connect", str(ctx_connect.exception).lower())

    def test_malformed_api_responses(self):
        mock_client = MagicMock(spec=httpx.Client)
        # Non-numeric embedding element
        bad_vec = [0.1] * 1535 + ["not-a-number"]
        mock_client.post.return_value = make_mock_embedding_response([bad_vec])
        with self.assertRaises(EmbeddingError):
            embed_text("Sample", client=mock_client, settings=MOCK_SETTINGS)

        # Missing embedding key
        mock_client.post.return_value = httpx.Response(
            200,
            json={"object": "list", "data": [{"index": 0}], "model": "text-embedding-3-small"},
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
        )
        with self.assertRaises(EmbeddingError):
            embed_text("Sample", client=mock_client, settings=MOCK_SETTINGS)


if __name__ == "__main__":
    unittest.main()
