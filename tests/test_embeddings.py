"""Unit tests for embedding generation service."""

import unittest
from unittest.mock import MagicMock
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
    OLLAMA_BASE_URL="http://127.0.0.1:11434",
    EMBEDDING_MODEL="nomic-embed-text:latest",
    JWT_SECRET_KEY="test-jwt-secret",
)


def make_mock_embedding_response(vectors: list, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response matching the Ollama /api/embed endpoint format."""
    return httpx.Response(
        status_code=status_code,
        json={"model": "nomic-embed-text:latest", "embeddings": vectors},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/embed"),
    )


class TestEmbeddings(unittest.TestCase):
    def test_embed_text_success(self):
        dummy_vector = [0.05] * 768
        mock_response = make_mock_embedding_response([dummy_vector])

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        text = "Regulatory compliance guideline."
        result = embed_text(text, client=mock_client, settings=MOCK_SETTINGS)

        self.assertEqual(len(result), 768)
        self.assertEqual(result, dummy_vector)
        # Verify request construction
        mock_client.post.assert_called_once()
        args, kwargs = mock_client.post.call_args
        called_url = args[0] if args else kwargs.get("url")
        self.assertEqual(called_url, "http://127.0.0.1:11434/api/embed")
        self.assertEqual(kwargs["json"]["input"], text)
        self.assertEqual(kwargs["json"]["model"], "nomic-embed-text:latest")
        self.assertNotIn("Authorization", kwargs.get("headers", {}))

    def test_embed_texts_batch_order_preservation(self):
        v0 = [0.1] * 768
        v1 = [0.2] * 768
        v2 = [0.3] * 768
        mock_response = make_mock_embedding_response([v0, v1, v2])
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        texts = ["Text A", "Text B", "Text C"]
        results = embed_texts(texts, client=mock_client, settings=MOCK_SETTINGS)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], v0)
        self.assertEqual(results[1], v1)
        self.assertEqual(results[2], v2)

    def test_embed_chunks_success(self):
        v0 = [0.11] * 768
        v1 = [0.22] * 768
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

    def test_no_openai_api_key_required(self):
        dummy_vector = [0.05] * 768
        mock_response = make_mock_embedding_response([dummy_vector])
        test_settings = [
            Settings(OPENAI_API_KEY=None, JWT_SECRET_KEY="sec"),
            Settings(OPENAI_API_KEY="", JWT_SECRET_KEY="sec"),
            Settings(OPENAI_API_KEY="your-openai-api-key-here", JWT_SECRET_KEY="sec"),
            Settings(OPENAI_API_KEY="placeholder_key_here", JWT_SECRET_KEY="sec"),
        ]
        for cfg in test_settings:
            mock_client = MagicMock(spec=httpx.Client)
            mock_client.post.return_value = mock_response
            result = embed_text("Sample", client=mock_client, settings=cfg)
            self.assertEqual(len(result), 768)
            _, kwargs = mock_client.post.call_args
            self.assertNotIn("Authorization", kwargs.get("headers", {}))

    def test_http_api_status_errors(self):
        mock_client = MagicMock(spec=httpx.Client)
        for status in [400, 404, 500]:
            mock_client.post.return_value = httpx.Response(
                status,
                text=f"Error {status}",
                request=httpx.Request("POST", "http://127.0.0.1:11434/api/embed"),
            )
            with self.assertRaises(EmbeddingAPIError) as ctx:
                embed_text("Sample", client=mock_client, settings=MOCK_SETTINGS)
            self.assertEqual(ctx.exception.status_code, status)

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
        bad_vec = [0.1] * 767 + ["not-a-number"]
        mock_client.post.return_value = make_mock_embedding_response([bad_vec])
        with self.assertRaises(EmbeddingError):
            embed_text("Sample", client=mock_client, settings=MOCK_SETTINGS)

        # Missing embeddings key
        mock_client.post.return_value = httpx.Response(
            200,
            json={"model": "nomic-embed-text:latest"},
            request=httpx.Request("POST", "http://127.0.0.1:11434/api/embed"),
        )
        with self.assertRaises(EmbeddingError):
            embed_text("Sample", client=mock_client, settings=MOCK_SETTINGS)

        # Count mismatch
        mock_client.post.return_value = make_mock_embedding_response([[0.1] * 768])
        with self.assertRaises(EmbeddingError):
            embed_texts(["A", "B"], client=mock_client, settings=MOCK_SETTINGS)


    # --- Batching Tests ---

    def test_embed_chunks_5_chunks_one_request(self):
        chunks = [
            Chunk(chunk_index=i, content=f"Chunk content {i}", char_count=15, start_char=i * 20, end_char=i * 20 + 15)
            for i in range(5)
        ]
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = make_mock_embedding_response([[0.1] * 768 for _ in range(5)])

        results = embed_chunks(chunks, client=mock_client, settings=MOCK_SETTINGS)

        self.assertEqual(len(results), 5)
        self.assertEqual(mock_client.post.call_count, 1)
        _, kwargs = mock_client.post.call_args
        self.assertEqual(len(kwargs["json"]["input"]), 5)

    def test_embed_chunks_10_chunks_one_request(self):
        chunks = [
            Chunk(chunk_index=i, content=f"Chunk content {i}", char_count=15, start_char=i * 20, end_char=i * 20 + 15)
            for i in range(10)
        ]
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = make_mock_embedding_response([[0.1] * 768 for _ in range(10)])

        results = embed_chunks(chunks, client=mock_client, settings=MOCK_SETTINGS)

        self.assertEqual(len(results), 10)
        self.assertEqual(mock_client.post.call_count, 1)
        _, kwargs = mock_client.post.call_args
        self.assertEqual(len(kwargs["json"]["input"]), 10)

    def test_embed_chunks_25_chunks_three_requests_10_10_5(self):
        chunks = [
            Chunk(chunk_index=i, content=f"Chunk content {i}", char_count=15, start_char=i * 20, end_char=i * 20 + 15)
            for i in range(25)
        ]
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.side_effect = [
            make_mock_embedding_response([[0.1] * 768 for _ in range(10)]),
            make_mock_embedding_response([[0.2] * 768 for _ in range(10)]),
            make_mock_embedding_response([[0.3] * 768 for _ in range(5)]),
        ]

        results = embed_chunks(chunks, client=mock_client, settings=MOCK_SETTINGS)

        self.assertEqual(len(results), 25)
        self.assertEqual(mock_client.post.call_count, 3)

        call_sizes = [len(call[1]["json"]["input"]) for call in mock_client.post.call_args_list]
        self.assertEqual(call_sizes, [10, 10, 5])

    def test_embed_chunks_batching_preserves_input_order(self):
        chunks = [
            Chunk(chunk_index=i, content=f"Content {i}", char_count=9, start_char=i * 10, end_char=i * 10 + 9)
            for i in range(15)
        ]
        batch1_vecs = [[float(i)] * 768 for i in range(10)]
        batch2_vecs = [[float(i)] * 768 for i in range(10, 15)]

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.side_effect = [
            make_mock_embedding_response(batch1_vecs),
            make_mock_embedding_response(batch2_vecs),
        ]

        results = embed_chunks(chunks, client=mock_client, settings=MOCK_SETTINGS)

        self.assertEqual(len(results), 15)
        for i in range(15):
            self.assertEqual(results[i], [float(i)] * 768)

    def test_embed_chunks_batch_failure_propagated(self):
        chunks = [
            Chunk(chunk_index=i, content=f"Chunk {i}", char_count=7, start_char=i * 8, end_char=i * 8 + 7)
            for i in range(25)
        ]
        mock_client = MagicMock(spec=httpx.Client)
        # First batch succeeds, second batch fails
        mock_client.post.side_effect = [
            make_mock_embedding_response([[0.1] * 768 for _ in range(10)]),
            httpx.TimeoutException("Batch 2 timed out"),
        ]

        with self.assertRaises(EmbeddingError) as ctx:
            embed_chunks(chunks, client=mock_client, settings=MOCK_SETTINGS)
        self.assertIn("timed out", str(ctx.exception).lower())
        self.assertEqual(mock_client.post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
