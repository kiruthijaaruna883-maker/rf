"""Unit and integration tests for document ingestion endpoint (POST /documents/upload)."""

import io
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.api.auth import get_current_user
from app.database.models import Document, DocumentChunk, User
from app.database.session import SessionLocal
from app.main import app
from app.services.chunking import Chunk
from app.services.embeddings import EmbeddingError


class TestDocumentsApi(unittest.TestCase):
    """Test suite covering RAG document upload and ingestion flow."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        self.db = SessionLocal()
        self.created_doc_ids: list[int] = []
        self.created_file_paths: list[Path] = []

        # Use an existing user from the database or create one
        existing_user = self.db.scalars(select(User).limit(1)).first()
        if existing_user is None:
            existing_user = User(
                username="test_regulatory_officer",
                email="officer@regulatory.test",
                hashed_password="fake_hashed_pw",
                is_active=True,
            )
            self.db.add(existing_user)
            self.db.commit()
            self.db.refresh(existing_user)
        self.test_user = existing_user
        app.dependency_overrides[get_current_user] = lambda: self.test_user

    def tearDown(self) -> None:
        app.dependency_overrides = {}
        try:
            self.db.rollback()
            for doc_id in self.created_doc_ids:
                doc = self.db.get(Document, doc_id)
                if doc:
                    if doc.file_path and Path(doc.file_path).exists():
                        try:
                            Path(doc.file_path).unlink()
                        except OSError:
                            pass
                    self.db.delete(doc)
            self.db.commit()
        finally:
            self.db.close()

        for fp in self.created_file_paths:
            if fp.exists():
                try:
                    fp.unlink()
                except OSError:
                    pass

    def _mock_embeddings(self, count: int, dim: int = 768) -> list[list[float]]:
        return [[0.01 * (j % 10) for j in range(dim)] for _ in range(count)]

    # 1. Successful document ingestion
    @patch("app.api.documents.embed_chunks")
    def test_successful_document_ingestion(self, mock_embed: MagicMock) -> None:
        mock_embed.side_effect = lambda chunks: self._mock_embeddings(len(chunks))

        file_content = (
            "REGULATORY STABILITY GUIDELINE\n\n"
            "Section 1: Scope\n"
            "This guidance addresses the information to be submitted in registration applications "
            "for new drug substances and associated medicinal products.\n\n"
            "Section 2: Storage Conditions\n"
            "Long term testing shall be conducted at 25 degrees C +/- 2 degrees C / 60% RH +/- 5% RH "
            "for a minimum of 12 months. Accelerated testing shall be conducted at 40 degrees C +/- 2 "
            "degrees C / 75% RH +/- 5% RH for a minimum of 6 months.\n"
        )
        file_obj = io.BytesIO(file_content.encode("utf-8"))

        response = self.client.post(
            "/documents/upload",
            files={"file": ("stability_guideline.txt", file_obj, "text/plain")},
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        data = response.json()
        doc_id = data["id"]
        self.created_doc_ids.append(doc_id)

        self.assertEqual(data["filename"], "stability_guideline.txt")
        self.assertEqual(data["file_type"], "txt")
        self.assertEqual(data["uploaded_by"], self.test_user.id)
        self.assertGreater(data["chunks_count"], 0)
        self.assertIn("successfully ingested", data["message"])

        # Verify DB Document record
        db_doc = self.db.get(Document, doc_id)
        self.assertIsNotNone(db_doc)
        self.assertEqual(db_doc.filename, "stability_guideline.txt")
        self.created_file_paths.append(Path(db_doc.file_path))

        # Verify DB DocumentChunk records and pgvector embedding persistence
        chunks = self.db.scalars(
            select(DocumentChunk).where(DocumentChunk.document_id == doc_id).order_by(DocumentChunk.chunk_index)
        ).all()
        self.assertEqual(len(chunks), data["chunks_count"])
        for chunk in chunks:
            self.assertEqual(len(chunk.embedding), 768)
            self.assertTrue(len(chunk.content) > 0)
            self.assertEqual(chunk.char_count, len(chunk.content))

    # 2. Text extraction failure
    @patch("app.api.documents.extract_document_text")
    def test_text_extraction_failure_cleans_up_and_returns_422(self, mock_extract: MagicMock) -> None:
        mock_extract.side_effect = RuntimeError("Corrupted document stream")

        file_obj = io.BytesIO(b"Corrupted binary stream")
        response = self.client.post(
            "/documents/upload",
            files={"file": ("corrupt.pdf", file_obj, "application/pdf")},
        )

        self.assertEqual(response.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)
        self.assertIn("Failed to extract text from document", response.json()["detail"])

        # Verify no document or chunks created in database
        doc = self.db.scalar(select(Document).where(Document.filename == "corrupt.pdf"))
        self.assertIsNone(doc)

    # 3. Empty / unusable document handling
    def test_empty_document_handling_returns_400(self) -> None:
        file_obj = io.BytesIO(b"   \r\n\t  \n  ")
        response = self.client.post(
            "/documents/upload",
            files={"file": ("empty.txt", file_obj, "text/plain")},
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("no extractable text", response.json()["detail"].lower())

        doc = self.db.scalar(select(Document).where(Document.filename == "empty.txt"))
        self.assertIsNone(doc)

    # 4. Embedding failure
    @patch("app.api.documents.embed_chunks")
    def test_embedding_failure_cleans_up_and_returns_502(self, mock_embed: MagicMock) -> None:
        mock_embed.side_effect = EmbeddingError("Ollama service timeout")

        file_obj = io.BytesIO(b"Valid guidance text that produces chunks for embedding.")
        response = self.client.post(
            "/documents/upload",
            files={"file": ("valid_text.txt", file_obj, "text/plain")},
        )

        self.assertEqual(response.status_code, status.HTTP_502_BAD_GATEWAY)
        self.assertIn("Embedding service failed", response.json()["detail"])

        doc = self.db.scalar(select(Document).where(Document.filename == "valid_text.txt"))
        self.assertIsNone(doc)

    # 5. Chunk / vector persistence failure
    @patch("app.api.documents.embed_chunks")
    @patch("app.api.documents.store_document_chunks")
    def test_chunk_persistence_failure_cleans_up_and_returns_500(
        self, mock_store: MagicMock, mock_embed: MagicMock
    ) -> None:
        mock_embed.side_effect = lambda chunks: self._mock_embeddings(len(chunks))
        mock_store.side_effect = RuntimeError("pgvector constraint violation")

        file_obj = io.BytesIO(b"Valid guidance text for chunking and embedding.")
        response = self.client.post(
            "/documents/upload",
            files={"file": ("persistence_fail.txt", file_obj, "text/plain")},
        )

        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("Failed to store document chunks", response.json()["detail"])

        doc = self.db.scalar(select(Document).where(Document.filename == "persistence_fail.txt"))
        self.assertIsNone(doc)

    # 6. Existing document upload behavior (auth, file types, missing file)
    def test_unauthenticated_request_returns_401(self) -> None:
        app.dependency_overrides = {}  # remove auth override
        file_obj = io.BytesIO(b"Valid content")
        response = self.client.post(
            "/documents/upload",
            files={"file": ("doc.txt", file_obj, "text/plain")},
        )
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_unsupported_file_extension_returns_400(self) -> None:
        file_obj = io.BytesIO(b"Binary data")
        response = self.client.post(
            "/documents/upload",
            files={"file": ("malicious.exe", file_obj, "application/octet-stream")},
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Unsupported file type", response.json()["detail"])

    def test_missing_filename_returns_400(self) -> None:
        file_obj = io.BytesIO(b"Some text")
        response = self.client.post(
            "/documents/upload",
            files={"file": ("   ", file_obj, "text/plain")},
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("No file was supplied", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
