"""End-to-end integration test for the Phase 4 RAG pipeline.

Tests the full sequential flow:
Document text -> Normalization -> Chunking -> Embedding (mocked API) -> pgvector Storage -> Similarity Retrieval
"""

from unittest.mock import MagicMock, patch
import unittest
import httpx
from sqlalchemy import func, select

from app.database.models import Document, DocumentChunk
from app.database.session import SessionLocal
from app.services.chunking import chunk_text
from app.services.retrieval import retrieve_chunks
from app.services.text_extraction import normalize_text
from app.services.vector_storage import store_document_chunks


class TestRAGPipelineE2E(unittest.TestCase):
    def setUp(self):
        self.db = SessionLocal()
        self.test_docs = []

    def tearDown(self):
        try:
            self.db.rollback()
            for doc in self.test_docs:
                existing = self.db.get(Document, doc.id)
                if existing:
                    self.db.delete(existing)
            self.db.commit()
            # Verify clean DB state
            doc_count = self.db.scalar(select(func.count(Document.id)))
            chunk_count = self.db.scalar(select(func.count(DocumentChunk.id)))
            self.assertEqual(doc_count, 0)
            self.assertEqual(chunk_count, 0)
        finally:
            self.db.close()

    def test_full_rag_pipeline_flow(self):
        """Execute complete flow: Raw text -> Normalization -> Chunking -> Embed -> Store -> Retrieve."""
        # 1. Raw regulatory guidance text
        raw_text = (
            "  GUIDANCE DOCUMENT FOR CLINICAL INVESTIGATIONS   \r\n\r\n"
            "Section 1: General Requirements.\r\n"
            "All clinical trials must be conducted in accordance with ethical principles that have "
            "their origin in the Declaration of Helsinki and are consistent with GCP.\r\n\r\n"
            "Section 2: Stability and Shelf-Life Testing.\r\n"
            "Accelerated stability testing shall be conducted at 40 degrees C plus or minus 2 degrees C "
            "and 75 percent RH plus or minus 5 percent RH for a minimum duration of six months.\r\n\r\n"
            "Section 3: Adverse Event Expedited Reporting.\r\n"
            "Any serious and unexpected adverse drug reaction must be reported to the regulatory authority "
            "within seven calendar days for fatal or life-threatening cases.\r\n"
        )

        # 2. Text Normalization
        normalized = normalize_text(raw_text)
        self.assertNotIn("\r", normalized)
        self.assertTrue(normalized.startswith("GUIDANCE DOCUMENT FOR CLINICAL INVESTIGATIONS"))

        # 3. Regulatory Chunking
        chunks = chunk_text(normalized, chunk_size=250, chunk_overlap=40)
        self.assertTrue(len(chunks) >= 3)

        # Find the target chunk discussing accelerated stability testing conditions
        target_chunk_idx = None
        for i, ch in enumerate(chunks):
            if "Accelerated stability testing" in ch.content:
                target_chunk_idx = i
                break
        self.assertIsNotNone(target_chunk_idx, "Target stability chunk not found among chunked texts")

        # 4. Generate deterministic mock embeddings
        # Assign an orthogonal basis vector to each chunk
        chunk_embeddings = []
        for i in range(len(chunks)):
            vec = [0.0] * 768
            vec[i] = 1.0
            chunk_embeddings.append(vec)

        # 5. Store in PostgreSQL + pgvector
        doc = Document(
            filename="ich_q1a_stability.pdf",
            file_path="/tmp/ich_q1a_stability.pdf",
            file_type="pdf",
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        self.test_docs.append(doc)

        stored_chunks = store_document_chunks(
            db=self.db,
            document_id=doc.id,
            chunks=chunks,
            embeddings=chunk_embeddings,
        )
        self.assertEqual(len(stored_chunks), len(chunks))

        # 6. Similarity Retrieval for a query targeted at stability testing
        query = "What are the accelerated stability testing conditions and duration?"
        target_query_vector = list(chunk_embeddings[target_chunk_idx])

        # Mock embed_text to return the target chunk's vector
        with patch("app.services.retrieval.embed_text") as mock_embed:
            mock_embed.return_value = target_query_vector
            mock_client = MagicMock(spec=httpx.Client)

            search_results = retrieve_chunks(
                db=self.db,
                query=query,
                top_k=3,
                document_id=doc.id,
                client=mock_client,
            )

            # 7. Verification of retrieval results
            self.assertTrue(len(search_results) >= 1)
            top_result = search_results[0]

            # Top result must be the exact stability chunk
            self.assertEqual(top_result.chunk_index, target_chunk_idx)
            self.assertIn("Accelerated stability testing", top_result.content)
            self.assertIn("40 degrees C", top_result.content)
            self.assertEqual(top_result.filename, "ich_q1a_stability.pdf")

            # Distance should be 0.0 (identical vector) and similarity should be 1.0
            self.assertAlmostEqual(top_result.distance, 0.0, places=4)
            self.assertAlmostEqual(top_result.similarity, 1.0, places=4)

            # Remaining results must have strictly higher distance
            for other_result in search_results[1:]:
                self.assertTrue(other_result.distance > top_result.distance)
                self.assertAlmostEqual(other_result.similarity, 1.0 - other_result.distance, places=6)


if __name__ == "__main__":
    unittest.main()
