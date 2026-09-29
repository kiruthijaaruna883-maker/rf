"""Integration tests for similarity retrieval against PostgreSQL and pgvector."""

import math
from unittest.mock import MagicMock, patch
import unittest
import httpx
from sqlalchemy import func, select

from app.database.models import Document, DocumentChunk
from app.database.session import SessionLocal
from app.services.retrieval import ChunkSearchResult, retrieve_chunks, search_similar_chunks


class TestRetrieval(unittest.TestCase):
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
        finally:
            self.db.close()

    def _setup_test_documents_and_chunks(self):
        """Create 2 documents with distinct known 1536-dimensional vectors."""
        doc1 = Document(
            filename="fda_guidance_2026.pdf",
            file_path="/tmp/fda_guidance_2026.pdf",
            file_type="pdf",
        )
        doc2 = Document(
            filename="ema_guidance_2026.pdf",
            file_path="/tmp/ema_guidance_2026.pdf",
            file_type="pdf",
        )
        self.db.add_all([doc1, doc2])
        self.db.commit()
        self.db.refresh(doc1)
        self.db.refresh(doc2)
        self.test_docs.extend([doc1, doc2])

        # Direction 0: aligned along axis 0
        v0 = [0.0] * 1536
        v0[0] = 1.0

        # Direction 1: 45 degrees between axis 0 and 1 (distance ~ 1 - 1/sqrt(2) = 0.2929)
        v1 = [0.0] * 1536
        v1[0] = 1.0 / math.sqrt(2)
        v1[1] = 1.0 / math.sqrt(2)

        # Direction 2: aligned along axis 1 (orthogonal to axis 0, distance = 1.0)
        v2 = [0.0] * 1536
        v2[1] = 1.0

        # Direction 3: doc2 chunk, distance ~ 0.5 to axis 0
        v3 = [0.0] * 1536
        v3[0] = 0.5
        v3[1] = math.sqrt(0.75)

        c0 = DocumentChunk(
            document_id=doc1.id,
            chunk_index=0,
            content="FDA Regulatory Overview and Scope.",
            char_count=34,
            start_char=0,
            end_char=34,
            metadata_json={"section": "1.0"},
            embedding=v0,
        )
        c1 = DocumentChunk(
            document_id=doc1.id,
            chunk_index=1,
            content="FDA Clinical Trial Methodology.",
            char_count=31,
            start_char=35,
            end_char=66,
            metadata_json={"section": "2.0"},
            embedding=v1,
        )
        c2 = DocumentChunk(
            document_id=doc1.id,
            chunk_index=2,
            content="FDA Safety Reporting.",
            char_count=21,
            start_char=67,
            end_char=88,
            metadata_json={"section": "3.0"},
            embedding=v2,
        )
        c3 = DocumentChunk(
            document_id=doc2.id,
            chunk_index=0,
            content="EMA Guideline Active Substances.",
            char_count=32,
            start_char=0,
            end_char=32,
            metadata_json={"annex": "A"},
            embedding=v3,
        )

        self.db.add_all([c0, c1, c2, c3])
        self.db.commit()
        return doc1, doc2, [c0, c1, c2, c3]

    def test_search_similar_chunks_cosine_ordering_and_similarity(self):
        doc1, doc2, chunks = self._setup_test_documents_and_chunks()

        # Query vector perfectly matches v0 (axis 0)
        query_vector = [0.0] * 1536
        query_vector[0] = 1.0

        results = search_similar_chunks(self.db, query_vector, top_k=5)
        self.assertEqual(len(results), 4)

        # First match must be chunk 0 of doc 1 (distance ~ 0.0)
        self.assertEqual(results[0].chunk_id, chunks[0].id)
        self.assertAlmostEqual(results[0].distance, 0.0, places=4)
        self.assertAlmostEqual(results[0].similarity, 1.0, places=4)
        self.assertEqual(results[0].filename, "fda_guidance_2026.pdf")
        self.assertEqual(results[0].metadata_json, {"section": "1.0"})

        # Second match must be chunk 1 (distance ~ 0.2929)
        self.assertEqual(results[1].chunk_id, chunks[1].id)
        self.assertTrue(results[0].distance <= results[1].distance <= results[2].distance)

        # Check similarity formula across all results
        for r in results:
            self.assertAlmostEqual(r.similarity, 1.0 - r.distance, places=6)

    def test_top_k_limiting_and_fewer_than_k(self):
        doc1, doc2, chunks = self._setup_test_documents_and_chunks()
        query_vector = [0.0] * 1536
        query_vector[0] = 1.0

        # top_k=2 limits to 2
        results_k2 = search_similar_chunks(self.db, query_vector, top_k=2)
        self.assertEqual(len(results_k2), 2)

        # top_k=10 returns all 4 available chunks without error
        results_k10 = search_similar_chunks(self.db, query_vector, top_k=10)
        self.assertEqual(len(results_k10), 4)

    def test_empty_database_returns_empty_list(self):
        # Database has no chunks matching
        query_vector = [0.0] * 1536
        query_vector[0] = 1.0
        results = search_similar_chunks(self.db, query_vector, top_k=5)
        self.assertEqual(results, [])

    def test_document_id_and_document_ids_filtering(self):
        doc1, doc2, chunks = self._setup_test_documents_and_chunks()
        query_vector = [0.0] * 1536
        query_vector[0] = 1.0

        # Single document_id filter
        res_doc1 = search_similar_chunks(self.db, query_vector, top_k=10, document_id=doc1.id)
        self.assertEqual(len(res_doc1), 3)
        for r in res_doc1:
            self.assertEqual(r.document_id, doc1.id)

        # Multiple document_ids filter
        res_doc2 = search_similar_chunks(self.db, query_vector, top_k=10, document_ids=[doc2.id])
        self.assertEqual(len(res_doc2), 1)
        self.assertEqual(res_doc2[0].document_id, doc2.id)

        # Ambiguous filtering rejected
        with self.assertRaises(ValueError):
            search_similar_chunks(self.db, query_vector, document_id=doc1.id, document_ids=[doc2.id])

    def test_max_distance_filtering(self):
        doc1, doc2, chunks = self._setup_test_documents_and_chunks()
        query_vector = [0.0] * 1536
        query_vector[0] = 1.0

        # Cutoff distance at 0.3 should include only chunks 0 (dist 0) and 1 (dist ~0.2929)
        results = search_similar_chunks(self.db, query_vector, top_k=10, max_distance=0.3)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertTrue(r.distance <= 0.3)

    def test_retrieval_is_strictly_read_only(self):
        doc1, doc2, chunks = self._setup_test_documents_and_chunks()
        query_vector = [0.0] * 1536
        query_vector[0] = 1.0

        count_before = self.db.scalar(select(func.count(DocumentChunk.id)))
        _ = search_similar_chunks(self.db, query_vector, top_k=5)
        count_after = self.db.scalar(select(func.count(DocumentChunk.id)))

        self.assertEqual(count_before, count_after)
        self.assertEqual(len(self.db.dirty), 0)
        self.assertEqual(len(self.db.new), 0)
        self.assertEqual(len(self.db.deleted), 0)

    def test_retrieve_chunks_delegates_to_embed_text(self):
        doc1, doc2, chunks = self._setup_test_documents_and_chunks()
        query_vector = [0.0] * 1536
        query_vector[0] = 1.0

        with patch("app.services.retrieval.embed_text") as mock_embed:
            mock_embed.return_value = query_vector
            mock_client = MagicMock(spec=httpx.Client)

            results = retrieve_chunks(
                db=self.db,
                query="FDA clinical scope",
                top_k=2,
                document_id=doc1.id,
                client=mock_client,
            )
            mock_embed.assert_called_once_with("FDA clinical scope", client=mock_client)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].chunk_id, chunks[0].id)

    def test_validation_errors(self):
        valid_vec = [0.0] * 1536

        # Bad query string
        with self.assertRaises(ValueError):
            retrieve_chunks(self.db, "")
        with self.assertRaises(ValueError):
            retrieve_chunks(self.db, "   ")

        # Bad vector dimension
        with self.assertRaises(ValueError):
            search_similar_chunks(self.db, [0.1] * 100)

        # Non-numeric or boolean in vector
        with self.assertRaises(ValueError):
            search_similar_chunks(self.db, [0.1] * 1535 + ["bad"])
        with self.assertRaises(ValueError):
            search_similar_chunks(self.db, [0.1] * 1535 + [True])

        # Bad top_k
        with self.assertRaises(ValueError):
            search_similar_chunks(self.db, valid_vec, top_k=0)
        with self.assertRaises(ValueError):
            search_similar_chunks(self.db, valid_vec, top_k=-5)

        # Bad max_distance
        with self.assertRaises(ValueError):
            search_similar_chunks(self.db, valid_vec, max_distance=-0.1)


if __name__ == "__main__":
    unittest.main()
