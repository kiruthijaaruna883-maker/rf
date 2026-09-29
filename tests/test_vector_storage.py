"""Integration tests for PostgreSQL and pgvector document chunk storage."""

import unittest
from sqlalchemy import func, select
from app.database.models import Document, DocumentChunk
from app.database.session import SessionLocal
from app.services.chunking import Chunk
from app.services.vector_storage import store_document_chunks


class TestVectorStorage(unittest.TestCase):
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

    def _create_test_document(self, filename: str = "test_guidance.pdf") -> Document:
        doc = Document(
            filename=filename,
            file_path=f"/tmp/{filename}",
            file_type="pdf",
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        self.test_docs.append(doc)
        return doc

    def test_store_document_chunks_and_pgvector_persistence(self):
        doc = self._create_test_document("test_persistence.pdf")

        chunks = [
            Chunk(chunk_index=0, content="Regulatory Intro", char_count=16, start_char=0, end_char=16),
            Chunk(chunk_index=1, content="Regulatory Scope", char_count=16, start_char=17, end_char=33),
        ]
        v0 = [0.01] * 1536
        v1 = [0.02] * 1536
        embeddings = [v0, v1]

        stored = store_document_chunks(self.db, doc.id, chunks, embeddings)
        self.assertEqual(len(stored), 2)
        self.assertEqual(stored[0].chunk_index, 0)
        self.assertEqual(stored[0].content, "Regulatory Intro")
        self.assertEqual(stored[1].chunk_index, 1)

        # Query database directly to confirm pgvector persistence
        direct_chunks = self.db.scalars(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index)
        ).all()
        self.assertEqual(len(direct_chunks), 2)
        # Vector dimension check
        self.assertEqual(len(direct_chunks[0].embedding), 1536)
        self.assertAlmostEqual(direct_chunks[0].embedding[0], 0.01, places=5)
        self.assertAlmostEqual(direct_chunks[1].embedding[0], 0.02, places=5)

    def test_document_chunk_relationship(self):
        doc = self._create_test_document("test_rel.pdf")
        chunks = [
            Chunk(chunk_index=0, content="Content A", char_count=9, start_char=0, end_char=9),
        ]
        embeddings = [[0.05] * 1536]
        store_document_chunks(self.db, doc.id, chunks, embeddings)

        self.db.refresh(doc)
        self.assertEqual(len(doc.chunks), 1)
        self.assertEqual(doc.chunks[0].content, "Content A")

    def test_cascade_deletion(self):
        doc = self._create_test_document("test_cascade.pdf")
        chunks = [
            Chunk(chunk_index=0, content="Content 1", char_count=9, start_char=0, end_char=9),
            Chunk(chunk_index=1, content="Content 2", char_count=9, start_char=10, end_char=19),
        ]
        embeddings = [[0.1] * 1536, [0.2] * 1536]
        store_document_chunks(self.db, doc.id, chunks, embeddings)

        # Verify chunks exist
        count_before = self.db.scalar(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == doc.id)
        )
        self.assertEqual(count_before, 2)

        # Delete document and verify chunks are cascade deleted
        self.db.delete(doc)
        self.db.commit()
        self.test_docs.remove(doc)

        count_after = self.db.scalar(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == doc.id)
        )
        self.assertEqual(count_after, 0)

    def test_duplicate_chunk_index_in_batch_rejected(self):
        doc = self._create_test_document("test_dup_batch.pdf")
        # Same chunk_index in batch
        chunks = [
            Chunk(chunk_index=0, content="First", char_count=5, start_char=0, end_char=5),
            Chunk(chunk_index=0, content="Duplicate", char_count=9, start_char=6, end_char=15),
        ]
        embeddings = [[0.1] * 1536, [0.2] * 1536]
        with self.assertRaises(ValueError):
            store_document_chunks(self.db, doc.id, chunks, embeddings)

    def test_duplicate_chunk_index_in_database_rolled_back(self):
        doc = self._create_test_document("test_dup_db.pdf")
        chunks1 = [Chunk(chunk_index=0, content="Initial", char_count=7, start_char=0, end_char=7)]
        embeddings1 = [[0.1] * 1536]
        store_document_chunks(self.db, doc.id, chunks1, embeddings1)

        # Try to store another chunk with chunk_index=0 for same document
        chunks2 = [Chunk(chunk_index=0, content="Conflict", char_count=8, start_char=0, end_char=8)]
        embeddings2 = [[0.2] * 1536]
        with self.assertRaises(RuntimeError):
            store_document_chunks(self.db, doc.id, chunks2, embeddings2)

        # Verify original chunk is still intact and no duplicate was added
        remaining = self.db.scalars(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        ).all()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].content, "Initial")

    def test_validation_errors(self):
        doc = self._create_test_document("test_validation.pdf")
        valid_chunk = Chunk(chunk_index=0, content="Content", char_count=7, start_char=0, end_char=7)
        valid_emb = [0.1] * 1536

        # Non-existent document_id
        with self.assertRaises(ValueError):
            store_document_chunks(self.db, 999999, [valid_chunk], [valid_emb])

        # Empty sequences
        with self.assertRaises(ValueError):
            store_document_chunks(self.db, doc.id, [], [])

        # Count mismatch
        with self.assertRaises(ValueError):
            store_document_chunks(self.db, doc.id, [valid_chunk], [valid_emb, valid_emb])

        # Wrong embedding dimension
        bad_dim_emb = [0.1] * 100
        with self.assertRaises(ValueError):
            store_document_chunks(self.db, doc.id, [valid_chunk], [bad_dim_emb])

        # Non-numeric embedding value
        non_num_emb = [0.1] * 1535 + ["text"]
        with self.assertRaises(ValueError):
            store_document_chunks(self.db, doc.id, [valid_chunk], [non_num_emb])


if __name__ == "__main__":
    unittest.main()
