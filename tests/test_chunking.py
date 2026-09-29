"""Unit tests for regulatory text chunking service."""

import unittest
from app.services.chunking import Chunk, chunk_text


class TestChunking(unittest.TestCase):
    def test_chunk_dataclass_fields(self):
        c = Chunk(
            chunk_index=0,
            content="Regulatory section.",
            char_count=19,
            start_char=0,
            end_char=19,
        )
        self.assertEqual(c.chunk_index, 0)
        self.assertEqual(c.content, "Regulatory section.")
        self.assertEqual(c.char_count, 19)
        self.assertEqual(c.start_char, 0)
        self.assertEqual(c.end_char, 19)

    def test_parameter_validation(self):
        text = "Sample text for chunking."
        # chunk_size <= 0
        with self.assertRaises(ValueError):
            chunk_text(text, chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_text(text, chunk_size=-100)

        # chunk_overlap < 0
        with self.assertRaises(ValueError):
            chunk_text(text, chunk_overlap=-1)

        # chunk_overlap >= chunk_size
        with self.assertRaises(ValueError):
            chunk_text(text, chunk_size=100, chunk_overlap=100)
        with self.assertRaises(ValueError):
            chunk_text(text, chunk_size=100, chunk_overlap=150)

    def test_empty_and_whitespace_input(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\t  "), [])

    def test_short_text_single_chunk(self):
        text = "A brief regulatory statement."
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].content, text)
        self.assertEqual(chunks[0].start_char, 0)
        self.assertEqual(chunks[0].end_char, len(text))
        self.assertEqual(chunks[0].char_count, len(text))

    def test_character_offsets_integrity(self):
        text = (
            "Section 1: General Principles.\n\n"
            "All investigational medicinal products must comply with Good Clinical Practice. "
            "Sponsors shall ensure adequate oversight of trial conduct.\n\n"
            "Section 2: Quality Standards.\n\n"
            "Active pharmaceutical ingredients must meet established pharmacopeial specifications."
        )
        chunks = chunk_text(text, chunk_size=120, chunk_overlap=30)
        self.assertTrue(len(chunks) > 1)

        for chunk in chunks:
            extracted_slice = text[chunk.start_char : chunk.end_char]
            self.assertEqual(extracted_slice, chunk.content)
            self.assertEqual(len(chunk.content), chunk.char_count)
            self.assertTrue(chunk.char_count <= 120)

    def test_zero_overlap(self):
        text = "First paragraph content here.\n\nSecond paragraph content here."
        chunks = chunk_text(text, chunk_size=40, chunk_overlap=0)
        self.assertTrue(len(chunks) >= 2)
        # Verify no overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            self.assertTrue(chunks[i + 1].start_char >= chunks[i].end_char)

    def test_paragraph_boundary_preservation(self):
        para1 = "Paragraph 1 contains important guidelines for clinical study sponsors."
        para2 = "Paragraph 2 outlines post-marketing safety reporting obligations."
        text = f"{para1}\n\n{para2}"

        # Size large enough to hold either paragraph, but not both
        chunks = chunk_text(text, chunk_size=len(para1) + 15, chunk_overlap=10)
        self.assertTrue(len(chunks) >= 2)
        self.assertEqual(chunks[0].content, para1)

    def test_regulatory_abbreviations_protection(self):
        # Text where period in 'e.g.' or 'sec.' should NOT split into separate sentences prematurely
        text = (
            "Requirements for stability testing (e.g. accelerated and long term) "
            "must follow ICH Q1A guidelines as outlined in sec. 3 of the guidance document."
        )
        # Choose a size that would fit the whole sentence if not split at e.g.
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=30)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, text)


if __name__ == "__main__":
    unittest.main()
