"""Text extraction and normalization service for Regulatory Affairs Assistant.

Supports PDF, DOCX, and TXT document formats.
Deterministic and conservative normalization for regulatory guidance documents.
"""

from pathlib import Path
import re
from typing import List
import unicodedata

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph
import pypdf


def extract_text_from_txt(path: Path) -> str:
    """Read and extract text from a plain text file.

    Prefers UTF-8 with fallback to common encodings (UTF-8-SIG, CP1252, Latin-1).
    Does not modify the source file.

    Args:
        path: Path to the text file.

    Returns:
        The extracted raw text string.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If reading the file fails.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {path}")

    try:
        raw_bytes = path.read_bytes()
    except Exception as e:
        raise RuntimeError(f"Failed to read text file '{path}': {e}") from e

    # Try common encodings in priority order
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue

    # Fallback to UTF-8 with replacement
    return raw_bytes.decode("utf-8", errors="replace")


def extract_text_from_pdf(path: Path) -> str:
    """Extract text from a PDF file using pypdf.PdfReader.

    Iterates through all pages in document order, preserving page reading order.
    Handles pages where extracted text is None.
    Does not modify the source file.

    Args:
        path: Path to the PDF file.

    Returns:
        The extracted raw text string across all pages.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If PDF parsing fails.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {path}")

    try:
        reader = pypdf.PdfReader(str(path))
        pages_text: List[str] = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

        return "\n\n".join(pages_text)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF file '{path}': {e}") from e


def extract_text_from_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx.

    Extracts paragraphs and table contents in document body order.
    Does not modify the source file.

    Args:
        path: Path to the DOCX file.

    Returns:
        The extracted raw text string.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If DOCX parsing fails.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {path}")

    try:
        doc = docx.Document(str(path))
        elements: List[str] = []

        # Iterate through document body elements in document order
        if hasattr(doc, "element") and hasattr(doc.element, "body"):
            for child in doc.element.body:
                if child.tag.endswith("p"):
                    p = Paragraph(child, doc)
                    if p.text and p.text.strip():
                        elements.append(p.text.strip())
                elif child.tag.endswith("tbl"):
                    t = Table(child, doc)
                    for row in t.rows:
                        row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_cells:
                            elements.append(" | ".join(row_cells))

        # Fallback to direct paragraphs/tables if body iteration yields nothing
        if not elements and (doc.paragraphs or doc.tables):
            for p in doc.paragraphs:
                if p.text and p.text.strip():
                    elements.append(p.text.strip())
            for t in doc.tables:
                for row in t.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        elements.append(" | ".join(row_cells))

        return "\n\n".join(elements)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from DOCX file '{path}': {e}") from e


def normalize_text(text: str) -> str:
    """Normalize extracted document text deterministically and conservatively.

    Preserves paragraph structure, regulatory terminology, citations,
    section numbers, and capitalization. Only performs whitespace, newline,
    and control character normalization.

    Args:
        text: Raw extracted document text string.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    # 1. Convert CRLF and CR to standard LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Replace non-breaking spaces and zero-width spaces with standard equivalents
    text = text.replace("\u00a0", " ")  # Non-breaking space
    text = text.replace("\u202f", " ")  # Narrow no-break space
    text = text.replace("\u2007", " ")  # Figure space
    text = text.replace("\ufeff", "")   # Byte-order mark
    text = text.replace("\u200b", "")   # Zero-width space
    text = text.replace("\u200c", "")   # Zero-width non-joiner
    text = text.replace("\u200d", "")   # Zero-width joiner

    # 3. Remove inappropriate control characters (retain \n and \t)
    cleaned_chars: List[str] = []
    for ch in text:
        if ch in ("\n", "\t"):
            cleaned_chars.append(ch)
        elif unicodedata.category(ch) == "Cc":
            continue
        else:
            cleaned_chars.append(ch)
    text = "".join(cleaned_chars)

    # 4. Normalize lines:
    # - normalize repeated horizontal whitespace within lines to a single space
    # - remove trailing whitespace from lines
    normalized_lines: List[str] = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line)
        line = line.strip()
        normalized_lines.append(line)
    text = "\n".join(normalized_lines)

    # 5. Collapse excessive blank lines (more than 2 consecutive newlines -> 2 newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6. Strip leading and trailing whitespace from the complete result
    return text.strip()


def extract_document_text(file_path: Path, file_type: str) -> str:
    """Extract and normalize text from a document based on its file type.

    Unified dispatcher for regulatory document ingestion pipeline.

    Args:
        file_path: Path to the document file.
        file_type: File extension or type identifier ('txt', 'pdf', 'docx').

    Returns:
        Clean, normalized text string.

    Raises:
        FileNotFoundError: If the document file does not exist.
        ValueError: If the file type is unsupported.
        RuntimeError: If text extraction fails.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {path}")

    normalized_type = file_type.lower().strip().lstrip(".")

    if normalized_type == "txt":
        raw_text = extract_text_from_txt(path)
    elif normalized_type == "pdf":
        raw_text = extract_text_from_pdf(path)
    elif normalized_type == "docx":
        raw_text = extract_text_from_docx(path)
    else:
        raise ValueError(
            f"Unsupported document type '{file_type}'. Supported types: 'txt', 'pdf', 'docx'"
        )

    return normalize_text(raw_text)
