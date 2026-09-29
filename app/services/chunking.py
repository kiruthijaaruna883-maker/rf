"""Regulatory text chunking service for Regulatory Affairs Assistant.

Provides deterministic, boundary-aware text chunking with exact character offsets
and configurable overlap, implemented using Python standard-library functionality only.
"""

from dataclasses import dataclass, field
import re
from typing import List, Optional, Set

# Default chunking parameters calibrated for regulatory text and embedding models
DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 200

# Common abbreviations in regulatory, legal, scientific, and medical documents
# that should not be treated as sentence boundaries when followed by a period.
REGULATORY_ABBREVIATIONS: Set[str] = {
    "e.g.",
    "i.e.",
    "vs.",
    "sec.",
    "no.",
    "vol.",
    "ref.",
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "al.",
    "approx.",
    "dept.",
    "fig.",
    "p.",
    "pp.",
    "cf.",
    "cfr.",
    "fda.",
    "reg.",
    "subpart.",
    "par.",
}


@dataclass
class Chunk:
    """Represents a discrete text chunk with source character offsets.

    Attributes:
        chunk_index: 0-based sequential index of the chunk within the document.
        content: The text content of the chunk.
        char_count: Length of content in characters (len(content)).
        start_char: Starting character offset in the source string (inclusive).
        end_char: Ending character offset in the source string (exclusive).
        metadata: Optional dictionary for tracking chunk or document metadata.
    """

    chunk_index: int
    content: str
    char_count: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


def _is_sentence_boundary(text: str, punct_pos: int) -> bool:
    """Determine if punctuation at punct_pos represents a true sentence boundary.

    Distinguishes sentence terminators from regulatory citations, abbreviations,
    decimal numbers, and acronyms.

    Args:
        text: Source text string.
        punct_pos: Index of the punctuation character ('.', '!', or '?').

    Returns:
        True if the punctuation marks a valid sentence boundary, False otherwise.
    """
    ch = text[punct_pos]
    if ch in ("!", "?"):
        return True

    # For periods, inspect the preceding word token
    prefix = text[: punct_pos + 1]
    m = re.search(r"([A-Za-z0-9._]+)$", prefix)
    if not m:
        return True

    token = m.group(1).lower()

    # Check known abbreviations (e.g., "e.g.", "sec.", "ref.")
    if token in REGULATORY_ABBREVIATIONS:
        return False

    # Check single-letter abbreviations (e.g., "A.", "U.")
    if re.match(r"^[a-z]\.$", token):
        return False

    # Check multi-part dotted acronyms (e.g., "u.s.", "c.f.r.")
    if re.match(r"^([a-z]\.)+$", token):
        return False

    return True


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """Split normalized text into deterministic chunks with exact character offsets.

    Preserves paragraph, sentence, and word boundaries wherever possible.
    Splits inside words only when an individual word exceeds chunk_size.

    Args:
        text: The source (normalized) text string to chunk.
        chunk_size: Maximum character count for each chunk. Must be > 0.
        chunk_overlap: Number of characters to overlap between adjacent chunks.
            Must be >= 0 and strictly less than chunk_size.

    Returns:
        A list of Chunk objects with exact source character offsets.

    Raises:
        ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
    """
    # Parameter validation
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    # Empty or whitespace-only input
    if not text or not text.strip():
        return []

    chunks: List[Chunk] = []
    chunk_index = 0
    current_start = 0
    prev_end = 0

    # Advance to initial non-whitespace character
    while current_start < len(text) and text[current_start].isspace():
        current_start += 1

    while current_start < len(text):
        # Skip leading whitespace for the current chunk
        while current_start < len(text) and text[current_start].isspace():
            current_start += 1
        if current_start >= len(text):
            break

        max_end = min(len(text), current_start + chunk_size)
        window = text[current_start:max_end]

        if max_end == len(text):
            raw_end = len(text)
        else:
            raw_end = None

            # Tier 1: Paragraph boundary (\n\n+)
            # Candidate split point must be > max(current_start, prev_end) to ensure forward progress
            para_matches = list(re.finditer(r"\n[ \t]*\n+", window))
            for pm in reversed(para_matches):
                cand = current_start + pm.start()
                if cand > max(current_start, prev_end) and text[current_start:cand].strip():
                    raw_end = cand
                    break

            # Tier 2: Sentence boundary ([.!?] followed by whitespace or end-of-string)
            if raw_end is None:
                sent_matches = list(re.finditer(r"([.!?]['\"\u201d\u2019)\]]*)(?:\s+|\Z)", window))
                for sm in reversed(sent_matches):
                    punct_idx = current_start + sm.start(1)
                    if _is_sentence_boundary(text, punct_idx):
                        cand = current_start + sm.end(1)
                        if cand > max(current_start, prev_end) and text[current_start:cand].strip():
                            raw_end = cand
                            break

            # Tier 3: Line boundary (single \n)
            if raw_end is None:
                line_matches = list(re.finditer(r"\n+", window))
                for lm in reversed(line_matches):
                    cand = current_start + lm.start()
                    if cand > max(current_start, prev_end) and text[current_start:cand].strip():
                        raw_end = cand
                        break

            # Tier 4: Word boundary (whitespace)
            if raw_end is None:
                word_matches = list(re.finditer(r"[ \t]+", window))
                for wm in reversed(word_matches):
                    cand = current_start + wm.start()
                    if cand > max(current_start, prev_end) and text[current_start:cand].strip():
                        raw_end = cand
                        break

            # Tier 5: Hard character split (when a single token exceeds chunk_size)
            if raw_end is None:
                raw_end = max_end

        # Trim trailing whitespace from chunk end
        actual_end = raw_end
        while actual_end > current_start and text[actual_end - 1].isspace():
            actual_end -= 1

        content = text[current_start:actual_end]
        if content:
            chunks.append(
                Chunk(
                    chunk_index=chunk_index,
                    content=content,
                    char_count=len(content),
                    start_char=current_start,
                    end_char=actual_end,
                )
            )
            chunk_index += 1
            prev_end = actual_end

        # If no non-whitespace text remains beyond this chunk, we are finished
        if not text[actual_end:].strip():
            break

        # Calculate starting position for the next chunk
        if chunk_overlap == 0:
            next_start = actual_end
            while next_start < len(text) and text[next_start].isspace():
                next_start += 1
            current_start = next_start
        else:
            target_overlap_start = actual_end - chunk_overlap
            min_start = current_start + 1
            candidate_start = max(min_start, target_overlap_start)

            # Within the overlap window [candidate_start, actual_end], try to snap
            # to a clean sentence or word boundary so chunks begin coherently.
            overlap_window = text[candidate_start:actual_end]
            best_boundary: Optional[int] = None

            # Try sentence start in overlap window
            sent_in_overlap = list(re.finditer(r"([.!?]['\"\u201d\u2019)\]]*)\s+", overlap_window))
            for sm in sent_in_overlap:
                punct_idx = candidate_start + sm.start(1)
                if _is_sentence_boundary(text, punct_idx):
                    cand_start = candidate_start + sm.end()
                    if cand_start < actual_end:
                        best_boundary = cand_start
                        break

            # Try word start in overlap window if no sentence boundary found
            if best_boundary is None:
                word_in_overlap = list(re.finditer(r"\s+", overlap_window))
                for wm in word_in_overlap:
                    cand_start = candidate_start + wm.end()
                    if cand_start < actual_end:
                        best_boundary = cand_start
                        break

            if best_boundary is not None:
                candidate_start = best_boundary

            # Advance past any leading whitespace at the candidate start
            while candidate_start < actual_end and text[candidate_start].isspace():
                candidate_start += 1

            if candidate_start < actual_end:
                current_start = candidate_start
            else:
                next_start = actual_end
                while next_start < len(text) and text[next_start].isspace():
                    next_start += 1
                current_start = next_start

    return chunks
