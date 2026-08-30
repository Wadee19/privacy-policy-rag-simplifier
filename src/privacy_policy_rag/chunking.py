from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A traceable source chunk with word offsets."""

    chunk_id: str
    text: str
    start_word: int
    end_word: int

    @property
    def word_count(self) -> int:
        return self.end_word - self.start_word


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 80,
    min_tail_size: int = 120,
) -> list[Chunk]:
    """Split text into overlapping chunks without silently dropping the tail."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    if min_tail_size < 0:
        raise ValueError("min_tail_size must be non-negative")

    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        remaining = len(words) - end
        if remaining and remaining < min_tail_size:
            end = len(words)

        chunks.append(
            Chunk(
                chunk_id=f"chunk-{index:04d}",
                text=" ".join(words[start:end]),
                start_word=start,
                end_word=end,
            )
        )
        if end >= len(words):
            break
        start = end - overlap
        index += 1

    return chunks


def source_coverage_ratio(chunks: list[Chunk], total_words: int) -> float:
    """Return the fraction of source word positions represented by chunks."""
    if total_words <= 0:
        return 1.0
    covered: set[int] = set()
    for chunk in chunks:
        covered.update(range(chunk.start_word, chunk.end_word))
    return len(covered) / total_words
