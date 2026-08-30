from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .chunking import Chunk, chunk_text
from .evaluation import RewriteMetrics, evaluate_rewrite


class Rewriter(Protocol):
    def rewrite(self, chunk: Chunk) -> str: ...


@dataclass(frozen=True)
class RewriteRecord:
    chunk_id: str
    start_word: int
    end_word: int
    source_text: str
    rewritten_text: str


def rewrite_document(
    text: str,
    rewriter: Rewriter,
    *,
    chunk_size: int = 500,
    overlap: int = 80,
    min_tail_size: int = 120,
) -> tuple[list[RewriteRecord], RewriteMetrics]:
    """Rewrite a document chunk-by-chunk while preserving source traceability."""
    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
        min_tail_size=min_tail_size,
    )
    outputs: dict[str, str] = {}
    records: list[RewriteRecord] = []

    for chunk in chunks:
        rewritten = rewriter.rewrite(chunk).strip()
        if not rewritten:
            raise RuntimeError(f"Empty rewrite returned for {chunk.chunk_id}")
        outputs[chunk.chunk_id] = rewritten
        records.append(
            RewriteRecord(
                chunk_id=chunk.chunk_id,
                start_word=chunk.start_word,
                end_word=chunk.end_word,
                source_text=chunk.text,
                rewritten_text=rewritten,
            )
        )

    metrics = evaluate_rewrite(text, chunks, outputs)
    return records, metrics


def stitch_rewrite(records: list[RewriteRecord]) -> str:
    """Join rewritten chunks in source order with visible trace anchors."""
    return "\n\n".join(
        f"<!-- {record.chunk_id} source_words={record.start_word}:{record.end_word} -->\n"
        f"{record.rewritten_text}"
        for record in records
    )
