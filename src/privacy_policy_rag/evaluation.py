from __future__ import annotations

from dataclasses import asdict, dataclass

from .chunking import Chunk, source_coverage_ratio


@dataclass(frozen=True)
class RewriteMetrics:
    source_words: int
    output_words: int
    expansion_ratio: float
    source_chunk_coverage: float
    flesch_kincaid_grade: float | None
    gunning_fog: float | None
    smog_index: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _readability(text: str) -> tuple[float | None, float | None, float | None]:
    try:
        import textstat
    except ImportError:
        return None, None, None
    if not text.strip():
        return None, None, None
    return (
        float(textstat.flesch_kincaid_grade(text)),
        float(textstat.gunning_fog(text)),
        float(textstat.smog_index(text)),
    )


def evaluate_rewrite(
    source_text: str,
    source_chunks: list[Chunk],
    rewritten_by_chunk: dict[str, str],
) -> RewriteMetrics:
    """Compute structural/readability metrics without claiming legal faithfulness."""
    source_words = len(source_text.split())
    ordered_outputs = [
        rewritten_by_chunk[chunk.chunk_id]
        for chunk in source_chunks
        if chunk.chunk_id in rewritten_by_chunk
    ]
    output_text = "\n\n".join(ordered_outputs)
    output_words = len(output_text.split())
    expansion = output_words / source_words if source_words else 0.0

    expected_ids = {chunk.chunk_id for chunk in source_chunks}
    completed_ids = set(rewritten_by_chunk).intersection(expected_ids)
    trace_coverage = len(completed_ids) / len(expected_ids) if expected_ids else 1.0

    if source_coverage_ratio(source_chunks, source_words) < 1.0:
        raise ValueError("Source chunking does not cover the complete document")

    fkgl, fog, smog = _readability(output_text)
    return RewriteMetrics(
        source_words=source_words,
        output_words=output_words,
        expansion_ratio=expansion,
        source_chunk_coverage=trace_coverage,
        flesch_kincaid_grade=fkgl,
        gunning_fog=fog,
        smog_index=smog,
    )
