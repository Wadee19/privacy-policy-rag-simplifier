"""Privacy-policy simplification and retrieval utilities."""

from .chunking import Chunk, chunk_text, source_coverage_ratio
from .evaluation import RewriteMetrics, evaluate_rewrite
from .pipeline import RewriteRecord, rewrite_document, stitch_rewrite
from .retrieval import RetrievedChunk, retrieve

__all__ = [
    "Chunk",
    "RetrievedChunk",
    "RewriteMetrics",
    "RewriteRecord",
    "chunk_text",
    "evaluate_rewrite",
    "retrieve",
    "rewrite_document",
    "source_coverage_ratio",
    "stitch_rewrite",
]
