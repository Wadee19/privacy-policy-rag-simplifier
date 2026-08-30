from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .chunking import Chunk


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def cosine_scores(query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
    """Compute cosine similarity with explicit zero-vector handling."""
    query = np.asarray(query_vector, dtype=float).reshape(-1)
    docs = np.asarray(document_vectors, dtype=float)
    if docs.ndim != 2:
        raise ValueError("document_vectors must be a 2D matrix")
    if docs.shape[1] != query.shape[0]:
        raise ValueError("query/document embedding dimensions do not match")

    query_norm = np.linalg.norm(query)
    doc_norms = np.linalg.norm(docs, axis=1)
    denom = doc_norms * query_norm
    numer = docs @ query
    return np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)


def retrieve(chunks: list[Chunk], query: str, embedder: Embedder, top_k: int = 5) -> list[RetrievedChunk]:
    """Return the highest-scoring source chunks for a query."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not chunks:
        return []

    document_vectors = np.asarray(embedder.encode([chunk.text for chunk in chunks]))
    query_vector = np.asarray(embedder.encode([query]))[0]
    scores = cosine_scores(query_vector, document_vectors)

    k = min(top_k, len(chunks))
    ranked = np.argsort(-scores, kind="stable")[:k]
    return [RetrievedChunk(chunk=chunks[int(i)], score=float(scores[int(i)])) for i in ranked]
