from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .prompting import SYSTEM_INSTRUCTIONS, build_rewrite_input


@dataclass
class OpenAIRewriter:
    """Lazy optional wrapper around the OpenAI Responses API."""

    model: str = os.getenv("OPENAI_MODEL", "gpt-5.2")
    max_output_tokens: int = 1800

    def rewrite(self, chunk: Chunk) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI support is optional. Install requirements-llm.txt first."
            ) from exc

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")

        client = OpenAI()
        response = client.responses.create(
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=build_rewrite_input(chunk),
            max_output_tokens=self.max_output_tokens,
        )
        text = response.output_text.strip()
        if not text:
            raise RuntimeError(f"Model returned empty text for {chunk.chunk_id}")
        return text


@dataclass
class SentenceTransformerEmbedder:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local embeddings are optional. Install requirements-llm.txt first."
            ) from exc
        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=float)
