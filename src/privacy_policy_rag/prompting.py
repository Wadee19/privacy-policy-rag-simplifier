from __future__ import annotations

from .chunking import Chunk


SYSTEM_INSTRUCTIONS = """You simplify privacy-policy text for non-lawyers.
Preserve every material fact, actor, condition, exception, right, obligation, purpose, recipient, retention statement, and limitation found in the supplied source.
Do not add legal conclusions, promises, advice, examples, or facts that are not supported by the source.
Do not summarize away details. Prefer shorter sentences and plain English while keeping the original meaning.
Return only the rewritten text for the supplied chunk.
"""


def build_rewrite_input(chunk: Chunk) -> str:
    return (
        f"SOURCE TRACE ID: {chunk.chunk_id}\n"
        f"SOURCE WORD RANGE: {chunk.start_word}:{chunk.end_word}\n\n"
        "Rewrite this source chunk in clear, plain English. Preserve its meaning and details.\n\n"
        f"SOURCE:\n{chunk.text}"
    )
