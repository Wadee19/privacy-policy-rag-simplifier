import numpy as np

from privacy_policy_rag.chunking import chunk_text, source_coverage_ratio
from privacy_policy_rag.evaluation import evaluate_rewrite
from privacy_policy_rag.retrieval import retrieve


class FakeEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([
                lowered.count("data") + lowered.count("sharing"),
                lowered.count("rights") + lowered.count("delete"),
            ])
        return np.asarray(vectors, dtype=float)


def test_chunker_keeps_complete_source_coverage():
    text = " ".join(f"w{i}" for i in range(1030))
    chunks = chunk_text(text, chunk_size=500, overlap=80, min_tail_size=120)
    assert chunks[0].start_word == 0
    assert chunks[-1].end_word == 1030
    assert source_coverage_ratio(chunks, 1030) == 1.0


def test_chunker_rejects_invalid_overlap():
    try:
        chunk_text("one two three", chunk_size=3, overlap=3)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid overlap to raise ValueError")


def test_retrieval_ranks_relevant_chunk_first():
    chunks = chunk_text(
        "data sharing advertising tracking " * 20
        + " rights delete access appeal " * 20,
        chunk_size=80,
        overlap=0,
        min_tail_size=0,
    )
    hits = retrieve(chunks, "rights delete", FakeEmbedder(), top_k=1)
    assert len(hits) == 1
    assert "rights" in hits[0].chunk.text


def test_evaluation_reports_trace_coverage():
    source = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(source, chunk_size=120, overlap=20, min_tail_size=0)
    outputs = {chunk.chunk_id: "plain language output" for chunk in chunks[:-1]}
    metrics = evaluate_rewrite(source, chunks, outputs)
    assert 0 < metrics.source_chunk_coverage < 1
    assert metrics.source_words == 300
