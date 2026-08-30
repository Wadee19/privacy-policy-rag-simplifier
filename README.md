# Privacy Policy RAG Simplifier

**MSc thesis PoC + post-thesis engineering V2** for simplifying and exploring long privacy policies with retrieval-augmented generation and traceable LLM workflows.

This repository has two deliberately separated layers:

- **V1 — MSc thesis submission / original PoC:** the original TikTok privacy-policy experiment is preserved unchanged for academic provenance.
- **V2 — post-thesis engineering upgrade:** the same research idea is packaged as testable Python components with source traceability, deterministic tests, CI, and a clearer distinction between retrieval and full-document rewriting.

> Educational and research project only. It is not legal advice and the generated text must not replace the legally binding source policy.

## Research question

Privacy policies are long, legally dense, and difficult for non-specialists to read. The thesis PoC explored whether a small RAG pipeline could retrieve relevant passages and generate a simpler English representation while reducing unsupported generation by grounding the model in source text.

## V1 — original MSc thesis PoC

The original submitted experiment is preserved in:

- `RAG_Privacy_Policy_Simplifier_(TikTok_PoC) (2).ipynb`
- `README.txt` — original experiment notes and settings

The TikTok trial used:

- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- overlapping ~500-word chunks
- Top-K semantic retrieval
- `mistralai/Mistral-7B-Instruct-v0.2` as the local generator in Colab
- readability metrics including FKGL, Gunning Fog, and SMOG

### Recorded V1 trial results

| Metric | Original policy | Retrieval-grounded output |
|---|---:|---:|
| Words | 8,237 | 374 |
| Flesch-Kincaid Grade Level | 14.82 | 13.48 |
| Gunning Fog | 17.66 | 16.49 |
| SMOG | 15.79 | 14.92 |

These values are retained as **historical thesis/PoC results**, not presented as a new benchmark. The original notes also document the main limitation: Top-K retrieval produced a much shorter output and therefore could not guarantee complete policy coverage.

## Why V2 exists

The V1 notebook answered a research/prototyping question. V2 answers a different engineering question: **how would the same idea be structured so that it is portable, testable, traceable, and harder to misuse?**

The biggest conceptual correction is that two tasks are now named separately:

1. **RAG retrieval mode** — retrieve policy passages relevant to a user question or risk area.
2. **Full-document simplification mode** — process every source chunk in order and rewrite it with an explicit source trace.

A full rewrite is not described as RAG simply because an LLM processes chunks.

## V2 architecture

```text
                         privacy policy
                               │
                               ▼
                    coverage-safe chunking
                    + source word offsets
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
          RETRIEVAL / RAG MODE      FULL REWRITE MODE
                  │                         │
         semantic embeddings        chunk trace IDs
                  │                         │
           cosine retrieval          constrained LLM rewrite
                  │                         │
        top-k grounded passages      ordered traceable output
                  │                         │
                  └────────────┬────────────┘
                               ▼
                  transparent evaluation
             coverage + readability + limits
```

### No silent tail loss

V2's chunker tracks source word offsets. If a short remainder would otherwise be dropped, it is folded into the final chunk. Tests assert 100% source-position coverage before generation begins.

### Traceability

Every V2 rewrite record includes:

- `chunk_id`
- source `start_word` / `end_word`
- original source text
- rewritten text

That makes it possible to audit an output paragraph back to the source span that produced it.

## Repository structure

```text
.
├── RAG_Privacy_Policy_Simplifier_(TikTok_PoC) (2).ipynb  # original MSc thesis PoC
├── README.txt                                               # original V1 notes
├── THESIS_V1.md                                             # provenance / V1-vs-V2 guide
├── src/privacy_policy_rag/
│   ├── chunking.py
│   ├── retrieval.py
│   ├── prompting.py
│   ├── providers.py
│   ├── pipeline.py
│   ├── evaluation.py
│   └── cli.py
├── tests/test_core.py
├── .github/workflows/tests.yml
├── .env.example
├── requirements.txt
├── requirements-llm.txt
└── pyproject.toml
```

## Quick start — deterministic core

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```

The core test suite requires **no API key, GPU, Hugging Face token, or network model download**.

## Optional LLM runtime

```bash
pip install -r requirements-llm.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in your environment. The model is configurable through `OPENAI_MODEL`.

### Full-document simplification

```bash
PYTHONPATH=src python -m privacy_policy_rag.cli rewrite \
  --input policy.txt \
  --output-dir artifacts
```

Outputs:

- `artifacts/simplified_policy.md`
- `artifacts/trace.jsonl`
- `artifacts/metrics.json`

### Retrieval / RAG mode

```bash
PYTHONPATH=src python -m privacy_policy_rag.cli retrieve \
  --input policy.txt \
  --query "How is my data shared with advertisers?" \
  --top-k 5
```

## Evaluation philosophy

Readability is useful evidence, but **readability is not semantic or legal faithfulness**.

V2 therefore keeps separate measurements for:

- preprocessing/source coverage
- completed trace coverage
- output length / expansion ratio
- readability metrics

It intentionally does **not** claim that automated readability scores prove that every legal condition, exception, or obligation was preserved. That would require a separately validated source-grounded evaluation protocol and/or expert review.

## Engineering improvements after the thesis PoC

- portable `src/` package instead of notebook-only execution
- explicit separation of retrieval from full-document rewriting
- source trace IDs and word offsets
- coverage-safe chunking
- injectable embedding layer for testability
- lazy optional model/API dependencies
- environment-variable secrets
- deterministic unit tests
- GitHub Actions CI
- evaluation language that does not confuse readability with truthfulness

## Limitations

- No lawyer-validated semantic benchmark.
- Generated simplifications may omit or alter legal nuance.
- JavaScript-heavy policy pages need a robust ingestion strategy; V2 currently expects clean input text at the core boundary.
- Local embeddings require a model download.
- Full-document generation has latency and cost proportional to source length.

## Author

Ahmed Wadee Moustafa
