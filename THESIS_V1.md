# V1 Thesis Provenance

This repository preserves the original MSc thesis proof of concept and adds a later engineering V2 without rewriting the project history.

## V1 — submitted MSc PoC

The original artifact is kept unchanged:

- `RAG_Privacy_Policy_Simplifier_(TikTok_PoC) (2).ipynb`
- `README.txt`

V1 explored a TikTok privacy-policy simplification workflow using overlapping chunks, MiniLM embeddings, semantic Top-K retrieval, and an instruction-tuned generator in Google Colab. Its recorded results and limitations remain in the original notes.

## V2 — post-thesis engineering work

The `src/privacy_policy_rag/` package, tests, CI, CLI, trace IDs, coverage-safe chunking, and explicit retrieval-vs-rewrite separation were added later as an engineering upgrade.

V2 must not be represented as part of the original submitted thesis experiment unless a specific file or result actually existed in V1. Likewise, V1 metrics are historical results and are not automatically evidence for the V2 runtime.

This separation is intentional: it keeps the academic provenance honest while showing how the original research prototype can be evolved into a cleaner software-engineering implementation.
