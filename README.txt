# RAG Privacy Policy Simplifier (TikTok PoC)

Imp, this is a PoC only.
Idea: take a privacy policy URL, use a small RAG pipeline to reduce hallucination, then rewrite the policy in simpler English.

Not legal advice. Not 100% coverage.

---

## What this does
- Fetch policy text from the URL
- Clean the text (whitespace + common broken encoding chars)
- Chunking with overlap (so meaning at chunk borders is not lost)
- Embeddings + Top-K retrieval using a query
- Rewrite using only the retrieved context (not the whole doc)
- Compare readability before vs after + plot the difference

---

## Input
Policy URL used in this PoC:
- https://www.tiktok.com/legal/privacy-policy?lang=en

Retrieval query:
- `data collection, sharing, retention, rights, security, advertising, tracking`

---

## Models used
Embeddings (retrieval):
- `sentence-transformers/all-MiniLM-L6-v2`

Generator (rewriter):
- `mistralai/Mistral-7B-Instruct-v0.2`  
Loaded with 4-bit quant so it fits GPU in Colab.

---

## Main settings I used
Chunking:
- `CHUNK_WORDS = 500`
- `CHUNK_OVERLAP = 80`
- `MIN_CHUNK_WORDS = 120`

Retrieval:
- `TOP_K = 12`  
I kept it 12 because the policy is long and small Top-K was missing important parts (rights, transfers, ads, etc).

Generation context:
- `TOP_K_FOR_GEN = 11`
- `per_chunk_chars = 1200` (caps each chunk to reduce token overflow)

Generation:
- `max_new_tokens = 1400`
- `do_sample = False` (more stable, less loops)
- `repetition_penalty = 1.12`
- `no_repeat_ngram_size = 8`

---

## How to run (Colab)
1) Run installs + imports + device check  
2) Paste `HF_TOKEN` (hidden input via getpass)  
3) Fetch + clean + preview head/tail (make sure the text is real and long)  
4) Chunking  
5) Embeddings + retrieval + preview top chunks + scores  
6) Load generator (Mistral 7B 4-bit)  
7) Rewrite + evaluate + plot

If the fetch looks incomplete:
- Use `MANUAL_POLICY_TEXT` and paste the full policy, then rerun.

---

## Current results (what I saw in the trial)
Policy text size (from preview):
- chars: **50903**
- words: **8237**
- The page showed “Other regions”, last updated **July 8, 2025**

Chunking output:
- chunks: **20**
- words min/avg/max: **257 / 487 / 500**

Rewrite output (After):
- words: **374**
- chars: **2473**

Readability metrics (Before -> After):
- FKGL: **14.82 -> 13.48**
- GunningFog: **17.66 -> 16.49**
- SMOG: **15.79 -> 14.92**

### What was good
- Output matched several key areas like location (approx + GPS), image/audio analysis, ads/analytics/payment sharing, and international transfers.
- The retrieval chunks looked relevant, not random trash.

### What was not good yet (honest)
- Output came out too short compared to the original policy (8237 words vs 374 words rewrite).
- The model sometimes used bullets even when the prompt says “no bullets”.
- Some lines got too general, like slightly guessy, so it can drift.
========================================
So yeah, good PoC, but not final legal-grade rewrite.
========================================
---

## Known risks / limitations
- JS-rendered pages: some sites don’t expose full policy in HTML. Fetch may return partial text. Manual paste is the backup.
- Coverage: Top-K retrieval means you might miss parts of the policy.
- Not legal advice: this is a tech demo, not lawyer review.

---

## How to improve accuracy later (without breaking the project)
These are upgrades that make sense, not random stuff.

### Plan A (keep same setup, just tighten it)
Coverage control:
- if we Add a target length ratio (example: aim output = **70–80%** of original words)
- Rewriteing chunk-by-chunk and then stitch paragraphs in original order (this is the clean way to keep coverage)

Hallucination guardrails:
- Block “we never”, “we do not sell”, “consent” type claims unless the meaning is actually in the retrieved text
- Add an evidence mode later: store chunk IDs behind each paragraph (internal only, not shown to user)

Retrieval quality:
- maybe be Use multiple retrieval queries, not one long string
- Optional reranking (cross-encoder) after initial retrieval, for better Top-K

### Plan B (try other models for better results)
This is the “ok let’s test stronger options” plan.

Retrieval model ideas (embeddings):
- wee can Try a stronger sentence-transformer embedding model (bigger one) and compare retrieval scores + chunk relevance.
- Keep the same pipeline, just swap the embedder and re-run.

Generator model ideas (rewriter):
- we could Try a different instruction model that follows formatting rules better and compare output length + drift.
- Same pipeline again, just swapping generator model and re-run .

About GPT-5.2:
- I wanted to test GPT-5.2 as a generator because it usually follows rules better and can reduce the “guessy” lines, but I couldn’t afford running the PoC with it right now.
- So I kept everything on HuggingFace + Colab and treated this as a baseline.

---

## Minimum system requirements (from my real run)
I’m putting this as minimum, not recommended.

- System RAM (min): **6.4 GB**
- GPU RAM (min): **7.1 GB**
- Disk (min): **52.4 GB**

---

## Disclaimer
This is for learning/testing and may contain mistakes. The official and legally binding text is always the original privacy policy.

Ahmed Wadee Moustafa
