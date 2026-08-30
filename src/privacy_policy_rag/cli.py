from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking import chunk_text
from .pipeline import rewrite_document, stitch_rewrite
from .providers import OpenAIRewriter, SentenceTransformerEmbedder
from .retrieval import retrieve


def _read_text(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")
    return text


def command_rewrite(args: argparse.Namespace) -> None:
    source = _read_text(args.input)
    rewriter = OpenAIRewriter(model=args.model)
    records, metrics = rewrite_document(
        source,
        rewriter,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "simplified_policy.md").write_text(
        stitch_rewrite(records), encoding="utf-8"
    )
    with (output_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics.as_dict(), indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics.as_dict(), indent=2))


def command_retrieve(args: argparse.Namespace) -> None:
    source = _read_text(args.input)
    chunks = chunk_text(source, chunk_size=args.chunk_size, overlap=args.overlap)
    embedder = SentenceTransformerEmbedder(model_name=args.embedding_model)
    hits = retrieve(chunks, args.query, embedder, top_k=args.top_k)
    for rank, hit in enumerate(hits, start=1):
        preview = hit.chunk.text[:500].replace("\n", " ")
        print(f"#{rank} {hit.chunk.chunk_id} score={hit.score:.4f}\n{preview}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Privacy-policy RAG and simplification tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rewrite = subparsers.add_parser("rewrite", help="Simplify an entire policy with traceable chunks")
    rewrite.add_argument("--input", required=True, help="UTF-8 text file containing the policy")
    rewrite.add_argument("--output-dir", default="artifacts")
    rewrite.add_argument("--model", default="gpt-5.2")
    rewrite.add_argument("--chunk-size", type=int, default=500)
    rewrite.add_argument("--overlap", type=int, default=80)
    rewrite.set_defaults(func=command_rewrite)

    search = subparsers.add_parser("retrieve", help="Retrieve policy chunks relevant to a question")
    search.add_argument("--input", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--chunk-size", type=int, default=500)
    search.add_argument("--overlap", type=int, default=80)
    search.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    search.set_defaults(func=command_retrieve)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
