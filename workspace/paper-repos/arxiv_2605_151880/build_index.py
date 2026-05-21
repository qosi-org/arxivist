"""
FutureSim — CCNews Index Builder
==================================
Ingests raw CCNews JSONL files and builds a LanceDB hybrid search index.

Usage:
  python build_index.py --corpus-path data/ccnews/ --index-path data/lancedb_index/
  python build_index.py --corpus-path data/ccnews/ --index-path data/lancedb_index/ --force
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from futuresim.corpus.retrieval import CCNewsIndexBuilder
from futuresim.utils.logging import get_logger

logger = get_logger("futuresim.build_index")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build LanceDB hybrid index from CCNews corpus")
    p.add_argument("--corpus-path", required=True, help="Root of CCNews JSONL article files")
    p.add_argument("--index-path", required=True, help="Output LanceDB index directory")
    p.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-8B",
        help="HuggingFace model ID for semantic embeddings",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Token chunk size (paper: 512)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild index even if it already exists",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    corpus_path = Path(args.corpus_path)
    index_path = Path(args.index_path)

    assert corpus_path.exists(), f"Corpus path not found: {corpus_path}"
    if index_path.exists() and not args.force:
        print(f"Index already exists at {index_path}. Use --force to rebuild.")
        return

    logger.info(f"Building index: corpus={corpus_path} → index={index_path}")
    logger.info(f"Embedding model: {args.embedding_model}, chunk_size={args.chunk_size}")

    builder = CCNewsIndexBuilder(
        corpus_path=str(corpus_path),
        index_path=str(index_path),
        embedding_model_name=args.embedding_model,
        chunk_size=args.chunk_size,
    )
    builder.build(force=args.force)
    logger.info("Index built successfully.")


if __name__ == "__main__":
    main()
