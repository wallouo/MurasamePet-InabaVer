"""Build the local Meguru knowledge index from Markdown/JSON files."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logic.knowledge import KnowledgeError, KnowledgeStore, ingest_paths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index curated Markdown/JSON knowledge into persistent Chroma."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="knowledge files or directories",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path(os.getenv("RAG_CHROMA_PATH", "data/knowledge/chroma")),
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("RAG_COLLECTION", "meguru_knowledge"),
    )
    parser.add_argument(
        "--embedding-model-path",
        type=Path,
        default=Path(
            os.getenv(
                "RAG_EMBEDDING_MODEL_PATH",
                "models/embeddings/multilingual-e5-small",
            )
        ),
    )
    parser.add_argument(
        "--target-chars",
        type=int,
        default=int(os.getenv("RAG_CHUNK_TARGET_CHARS", "700")),
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=int(os.getenv("RAG_CHUNK_OVERLAP_CHARS", "80")),
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="delete the existing collection before indexing",
    )
    args = parser.parse_args()
    try:
        store = KnowledgeStore(
            persist_path=args.chroma_path,
            collection_name=args.collection,
            embedding_model_path=args.embedding_model_path,
        )
        if args.replace:
            store.reset()
        result = ingest_paths(
            args.paths,
            store,
            target_chars=args.target_chars,
            overlap_chars=args.overlap_chars,
        )
    except (KnowledgeError, OSError) as exc:
        parser.error(str(exc))
    print(
        "Indexed {documents} documents / {chunks} chunks; collection now has "
        "{indexed} chunks.".format(**result)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
