"""Run real multilingual semantic-search checks against the local Chroma index."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logic.knowledge import KnowledgeError, KnowledgeStore  # noqa: E402


CASES = (
    {
        "name": "japanese_question",
        "query": "因幡めぐるは公式サイトでどのように紹介されていますか？",
        "expected": {"sanoba-witch-meguru-official"},
    },
    {
        "name": "traditional_chinese_question",
        "query": "《魔女的夜宴》中因幡めぐる是谁？",
        "expected": {
            "sanoba-witch-meguru-official",
            "sanoba-witch-meguru-localization",
        },
    },
    {
        "name": "english_question",
        "query": "Who is Meguru Inaba?",
        "expected": {
            "sanoba-witch-meguru-official",
            "sanoba-witch-meguru-localization",
        },
    },
    {
        "name": "character_aliases",
        "query": "因幡めぐる / Meguru Inaba",
        "expected": {
            "sanoba-witch-meguru-official",
            "sanoba-witch-meguru-localization",
        },
    },
    {
        "name": "unrelated_question",
        "query": "Who won the 2018 World Cup?",
        "expected": set(),
    },
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--min-similarity", type=float, default=0.70)
    args = parser.parse_args()

    try:
        store = KnowledgeStore(
            persist_path=args.chroma_path,
            collection_name=args.collection,
            embedding_model_path=args.embedding_model_path,
        )
        checks = []
        for case in CASES:
            results = store.search(
                case["query"], min_similarity=args.min_similarity, limit=3
            )
            ids = [str(result.metadata.get("document_id")) for result in results]
            expected = case["expected"]
            passed = (not expected and not ids) or (
                bool(expected) and bool(ids) and ids[0] in expected
            )
            checks.append(
                {
                    "name": case["name"],
                    "query": case["query"],
                    "passed": passed,
                    "results": [
                        {
                            "document_id": result.metadata.get("document_id"),
                            "language": result.metadata.get("language"),
                            "similarity": round(result.similarity, 4),
                            "source_url": result.metadata.get("source_url"),
                        }
                        for result in results
                    ],
                }
            )
    except (KnowledgeError, OSError) as exc:
        parser.error(str(exc))

    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if all(check["passed"] for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
