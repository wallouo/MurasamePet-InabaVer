import json
import gc
import tempfile
import unittest
from pathlib import Path

from logic.knowledge import (
    KnowledgeError,
    KnowledgeDocument,
    KnowledgeChunk,
    chunk_document,
    load_documents,
)

try:
    import chromadb  # noqa: F401
except ImportError:
    CHROMA_AVAILABLE = False
else:
    CHROMA_AVAILABLE = True


class KnowledgeTests(unittest.TestCase):
    def test_markdown_frontmatter_and_metadata_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "meguru.md"
            path.write_text(
                "---\n"
                "id: meguru-profile\n"
                "title: Meguru profile\n"
                "source_url: https://example.test/meguru\n"
                "route_scope: meguru\n"
                "perspective_status: lived\n"
                "source_authority: curated\n"
                "language: ja\n"
                "character_tags: [meguru, protagonist] \n"
                "relationship_tags: [senpai]\n"
                "---\n\n"
                "めぐるはゲームが好き。\n\n"
                "センパイとは同じ学園に通っている。",
                encoding="utf-8",
            )

            [document] = load_documents(path)
            chunks = chunk_document(document, target_chars=100, overlap_chars=10)

        self.assertEqual(document.document_id, "meguru-profile")
        self.assertEqual(document.source_url, "https://example.test/meguru")
        self.assertEqual(document.character_tags, ("meguru", "protagonist"))
        self.assertEqual(document.relationship_tags, ("senpai",))
        self.assertEqual(chunks[0].metadata["route_scope"], "meguru")
        self.assertEqual(chunks[0].metadata["source_authority"], "curated")

    def test_overlap_never_makes_chunks_exceed_target(self):
        source = KnowledgeDocument(
            document_id="long",
            title="long",
            text=("a" * 95) + "\n\n" + ("b" * 95),
            source_path="memory",
            source_url="",
            document_format="json",
            source_authority="curated",
            route_scope="meguru",
            perspective_status="universal",
            language="en",
        )
        chunks = chunk_document(source, target_chars=100, overlap_chars=80)

        self.assertTrue(all(len(chunk.text) <= 100 for chunk in chunks))

    def test_e5_embedder_separates_passage_and_query_prefixes(self):
        from logic.knowledge import _E5EmbeddingFunction

        class RecordingModel:
            def __init__(self):
                self.inputs = []

            def encode(self, inputs, **kwargs):
                self.inputs.append((list(inputs), kwargs))
                return [[1.0, 0.0] for _ in inputs]

        model = RecordingModel()
        embedding = _E5EmbeddingFunction(model, model_path="models/embeddings/multilingual-e5-small")

        embedding(["fact"])
        embedding.embed_query(["question"])

        self.assertEqual(model.inputs[0][0], ["passage: fact"])
        self.assertEqual(model.inputs[1][0], ["query: question"])
        self.assertEqual(embedding.get_config()["model_path"], "models/embeddings/multilingual-e5-small")

    @unittest.skipUnless(CHROMA_AVAILABLE, "chromadb is not installed")
    def test_search_omits_weak_nearest_neighbours(self):
        from logic.knowledge import KnowledgeStore

        class VectorEmbedding:
            def __call__(self, input):
                return [[1.0, 0.0] if "match" in text else [0.0, 1.0] for text in input]

            @staticmethod
            def name():
                return "test_vector_embedding"

            @staticmethod
            def build_from_config(config):
                return VectorEmbedding()

            def is_legacy(self):
                return False

            def supported_spaces(self):
                return ["cosine"]

            def get_config(self):
                return {"name": self.name(), "space": "cosine"}

            def embed_query(self, input):
                return [
                    [1.0, 0.0]
                    if text == "match"
                    else ([0.5, 0.5] if text == "unrelated" else [0.0, 1.0])
                    for text in input
                ]

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            store = KnowledgeStore(
                persist_path=Path(temporary) / "chroma",
                collection_name="search_test",
                embedding_model_path="unused-for-test",
                embedding_function=VectorEmbedding(),
            )
            store.upsert(
                [
                    KnowledgeChunk(
                        chunk_id="match",
                        text="match fact",
                        metadata={"document_id": "match-doc"},
                    ),
                    KnowledgeChunk(
                        chunk_id="other",
                        text="other fact",
                        metadata={"document_id": "other-doc"},
                    ),
                ]
            )

            matches = store.search("match", min_similarity=0.70)
            unrelated = store.search("unrelated", min_similarity=0.80)

        self.assertEqual([item.metadata["document_id"] for item in matches], ["match-doc"])
        self.assertEqual(unrelated, [])

    def test_json_list_is_supported_and_chunk_ids_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "facts.json"
            payload = [
                {
                    "id": "fact-a",
                    "content": "A short fact.",
                    "source_url": "https://example.test/fact-a",
                    "route_scope": "global",
                    "perspective_status": "universal",
                    "source_authority": "official",
                    "language": "en",
                },
                {
                    "id": "fact-b",
                    "text": "別の短い事実。",
                    "route_scope": "common",
                    "perspective_status": "alternate",
                    "source_authority": "official_localization",
                    "language": "ja",
                },
            ]
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            documents = load_documents(path)

        first = chunk_document(documents[0], target_chars=100)
        second = chunk_document(documents[0], target_chars=100)
        self.assertEqual([item.document_id for item in documents], ["fact-a", "fact-b"])
        self.assertEqual(first[0].chunk_id, second[0].chunk_id)

    def test_invalid_authority_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "invalid",
                        "text": "fact",
                        "route_scope": "meguru",
                        "perspective_status": "lived",
                        "source_authority": "markdown",
                        "language": "ja",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(KnowledgeError):
                load_documents(path)

    @unittest.skipUnless(CHROMA_AVAILABLE, "chromadb is not installed")
    def test_chroma_persists_indexed_chunks(self):
        from logic.knowledge import KnowledgeStore, ingest_paths

        class HashEmbedding:
            def __call__(self, input):
                return [
                    [float(len(text)), float(sum(map(ord, text)) % 997)]
                    for text in input
                ]

            @staticmethod
            def name():
                return "test_hash_embedding"

            @staticmethod
            def build_from_config(config):
                return HashEmbedding()

            def is_legacy(self):
                return False

            def supported_spaces(self):
                return ["cosine"]

            def get_config(self):
                return {"name": self.name(), "space": "cosine"}

            def embed_query(self, input):
                return self(input)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            store = KnowledgeStore(
                persist_path=Path(temporary) / "chroma",
                collection_name="test_knowledge",
                embedding_model_path="unused-for-test",
                embedding_function=HashEmbedding(),
            )
            result = ingest_paths(
                [Path("data/knowledge/examples")],
                store,
                target_chars=180,
                overlap_chars=20,
            )
            reopened = KnowledgeStore(
                persist_path=Path(temporary) / "chroma",
                collection_name="test_knowledge",
                embedding_model_path="unused-for-test",
                embedding_function=HashEmbedding(),
            )
            count = reopened.count()
            del reopened, store
            gc.collect()

        self.assertGreater(result["chunks"], 0)
        self.assertEqual(count, result["indexed"])


if __name__ == "__main__":
    unittest.main()
