import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from logic.knowledge import IndexSnapshot, corpus_version
from logic.knowledge_gate import (
    GateRuntimeConfig,
    evaluate_gate_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "knowledge" / "examples"
INVENTORY = ROOT / "data" / "knowledge" / "gate_inventory.json"


class KnowledgeGateReadinessTests(unittest.TestCase):
    def _config(self, **overrides):
        values = {
            "inventory_path": INVENTORY,
            "corpus_path": CORPUS,
            "repo_root": ROOT,
            "chroma_path": ROOT / "data" / "knowledge" / "chroma",
            "collection_name": "meguru_knowledge",
            "embedding_model_id": "intfloat_multilingual_e5_small",
            "embedding_dimension": 384,
            "chunk_schema_version": 1,
            "chunk_target_chars": 700,
            "chunk_overlap_chars": 80,
        }
        values.update(overrides)
        return GateRuntimeConfig(**values)

    def _snapshot(self, **metadata):
        values = {
            "hnsw:space": "cosine",
            "index_state": "ready",
            "corpus_version": corpus_version(CORPUS),
            "embedding_model_id": "intfloat_multilingual_e5_small",
            "embedding_dimension": 384,
            "chunk_schema_version": 1,
            "chunk_target_chars": 700,
            "chunk_overlap_chars": 80,
        }
        values.update(metadata)
        return IndexSnapshot(
            metadata=values,
            document_ids=frozenset(
                {
                    "sanoba-witch-meguru-official",
                    "sanoba-witch-meguru-localization",
                    "sanoba-witch-meguru-song-official",
                }
            ),
        )

    def test_matching_inventory_corpus_and_index_are_ready(self):
        readiness = evaluate_gate_readiness(
            self._config(), index_reader=lambda *_: self._snapshot()
        )
        self.assertTrue(readiness.ready)
        self.assertEqual(readiness.reason, "ready")
        self.assertIsNotNone(readiness.gate)

    def test_unavailable_index_fails_closed_with_a_stable_code(self):
        def unavailable(*_args):
            raise RuntimeError("C:/private/index/chroma.sqlite3 is unavailable")

        readiness = evaluate_gate_readiness(
            self._config(), index_reader=unavailable
        )
        self.assertFalse(readiness.ready)
        self.assertEqual(readiness.reason, "gate_index_unavailable")

    def test_missing_or_stale_collection_corpus_fails_closed(self):
        cases = (
            ({"corpus_version": None}, "missing_version"),
            ({"corpus_version": "stale"}, "stale_version"),
            ({"index_state": "unverified"}, "unverified_state"),
        )
        for mutation, name in cases:
            with self.subTest(name=name):
                snapshot = self._snapshot()
                metadata = dict(snapshot.metadata)
                if mutation.get("corpus_version", object()) is None:
                    metadata.pop("corpus_version")
                else:
                    metadata.update(mutation)
                readiness = evaluate_gate_readiness(
                    self._config(),
                    index_reader=lambda *_args, metadata=metadata: IndexSnapshot(
                        metadata=metadata,
                        document_ids=snapshot.document_ids,
                    ),
                )
                self.assertFalse(readiness.ready)
                self.assertEqual(readiness.reason, "gate_index_corpus_mismatch")

    def test_changed_evidence_excerpt_has_a_stable_readiness_code(self):
        inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
        inventory["entities"][0]["supported_facts"]["identity_profile"][
            "evidence"
        ][0]["excerpt"] = "stale evidence that is absent from the corpus"
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "inventory.json"
            path.write_text(json.dumps(inventory), encoding="utf-8")
            readiness = evaluate_gate_readiness(
                self._config(inventory_path=path),
                index_reader=lambda *_: self._snapshot(),
            )
        self.assertFalse(readiness.ready)
        self.assertEqual(readiness.reason, "gate_evidence_invalid")

    def test_inventory_is_bound_to_the_exact_configured_corpus_path(self):
        with TemporaryDirectory() as temporary:
            other = Path(temporary) / "private"
            other.mkdir()
            readiness = evaluate_gate_readiness(
                self._config(corpus_path=other),
                index_reader=lambda *_: self._snapshot(),
            )
        self.assertFalse(readiness.ready)
        self.assertEqual(readiness.reason, "gate_corpus_path_mismatch")

    def test_embedding_schema_and_index_documents_are_checked(self):
        cases = (
            ({"embedding_dimension": 999}, "gate_index_embedding_mismatch"),
            ({"chunk_schema_version": 2}, "gate_index_schema_mismatch"),
        )
        for metadata, reason in cases:
            with self.subTest(reason=reason):
                readiness = evaluate_gate_readiness(
                    self._config(),
                    index_reader=lambda *_args, metadata=metadata: self._snapshot(
                        **metadata
                    ),
                )
                self.assertEqual(readiness.reason, reason)

        snapshot = self._snapshot()
        readiness = evaluate_gate_readiness(
            self._config(),
            index_reader=lambda *_: IndexSnapshot(
                metadata=snapshot.metadata,
                document_ids=frozenset({"sanoba-witch-meguru-official"}),
            ),
        )
        self.assertEqual(readiness.reason, "gate_index_document_missing")


if __name__ == "__main__":
    unittest.main()
