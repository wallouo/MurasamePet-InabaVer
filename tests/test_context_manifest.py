import copy
import json
import tempfile
import unittest
from pathlib import Path

from logic.context_manifest import (
    canonical_ollama_profile,
    evaluate_rag_readiness,
    sha256_file,
)


class ContextManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.gguf = self.root / "meguru.gguf"
        self.artifact = self.root / "ollama-blob"
        self.gguf.write_bytes(b"same-model")
        self.artifact.write_bytes(b"same-model")
        self.show = {
            "modelfile": f'FROM "{self.artifact}"',
            "template": "template /no_think",
            "system": "system",
            "parameters": (
                'temperature 0.8\nnum_ctx 4096\nnum_predict 300\n'
                'stop "<|im_end|>"\nstop "user\\\\n"'
            ),
        }
        self.manifest_path = self.root / "manifest.json"
        self.manifest = {
            "schema_version": 2,
            "gguf": {"file": {"sha256": sha256_file(self.gguf)}},
            "ollama": {
                "validation_profile": canonical_ollama_profile(
                    self.show, model="meguru"
                )
            },
            "counter": {"mode": "gguf_native"},
            "validation": {"status": "passed"},
        }
        self._write_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def _readiness(self, **overrides):
        values = {
            "requested": True,
            "manifest_path": self.manifest_path,
            "gguf_path": self.gguf,
            "model": "meguru",
            "ollama_show": self.show,
            "tokenizer_mode": "gguf_native",
        }
        values.update(overrides)
        return evaluate_rag_readiness(**values)

    def test_matching_snapshot_is_ready(self):
        self.assertEqual(self._readiness().reason, "ready")
        self.assertTrue(self._readiness().ready)

    def test_disabled_configuration_short_circuits(self):
        readiness = self._readiness(requested=False)
        self.assertFalse(readiness.ready)
        self.assertEqual(readiness.reason, "disabled_by_config")

    def test_missing_invalid_and_old_manifests_are_rejected(self):
        missing = self._readiness(manifest_path=self.root / "missing.json")
        self.assertEqual(missing.reason, "manifest_missing")

        self.manifest_path.write_text("not json", encoding="utf-8")
        self.assertEqual(self._readiness().reason, "manifest_invalid")

        self.manifest["schema_version"] = 1
        self._write_manifest()
        self.assertEqual(self._readiness().reason, "manifest_invalid")

    def test_validation_and_tokenizer_must_pass(self):
        self.manifest["validation"]["status"] = "failed"
        self._write_manifest()
        self.assertEqual(self._readiness().reason, "validation_failed")

        self.manifest["validation"]["status"] = "passed"
        self._write_manifest()
        self.assertEqual(
            self._readiness(tokenizer_mode="utf8_upper_bound").reason,
            "tokenizer_mode_mismatch",
        )

    def test_each_canonical_profile_field_is_checked(self):
        cases = {
            "model": "ollama_model_mismatch",
            "artifact_sha256": "ollama_artifact_mismatch",
            "template_sha256": "template_mismatch",
            "system_sha256": "system_mismatch",
            "num_ctx": "runtime_parameters_mismatch",
            "num_predict": "runtime_parameters_mismatch",
            "stop_tokens": "stop_tokens_mismatch",
        }
        for field, reason in cases.items():
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.manifest)
                value = manifest["ollama"]["validation_profile"][field]
                manifest["ollama"]["validation_profile"][field] = (
                    ["different"] if isinstance(value, list) else "different"
                )
                if isinstance(value, int):
                    manifest["ollama"]["validation_profile"][field] = value + 1
                self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                self.assertEqual(self._readiness().reason, reason)

    def test_gguf_hash_is_checked(self):
        self.gguf.write_bytes(b"changed")
        self.assertEqual(self._readiness().reason, "gguf_mismatch")

    def test_stop_order_and_generation_tuning_are_ignored(self):
        changed = dict(self.show)
        changed["parameters"] = (
            'top_p 0.1\nstop "user\\\\n"\nnum_predict 300\n'
            'stop "<|im_end|>"\nnum_ctx 4096\ntemperature 0.2'
        )
        self.assertEqual(self._readiness(ollama_show=changed).reason, "ready")

    def test_missing_runtime_profile_disables_only_rag(self):
        readiness = self._readiness(ollama_show=None)
        self.assertFalse(readiness.ready)
        self.assertEqual(readiness.reason, "ollama_profile_unavailable")

    def test_missing_budget_parameters_never_validate_each_other(self):
        show = dict(self.show)
        show["parameters"] = 'temperature 0.8\nstop "<|im_end|>"'
        self.manifest["ollama"]["validation_profile"] = canonical_ollama_profile(
            show, model="meguru"
        )
        self._write_manifest()

        readiness = self._readiness(ollama_show=show)

        self.assertFalse(readiness.ready)
        self.assertEqual(readiness.reason, "runtime_parameters_mismatch")


if __name__ == "__main__":
    unittest.main()
