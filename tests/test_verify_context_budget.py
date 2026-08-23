import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools import verify_context_budget


class VerifyContextBudgetCliTests(unittest.TestCase):
    def _run(self, status):
        manifest = {"validation": {"status": status}}
        with TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"

            def write_manifest(**kwargs):
                kwargs["manifest_path"].write_text("written", encoding="utf-8")
                return manifest

            with (
                patch.object(
                    verify_context_budget,
                    "build_manifest",
                    side_effect=write_manifest,
                ),
                patch(
                    "sys.argv",
                    [
                        "verify_context_budget.py",
                        "--manifest",
                        str(manifest_path),
                    ],
                ),
            ):
                result = verify_context_budget.main()
            self.assertTrue(manifest_path.is_file())
            return result

    def test_passed_manifest_exits_zero(self):
        self.assertEqual(self._run("passed"), 0)

    def test_failed_manifest_exits_nonzero(self):
        self.assertNotEqual(self._run("failed"), 0)

    def test_not_validated_manifest_exits_nonzero(self):
        self.assertNotEqual(self._run("not_validated"), 0)


if __name__ == "__main__":
    unittest.main()
