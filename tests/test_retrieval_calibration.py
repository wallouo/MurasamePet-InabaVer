import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from logic.knowledge import KnowledgeError, corpus_version
from tools.check_knowledge_search import (
    CaseScores,
    release_status,
    runtime_threshold,
    select_policy,
)


def _case(name, score, *, expected=False, second=0.0, language="en"):
    document = "expected" if expected else "wrong"
    return CaseScores(
        name=name,
        language=language,
        expected=frozenset({"expected"}) if expected else frozenset(),
        results=(
            {"document_id": document, "similarity": score},
            {"document_id": "other", "similarity": second},
        ),
    )


class CorpusVersionTests(unittest.TestCase):
    def test_version_is_stable_and_changes_with_path_or_content(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.md").write_text("a", encoding="utf-8")
            first = corpus_version(root)
            self.assertEqual(first, corpus_version(root))
            (root / "a.md").write_text("changed", encoding="utf-8")
            self.assertNotEqual(first, corpus_version(root))

    def test_generated_and_evaluation_files_are_excluded(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fact.md").write_text("fact", encoding="utf-8")
            first = corpus_version(root)
            (root / "evaluation").mkdir()
            (root / "evaluation" / "cases.json").write_text("{}", encoding="utf-8")
            self.assertEqual(first, corpus_version(root))


class CalibrationTests(unittest.TestCase):
    def test_threshold_only_is_preferred_when_it_passes(self):
        selected = select_policy(
            [_case("positive", 0.82, expected=True), _case("negative", 0.71)]
        )
        self.assertIsNotNone(selected)
        self.assertIsNone(selected["margin"])

    def test_margin_is_used_only_when_scores_overlap_and_gaps_separate(self):
        selected = select_policy(
            [
                _case("positive", 0.80, expected=True, second=0.50),
                _case("negative", 0.81, second=0.80),
            ]
        )
        self.assertIsNotNone(selected)
        self.assertIsNotNone(selected["margin"])

    def test_failure_is_reported_when_no_policy_can_separate_cases(self):
        selected = select_policy(
            [
                _case("positive", 0.80, expected=True, second=0.70),
                _case("negative", 0.80, second=0.70),
            ]
        )
        self.assertIsNone(selected)

    def test_discovered_candidate_does_not_pass_until_configured(self):
        passed, reason = release_status(
            {"passed": False}, {"threshold": 0.82, "margin": None}
        )
        self.assertFalse(passed)
        self.assertEqual(reason, "calibration_update_required")

        passed, reason = release_status({"passed": True}, None)
        self.assertTrue(passed)
        self.assertEqual(reason, "ready")

    def test_release_threshold_matches_effective_runtime_environment(self):
        with patch.dict(os.environ, {"RAG_MIN_SIMILARITY": "0.82"}):
            self.assertEqual(runtime_threshold(None), 0.82)
            self.assertEqual(runtime_threshold(0.82), 0.82)
            with self.assertRaisesRegex(KnowledgeError, "baseline_threshold_mismatch"):
                runtime_threshold(0.70)


if __name__ == "__main__":
    unittest.main()
