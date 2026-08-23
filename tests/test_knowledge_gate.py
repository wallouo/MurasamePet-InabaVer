import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from logic.knowledge import KnowledgeError
from logic.knowledge_gate import classify_query, load_inventory
from tools.prototype_knowledge_gate import (
    evaluate_gate,
    validate_heldout_separation,
)


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "data" / "knowledge" / "gate_inventory.json"
CORPUS_PATH = ROOT / "data" / "knowledge" / "examples"


class KnowledgeGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = load_inventory(INVENTORY_PATH, CORPUS_PATH, repo_root=ROOT)

    def test_supported_identity_and_song_queries_are_allowed(self):
        cases = {
            "Who is Meguru Inaba?": "identity_profile",
            "因幡めぐる / Meguru Inaba": "identity_profile",
            "因幡めぐるのキャラクターソングは何？": "character_song",
            "因幡めぐる的角色歌是什么？": "character_song",
        }
        for query, fact_key in cases.items():
            with self.subTest(query=query):
                result = classify_query(query, self.inventory)
                self.assertEqual(result.decision, "allow_retrieval")
                self.assertEqual(result.fact_key, fact_key)
                self.assertEqual(result.entities, ("meguru",))

    def test_unknown_domain_unsupported_fact_and_ambiguity_are_distinct(self):
        cases = {
            "今天倫敦的天氣如何？": "abstain_unknown_domain",
            "因幡めぐる的生日是幾月幾日？": "abstain_unsupported_fact",
            "她最喜歡什麼？": "abstain_ambiguous_query",
            "Who is Nene Ayachi in Sanoba Witch?": "abstain_unknown_entity",
            "What does Arca mean in Sanoba Witch?": "abstain_unsupported_fact",
        }
        for query, decision in cases.items():
            with self.subTest(query=query):
                self.assertEqual(
                    classify_query(query, self.inventory).decision, decision
                )

    def test_tags_without_textual_evidence_do_not_create_supported_facts(self):
        for query, expected_fact in (
            ("What is Meguru Inaba's relationship with the protagonist?", "relationship"),
            ("因幡めぐるルートでは何が起きる？", "route_event"),
        ):
            with self.subTest(query=query):
                result = classify_query(query, self.inventory)
                self.assertEqual(result.fact_key, expected_fact)
                self.assertEqual(result.decision, "abstain_unsupported_fact")

    def test_domain_support_requires_a_current_evidence_document(self):
        inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        inventory["domains"][0]["supported_facts"] = {
            "setting_terminology": {
                "evidence": [
                    {"document_id": "missing-document", "excerpt": "missing"}
                ]
            }
        }
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "inventory.json"
            path.write_text(json.dumps(inventory), encoding="utf-8")
            with self.assertRaisesRegex(KnowledgeError, "document_missing"):
                load_inventory(path, CORPUS_PATH, repo_root=ROOT)

    def test_supported_fact_excerpt_must_exist_in_the_evidence_document(self):
        inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        inventory["entities"][0]["supported_facts"]["identity_profile"][
            "evidence"
        ][0]["excerpt"] = "text that is not in the source document"
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "inventory.json"
            path.write_text(json.dumps(inventory), encoding="utf-8")
            with self.assertRaisesRegex(KnowledgeError, "evidence_excerpt_invalid"):
                load_inventory(path, CORPUS_PATH, repo_root=ROOT)

    def test_inventory_references_require_known_domains_and_entities(self):
        mutations = (
            (
                lambda inventory: inventory["entities"][0].update(
                    {"domain_id": "missing-domain"}
                ),
                "domain_missing",
            ),
            (
                lambda inventory: inventory["relationships"].append(
                    {
                        "entity_ids": ["meguru", "missing-character"],
                        "supported_facts": {},
                    }
                ),
                "relationship_entity_missing",
            ),
        )
        for mutate, error in mutations:
            with self.subTest(error=error), TemporaryDirectory() as temporary:
                inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
                mutate(inventory)
                path = Path(temporary) / "inventory.json"
                path.write_text(json.dumps(inventory), encoding="utf-8")
                with self.assertRaisesRegex(KnowledgeError, error):
                    load_inventory(path, CORPUS_PATH, repo_root=ROOT)

    def test_complete_fixture_meets_gate_acceptance_bar(self):
        existing = json.loads(
            (ROOT / "data" / "knowledge" / "evaluation" / "retrieval_cases.json")
            .read_text(encoding="utf-8")
        )["cases"]
        additional = json.loads(
            (ROOT / "data" / "knowledge" / "evaluation" / "gate_cases.json")
            .read_text(encoding="utf-8")
        )["cases"]
        heldout = json.loads(
            (ROOT / "data" / "knowledge" / "evaluation" / "gate_heldout_cases.json")
            .read_text(encoding="utf-8")
        )["cases"]

        report = evaluate_gate(existing, additional, self.inventory, heldout)

        self.assertEqual(report["existing_positive_recall"], 1.0)
        self.assertEqual(report["existing_positive_count"], 4)
        self.assertEqual(report["existing_hard_negative_count"], 12)
        self.assertEqual(report["existing_hard_negative_acceptance_count"], 0)
        self.assertEqual(report["suite_metrics"]["heldout"]["total"], 24)
        self.assertEqual(report["suite_metrics"]["heldout"]["passed"], 24)
        self.assertGreaterEqual(report["heldout_separation"]["structural_cases"], 8)
        self.assertTrue(report["all_expectations_passed"])

    def test_heldout_fixture_rejects_development_duplicates(self):
        existing = []
        development = [
            {"query": "Who is Meguru Inaba?", "language": "en"}
        ]
        heldout = [
            {"query": "Who is Meguru Inaba?", "language": "en"}
        ]
        with self.assertRaisesRegex(KnowledgeError, "query_duplicate"):
            validate_heldout_separation(
                existing, development, heldout, self.inventory
            )


if __name__ == "__main__":
    unittest.main()
