"""Evaluate the shared corpus gate; this fixture/report code is not runtime."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logic.knowledge import KnowledgeError  # noqa: E402
from logic.knowledge_gate import (  # noqa: E402
    DECISIONS,
    classify_query,
    contains_recognizer_phrase,
    load_inventory,
)


EXPECTED_EXISTING_POSITIVE_COUNT = 4
EXPECTED_EXISTING_HARD_NEGATIVE_COUNT = 12


def _inventory_report(inventory: Mapping[str, Any]) -> dict[str, Any]:
    documents = inventory["_documents"]
    entities = []
    for entity in inventory["entities"]:
        facts = {}
        for fact_key, support in entity.get("supported_facts", {}).items():
            ids = [str(item["document_id"]) for item in support["evidence"]]
            facts[fact_key] = {
                "evidence": support["evidence"],
                "route_scopes": sorted({documents[item]["route_scope"] for item in ids}),
            }
        entities.append({**entity, "supported_facts": facts})
    return {
        "metadata_schema": inventory["metadata_schema"],
        "entities": entities,
        "domains": inventory["domains"],
        "relationships": inventory.get("relationships", []),
        "maintenance": inventory["maintenance"],
    }


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def validate_heldout_separation(
    existing: Sequence[Mapping[str, Any]],
    development: Sequence[Mapping[str, Any]],
    heldout: Sequence[Mapping[str, Any]],
    inventory: Mapping[str, Any],
) -> dict[str, int]:
    prior = {_normalized(str(case["query"])) for case in (*existing, *development)}
    bare = {
        _normalized(str(value))
        for owner in (*inventory["entities"], *inventory["domains"])
        for value in owner["aliases"]
    } | {
        _normalized(str(phrase))
        for recognizer in inventory["fact_recognizers"]
        for phrase in recognizer["phrases"]
    }
    seen: set[str] = set()
    structural_by_language: dict[str, int] = {}
    for case in heldout:
        query = str(case["query"])
        normalized = _normalized(query)
        if normalized in prior or normalized in seen:
            raise KnowledgeError("gate_heldout_query_duplicate")
        if normalized in bare:
            raise KnowledgeError("gate_heldout_query_is_bare_phrase")
        if not any(mark in query for mark in "?？。.!！"):
            raise KnowledgeError("gate_heldout_query_not_natural_utterance")
        seen.add(normalized)
        if not contains_recognizer_phrase(query, inventory):
            language = str(case["language"])
            structural_by_language[language] = structural_by_language.get(language, 0) + 1
    languages = {str(case["language"]) for case in heldout}
    if sum(structural_by_language.values()) < 8 or any(
        structural_by_language.get(language, 0) < 1 for language in languages
    ):
        raise KnowledgeError("gate_heldout_structural_coverage_missing")
    return {
        "cases": len(seen),
        "structural_cases": sum(structural_by_language.values()),
    }


def evaluate_gate(
    existing_cases: Sequence[Mapping[str, Any]],
    additional_cases: Sequence[Mapping[str, Any]],
    inventory: Mapping[str, Any],
    heldout_cases: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    separation = (
        validate_heldout_separation(
            existing_cases, additional_cases, heldout_cases, inventory
        )
        if heldout_cases
        else {"cases": 0, "structural_cases": 0}
    )
    results = []
    suite_metrics: dict[str, dict[str, int]] = {}
    language_results: dict[str, dict[str, int]] = {}
    existing_positive_total = existing_positive_passed = 0
    existing_negative_total = existing_negative_accepted = 0
    suites = (
        ("existing", existing_cases),
        ("development", additional_cases),
        ("heldout", heldout_cases),
    )
    for source, cases in suites:
        suite = suite_metrics.setdefault(
            source, {"total": 0, "passed": 0, "allowed": 0, "abstained": 0}
        )
        for case in cases:
            decision = classify_query(str(case["query"]), inventory)
            expected_decision = str(case["expected_gate_decision"])
            if expected_decision not in DECISIONS:
                raise KnowledgeError("gate_case_decision_invalid")
            expected_fact = case.get("expected_fact_key")
            expected_entities = tuple(map(str, case["expected_entities"]))
            expected_domains = tuple(map(str, case["expected_domains"]))
            passed = (
                decision.decision == expected_decision
                and decision.fact_key == expected_fact
                and decision.entities == expected_entities
                and decision.domains == expected_domains
            )
            language = str(case["language"])
            language_bucket = language_results.setdefault(
                language, {"total": 0, "passed": 0, "allowed": 0, "abstained": 0}
            )
            for bucket in (suite, language_bucket):
                bucket["total"] += 1
                bucket["passed"] += int(passed)
                bucket[
                    "allowed" if decision.decision == "allow_retrieval" else "abstained"
                ] += 1
            if source == "existing":
                if case.get("expected_document_ids"):
                    existing_positive_total += 1
                    existing_positive_passed += int(passed)
                else:
                    existing_negative_total += 1
                    existing_negative_accepted += int(
                        decision.decision == "allow_retrieval"
                    )
            results.append(
                {
                    "source": source,
                    "name": case["name"],
                    "query": case["query"],
                    "language": language,
                    "expected_decision": expected_decision,
                    "expected_fact_key": expected_fact,
                    "expected_entities": expected_entities,
                    "expected_domains": expected_domains,
                    **asdict(decision),
                    "passed": passed,
                }
            )
    all_passed = all(item["passed"] for item in results)
    acceptance_bar = (
        existing_positive_total == EXPECTED_EXISTING_POSITIVE_COUNT
        and existing_negative_total == EXPECTED_EXISTING_HARD_NEGATIVE_COUNT
        and existing_positive_passed == existing_positive_total
        and existing_negative_accepted == 0
        and suite_metrics["heldout"]["passed"] == suite_metrics["heldout"]["total"]
    )
    return {
        "schema_version": 1,
        "corpus_version": inventory["corpus_version"],
        "inventory": _inventory_report(inventory),
        "heldout_separation": separation,
        "existing_positive_recall": existing_positive_passed / existing_positive_total,
        "existing_positive_count": existing_positive_total,
        "existing_hard_negative_count": existing_negative_total,
        "existing_hard_negative_acceptance_count": existing_negative_accepted,
        "suite_metrics": suite_metrics,
        "per_language": language_results,
        "cannot_confidently_classify": [
            item["name"]
            for item in results
            if item["decision"] in {"abstain_ambiguous_query", "abstain_unknown_entity"}
        ],
        "all_expectations_passed": all_passed,
        "acceptance_bar_passed": acceptance_bar,
        "recommendation": "A" if all_passed and acceptance_bar else "B",
        "cases": results,
    }


def _read_cases(path: Path) -> list[Mapping[str, Any]]:
    return list(json.loads(path.read_text(encoding="utf-8"))["cases"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=ROOT / "data/knowledge/gate_inventory.json")
    parser.add_argument("--corpus-path", type=Path, default=ROOT / "data/knowledge/examples")
    parser.add_argument("--existing-cases", type=Path, default=ROOT / "data/knowledge/evaluation/retrieval_cases.json")
    parser.add_argument("--additional-cases", type=Path, default=ROOT / "data/knowledge/evaluation/gate_cases.json")
    parser.add_argument("--heldout-cases", type=Path, default=ROOT / "data/knowledge/evaluation/gate_heldout_cases.json")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        inventory = load_inventory(args.inventory, args.corpus_path, repo_root=ROOT)
        report = evaluate_gate(
            _read_cases(args.existing_cases),
            _read_cases(args.additional_cases),
            inventory,
            _read_cases(args.heldout_cases),
        )
    except (KeyError, OSError, json.JSONDecodeError, KnowledgeError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    passed = report["all_expectations_passed"] and report["acceptance_bar_passed"]
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
        print(f"gate_evaluation={'passed' if passed else 'failed'}")
    else:
        sys.stdout.buffer.write(rendered.encode("utf-8"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
