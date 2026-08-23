"""Corpus-versioned multilingual retrieval calibration using the local E5 index."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logic.context_manifest import sha256_file  # noqa: E402
from logic.environment import load_project_env  # noqa: E402
from logic.knowledge import (  # noqa: E402
    KnowledgeError,
    KnowledgeStore,
    iter_knowledge_files,
)

load_project_env(ROOT / "api.py")


@dataclass(frozen=True)
class CaseScores:
    name: str
    language: str
    expected: frozenset[str]
    results: tuple[dict[str, Any], ...]

    @property
    def top_score(self) -> float:
        return float(self.results[0]["similarity"]) if self.results else 0.0

    @property
    def margin(self) -> float:
        if len(self.results) < 2:
            return 1.0 if self.results else 0.0
        return self.top_score - float(self.results[1]["similarity"])


def corpus_version(corpus_path: Path) -> str:
    """Hash sorted relative paths and source hashes, excluding generated state."""

    files = [
        path
        for path in iter_knowledge_files([corpus_path])
        if not {"chroma", "evaluation"}.intersection(
            part.lower() for part in path.relative_to(corpus_path).parts
        )
    ]
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(corpus_path).as_posix()):
        relative = path.relative_to(corpus_path).as_posix()
        source_hash = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def evaluate_policy(
    cases: Sequence[CaseScores], threshold: float, margin: float | None = None
) -> dict[str, Any]:
    false_positives: dict[str, list[str]] = {}
    positive_passed = 0
    positive_total = 0
    case_results: list[dict[str, Any]] = []
    for case in cases:
        accepted = bool(case.results) and case.top_score >= threshold
        if margin is not None:
            accepted = accepted and case.margin >= margin
        top_document = (
            str(case.results[0].get("document_id")) if case.results else None
        )
        if case.expected:
            positive_total += 1
            passed = accepted and top_document in case.expected
            positive_passed += int(passed)
        else:
            passed = not accepted
            if not passed:
                false_positives.setdefault(case.language, []).append(case.name)
        case_results.append(
            {"name": case.name, "accepted": accepted, "passed": passed}
        )
    return {
        "threshold": threshold,
        "margin": margin,
        "positive_recall": (
            positive_passed / positive_total if positive_total else 1.0
        ),
        "positive_passed": positive_passed,
        "positive_total": positive_total,
        "false_positive_count": sum(map(len, false_positives.values())),
        "false_positives_by_language": false_positives,
        "cases": case_results,
        "passed": positive_passed == positive_total and not false_positives,
    }


def select_policy(cases: Sequence[CaseScores]) -> dict[str, Any] | None:
    scores = {0.0}
    for case in cases:
        scores.add(case.top_score)
        scores.add(math.nextafter(case.top_score, math.inf))
    thresholds = sorted(scores)
    for threshold in thresholds:
        result = evaluate_policy(cases, threshold)
        if result["passed"]:
            return result

    margins = {0.0}
    for case in cases:
        margins.add(case.margin)
        margins.add(math.nextafter(case.margin, math.inf))
    for threshold in thresholds:
        for margin in sorted(margins):
            result = evaluate_policy(cases, threshold, margin)
            if result["passed"]:
                return result
    return None


def release_status(
    baseline: Mapping[str, Any], selected: Mapping[str, Any] | None
) -> tuple[bool, str]:
    """Gate on the configured runtime policy, not an unapplied candidate."""

    if baseline.get("passed") is True:
        return True, "ready"
    if selected is not None:
        return False, "calibration_update_required"
    return False, "calibration_failed"


def runtime_threshold(cli_value: float | None) -> float:
    """Resolve the same dotenv-backed threshold used by the backend."""

    configured = float(os.getenv("RAG_MIN_SIMILARITY", "0.70"))
    if cli_value is not None and not math.isclose(
        cli_value, configured, rel_tol=0.0, abs_tol=1e-12
    ):
        raise KnowledgeError("baseline_threshold_mismatch")
    return configured


def _raw_case_scores(store: KnowledgeStore, case: Mapping[str, Any]) -> CaseScores:
    results = store.search(str(case["query"]), min_similarity=0.0, limit=3)
    return CaseScores(
        name=str(case["name"]),
        language=str(case["language"]),
        expected=frozenset(map(str, case.get("expected_document_ids", []))),
        results=tuple(
            {
                "document_id": result.metadata.get("document_id"),
                "chunk_id": result.chunk_id,
                "language": result.metadata.get("language"),
                "similarity": result.similarity,
            }
            for result in results
        ),
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/knowledge/evaluation/retrieval_cases.json"),
    )
    parser.add_argument(
        "--corpus-path", type=Path, default=Path("data/knowledge/examples")
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path(os.getenv("RAG_CHROMA_PATH", "data/knowledge/chroma")),
    )
    parser.add_argument(
        "--collection", default=os.getenv("RAG_COLLECTION", "meguru_knowledge")
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
        "--baseline-threshold",
        type=float,
        default=None,
    )
    args = parser.parse_args()

    try:
        baseline_threshold = runtime_threshold(args.baseline_threshold)
        fixture = json.loads(args.cases.read_text(encoding="utf-8"))
        actual_version = corpus_version(args.corpus_path)
        if fixture.get("corpus_version") != actual_version:
            print(
                json.dumps(
                    {
                        "passed": False,
                        "reason": "corpus_version_mismatch",
                        "recorded_corpus_version": fixture.get("corpus_version"),
                        "actual_corpus_version": actual_version,
                    },
                    indent=2,
                )
            )
            return 1
        store = KnowledgeStore(
            persist_path=args.chroma_path,
            collection_name=args.collection,
            embedding_model_path=args.embedding_model_path,
        )
        cases = tuple(_raw_case_scores(store, case) for case in fixture["cases"])
    except (KeyError, json.JSONDecodeError, KnowledgeError, OSError) as exc:
        parser.error(str(exc))

    baseline = evaluate_policy(cases, baseline_threshold)
    selected = select_policy(cases)
    passed, reason = release_status(baseline, selected)
    output = {
        "corpus_version": actual_version,
        "baseline": baseline,
        "selected": selected,
        "reason": reason,
        "case_scores": [
            {
                "name": case.name,
                "language": case.language,
                "expected_document_ids": sorted(case.expected),
                "top_score": case.top_score,
                "score_margin": case.margin,
                "results": list(case.results),
            }
            for case in cases
        ],
        # Candidate discovery is advisory. Release verification succeeds only
        # when the configured runtime threshold itself passes this corpus.
        "passed": passed,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
