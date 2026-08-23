"""Corpus-aware query gate and fail-closed runtime evidence validation."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .knowledge import (
    IndexSnapshot,
    KnowledgeError,
    corpus_version,
    iter_knowledge_files,
    load_documents,
    read_index_snapshot,
)


DECISIONS = frozenset(
    {
        "allow_retrieval",
        "abstain_unknown_domain",
        "abstain_unknown_entity",
        "abstain_unsupported_fact",
        "abstain_ambiguous_query",
    }
)


@dataclass(frozen=True)
class GateDecision:
    entities: tuple[str, ...]
    domains: tuple[str, ...]
    fact_key: str | None
    decision: str
    rationale: str


@dataclass(frozen=True)
class GateRuntimeConfig:
    inventory_path: Path
    corpus_path: Path
    repo_root: Path
    chroma_path: Path
    collection_name: str
    embedding_model_id: str
    embedding_dimension: int
    chunk_schema_version: int
    chunk_target_chars: int
    chunk_overlap_chars: int


@dataclass(frozen=True)
class GateReadiness:
    ready: bool
    reason: str
    gate: Mapping[str, Any] | None = None


def _contains_phrase(text: str, phrase: str) -> bool:
    text = text.casefold()
    phrase = phrase.casefold()
    if phrase.isascii() and phrase[:1].isalnum() and phrase[-1:].isalnum():
        return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None
    return phrase in text


def _matching_ids(query: str, items: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        str(item["id"])
        for item in items
        if any(_contains_phrase(query, str(alias)) for alias in item["aliases"])
    )


def _infer_fact_key(query: str, inventory: Mapping[str, Any]) -> str | None:
    for recognizer in inventory["fact_recognizers"]:
        if any(
            _contains_phrase(query, str(phrase))
            for phrase in recognizer["phrases"]
        ):
            return str(recognizer["key"])
    return None


def contains_recognizer_phrase(query: str, inventory: Mapping[str, Any]) -> bool:
    return any(
        _contains_phrase(query, str(phrase))
        for recognizer in inventory["fact_recognizers"]
        for phrase in recognizer["phrases"]
    )


def _is_alias_only(query: str, inventory: Mapping[str, Any]) -> bool:
    remaining = query
    for item in (*inventory["entities"], *inventory["domains"]):
        for alias in sorted(item["aliases"], key=len, reverse=True):
            remaining = re.sub(re.escape(str(alias)), "", remaining, flags=re.IGNORECASE)
    return not re.sub(r"[\s/|,，、?？.!。・_'\"-]+", "", remaining)


def _canonical_path(value: str | Path, repo_root: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return os.path.normcase(str(path.resolve()))


def load_inventory(
    path: Path, corpus_path: Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Load and bind the reviewed inventory to one exact source corpus."""

    repo_root = (repo_root or Path.cwd()).resolve()
    try:
        inventory = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise KnowledgeError("gate_inventory_missing") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeError("gate_inventory_invalid") from exc
    if inventory.get("schema_version") != 1:
        raise KnowledgeError("gate_inventory_schema_invalid")
    if _canonical_path(str(inventory.get("corpus_path", "")), repo_root) != (
        _canonical_path(corpus_path, repo_root)
    ):
        raise KnowledgeError("gate_corpus_path_mismatch")
    active_version = corpus_version(corpus_path)
    if inventory.get("corpus_version") != active_version:
        raise KnowledgeError("gate_inventory_corpus_version_mismatch")

    documents = {}
    for source in iter_knowledge_files([corpus_path]):
        for document in load_documents(source):
            documents[document.document_id] = document
    recognizer_keys = {
        str(item.get("key")) for item in inventory.get("fact_recognizers", [])
    }

    def validate_support(owner: Mapping[str, Any]) -> None:
        for fact_key, support in owner.get("supported_facts", {}).items():
            if fact_key not in recognizer_keys:
                raise KnowledgeError("gate_inventory_fact_recognizer_missing")
            evidence = support.get("evidence", [])
            if not evidence:
                raise KnowledgeError("gate_inventory_fact_evidence_missing")
            for item in evidence:
                document_id = str(item.get("document_id", ""))
                excerpt = str(item.get("excerpt", ""))
                if document_id not in documents:
                    raise KnowledgeError("gate_inventory_document_missing")
                if not excerpt or excerpt not in documents[document_id].text:
                    raise KnowledgeError("gate_inventory_evidence_excerpt_invalid")

    entities = inventory.get("entities", [])
    domains = inventory.get("domains", [])
    entity_ids = {str(entity.get("id")) for entity in entities}
    domain_ids = {str(domain.get("id")) for domain in domains}
    for entity in entities:
        if str(entity.get("domain_id")) not in domain_ids:
            raise KnowledgeError("gate_inventory_domain_missing")
    for owner in (*entities, *domains):
        if not owner.get("aliases"):
            raise KnowledgeError("gate_inventory_aliases_missing")
        validate_support(owner)
    for relationship in inventory.get("relationships", []):
        relationship_entities = set(map(str, relationship.get("entity_ids", [])))
        if len(relationship_entities) < 2:
            raise KnowledgeError("gate_inventory_relationship_entities_invalid")
        if not relationship_entities <= entity_ids:
            raise KnowledgeError("gate_inventory_relationship_entity_missing")
        validate_support(relationship)
    bound = dict(inventory)
    bound["_active_corpus_version"] = active_version
    bound["_documents"] = {
        document_id: {
            "title": document.title,
            "route_scope": document.route_scope,
            "language": document.language,
        }
        for document_id, document in documents.items()
    }
    return bound


def evidence_document_ids(inventory: Mapping[str, Any]) -> frozenset[str]:
    owners = (
        *inventory.get("entities", []),
        *inventory.get("domains", []),
        *inventory.get("relationships", []),
    )
    return frozenset(
        str(item["document_id"])
        for owner in owners
        for support in owner.get("supported_facts", {}).values()
        for item in support.get("evidence", [])
    )


def evaluate_gate_readiness(
    config: GateRuntimeConfig,
    *,
    index_reader: Callable[[Path, str, Sequence[str]], IndexSnapshot] = (
        read_index_snapshot
    ),
) -> GateReadiness:
    try:
        gate = load_inventory(
            config.inventory_path,
            config.corpus_path,
            repo_root=config.repo_root,
        )
    except KnowledgeError as exc:
        code = str(exc)
        if code in {"gate_inventory_missing", "gate_corpus_path_mismatch"}:
            reason = code
        elif code == "gate_inventory_corpus_version_mismatch":
            reason = "gate_corpus_mismatch"
        elif code in {
            "gate_inventory_document_missing",
            "gate_inventory_evidence_excerpt_invalid",
        }:
            reason = "gate_evidence_invalid"
        else:
            reason = "gate_inventory_invalid"
        return GateReadiness(False, reason)
    except (OSError, ValueError):
        return GateReadiness(False, "gate_inventory_invalid")

    try:
        required_document_ids = evidence_document_ids(gate)
        snapshot = index_reader(
            config.chroma_path,
            config.collection_name,
            tuple(sorted(required_document_ids)),
        )
    except Exception:
        return GateReadiness(False, "gate_index_unavailable")
    metadata = snapshot.metadata
    active_version = str(gate["_active_corpus_version"])
    if (
        metadata.get("index_state") != "ready"
        or metadata.get("corpus_version") != active_version
    ):
        return GateReadiness(False, "gate_index_corpus_mismatch")
    if (
        metadata.get("embedding_model_id") != config.embedding_model_id
        or metadata.get("embedding_dimension") != config.embedding_dimension
    ):
        return GateReadiness(False, "gate_index_embedding_mismatch")
    if any(
        metadata.get(name) != expected
        for name, expected in (
            ("chunk_schema_version", config.chunk_schema_version),
            ("chunk_target_chars", config.chunk_target_chars),
            ("chunk_overlap_chars", config.chunk_overlap_chars),
        )
    ):
        return GateReadiness(False, "gate_index_schema_mismatch")
    if not required_document_ids <= snapshot.document_ids:
        return GateReadiness(False, "gate_index_document_missing")
    return GateReadiness(True, "ready", gate)


def classify_query(query: str, inventory: Mapping[str, Any]) -> GateDecision:
    entities = _matching_ids(query, inventory["entities"])
    domains = set(_matching_ids(query, inventory["domains"]))
    entity_index = {str(item["id"]): item for item in inventory["entities"]}
    domain_index = {str(item["id"]): item for item in inventory["domains"]}
    domains.update(
        str(entity_index[entity].get("domain_id"))
        for entity in entities
        if entity_index[entity].get("domain_id")
    )
    fact_key = _infer_fact_key(query, inventory)
    if entities and fact_key is None and _is_alias_only(query, inventory):
        fact_key = "identity_profile"
    pronoun = any(_contains_phrase(query, item) for item in inventory["pronouns"])

    if not entities and pronoun:
        return GateDecision(
            (), tuple(sorted(domains)), fact_key, "abstain_ambiguous_query",
            "A pronoun is present but no inventory entity can be resolved.",
        )
    if not entities and not domains:
        return GateDecision(
            (), (), fact_key, "abstain_unknown_domain",
            "No corpus entity or domain alias was recognised.",
        )
    if not entities:
        if fact_key == "setting_terminology":
            supported = any(
                fact_key in domain_index[domain].get("supported_facts", {})
                for domain in domains
            )
            return GateDecision(
                (), tuple(sorted(domains)), fact_key,
                "allow_retrieval" if supported else "abstain_unsupported_fact",
                "The known domain does or does not contain this evidence.",
            )
        return GateDecision(
            (), tuple(sorted(domains)), fact_key, "abstain_unknown_entity",
            "The domain is known, but no inventory entity matched.",
        )
    if fact_key is None:
        return GateDecision(
            entities, tuple(sorted(domains)), None, "abstain_ambiguous_query",
            "An entity is known, but the requested fact cannot be inferred.",
        )

    supporting_documents: set[str] = set()
    if fact_key == "relationship":
        for relationship in inventory.get("relationships", []):
            if set(map(str, relationship["entity_ids"])) == set(entities):
                support = relationship.get("supported_facts", {}).get(fact_key, {})
                supporting_documents.update(
                    str(item["document_id"])
                    for item in support.get("evidence", [])
                )
    else:
        evidence_sets = [
            {
                str(item["document_id"])
                for item in entity_index[entity]
                .get("supported_facts", {})
                .get(fact_key, {})
                .get("evidence", [])
            }
            for entity in entities
        ]
        if evidence_sets and all(evidence_sets):
            supporting_documents.update(set().union(*evidence_sets))
    if supporting_documents:
        return GateDecision(
            entities, tuple(sorted(domains)), fact_key, "allow_retrieval",
            "Corpus evidence exists for this entity and fact key.",
        )
    return GateDecision(
        entities, tuple(sorted(domains)), fact_key, "abstain_unsupported_fact",
        "The entity is known, but the corpus does not support this fact key.",
    )
