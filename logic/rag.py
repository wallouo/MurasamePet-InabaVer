"""Runtime retrieval for the optional local Meguru knowledge index."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .knowledge import (
    KnowledgeError,
    KnowledgeSearchResult,
    KnowledgeStore,
    PERSPECTIVE_STATUSES,
    ROUTE_SCOPES,
    validate_prompt_safe_text,
)
from .prompt_boundaries import RAG_BLOCK_CLOSE, RAG_BLOCK_OPEN


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RAGSettings:
    """Small, environment-backed runtime configuration for local retrieval."""

    chroma_path: str = "data/knowledge/chroma"
    collection_name: str = "meguru_knowledge"
    embedding_model_path: str = "models/embeddings/multilingual-e5-small"
    corpus_path: str = "data/knowledge/examples"
    gate_inventory_path: str = "data/knowledge/gate_inventory.json"
    max_results: int = 3
    min_similarity: float = 0.70
    route_scope: str = "meguru"
    chunk_target_chars: int = 700
    chunk_overlap_chars: int = 80

    @classmethod
    def from_env(cls) -> "RAGSettings":
        requested_scope = os.getenv("RAG_ROUTE_SCOPE", cls.route_scope).strip()
        if requested_scope not in ROUTE_SCOPES:
            requested_scope = cls.route_scope
        return cls(
            chroma_path=os.getenv("RAG_CHROMA_PATH", cls.chroma_path),
            collection_name=os.getenv("RAG_COLLECTION", cls.collection_name),
            embedding_model_path=os.getenv(
                "RAG_EMBEDDING_MODEL_PATH", cls.embedding_model_path
            ),
            corpus_path=os.getenv("RAG_CORPUS_PATH", cls.corpus_path),
            gate_inventory_path=os.getenv(
                "RAG_GATE_INVENTORY_PATH", cls.gate_inventory_path
            ),
            max_results=_env_int("RAG_MAX_RESULTS", cls.max_results),
            min_similarity=_env_float("RAG_MIN_SIMILARITY", cls.min_similarity),
            route_scope=requested_scope,
            chunk_target_chars=_env_int(
                "RAG_CHUNK_TARGET_CHARS", cls.chunk_target_chars, minimum=100
            ),
            chunk_overlap_chars=_env_int(
                "RAG_CHUNK_OVERLAP_CHARS", cls.chunk_overlap_chars, minimum=0
            ),
        )

    @property
    def preferred_route_scopes(self) -> tuple[str, ...]:
        """Meguru/shared facts win only exact-similarity ties."""

        return tuple(
            dict.fromkeys(("meguru", "common", "global", self.route_scope))
        )


class KnowledgeRetriever:
    """Lazy Chroma adapter that turns all index failures into empty retrieval."""

    def __init__(self, settings: RAGSettings, *, store: Any | None = None) -> None:
        self.settings = settings
        self._store = store
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _get_store(self) -> Any:
        if self._store is None:
            self._store = KnowledgeStore(
                persist_path=Path(self.settings.chroma_path),
                collection_name=self.settings.collection_name,
                embedding_model_path=Path(self.settings.embedding_model_path),
            )
        return self._store

    def search(self, query: str) -> list[KnowledgeSearchResult]:
        query = query.strip()
        if not query:
            return []
        try:
            results = self._get_store().search(
                query=query,
                # Search every route, then rank locally. The small multiplier
                # gives alternate routes a chance to survive before the
                # preference-only tie-breaker is applied.
                limit=max(
                    self.settings.max_results,
                    self.settings.max_results * len(ROUTE_SCOPES),
                ),
                min_similarity=self.settings.min_similarity,
                route_scopes=None,
            )
        except Exception as exc:  # normal chat must survive an unavailable index
            self._last_error = f"{type(exc).__name__}: {exc}"
            print(f"[RAG] retrieval failed locally: {self._last_error}")
            return []
        self._last_error = None
        preferred = self.settings.preferred_route_scopes
        preference_rank = {
            route_scope: index for index, route_scope in enumerate(preferred)
        }
        ranked = sorted(
            results,
            key=lambda result: (
                -result.similarity,
                preference_rank.get(
                    str(result.metadata.get("route_scope", "")), len(preferred)
                ),
            ),
        )
        return list(ranked[: self.settings.max_results])


def format_knowledge_block(result: KnowledgeSearchResult) -> str:
    """Render only grounding fields; keep indexing metadata out of the prompt."""

    metadata = result.metadata
    title = str(metadata.get("title", "")).strip()
    content = result.text.strip()
    try:
        validate_prompt_safe_text(title, "title")
        validate_prompt_safe_text(content, "content")
    except KnowledgeError:
        return ""
    route_scope = str(metadata.get("route_scope", "")).strip()
    perspective_status = str(metadata.get("perspective_status", "")).strip()
    try:
        validate_prompt_safe_text(route_scope, "route_scope")
        validate_prompt_safe_text(perspective_status, "perspective_status")
    except KnowledgeError:
        return ""
    if route_scope not in ROUTE_SCOPES or perspective_status not in PERSPECTIVE_STATUSES:
        return ""
    perspective_note = (
        "alternate-route knowledge; do not present it as Meguru and Senpai's "
        "lived history."
        if perspective_status == "alternate"
        else ""
    )
    fields = (
        ("title", title),
        ("route_scope", route_scope),
        ("perspective_status", perspective_status),
        ("perspective_note", perspective_note),
    )
    lines = [
        RAG_BLOCK_OPEN,
        *[
            f"{name}: {str(value).strip()}"
            for name, value in fields
            if str(value).strip()
        ],
        "content:",
        content,
        RAG_BLOCK_CLOSE,
    ]
    return "\n".join(lines)


def format_knowledge_blocks(
    results: Sequence[KnowledgeSearchResult],
) -> tuple[str, ...]:
    blocks = (format_knowledge_block(result) for result in results)
    return tuple(block for block in blocks if block)


def knowledge_result_diagnostics(result: KnowledgeSearchResult) -> dict[str, Any]:
    """Expose complete retrieval provenance without putting it in the prompt."""

    metadata = dict(result.metadata)
    metadata.pop("source_path", None)
    for key in ("character_tags", "relationship_tags"):
        value = metadata.get(key)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value
            metadata[key] = parsed
    return {
        "chunk_id": result.chunk_id,
        "text": result.text,
        "distance": result.distance,
        "similarity": result.similarity,
        "metadata": metadata,
    }
