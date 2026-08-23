"""Curated knowledge documents, deterministic chunks, and local Chroma storage."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .prompt_boundaries import RAG_BLOCK_CLOSE, RAG_BLOCK_OPEN, USER_MESSAGE_MARKER


ROUTE_SCOPES = frozenset(
    {"nene", "meguru", "tsumugi", "touko", "wakana", "common", "global"}
)
PERSPECTIVE_STATUSES = frozenset({"lived", "alternate", "universal"})
DOCUMENT_FORMATS = frozenset({"markdown", "json"})
SOURCE_AUTHORITIES = frozenset(
    {"official", "official_localization", "secondary", "curated", "fan_verified"}
)
_CHATML_CONTROL_TOKEN = re.compile(r"<\|[^|\r\n<>]+\|>", re.IGNORECASE)


class KnowledgeError(ValueError):
    """A knowledge document or local index cannot be used safely."""


def validate_prompt_safe_text(value: str, field: str) -> None:
    """Reject text that can cross the model-facing prompt boundary."""

    if (
        _CHATML_CONTROL_TOKEN.search(value)
        or USER_MESSAGE_MARKER in value
        or RAG_BLOCK_OPEN in value
        or RAG_BLOCK_CLOSE in value
    ):
        raise KnowledgeError(f"{field} contains a reserved prompt marker")


def _required_choice(data: Mapping[str, Any], name: str, choices: set[str]) -> str:
    value = str(data.get(name) or "").strip()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise KnowledgeError(f"{name} must be one of: {allowed}")
    return value


def _tags(value: Any, name: str) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple)):
        values = [str(item).strip() for item in value]
    else:
        raise KnowledgeError(f"{name} must be a string or list of strings")
    return tuple(item for item in values if item)


@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str
    title: str
    text: str
    source_path: str
    source_url: str
    document_format: str
    source_authority: str
    route_scope: str
    perspective_status: str
    language: str
    character_tags: tuple[str, ...] = ()
    relationship_tags: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        source_path: Path,
        document_format: str,
    ) -> "KnowledgeDocument":
        if document_format not in DOCUMENT_FORMATS:
            raise KnowledgeError(f"unsupported document format: {document_format}")
        declared_format = str(data.get("document_format") or document_format)
        if declared_format != document_format:
            raise KnowledgeError(
                f"{source_path}: document_format conflicts with its file extension"
            )
        document_id = str(data.get("id") or data.get("document_id") or "").strip()
        if not document_id:
            raise KnowledgeError(f"{source_path}: missing id")
        text = str(data.get("text") or data.get("content") or "").strip()
        if not text:
            raise KnowledgeError(f"{source_path}: missing text/content")
        title = str(data.get("title") or document_id).strip()
        validate_prompt_safe_text(title, "title")
        validate_prompt_safe_text(text, "content")
        language = str(data.get("language") or "").strip()
        if not language:
            raise KnowledgeError(f"{source_path}: missing language")
        return cls(
            document_id=document_id,
            title=title,
            text=text,
            source_path=str(source_path),
            source_url=str(data.get("source_url") or "").strip(),
            document_format=declared_format,
            source_authority=_required_choice(
                data, "source_authority", set(SOURCE_AUTHORITIES)
            ),
            route_scope=_required_choice(data, "route_scope", set(ROUTE_SCOPES)),
            perspective_status=_required_choice(
                data, "perspective_status", set(PERSPECTIVE_STATUSES)
            ),
            language=language,
            character_tags=_tags(data.get("character_tags"), "character_tags"),
            relationship_tags=_tags(
                data.get("relationship_tags"), "relationship_tags"
            ),
        )


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class KnowledgeSearchResult:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float
    similarity: float


def _frontmatter_value(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return ""
    if raw.startswith("[") or raw.startswith("{") or raw.startswith('"'):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if raw.startswith("[") and raw.endswith("]"):
                return [item.strip().strip("'\"") for item in raw[1:-1].split(",")]
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    return raw


def _read_markdown(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", text, re.DOTALL)
    if not match:
        raise KnowledgeError(
            f"{path}: Markdown needs a front matter block delimited by '---'"
        )
    metadata: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        if key == "title":
            validate_prompt_safe_text(raw.strip(), "title")
        metadata[key] = _frontmatter_value(raw)
    metadata["text"] = text[match.end() :].strip()
    return metadata


def load_documents(path: str | Path) -> list[KnowledgeDocument]:
    """Load one Markdown/JSON file; JSON may contain one object or a list."""

    path = Path(path)
    if path.suffix.lower() == ".md":
        return [
            KnowledgeDocument.from_mapping(
                _read_markdown(path), source_path=path, document_format="markdown"
            )
        ]
    if path.suffix.lower() != ".json":
        raise KnowledgeError(f"unsupported knowledge file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeError(f"cannot read {path}: {exc}") from exc
    items = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(item, Mapping) for item in items):
        raise KnowledgeError(f"{path}: JSON must contain an object or list of objects")
    return [
        KnowledgeDocument.from_mapping(
            item, source_path=path, document_format="json"
        )
        for item in items
    ]


def iter_knowledge_files(paths: Sequence[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.suffix.lower() in {".md", ".json"}:
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate
                for candidate in sorted(path.rglob("*"))
                if candidate.is_file() and candidate.suffix.lower() in {".md", ".json"}
            )
        else:
            raise KnowledgeError(f"knowledge path does not exist: {path}")
    return sorted(set(files), key=lambda item: str(item).lower())


def _split_long_piece(text: str, target_chars: int) -> list[str]:
    if len(text) <= target_chars:
        return [text]
    pieces: list[str] = []
    remaining = text
    boundaries = "。！？!?\n"
    while len(remaining) > target_chars:
        cut = max((remaining.rfind(mark, 0, target_chars + 1) for mark in boundaries), default=-1)
        if cut < target_chars // 2:
            cut = target_chars
        else:
            cut += 1
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _load_local_e5_model(model_path: Path) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise KnowledgeError(
            "SentenceTransformers is required for the local embedding model"
        ) from exc
    if not model_path.is_dir():
        raise KnowledgeError(
            "embedding model path is missing; runtime will not download it: "
            f"{model_path}"
        )
    try:
        return SentenceTransformer(
            str(model_path), device="cpu", local_files_only=True
        )
    except TypeError as exc:
        raise KnowledgeError(
            "installed sentence-transformers lacks local_files_only support"
        ) from exc


def chunk_document(
    document: KnowledgeDocument,
    *,
    target_chars: int = 700,
    overlap_chars: int = 80,
) -> list[KnowledgeChunk]:
    """Split on paragraphs, then sentences/characters, retaining small overlap."""

    if target_chars < 100:
        raise KnowledgeError("target_chars must be at least 100")
    if overlap_chars < 0 or overlap_chars >= target_chars:
        raise KnowledgeError("overlap_chars must be >= 0 and smaller than target_chars")
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", document.text) if item.strip()]
    pieces = [piece for paragraph in paragraphs for piece in _split_long_piece(paragraph, target_chars)]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if current and len(candidate) > target_chars:
            chunks.append(current)
            overlap = current[-overlap_chars:] if overlap_chars else ""
            carried = f"{overlap}\n{piece}".strip() if overlap else piece
            # A near-target paragraph gets no overlap rather than producing an
            # oversized chunk and violating the configured bound.
            current = carried if len(carried) <= target_chars else piece
        else:
            current = candidate
    if current:
        chunks.append(current)

    result: list[KnowledgeChunk] = []
    for index, text in enumerate(chunks):
        chunk_id = hashlib.sha256(
            f"{document.document_id}:{index}:{text}".encode("utf-8")
        ).hexdigest()[:24]
        metadata: dict[str, str | int] = {
            "document_id": document.document_id,
            "title": document.title,
            "source_path": document.source_path,
            "source_url": document.source_url,
            "document_format": document.document_format,
            "source_authority": document.source_authority,
            "route_scope": document.route_scope,
            "perspective_status": document.perspective_status,
            "language": document.language,
            "character_tags": json.dumps(document.character_tags, ensure_ascii=False),
            "relationship_tags": json.dumps(
                document.relationship_tags, ensure_ascii=False
            ),
            "chunk_index": index,
        }
        result.append(KnowledgeChunk(chunk_id=chunk_id, text=text, metadata=metadata))
    return result


class _E5EmbeddingFunction:
    """Chroma document embedder plus correctly prefixed E5 query embeddings."""

    def __init__(self, model: Any, *, model_path: str | None = None) -> None:
        self._model = model
        self._model_path = model_path

    def _encode(self, prefix: str, input: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            [f"{prefix}{text}" for text in input],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist() if hasattr(vectors, "tolist") else [list(item) for item in vectors]

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        return self._encode("passage: ", input)

    def embed_query(self, input: Sequence[str]) -> list[list[float]]:
        return self._encode("query: ", input)

    @staticmethod
    def name() -> str:
        return "intfloat_multilingual_e5_small"

    @staticmethod
    def build_from_config(config: Mapping[str, Any]) -> "_E5EmbeddingFunction":
        model_path = str(config.get("model_path") or "").strip()
        if not model_path:
            raise KnowledgeError("E5 embedding configuration is missing model_path")
        model = _load_local_e5_model(Path(model_path))
        return _E5EmbeddingFunction(model, model_path=model_path)

    def is_legacy(self) -> bool:
        return False

    def supported_spaces(self) -> list[str]:
        return ["cosine"]

    def get_config(self) -> dict[str, str]:
        return {
            "name": self.name(),
            "space": "cosine",
            "model_path": self._model_path or "",
        }


class KnowledgeStore:
    """Thin Chroma adapter; imports optional heavy dependencies only on use."""

    def __init__(
        self,
        *,
        persist_path: str | Path,
        collection_name: str,
        embedding_model_path: str | Path,
        embedding_function: Any | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise KnowledgeError(
                "Phase 2 indexing requires chromadb"
            ) from exc
        if embedding_function is None:
            model_path = Path(embedding_model_path)
            model = _load_local_e5_model(model_path)
            embedding_function = _E5EmbeddingFunction(
                model, model_path=str(model_path)
            )

        self._client = chromadb.PersistentClient(path=str(Path(persist_path)))
        self._embedding_function = embedding_function
        self._collection_name = collection_name
        self._collection = self._open_collection()

    def _open_collection(self) -> Any:
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_function,
        )

    def upsert(self, chunks: Sequence[KnowledgeChunk]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def count(self) -> int:
        return int(self._collection.count())

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        embed_query = getattr(self._embedding_function, "embed_query", None)
        return embed_query(texts) if embed_query else self._embedding_function(texts)

    def search(
        self,
        query: str,
        *,
        limit: int = 3,
        min_similarity: float = 0.70,
        route_scope: str | None = None,
        route_scopes: Sequence[str] | None = None,
    ) -> list[KnowledgeSearchResult]:
        """Return relevant chunks, omitting weak nearest-neighbour matches."""

        query = query.strip()
        if not query or limit < 1:
            return []
        if not 0.0 <= min_similarity <= 1.0:
            raise KnowledgeError("min_similarity must be between 0 and 1")
        scopes = tuple(
            dict.fromkeys(
                scope.strip()
                for scope in (route_scopes or ())
                if isinstance(scope, str) and scope.strip()
            )
        )
        if scopes:
            where = {"route_scope": scopes[0]} if len(scopes) == 1 else {
                "route_scope": {"$in": list(scopes)}
            }
        else:
            where = {"route_scope": route_scope} if route_scope else None
        result = self._collection.query(
            query_embeddings=self.embed_queries([query]),
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids") or [[]]
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        matches: list[KnowledgeSearchResult] = []
        for index, chunk_id in enumerate(ids[0]):
            distance = float(distances[0][index])
            similarity = 1.0 - distance
            if similarity < min_similarity:
                continue
            matches.append(
                KnowledgeSearchResult(
                    chunk_id=str(chunk_id),
                    text=str(documents[0][index]),
                    metadata=dict(metadatas[0][index] or {}),
                    distance=distance,
                    similarity=similarity,
                )
            )
        return matches

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._open_collection()


def ingest_paths(
    paths: Sequence[str | Path],
    store: KnowledgeStore,
    *,
    target_chars: int = 700,
    overlap_chars: int = 80,
) -> dict[str, int]:
    documents: list[KnowledgeDocument] = []
    seen_ids: set[str] = set()
    for path in iter_knowledge_files(paths):
        for document in load_documents(path):
            if document.document_id in seen_ids:
                raise KnowledgeError(f"duplicate document id: {document.document_id}")
            seen_ids.add(document.document_id)
            documents.append(document)
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(
            document, target_chars=target_chars, overlap_chars=overlap_chars
        )
    ]
    store.upsert(chunks)
    return {"documents": len(documents), "chunks": len(chunks), "indexed": store.count()}
