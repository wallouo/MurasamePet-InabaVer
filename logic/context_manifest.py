"""Context-budget manifest creation helpers and the runtime RAG readiness gate."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTEXT_MANIFEST_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class RAGReadiness:
    ready: bool
    reason: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parameter_int(parameters: str, name: str) -> int | None:
    match = re.search(rf"(?:^|\s){re.escape(name)}\s+(\d+)", parameters)
    return int(match.group(1)) if match else None


def canonical_stop_tokens(parameters: str) -> tuple[str, ...]:
    """Return stop strings independent of whitespace and declaration order."""

    stops: set[str] = set()
    for line in parameters.splitlines():
        match = re.match(r"^\s*stop\s+(.+?)\s*$", line)
        if not match:
            continue
        raw = match.group(1)
        try:
            value = json.loads(raw) if raw.startswith('"') else raw
        except json.JSONDecodeError:
            value = raw.strip('"')
        stops.add(str(value))
    return tuple(sorted(stops))


def ollama_artifact_path(show: Mapping[str, Any]) -> Path | None:
    modelfile = str(show.get("modelfile") or "")
    match = re.search(r"^FROM\s+(.+)$", modelfile, re.MULTILINE)
    if not match:
        return None
    candidate = Path(match.group(1).strip().strip('"'))
    return candidate if candidate.is_file() else None


def canonical_ollama_profile(
    show: Mapping[str, Any], *, model: str, artifact_sha256: str | None = None
) -> dict[str, Any]:
    parameters = str(show.get("parameters") or "")
    artifact = ollama_artifact_path(show)
    if artifact is None:
        raise OSError("active Ollama artifact is unavailable")
    return {
        "model": model,
        "artifact_sha256": artifact_sha256 or sha256_file(artifact),
        "template_sha256": sha256_text(str(show.get("template") or "")),
        "system_sha256": sha256_text(str(show.get("system") or "")),
        "num_ctx": parameter_int(parameters, "num_ctx"),
        "num_predict": parameter_int(parameters, "num_predict"),
        "stop_tokens": list(canonical_stop_tokens(parameters)),
    }


def runtime_parameters_present(profile: Mapping[str, Any]) -> bool:
    return all(isinstance(profile.get(field), int) for field in ("num_ctx", "num_predict"))


def _manifest_profile(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    ollama = manifest.get("ollama")
    if not isinstance(ollama, Mapping):
        raise ValueError("manifest ollama profile is missing")
    profile = ollama.get("validation_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("manifest validation profile is missing")
    return profile


def evaluate_rag_readiness(
    *,
    requested: bool,
    manifest_path: Path,
    gguf_path: Path,
    model: str,
    ollama_show: Mapping[str, Any] | None,
    tokenizer_mode: str,
) -> RAGReadiness:
    """Validate one conservative startup snapshot without leaking details."""

    if not requested:
        return RAGReadiness(False, "disabled_by_config")
    if not manifest_path.is_file():
        return RAGReadiness(False, "manifest_missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise ValueError("manifest must be an object")
        if manifest.get("schema_version") != CONTEXT_MANIFEST_SCHEMA_VERSION:
            return RAGReadiness(False, "manifest_invalid")
        validation = manifest.get("validation")
        if not isinstance(validation, Mapping) or validation.get("status") != "passed":
            return RAGReadiness(False, "validation_failed")
        counter = manifest.get("counter")
        if not isinstance(counter, Mapping):
            raise ValueError("manifest counter is missing")
        if tokenizer_mode != "gguf_native" or counter.get("mode") != tokenizer_mode:
            return RAGReadiness(False, "tokenizer_mode_mismatch")
        if ollama_show is None:
            return RAGReadiness(False, "ollama_profile_unavailable")
        gguf = manifest.get("gguf")
        gguf_file = gguf.get("file") if isinstance(gguf, Mapping) else None
        expected_gguf = gguf_file.get("sha256") if isinstance(gguf_file, Mapping) else None
        if not gguf_path.is_file():
            return RAGReadiness(False, "gguf_mismatch")
        current_gguf_hash = sha256_file(gguf_path)
        if current_gguf_hash != expected_gguf:
            return RAGReadiness(False, "gguf_mismatch")
        expected = _manifest_profile(manifest)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return RAGReadiness(False, "manifest_invalid")
    try:
        artifact = ollama_artifact_path(ollama_show)
        same_artifact = artifact is not None and artifact.samefile(gguf_path)
        current = canonical_ollama_profile(
            ollama_show,
            model=model,
            artifact_sha256=current_gguf_hash if same_artifact else None,
        )
    except (OSError, ValueError, TypeError):
        return RAGReadiness(False, "ollama_profile_unavailable")

    comparisons: Sequence[tuple[str, str]] = (
        ("model", "ollama_model_mismatch"),
        ("artifact_sha256", "ollama_artifact_mismatch"),
        ("template_sha256", "template_mismatch"),
        ("system_sha256", "system_mismatch"),
    )
    for field, reason in comparisons:
        if current.get(field) != expected.get(field):
            return RAGReadiness(False, reason)
    if current["artifact_sha256"] != expected_gguf:
        return RAGReadiness(False, "ollama_artifact_mismatch")
    if not runtime_parameters_present(current) or not runtime_parameters_present(expected):
        return RAGReadiness(False, "runtime_parameters_mismatch")
    if (
        current.get("num_ctx") != expected.get("num_ctx")
        or current.get("num_predict") != expected.get("num_predict")
    ):
        return RAGReadiness(False, "runtime_parameters_mismatch")
    if list(current.get("stop_tokens") or ()) != list(expected.get("stop_tokens") or ()):
        return RAGReadiness(False, "stop_tokens_mismatch")
    return RAGReadiness(True, "ready")
