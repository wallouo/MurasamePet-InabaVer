"""Inspect the retained Meguru GGUF and verify local token accounting.

This is an offline-first diagnostic.  It never downloads a tokenizer.  When
``llama-cpp-python`` is installed it probes the vocabulary embedded in the
exact GGUF; otherwise it records the conservative UTF-8 fallback and leaves
RAG disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Mapping
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logic.token_budget import (  # noqa: E402
    PromptProfile,
    TOKENIZER_VALIDATION_TOLERANCE_TOKENS,
    TokenCounter,
)


GGUF_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}
STRUCT_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF metadata")
    return data


def _read_u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _read_string(handle: BinaryIO) -> str:
    size = _read_u64(handle)
    if size > 64 * 1024 * 1024:
        raise ValueError("unreasonable GGUF string length")
    return _read_exact(handle, size).decode("utf-8", errors="replace")


def _read_value(handle: BinaryIO, value_type: int) -> Any:
    if value_type == 8:
        return _read_string(handle)
    if value_type == 9:
        item_type = struct.unpack("<I", _read_exact(handle, 4))[0]
        count = _read_u64(handle)
        if item_type not in GGUF_TYPES or count > 2_000_000:
            raise ValueError("unsupported or unreasonable GGUF array")
        return [_read_value(handle, item_type) for _ in range(count)]
    fmt = STRUCT_FORMATS.get(value_type)
    if fmt is None:
        raise ValueError(f"unsupported GGUF metadata type {value_type}")
    return struct.unpack(fmt, _read_exact(handle, struct.calcsize(fmt)))[0]


def read_gguf_metadata(path: Path) -> tuple[dict[str, Any], dict[str, int]]:
    """Read the GGUF header and metadata without touching tensor payloads."""

    with path.open("rb") as handle:
        if _read_exact(handle, 4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        version = struct.unpack("<I", _read_exact(handle, 4))[0]
        tensor_count = _read_u64(handle)
        metadata_count = _read_u64(handle)
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = _read_string(handle)
            value_type = struct.unpack("<I", _read_exact(handle, 4))[0]
            metadata[key] = _read_value(handle, value_type)
    return metadata, {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_kv_count": metadata_count,
    }


def _summarize_array(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None
    return {
        "count": len(value),
        "head": [str(item) for item in value[:3]],
        "tail": [str(item) for item in value[-3:]],
    }


def summarize_gguf(path: Path) -> dict[str, Any]:
    metadata, header = read_gguf_metadata(path)
    tokenizer_keys = {
        key: value
        for key, value in metadata.items()
        if key.startswith("tokenizer.")
    }
    token_ids = {
        key: value
        for key, value in tokenizer_keys.items()
        if key.endswith("_token_id") or key.endswith("_bos_token") or key.endswith("_eos_token")
    }
    return {
        "header": header,
        "file": {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "model": {
            "architecture": metadata.get("general.architecture"),
            "name": metadata.get("general.name"),
            "type": metadata.get("general.type"),
            "quantized_by": metadata.get("general.quantized_by"),
            "size_label": metadata.get("general.size_label"),
            "context_length": metadata.get(
                f"{metadata.get('general.architecture', '')}.context_length"
            ),
        },
        "tokenizer": {
            "model": metadata.get("tokenizer.ggml.model"),
            "pre": metadata.get("tokenizer.ggml.pre"),
            "tokens": _summarize_array(tokenizer_keys.get("tokenizer.ggml.tokens")),
            "token_type": _summarize_array(tokenizer_keys.get("tokenizer.ggml.token_type")),
            "merges": _summarize_array(tokenizer_keys.get("tokenizer.ggml.merges")),
            "special_token_ids": token_ids,
            "add_bos_token": metadata.get("tokenizer.ggml.add_bos_token"),
            "add_eos_token": metadata.get("tokenizer.ggml.add_eos_token"),
            "chat_template": tokenizer_keys.get("tokenizer.chat_template")
            or tokenizer_keys.get("tokenizer.ggml.chat_template"),
        },
    }


def ollama_show(endpoint: str, model: str) -> dict[str, Any]:
    request = Request(
        f"{endpoint.rstrip('/')}/api/show",
        data=json.dumps({"model": model}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_chat(
    endpoint: str, model: str, text: str, *, num_ctx: int = 4096
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": 1},
    }
    request = Request(
        f"{endpoint.rstrip('/')}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_version(endpoint: str) -> str | None:
    try:
        request = Request(f"{endpoint.rstrip('/')}/api/version", method="GET")
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("version"):
            return str(payload["version"])
    except Exception:
        pass

    candidates = [
        os.getenv("OLLAMA_BIN"),
        shutil.which("ollama"),
        r"C:\Users\kanek\AppData\Local\Programs\Ollama\ollama.exe",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            result = subprocess.run(
                [candidate, "version"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        output = (result.stdout or result.stderr).strip()
        if output:
            return output
    return None


def _ollama_artifact_path(show: Mapping[str, Any]) -> Path | None:
    modelfile = str(show.get("modelfile") or "")
    match = re.search(r"^FROM\s+(.+)$", modelfile, re.MULTILINE)
    if not match:
        return None
    candidate = Path(match.group(1).strip().strip('"'))
    return candidate if candidate.is_file() else None


def validate_counter(
    *,
    endpoint: str,
    model: str,
    profile: PromptProfile,
    counter: TokenCounter,
    tolerance: int,
) -> dict[str, Any]:
    fixtures = [
        ("japanese", "センパイ、おはよう。めぐるの設定を教えて。"),
        ("traditional_chinese", "學長，今天過得怎麼樣？"),
        ("simplified_chinese", "学长，你今天还好吗？"),
        ("english", "Senpai, tell me about Meguru."),
        ("mixed_language", "センパイ、今天のゲームは fun だった？"),
        ("newline_emoji", "ちゃろー！\n今日はどう？🎮✨"),
        (
            "near_budget",
            " ".join(
                "センパイとめぐるのゲームの思い出を確認するためのテスト文です。"
                for _ in range(180)
            ),
        ),
    ]
    results: list[dict[str, Any]] = []
    for name, text in fixtures:
        rendered = profile.render(text)
        local_count = counter.count(rendered)
        try:
            ollama_result = ollama_chat(
                endpoint,
                model,
                text,
                num_ctx=profile.num_ctx or 4096,
            )
            remote_count = ollama_result.get("prompt_eval_count")
            delta = local_count - remote_count if isinstance(remote_count, int) else None
            within = isinstance(delta, int) and abs(delta) <= tolerance
            error = None
        except Exception as exc:  # diagnostics must never break chat startup
            remote_count = None
            delta = None
            within = False
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "name": name,
                "input_chars": len(text),
                "local_count": local_count,
                "ollama_prompt_eval_count": remote_count,
                "delta": delta,
                "within_tolerance": within,
                "error": error,
            }
        )

    if counter.exact:
        passed = all(item["within_tolerance"] for item in results)
        status = "passed" if passed else "failed"
        reason = None if passed else "GGUF-native counts exceeded the configured tolerance"
    else:
        status = "not_validated"
        reason = "llama-cpp-python GGUF tokenizer is unavailable"
    return {
        "status": status,
        "reason": reason,
        "counter_mode": counter.mode,
        "tolerance_tokens": tolerance,
        "fixtures": results,
    }


def build_manifest(
    *,
    gguf_path: Path,
    endpoint: str,
    model: str,
    manifest_path: Path,
    tolerance: int,
) -> dict[str, Any]:
    gguf = summarize_gguf(gguf_path)
    show = ollama_show(endpoint, model)
    profile = PromptProfile.from_ollama_response(show)
    counter = TokenCounter.load(gguf_path)
    artifact = _ollama_artifact_path(show)
    artifact_info = None
    if artifact:
        artifact_info = {
            "path": str(artifact.resolve()),
            "size_bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        }
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gguf": gguf,
        "ollama": {
            "endpoint": endpoint,
            "model": model,
            "version": ollama_version(endpoint),
            "modified_at": show.get("modified_at"),
            "artifact": artifact_info,
            "template_sha256": sha256_text(str(show.get("template") or "")),
            "system_sha256": sha256_text(str(show.get("system") or "")),
            "parameters": show.get("parameters"),
            "parameters_sha256": sha256_text(str(show.get("parameters") or "")),
            "details": show.get("details"),
        },
        "counter": {
            "requested": "gguf_native_if_available",
            "mode": counter.mode,
            "exact": counter.exact,
            "fallback": "utf8_upper_bound",
            "rag_default_enabled": False,
        },
        "validation": validate_counter(
            endpoint=endpoint,
            model=model,
            profile=profile,
            counter=counter,
            tolerance=tolerance,
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gguf",
        type=Path,
        default=ROOT / "tools" / "model_training" / "meguru_q4_k_m.gguf",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "context_budget_manifest.json",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "meguru"))
    parser.add_argument(
        "--tolerance",
        type=int,
        default=TOKENIZER_VALIDATION_TOLERANCE_TOKENS,
    )
    args = parser.parse_args()
    manifest = build_manifest(
        gguf_path=args.gguf.resolve(),
        endpoint=args.endpoint,
        model=args.model,
        manifest_path=args.manifest.resolve(),
        tolerance=args.tolerance,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
