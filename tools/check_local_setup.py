"""Verify that the constrained, local-only RAG runtime is reproducible."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REQUIRED_IMPORTS = ("chromadb", "sentence_transformers", "llama_cpp")
REQUIRED_DISTRIBUTIONS = (
    "chromadb",
    "sentence-transformers",
    "llama-cpp-python",
)
LLAMA_RELEASE_PREFIX = (
    "https://github.com/abetlen/llama-cpp-python/releases/download/"
)
WINDOWS_WHEEL_TAG = "py3-none-win_amd64"


class SetupCheckError(RuntimeError):
    pass


def read_constraints(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        pins[name.strip().lower()] = version.strip()
    return pins


def verify_install_report(payload: Mapping[str, Any], expected_version: str) -> None:
    candidates = [
        item
        for item in payload.get("install", [])
        if str(item.get("metadata", {}).get("name", ""))
        .lower()
        .replace("_", "-")
        == "llama-cpp-python"
    ]
    if len(candidates) != 1:
        raise SetupCheckError("llama_cpp_install_report_missing")
    item = candidates[0]
    if str(item.get("metadata", {}).get("version")) != expected_version:
        raise SetupCheckError("llama_cpp_version_mismatch")
    url = str(item.get("download_info", {}).get("url", ""))
    filename = url.rsplit("/", 1)[-1]
    if not url.startswith(LLAMA_RELEASE_PREFIX) or not filename.endswith(
        f"-{WINDOWS_WHEEL_TAG}.whl"
    ):
        raise SetupCheckError("llama_cpp_unexpected_download")


def verify_wheel_metadata(distribution: Any, expected_version: str) -> None:
    if distribution.version != expected_version:
        raise SetupCheckError("llama_cpp_version_mismatch")
    wheel_text = distribution.read_text("WHEEL") or ""
    tags = {
        line.split(":", 1)[1].strip()
        for line in wheel_text.splitlines()
        if line.startswith("Tag:")
    }
    if WINDOWS_WHEEL_TAG not in tags:
        raise SetupCheckError("llama_cpp_wrong_wheel_tag")


def check_setup(
    *,
    constraints_path: Path,
    e5_path: Path,
    gguf_path: Path,
    install_report_path: Path | None,
) -> None:
    pins = read_constraints(constraints_path)
    for name in REQUIRED_DISTRIBUTIONS:
        expected = pins.get(name)
        if not expected:
            raise SetupCheckError(f"constraint_missing:{name}")
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise SetupCheckError(f"package_missing:{name}") from exc
        if installed != expected:
            raise SetupCheckError(f"package_version_mismatch:{name}")
    for module_name in REQUIRED_IMPORTS:
        importlib.import_module(module_name)
    if not e5_path.is_dir():
        raise SetupCheckError("embedding_model_missing")
    if not gguf_path.is_file():
        raise SetupCheckError("gguf_missing")

    if platform.system() == "Windows" and platform.machine().upper() in {
        "AMD64",
        "X86_64",
    }:
        llama_dist = importlib.metadata.distribution("llama-cpp-python")
        verify_wheel_metadata(llama_dist, pins["llama-cpp-python"])
        if install_report_path is None:
            raise SetupCheckError("install_report_missing")
        try:
            report = json.loads(install_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SetupCheckError("install_report_invalid") from exc
        verify_install_report(report, pins["llama-cpp-python"])

    from logic.token_budget import TokenCounter

    if TokenCounter.load(gguf_path).mode != "gguf_native":
        raise SetupCheckError("gguf_tokenizer_unavailable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--constraints", type=Path, default=ROOT / "constraints.txt")
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=ROOT / "models" / "embeddings" / "multilingual-e5-small",
    )
    parser.add_argument(
        "--gguf",
        type=Path,
        default=ROOT / "tools" / "model_training" / "meguru_q4_k_m.gguf",
    )
    parser.add_argument("--install-report", type=Path)
    args = parser.parse_args()
    try:
        check_setup(
            constraints_path=args.constraints,
            e5_path=args.embedding_model,
            gguf_path=args.gguf,
            install_report_path=args.install_report,
        )
    except SetupCheckError as exc:
        print(f"setup_check=failed reason={exc}")
        return 1
    print("setup_check=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
