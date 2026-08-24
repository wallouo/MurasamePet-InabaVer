"""Load repository-local configuration without overriding process settings."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_project_env(project_file: str | Path) -> None:
    """Load the ``.env`` beside a project entry point, preserving explicit env."""

    load_dotenv(Path(project_file).resolve().with_name(".env"), override=False)
