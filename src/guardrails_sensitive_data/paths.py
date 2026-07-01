"""Shared project paths."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NETFLIX_DIR = PROJECT_ROOT / "data" / "netflix"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"
DEFAULT_IMDB_CSV = PROJECT_ROOT / "notebooks" / "imdb_data.csv"


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path
