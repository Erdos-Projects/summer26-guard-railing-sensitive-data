"""Readers and validators for the Netflix Prize file format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tarfile
import zipfile

import pandas as pd
import requests


COMBINED_FILENAMES = tuple(f"combined_data_{part}.txt" for part in range(1, 5))
OPTIONAL_FILENAMES = ("probe.txt", "qualifying.txt", "movie_titles.csv", "movie_titles.txt")


@dataclass(frozen=True)
class NetflixPaths:
    data_dir: Path
    combined_files: tuple[Path, ...]
    probe_file: Path
    qualifying_file: Path
    movie_titles_file: Path


def netflix_paths(data_dir: Path) -> NetflixPaths:
    """Return conventional Netflix Prize paths for a data directory."""

    data_dir = Path(data_dir)
    movie_titles_csv = data_dir / "movie_titles.csv"
    movie_titles_txt = data_dir / "movie_titles.txt"
    return NetflixPaths(
        data_dir=data_dir,
        combined_files=tuple(data_dir / name for name in COMBINED_FILENAMES),
        probe_file=data_dir / "probe.txt",
        qualifying_file=data_dir / "qualifying.txt",
        movie_titles_file=movie_titles_csv if movie_titles_csv.exists() else movie_titles_txt,
    )


def missing_netflix_files(data_dir: Path, require_probe: bool = False) -> list[Path]:
    """List required Netflix files that are absent."""

    paths = netflix_paths(data_dir)
    required = [*paths.combined_files, paths.movie_titles_file]
    if require_probe:
        required.append(paths.probe_file)
    return [path for path in required if not path.exists()]


def verify_netflix_files(data_dir: Path, require_probe: bool = False) -> dict[str, object]:
    """Return a small validation report for local Netflix Prize files."""

    paths = netflix_paths(data_dir)
    missing = missing_netflix_files(data_dir, require_probe=require_probe)
    present = [path for path in [*paths.combined_files, paths.movie_titles_file, paths.probe_file] if path.exists()]
    return {
        "data_dir": str(paths.data_dir),
        "ok": not missing,
        "missing": [str(path) for path in missing],
        "present": {path.name: path.stat().st_size for path in present},
    }


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    """Download a file from an explicit URL.

    The Netflix Prize data license restricts redistribution, so the project does
    not bake in a third-party mirror. Pass a URL only when you have obtained the
    dataset through an authorized channel.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
    return destination


def unpack_archive(archive_path: Path, data_dir: Path) -> Path:
    """Unpack a zip or tar archive containing Netflix Prize files."""

    archive_path = Path(archive_path)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    data_root = data_dir.resolve()

    def validate_member(member_name: str) -> None:
        target = (data_root / member_name).resolve()
        if not target.is_relative_to(data_root):
            raise ValueError(f"Archive member escapes target directory: {member_name}")

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            for member_name in archive.namelist():
                validate_member(member_name)
            archive.extractall(data_dir)
        return data_dir

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                validate_member(member.name)
            archive.extractall(data_dir)
        return data_dir

    raise ValueError(f"Unsupported archive type: {archive_path}")


def iter_combined_ratings(
    combined_files: list[Path] | tuple[Path, ...],
    movie_ids: set[int] | None = None,
    exclude_pairs: set[tuple[int, int]] | None = None,
):
    """Yield ``(movie_id, customer_id, rating, date)`` from combined_data files."""

    for path in combined_files:
        movie_id: int | None = None
        keep_movie = True
        with Path(path).open("rt", encoding="latin1") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    continue

                if line.endswith(":"):
                    movie_id = int(line[:-1])
                    keep_movie = movie_ids is None or movie_id in movie_ids
                    continue

                if movie_id is None or not keep_movie:
                    continue

                customer_text, rating_text, date_text = line.split(",")
                customer_id = int(customer_text)
                if exclude_pairs and (movie_id, customer_id) in exclude_pairs:
                    continue
                yield movie_id, customer_id, int(rating_text), date_text


def read_netflix_ratings(
    data_dir: Path,
    max_rows: int | None = None,
    movie_ids: set[int] | None = None,
    include_date: bool = True,
    exclude_pairs: set[tuple[int, int]] | None = None,
    combined_files: list[Path] | tuple[Path, ...] | None = None,
) -> pd.DataFrame:
    """Read Netflix ratings into a DataFrame.

    ``max_rows`` is intended for experiments and tests. Leave it as ``None`` for
    full-data runs.
    """

    paths = netflix_paths(data_dir)
    files = tuple(combined_files) if combined_files is not None else paths.combined_files
    movie_values: list[int] = []
    customer_values: list[int] = []
    rating_values: list[int] = []
    date_values: list[str] = []

    for movie_id, customer_id, rating, date_text in iter_combined_ratings(
        files,
        movie_ids=movie_ids,
        exclude_pairs=exclude_pairs,
    ):
        movie_values.append(movie_id)
        customer_values.append(customer_id)
        rating_values.append(rating)
        if include_date:
            date_values.append(date_text)
        if max_rows is not None and len(rating_values) >= max_rows:
            break

    data: dict[str, object] = {
        "movie_id": pd.Series(movie_values, dtype="int32"),
        "customer_id": pd.Series(customer_values, dtype="int32"),
        "rating": pd.Series(rating_values, dtype="int8"),
    }
    if include_date:
        data["date"] = pd.Series(date_values, dtype="string")
    return pd.DataFrame(data)


def read_movie_titles(path: Path) -> pd.DataFrame:
    """Read Netflix movie titles, preserving titles that contain commas."""

    movie_ids: list[int] = []
    years: list[int | None] = []
    titles: list[str] = []

    with Path(path).open("rt", encoding="latin1") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.lower().startswith("id,year,title"):
                continue
            parts = line.split(",", 2)
            if len(parts) != 3:
                continue
            movie_id_text, year_text, title = parts
            movie_ids.append(int(movie_id_text))
            years.append(int(year_text) if year_text.strip().isdigit() else None)
            titles.append(title)

    return pd.DataFrame(
        {
            "movie_id": pd.Series(movie_ids, dtype="int32"),
            "year": pd.Series(years, dtype="Int64"),
            "title": pd.Series(titles, dtype="string"),
        }
    )


def read_probe(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    """Read ``probe.txt`` into ``movie_id, customer_id`` rows."""

    movie_values: list[int] = []
    customer_values: list[int] = []
    movie_id: int | None = None

    with Path(path).open("rt", encoding="latin1") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            if line.endswith(":"):
                movie_id = int(line[:-1])
                continue
            if movie_id is None:
                continue
            movie_values.append(movie_id)
            customer_values.append(int(line))
            if max_rows is not None and len(customer_values) >= max_rows:
                break

    return pd.DataFrame(
        {
            "movie_id": pd.Series(movie_values, dtype="int32"),
            "customer_id": pd.Series(customer_values, dtype="int32"),
        }
    )
