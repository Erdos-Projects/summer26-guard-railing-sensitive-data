"""IMDb ratings ingestion, scraping, and title matching."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
import html
import json
import re
from pathlib import Path
import time
import unicodedata

import numpy as np
import pandas as pd
import requests


WORD_PATTERN = re.compile(r"[a-z0-9]+")
YEAR_PATTERN = re.compile(r"(\d{4})")


@dataclass(frozen=True)
class MatchConfig:
    fuzzy: bool = True
    fuzzy_threshold: float = 0.93
    fuzzy_margin: float = 0.03


def normalize_title(value: object) -> str:
    """Normalize a movie title for conservative title matching."""

    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    words = WORD_PATTERN.findall(text)
    if words and words[0] in {"a", "an", "the"}:
        words = words[1:]
    return " ".join(words)


def parse_year(value: object) -> int | None:
    """Return the first four-digit year in a value."""

    if pd.isna(value):
        return None
    match = YEAR_PATTERN.search(str(value))
    return int(match.group(1)) if match else None


def imdb_rating_to_netflix_expected(value: object) -> float:
    """Map a 10-star IMDb rating to the 1-5 Netflix scale."""

    rating = float(value)
    return float(np.clip(rating / 2.0, 1.0, 5.0))


def imdb_rating_bounds(value: object) -> tuple[int, int]:
    """Return plausible integer Netflix ratings for a 10-star IMDb rating."""

    expected = imdb_rating_to_netflix_expected(value)
    low = int(np.floor(expected))
    high = int(np.ceil(expected))
    return max(low, 1), min(high, 5)


def preference_bucket_from_netflix_rating(value: object) -> str:
    rating = float(value)
    if rating >= 4:
        return "liked"
    if rating <= 2:
        return "disliked"
    return "neutral"


def read_imdb_ratings_csv(path: Path) -> pd.DataFrame:
    """Read cached IMDb user ratings from a CSV.

    Expected columns are ``title``, ``year``, ``user``, and ``rating``. The
    reader tolerates an exported pandas index column.
    """

    frame = pd.read_csv(path)
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed:")]
    if unnamed:
        frame = frame.drop(columns=unnamed)

    required = {"title", "year", "user", "rating"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"IMDb CSV missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["title"] = frame["title"].astype("string")
    frame["user"] = frame["user"].astype("string")
    frame["year"] = frame["year"].map(parse_year).astype("Int64")
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame = frame.dropna(subset=["title", "user", "rating"]).reset_index(drop=True)
    return frame


def _coerce_movie_titles(movie_titles: pd.DataFrame) -> pd.DataFrame:
    frame = movie_titles.copy()
    if "movie_id" not in frame.columns and "id" in frame.columns:
        frame = frame.rename(columns={"id": "movie_id"})
    required = {"movie_id", "title", "year"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Netflix movie titles missing columns: {sorted(missing)}")
    frame["movie_id"] = pd.to_numeric(frame["movie_id"], errors="raise").astype("int32")
    frame["title"] = frame["title"].astype("string")
    frame["year"] = frame["year"].map(parse_year).astype("Int64")
    frame["normalized_title"] = frame["title"].map(normalize_title)
    return frame


def _unique_group_map(frame: pd.DataFrame, key_columns: list[str]) -> dict[object, int]:
    mapping: dict[object, int] = {}
    for key, group in frame.groupby(key_columns, dropna=True, sort=False):
        unique_ids = group["movie_id"].drop_duplicates().tolist()
        if len(unique_ids) == 1:
            mapping[key] = int(unique_ids[0])
    return mapping


def _best_fuzzy_match(
    normalized_title: str,
    candidates: pd.DataFrame,
    threshold: float,
    margin: float,
) -> tuple[int | None, float]:
    if not normalized_title or candidates.empty:
        return None, 0.0

    scored: list[tuple[float, int]] = []
    for row in candidates[["movie_id", "normalized_title"]].itertuples(index=False):
        score = SequenceMatcher(None, normalized_title, row.normalized_title).ratio()
        scored.append((score, int(row.movie_id)))

    scored.sort(reverse=True)
    best_score, best_movie_id = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= threshold and best_score - second_score >= margin:
        return best_movie_id, best_score
    return None, best_score


def match_imdb_to_netflix(
    imdb_ratings: pd.DataFrame,
    movie_titles: pd.DataFrame,
    config: MatchConfig | None = None,
) -> pd.DataFrame:
    """Attach Netflix movie ids to IMDb user-rating rows."""

    config = config or MatchConfig()
    netflix = _coerce_movie_titles(movie_titles)
    netflix_by_id = netflix.set_index("movie_id")
    title_year_map = _unique_group_map(netflix, ["normalized_title", "year"])
    title_map = _unique_group_map(netflix, ["normalized_title"])
    candidates_by_year = {
        int(year): group
        for year, group in netflix.dropna(subset=["year"]).groupby("year", sort=False)
    }

    rows: list[dict[str, object]] = []
    for row in imdb_ratings.itertuples(index=False):
        title = getattr(row, "title")
        year = parse_year(getattr(row, "year"))
        user = getattr(row, "user")
        imdb_rating = float(getattr(row, "rating"))
        normalized = normalize_title(title)

        movie_id: int | None = None
        method = "unmatched"
        match_score = 0.0

        if year is not None:
            key = (normalized, year)
            movie_id = title_year_map.get(key)
            if movie_id is not None:
                method = "exact_title_year"
                match_score = 1.0

        if movie_id is None:
            movie_id = title_map.get(normalized)
            if movie_id is not None:
                method = "unique_title"
                match_score = 1.0

        if movie_id is None and config.fuzzy:
            candidates = candidates_by_year.get(year, pd.DataFrame()) if year is not None else netflix
            movie_id, match_score = _best_fuzzy_match(
                normalized,
                candidates,
                threshold=config.fuzzy_threshold,
                margin=config.fuzzy_margin,
            )
            if movie_id is not None:
                method = "fuzzy_title"

        low, high = imdb_rating_bounds(imdb_rating)
        matched = netflix_by_id.loc[movie_id] if movie_id is not None else None
        rows.append(
            {
                "user": user,
                "imdb_title": title,
                "imdb_year": year,
                "imdb_rating": imdb_rating,
                "netflix_movie_id": movie_id,
                "netflix_title": None if matched is None else matched["title"],
                "netflix_year": None if matched is None else parse_year(matched["year"]),
                "match_method": method,
                "match_score": match_score,
                "expected_netflix_rating": imdb_rating_to_netflix_expected(imdb_rating),
                "rating_low": low,
                "rating_high": high,
                "preference": preference_bucket_from_netflix_rating((low + high) / 2),
            }
        )

    matched_frame = pd.DataFrame(rows)
    if not matched_frame.empty:
        matched_frame["netflix_movie_id"] = matched_frame["netflix_movie_id"].astype("Int64")
    return matched_frame


def build_imdb_ratings_url(identifier: str) -> str:
    """Build an IMDb ratings URL from a full URL, user id, or username-like id."""

    if identifier.startswith("http://") or identifier.startswith("https://"):
        return identifier
    cleaned = identifier.strip("/")
    if cleaned.startswith("user/"):
        cleaned = cleaned.split("/", 1)[1]
    return f"https://www.imdb.com/user/{cleaned}/ratings"


def _json_script_payloads(page_html: str) -> Iterable[object]:
    pattern = re.compile(
        r"<script[^>]+(?:id=\"__NEXT_DATA__\"|type=\"application/ld\+json\")[^>]*>(.*?)</script>",
        flags=re.DOTALL,
    )
    for match in pattern.finditer(page_html):
        payload = html.unescape(match.group(1)).strip()
        if not payload:
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def _walk_json(value: object):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _text_from_json_field(value: object) -> str | None:
    if isinstance(value, dict):
        if "text" in value:
            return str(value["text"])
        if "plainText" in value:
            return str(value["plainText"])
    if isinstance(value, str):
        return value
    return None


def _extract_json_rows(page_html: str, username: str | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int | None, float]] = set()
    for payload in _json_script_payloads(page_html):
        for item in _walk_json(payload):
            title = _text_from_json_field(item.get("titleText") or item.get("originalTitleText"))
            user_rating = item.get("userRating") or item.get("rating")
            if isinstance(user_rating, dict):
                user_rating = user_rating.get("value") or user_rating.get("rating")
            release_year = item.get("releaseYear") or item.get("year")
            if isinstance(release_year, dict):
                release_year = release_year.get("year")
            if not title or user_rating is None:
                continue
            try:
                rating = float(user_rating)
            except (TypeError, ValueError):
                continue
            year = parse_year(release_year)
            key = (title, year, rating)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"title": title, "year": year, "user": username, "rating": rating})
    return rows


def _extract_bs4_rows(page_html: str, username: str | None) -> list[dict[str, object]]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(page_html, "html.parser")
    rows: list[dict[str, object]] = []
    for item in soup.select("li.ipc-metadata-list-summary-item"):
        title_node = item.select_one(".ipc-title__text")
        rating_nodes = item.select(".ipc-rating-star--rating")
        year_node = item.select_one(".ipc-inline-list__item")
        if title_node is None or not rating_nodes:
            continue
        title = title_node.get_text(" ", strip=True)
        title = re.sub(r"^\d+\.\s*", "", title)
        rating_text = rating_nodes[-1].get_text(" ", strip=True)
        try:
            rating = float(rating_text)
        except ValueError:
            continue
        rows.append(
            {
                "title": title,
                "year": parse_year(year_node.get_text(" ", strip=True) if year_node else None),
                "user": username,
                "rating": rating,
            }
        )
    return rows


def extract_imdb_user_ratings(page_html: str, username: str | None = None) -> pd.DataFrame:
    """Extract IMDb rating rows from a fetched ratings page."""

    rows = _extract_bs4_rows(page_html, username=username)
    if not rows:
        rows = _extract_json_rows(page_html, username=username)
    return pd.DataFrame(rows, columns=["title", "year", "user", "rating"])


def scrape_imdb_user_ratings(
    identifier: str,
    username: str | None = None,
    sleep_seconds: float = 1.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch and parse the first IMDb ratings page for a public user.

    IMDb changes its markup frequently and may block automated traffic. Prefer
    cached CSVs for reproducible experiments, and use this helper sparingly with
    public pages only.
    """

    url = build_imdb_ratings_url(identifier)
    request_session = session or requests.Session()
    response = request_session.get(
        url,
        params={"sort": "num_votes,desc"},
        headers={
            "User-Agent": "guard-rails-sensitive-data/0.1 research scraper",
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=30,
    )
    response.raise_for_status()
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    frame = extract_imdb_user_ratings(response.text, username=username or identifier)
    if frame.empty:
        raise ValueError(
            "No IMDb ratings were parsed. Supply a cached CSV or inspect whether the "
            "ratings page requires login/JavaScript."
        )
    return frame
