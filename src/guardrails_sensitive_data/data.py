"""Data loading, IMDb matching, and synthetic Netflix-like data."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from html import unescape
from html.parser import HTMLParser
import json
import re
from pathlib import Path
import time
import unicodedata
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


COMBINED_FILENAMES = tuple(f"combined_data_{part}.txt" for part in range(1, 5))
WORD_PATTERN = re.compile(r"[a-z0-9]+")
YEAR_PATTERN = re.compile(r"(\d{4})")
IMDB_BASE_URL = "https://www.imdb.com"
IMDB_RATINGS_COLUMNS = ["title", "year", "user", "rating"]
IMDB_PROFILE_DATE_KEYS = {
    "accountCreated",
    "accountCreationDate",
    "created",
    "createdAt",
    "createdDate",
    "dateCreated",
    "dateJoined",
    "joined",
    "joinedDate",
    "memberSince",
    "memberSinceDate",
    "registrationDate",
    "userSince",
}
IMDB_PROFILE_DATE_KEYS_LOWER = {key.lower() for key in IMDB_PROFILE_DATE_KEYS}


@dataclass(frozen=True)
class NetflixPaths:
    data_dir: Path
    combined_files: tuple[Path, ...]
    probe_file: Path
    qualifying_file: Path
    movie_titles_file: Path


@dataclass(frozen=True)
class MatchConfig:
    fuzzy: bool = True
    fuzzy_threshold: float = 0.93
    fuzzy_margin: float = 0.03


@dataclass(frozen=True)
class SyntheticProfile:
    target_user: int
    profile_label: str
    imdb_ratings: pd.DataFrame


@dataclass(frozen=True)
class ImdbScrapeResult:
    ratings: pd.DataFrame
    source_url: str
    pages_fetched: int
    profile_created_year: int | None = None
    earliest_rating_year: int | None = None
    warnings: tuple[str, ...] = ()


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
    """Read Netflix ratings into a DataFrame."""

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
            if not line or line.lower().startswith("id,year,title"):
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

    if value is None:
        return None
    if not isinstance(value, (dict, list, tuple, set)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    match = YEAR_PATTERN.search(str(value))
    return int(match.group(1)) if match else None


def imdb_rating_to_netflix_expected(value: object) -> float:
    """Map a 10-star IMDb rating to the 1-5 Netflix scale."""

    return float(np.clip(float(value) / 2.0, 1.0, 5.0))


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
    """Read cached IMDb user ratings from a CSV."""

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
    return frame.dropna(subset=["title", "user", "rating"]).reset_index(drop=True)


def empty_imdb_ratings_frame() -> pd.DataFrame:
    """Return an empty IMDb ratings frame with the maintained schema."""

    return pd.DataFrame(
        {
            "title": pd.Series(dtype="string"),
            "year": pd.Series(dtype="Int64"),
            "user": pd.Series(dtype="string"),
            "rating": pd.Series(dtype="float64"),
        }
    )


def _looks_like_imdb_url(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return False
    parsed = urlparse(text if "://" in text else f"https://{text}")
    return "imdb.com" in parsed.netloc.lower()


def _with_default_imdb_sort(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    query.setdefault("sort", ["num_votes,desc"])
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True), fragment=""))


def imdb_identifier_to_user_label(identifier: str) -> str:
    """Return a stable local label for an IMDb username, user id, or profile URL."""

    text = str(identifier).strip()
    if not _looks_like_imdb_url(text):
        return text.strip("/") or "imdb_profile"

    parsed = urlparse(text if "://" in text else f"https://{text}")
    parts = [part for part in parsed.path.split("/") if part]
    if "user" in parts:
        user_index = parts.index("user")
        if user_index + 1 < len(parts):
            return parts[user_index + 1]
    for part in reversed(parts):
        if part != "ratings":
            return part
    return "imdb_profile"


def imdb_ratings_url_candidates(identifier: str) -> tuple[str, ...]:
    """Return likely IMDb ratings URLs for a username, user id, or profile URL."""

    text = str(identifier).strip()
    if not text:
        raise ValueError("IMDb identifier cannot be empty.")

    if _looks_like_imdb_url(text):
        parsed = urlparse(text if "://" in text else f"https://{text}")
        parts = [part for part in parsed.path.split("/") if part]
        path = parsed.path.rstrip("/")
        if "user" in parts and parts.index("user") + 1 < len(parts):
            user_id = parts[parts.index("user") + 1]
            path = f"/user/{user_id}/ratings"
        elif "profile" in parts and parts.index("profile") + 1 < len(parts):
            profile_id = parts[parts.index("profile") + 1]
            path = f"/profile/{profile_id}/ratings"
        else:
            user_match = re.search(r"(/user/[^/]+)", path)
            if user_match:
                path = user_match.group(1)
            if not path.endswith("/ratings"):
                path = f"{path}/ratings"
            if not path.startswith("/"):
                path = f"/{path}"
        normalized = urlunparse(("https", "www.imdb.com", f"{path.rstrip('/')}/", "", "", ""))
        return (_with_default_imdb_sort(normalized),)

    cleaned = text.strip("/")
    quoted = quote(cleaned)
    urls = [f"{IMDB_BASE_URL}/user/{quoted}/ratings/"]
    if not re.fullmatch(r"(?:ur\d+|p\.[A-Za-z0-9_.-]+)", cleaned):
        urls.append(f"{IMDB_BASE_URL}/profile/{quoted}/ratings/")
    return tuple(_with_default_imdb_sort(url) for url in urls)


def _fetch_url_text(url: str, timeout: float = 20.0) -> str:
    request = Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0 Safari/537.36"
            ),
        },
    )
    with urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise ValueError(f"IMDb returned HTTP {status}; live scraping may be blocked.")
        charset = response.headers.get_content_charset() or "utf-8"
        text = response.read().decode(charset, errors="replace")
        if not text.strip():
            raise ValueError("IMDb returned an empty response; live scraping may be blocked.")
        return text


class _JsonScriptParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[str] = []
        self._capturing = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "script":
            return
        attr_map = {key.lower(): value or "" for key, value in attrs}
        script_type = attr_map.get("type", "").lower()
        script_id = attr_map.get("id", "")
        if script_id == "__NEXT_DATA__" or "json" in script_type or not script_type:
            self._capturing = True
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capturing:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "script" or not self._capturing:
            return
        payload = unescape("".join(self._parts)).strip()
        if payload.startswith(("{", "[")):
            self.scripts.append(payload)
        self._capturing = False
        self._parts = []


def _json_payloads_from_html(page_html: str) -> list[object]:
    parser = _JsonScriptParser()
    parser.feed(page_html)
    payloads: list[object] = []
    for script in parser.scripts:
        try:
            payloads.append(json.loads(script))
        except json.JSONDecodeError:
            continue
    return payloads


def _walk_json(value: object):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _coerce_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = unescape(value).strip()
        return text or None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        for key in ("text", "plainText", "name", "title", "titleText"):
            if key in value:
                text = _coerce_text(value[key])
                if text:
                    return text
    if isinstance(value, list):
        for child in value:
            text = _coerce_text(child)
            if text:
                return text
    return None


def _coerce_rating(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        rating = float(value)
        return rating if 0 < rating <= 10 else None
    if isinstance(value, str):
        match = re.search(r"\b(10|[1-9])(?:\.0+)?\b", value)
        if match:
            return float(match.group(1))
        return None
    if isinstance(value, dict):
        for key in ("rating", "value", "ratingValue", "score"):
            if key in value:
                rating = _coerce_rating(value[key])
                if rating is not None:
                    return rating
    if isinstance(value, list):
        for child in value:
            rating = _coerce_rating(child)
            if rating is not None:
                return rating
    return None


def _extract_user_rating_value(item: dict[str, object]) -> float | None:
    for key in (
        "userRating",
        "otherUserRating",
        "yourRating",
        "viewerRating",
        "ratingByUser",
        "userRatingValue",
        "personalRating",
    ):
        if key in item:
            rating = _coerce_rating(item[key])
            if rating is not None:
                return rating
    return None


def _extract_title_value(item: dict[str, object]) -> str | None:
    for key in ("titleText", "originalTitleText", "primaryTitle", "displayTitle", "title", "name"):
        if key in item:
            title = _coerce_text(item[key])
            if title:
                return re.sub(r"^\s*\d+\.\s*", "", title).strip()
    for key in ("node", "item", "titleNode"):
        child = item.get(key)
        if isinstance(child, dict):
            title = _extract_title_value(child)
            if title:
                return title
    return None


def _extract_year_value(item: dict[str, object]) -> int | None:
    for key in ("releaseYear", "year", "startYear", "titleReleaseText"):
        if key in item:
            year = parse_year(item[key])
            if year is not None:
                return year
    for key in ("node", "item", "title", "titleNode"):
        child = item.get(key)
        if isinstance(child, dict):
            year = _extract_year_value(child)
            if year is not None:
                return year
    return None


class _ImdbRatingsHTMLParser(HTMLParser):
    def __init__(self, username: str) -> None:
        super().__init__()
        self.username = username
        self.rows: list[dict[str, object]] = []
        self._in_item = False
        self._item_depth = 0
        self._rating_group_depth = 0
        self._active_field: str | None = None
        self._active_tag: str | None = None
        self._field_parts: list[str] = []
        self._current: dict[str, object] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attr_map = {key.lower(): value or "" for key, value in attrs}
        class_name = attr_map.get("class", "")

        if tag == "li" and "ipc-metadata-list-summary-item" in class_name:
            self._current = {}
            self._in_item = True
            self._item_depth = 1
            self._rating_group_depth = 0
            return

        if not self._in_item:
            return

        self._item_depth += 1
        if self._rating_group_depth:
            self._rating_group_depth += 1

        if tag in {"h3", "h4"} and "ipc-title__text" in class_name:
            self._start_field("title", tag)
        elif tag == "li" and "ipc-inline-list__item" in class_name and "year" not in self._current:
            self._start_field("year", tag)
        elif tag == "span" and attr_map.get("data-testid") == "ratingGroup--other-user-rating":
            self._rating_group_depth = 1
        elif (
            tag == "span"
            and self._rating_group_depth
            and "ipc-rating-star--rating" in class_name
            and "rating" not in self._current
        ):
            self._start_field("rating", tag)

    def handle_data(self, data: str) -> None:
        if self._active_field:
            self._field_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._active_field and tag == self._active_tag:
            self._finish_field()

        if not self._in_item:
            return

        if self._rating_group_depth:
            self._rating_group_depth -= 1
        self._item_depth -= 1
        if self._item_depth <= 0:
            self._finish_item()

    def _start_field(self, field: str, tag: str) -> None:
        self._active_field = field
        self._active_tag = tag
        self._field_parts = []

    def _finish_field(self) -> None:
        if self._active_field:
            value = unescape(" ".join(self._field_parts)).strip()
            if value:
                self._current[self._active_field] = value
        self._active_field = None
        self._active_tag = None
        self._field_parts = []

    def _finish_item(self) -> None:
        title = _coerce_text(self._current.get("title"))
        rating = _coerce_rating(self._current.get("rating"))
        if title and rating is not None:
            self.rows.append(
                {
                    "title": re.sub(r"^\s*\d+\.\s*", "", title).strip(),
                    "year": parse_year(self._current.get("year")),
                    "user": self.username,
                    "rating": rating,
                }
            )
        self._current = {}
        self._in_item = False
        self._item_depth = 0
        self._rating_group_depth = 0


def _clean_imdb_ratings_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return empty_imdb_ratings_frame()
    frame = pd.DataFrame(rows, columns=IMDB_RATINGS_COLUMNS)
    frame["title"] = frame["title"].astype("string").str.strip()
    frame["user"] = frame["user"].astype("string").str.strip()
    frame["year"] = frame["year"].map(parse_year).astype("Int64")
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame = frame.dropna(subset=["title", "user", "rating"])
    frame = frame[(frame["title"] != "") & frame["rating"].between(1, 10)]
    return frame.drop_duplicates(subset=IMDB_RATINGS_COLUMNS).reset_index(drop=True)


def extract_imdb_user_ratings(page_html: str, username: str = "imdb_profile") -> pd.DataFrame:
    """Parse IMDb user ratings from a ratings page HTML payload."""

    rows: list[dict[str, object]] = []
    for payload in _json_payloads_from_html(page_html):
        for item in _walk_json(payload):
            rating = _extract_user_rating_value(item)
            if rating is None:
                continue
            title = _extract_title_value(item)
            if not title:
                continue
            rows.append(
                {
                    "title": title,
                    "year": _extract_year_value(item),
                    "user": username,
                    "rating": rating,
                }
            )

    html_parser = _ImdbRatingsHTMLParser(username=username)
    html_parser.feed(page_html)
    rows.extend(html_parser.rows)
    return _clean_imdb_ratings_frame(rows)


def extract_imdb_profile_year(page_html: str) -> int | None:
    """Best-effort extraction of an IMDb profile creation/join year."""

    for payload in _json_payloads_from_html(page_html):
        for item in _walk_json(payload):
            for key, value in item.items():
                if key.lower() in IMDB_PROFILE_DATE_KEYS_LOWER:
                    year = parse_year(value)
                    if year is not None:
                        return year

    plain = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", page_html)))
    match = re.search(
        r"(?:member since|joined|created|date created|registration date|user since)"
        r".{0,80}?((?:19|20)\d{2})",
        plain,
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def extract_imdb_earliest_rating_year(page_html: str) -> int | None:
    """Best-effort extraction of the earliest visible IMDb rating activity year."""

    plain = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", page_html)))
    years = [int(match.group(1)) for match in re.finditer(r"Rated on .{3,30}?((?:19|20)\d{2})", plain)]
    return min(years) if years else None


def _imdb_browser_failure_reason(page_html: str, title: str = "") -> str | None:
    plain = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", page_html))).strip()
    combined = f"{title} {plain}".strip()
    lowered = combined.lower()
    if "403 forbidden" in lowered:
        return "IMDb returned 403 Forbidden in the browser; the profile may be private, unavailable, or blocked."
    if "404" in lowered and "not found" in lowered:
        return "IMDb returned a not-found page for this profile."
    if "robot check" in lowered or "captcha" in lowered:
        return "IMDb showed a robot/captcha challenge in the browser."
    if "sign in" in lowered and "ratings" not in lowered and len(plain) < 2_000:
        return "IMDb did not render a public ratings page; the profile may require sign-in or may not expose ratings."
    return None


def find_next_imdb_ratings_url(page_html: str, current_url: str) -> str | None:
    """Return the next ratings page URL when IMDb exposes it as a link."""

    for match in re.finditer(r"<a\b[^>]*>", page_html, flags=re.IGNORECASE):
        tag = match.group(0)
        if not re.search(r"(aria-label=['\"]Next|data-testid=['\"][^'\"]*next|pagination[^>]*next)", tag, re.I):
            continue
        if re.search(r"aria-disabled=['\"]true|disabled", tag, re.I):
            continue
        href = re.search(r"href=['\"]([^'\"]+)['\"]", tag)
        if href:
            return urljoin(current_url, unescape(href.group(1)))
    return None


def extract_imdb_user_ratings_with_beautifulsoup(
    page_html: str,
    username: str = "imdb_profile",
) -> pd.DataFrame:
    """Parse IMDb user-rating rows with BeautifulSoup selectors."""

    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ValueError("BeautifulSoup parsing requires: python -m pip install -e '.[scrape]'") from exc

    soup = BeautifulSoup(page_html, "html.parser")
    rows: list[dict[str, object]] = []
    for item in soup.select("li.ipc-metadata-list-summary-item"):
        title_node = item.select_one("h3.ipc-title__text, h4.ipc-title__text")
        year_node = item.select_one("li.ipc-inline-list__item")
        rating_group = item.select_one('[data-testid="ratingGroup--other-user-rating"]')

        if rating_group is not None:
            rating_nodes = rating_group.select(".ipc-rating-star--rating")
        else:
            rating_nodes = item.select(".ipc-rating-star--rating")

        title = _coerce_text(title_node.get_text(" ", strip=True) if title_node else None)
        rating = _coerce_rating(rating_nodes[-1].get_text(" ", strip=True) if rating_nodes else None)
        if not title or rating is None:
            continue

        rows.append(
            {
                "title": re.sub(r"^\s*\d+\.\s*", "", title).strip(),
                "year": parse_year(year_node.get_text(" ", strip=True) if year_node else None),
                "user": username,
                "rating": rating,
            }
        )

    return _clean_imdb_ratings_frame(rows)


def _scrape_imdb_user_ratings_from_fetcher(
    identifier: str,
    *,
    username: str,
    max_pages: int,
    sleep_seconds: float,
    fetcher: Callable[[str], str],
) -> ImdbScrapeResult:
    failures: list[str] = []

    for initial_url in imdb_ratings_url_candidates(identifier):
        rows: list[pd.DataFrame] = []
        warnings: list[str] = []
        pages_fetched = 0
        profile_year: int | None = None
        earliest_rating_year: int | None = None
        seen_urls: set[str] = set()
        next_url: str | None = initial_url

        try:
            while next_url and pages_fetched < max_pages and next_url not in seen_urls:
                seen_urls.add(next_url)
                page_html = fetcher(next_url)
                pages_fetched += 1
                if profile_year is None:
                    profile_year = extract_imdb_profile_year(page_html)
                page_earliest_rating_year = extract_imdb_earliest_rating_year(page_html)
                if page_earliest_rating_year is not None:
                    earliest_rating_year = (
                        page_earliest_rating_year
                        if earliest_rating_year is None
                        else min(earliest_rating_year, page_earliest_rating_year)
                    )
                page_ratings = extract_imdb_user_ratings(page_html, username=username)
                if not page_ratings.empty:
                    rows.append(page_ratings)
                next_link = find_next_imdb_ratings_url(page_html, next_url)
                next_url = _with_default_imdb_sort(next_link) if next_link else None
                if next_url and sleep_seconds > 0:
                    time.sleep(sleep_seconds)
        except Exception as exc:
            failures.append(f"{initial_url}: {type(exc).__name__}: {exc}")
            continue

        if rows:
            ratings = _clean_imdb_ratings_frame(pd.concat(rows, ignore_index=True).to_dict("records"))
            if next_url:
                warnings.append(f"Stopped after --imdb-max-pages={max_pages}; more IMDb ratings pages may exist.")
            return ImdbScrapeResult(
                ratings=ratings,
                source_url=initial_url,
                pages_fetched=pages_fetched,
                profile_created_year=profile_year,
                earliest_rating_year=earliest_rating_year,
                warnings=tuple(warnings),
            )
        failures.append(f"{initial_url}: fetched {pages_fetched} page(s), but parsed no rating rows")

    failure_text = "; ".join(failures) if failures else "no IMDb URLs were attempted"
    raise ValueError(f"IMDb HTTP scrape failed for {identifier!r}: {failure_text}")


def _new_selenium_driver(headless: bool = True):
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError as exc:
        raise ValueError("Selenium browser scraping requires: python -m pip install -e '.[scrape]'") from exc

    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.page_load_strategy = "eager"
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--lang=en-US")
    options.add_argument("--window-size=1400,1800")
    options.add_argument(
        "--user-agent="
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)


def _scrape_imdb_user_ratings_with_selenium(
    identifier: str,
    *,
    username: str,
    max_pages: int,
    sleep_seconds: float,
    wait_seconds: float,
    headless: bool,
) -> ImdbScrapeResult:
    try:
        from selenium.common.exceptions import TimeoutException, WebDriverException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError as exc:
        raise ValueError("Selenium browser scraping requires: python -m pip install -e '.[scrape]'") from exc

    failures: list[str] = []
    driver = _new_selenium_driver(headless=headless)
    driver.set_page_load_timeout(wait_seconds)
    try:
        for initial_url in imdb_ratings_url_candidates(identifier):
            rows: list[pd.DataFrame] = []
            warnings: list[str] = []
            pages_fetched = 0
            profile_year: int | None = None
            earliest_rating_year: int | None = None

            try:
                try:
                    driver.get(initial_url)
                except TimeoutException:
                    warnings.append("Timed out loading IMDb page; parsing current page source.")
                failure_reason = _imdb_browser_failure_reason(driver.page_source, driver.title)
                if failure_reason:
                    raise ValueError(f"{failure_reason} URL: {driver.current_url}")
                wait = WebDriverWait(driver, wait_seconds)
                while pages_fetched < max_pages:
                    try:
                        wait.until(
                            EC.presence_of_element_located(
                                (By.CSS_SELECTOR, "li.ipc-metadata-list-summary-item")
                            )
                        )
                    except TimeoutException:
                        warnings.append("Timed out waiting for IMDb ratings to render; parsing current page source.")

                    page_html = driver.page_source
                    failure_reason = _imdb_browser_failure_reason(page_html, driver.title)
                    if failure_reason:
                        raise ValueError(f"{failure_reason} URL: {driver.current_url}")
                    pages_fetched += 1
                    if profile_year is None:
                        profile_year = extract_imdb_profile_year(page_html)
                    page_earliest_rating_year = extract_imdb_earliest_rating_year(page_html)
                    if page_earliest_rating_year is not None:
                        earliest_rating_year = (
                            page_earliest_rating_year
                            if earliest_rating_year is None
                            else min(earliest_rating_year, page_earliest_rating_year)
                        )

                    page_ratings = extract_imdb_user_ratings_with_beautifulsoup(page_html, username=username)
                    if page_ratings.empty:
                        page_ratings = extract_imdb_user_ratings(page_html, username=username)
                    if not page_ratings.empty:
                        rows.append(page_ratings)

                    next_buttons = driver.find_elements(
                        By.CSS_SELECTOR,
                        "button[data-testid='index-pagination-nxt'], a[aria-label='Next']",
                    )
                    next_button = next_buttons[0] if next_buttons else None
                    if next_button is None:
                        break
                    disabled = (next_button.get_attribute("aria-disabled") or "").lower() == "true"
                    disabled = disabled or next_button.get_attribute("disabled") is not None
                    if disabled:
                        break
                    if pages_fetched >= max_pages:
                        warnings.append(f"Stopped after --imdb-max-pages={max_pages}; more IMDb ratings pages may exist.")
                        break

                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    driver.execute_script("arguments[0].click();", next_button)
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)

            except (WebDriverException, ValueError) as exc:
                failures.append(f"{initial_url}: {type(exc).__name__}: {exc}")
                continue

            if rows:
                ratings = _clean_imdb_ratings_frame(pd.concat(rows, ignore_index=True).to_dict("records"))
                return ImdbScrapeResult(
                    ratings=ratings,
                    source_url=initial_url,
                    pages_fetched=pages_fetched,
                    profile_created_year=profile_year,
                    earliest_rating_year=earliest_rating_year,
                    warnings=tuple(warnings),
                )
            failures.append(f"{initial_url}: browser fetched {pages_fetched} page(s), but parsed no rating rows")
    finally:
        driver.quit()

    failure_text = "; ".join(failures) if failures else "no IMDb URLs were attempted"
    raise ValueError(f"IMDb Selenium scrape failed for {identifier!r}: {failure_text}")


def scrape_imdb_user_ratings(
    identifier: str,
    *,
    username: str | None = None,
    max_pages: int = 5,
    sleep_seconds: float = 1.0,
    fetch_method: str = "auto",
    browser_headless: bool = True,
    browser_wait_seconds: float = 15.0,
    fetcher: Callable[[str], str] | None = None,
) -> ImdbScrapeResult:
    """Fetch and parse a public IMDb ratings profile at runtime.

    IMDb may block automated traffic or change markup. This helper is therefore
    intentionally best-effort and reports parse/fetch failures to the caller.
    ``fetch_method="auto"`` tries plain HTTP first, then Selenium browser
    rendering if the HTTP path fails.
    """

    if max_pages <= 0:
        raise ValueError("max_pages must be positive.")
    if browser_wait_seconds <= 0:
        raise ValueError("browser_wait_seconds must be positive.")
    method = fetch_method.lower()
    if method not in {"auto", "http", "browser"}:
        raise ValueError("fetch_method must be one of: auto, http, browser.")
    if fetcher is not None and method == "auto":
        method = "http"

    label = username or imdb_identifier_to_user_label(identifier)
    http_error: ValueError | None = None

    if method in {"auto", "http"}:
        try:
            return _scrape_imdb_user_ratings_from_fetcher(
                identifier,
                username=label,
                max_pages=max_pages,
                sleep_seconds=sleep_seconds,
                fetcher=fetcher or _fetch_url_text,
            )
        except ValueError as exc:
            if method == "http":
                raise ValueError(f"IMDb scrape failed for {identifier!r}: {exc}") from exc
            http_error = exc

    try:
        result = _scrape_imdb_user_ratings_with_selenium(
            identifier,
            username=label,
            max_pages=max_pages,
            sleep_seconds=sleep_seconds,
            wait_seconds=browser_wait_seconds,
            headless=browser_headless,
        )
    except ValueError as exc:
        if http_error is not None:
            raise ValueError(
                f"IMDb scrape failed for {identifier!r}: HTTP attempt failed ({http_error}); "
                f"Selenium attempt failed ({exc})"
            ) from exc
        raise ValueError(f"IMDb scrape failed for {identifier!r}: {exc}") from exc

    if http_error is None:
        return result
    return ImdbScrapeResult(
        ratings=result.ratings,
        source_url=result.source_url,
        pages_fetched=result.pages_fetched,
        profile_created_year=result.profile_created_year,
        earliest_rating_year=result.earliest_rating_year,
        warnings=(f"HTTP scrape failed first: {http_error}", *result.warnings),
    )


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
            movie_id = title_year_map.get((normalized, year))
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


def make_synthetic_movie_titles(n_movies: int = 160, seed: int = 333) -> pd.DataFrame:
    """Create deterministic synthetic movie metadata."""

    if n_movies <= 0:
        raise ValueError("n_movies must be positive.")

    rng = np.random.default_rng(seed)
    adjectives = np.array(["Crimson", "Silent", "Electric", "Hidden", "Last", "Golden", "Midnight"])
    nouns = np.array(["Signal", "Harbor", "Orbit", "Archive", "Summer", "Equation", "Witness"])
    years = rng.integers(1980, 2006, size=n_movies)
    titles = [f"{rng.choice(adjectives)} {rng.choice(nouns)} {movie_id:03d}" for movie_id in range(1, n_movies + 1)]
    return pd.DataFrame(
        {
            "movie_id": pd.Series(np.arange(1, n_movies + 1), dtype="int32"),
            "year": pd.Series(years, dtype="Int64"),
            "title": pd.Series(titles, dtype="string"),
        }
    )


def make_synthetic_netflix_ratings(
    n_users: int = 600,
    n_movies: int = 160,
    mean_ratings_per_user: int = 45,
    seed: int = 333,
) -> pd.DataFrame:
    """Generate a sparse, Netflix-like user/movie/rating/date table."""

    if n_users <= 0 or n_movies <= 0 or mean_ratings_per_user <= 0:
        raise ValueError("n_users, n_movies, and mean_ratings_per_user must be positive.")

    rng = np.random.default_rng(seed)
    latent_dim = 6
    customer_ids = np.arange(100_001, 100_001 + n_users, dtype=np.int32)
    movie_ids = np.arange(1, n_movies + 1, dtype=np.int32)
    movie_popularity = rng.lognormal(mean=0.0, sigma=1.1, size=n_movies)
    movie_probabilities = movie_popularity / movie_popularity.sum()
    movie_quality = rng.normal(0.0, 0.55, size=n_movies)
    user_bias = rng.normal(0.0, 0.35, size=n_users)
    user_factors = rng.normal(0.0, 0.55, size=(n_users, latent_dim))
    movie_factors = rng.normal(0.0, 0.55, size=(n_movies, latent_dim))
    start_date = pd.Timestamp("2001-01-01")

    rows: list[tuple[int, int, int, str]] = []
    for user_index, customer_id in enumerate(customer_ids):
        history_size = int(np.clip(rng.poisson(mean_ratings_per_user), 8, n_movies))
        rated_movies = rng.choice(movie_ids, size=history_size, replace=False, p=movie_probabilities)
        for movie_id in rated_movies:
            movie_index = int(movie_id) - 1
            affinity = float(user_factors[user_index] @ movie_factors[movie_index] / np.sqrt(latent_dim))
            raw_rating = 3.45 + user_bias[user_index] + movie_quality[movie_index] + affinity
            raw_rating += rng.normal(0.0, 0.85)
            rating = int(np.clip(np.rint(raw_rating), 1, 5))
            day_offset = int(rng.integers(0, 365 * 5))
            date_text = (start_date + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
            rows.append((int(customer_id), int(movie_id), rating, date_text))

    frame = pd.DataFrame(rows, columns=["customer_id", "movie_id", "rating", "date"])
    return frame.astype({"customer_id": "int32", "movie_id": "int32", "rating": "int8"}).sort_values(
        ["movie_id", "customer_id"],
        ignore_index=True,
    )


def make_synthetic_imdb_profile(
    ratings: pd.DataFrame,
    movie_titles: pd.DataFrame,
    *,
    target_user: int | None = None,
    n_known: int = 18,
    profile_label: str = "synthetic_public_profile",
    rating_noise_probability: float = 0.15,
    seed: int = 333,
) -> SyntheticProfile:
    """Create a noisy external profile sampled from one synthetic user."""

    if not 0 <= rating_noise_probability <= 1:
        raise ValueError("rating_noise_probability must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    user_counts = ratings["customer_id"].value_counts()
    eligible_users = user_counts[user_counts >= n_known].index.to_numpy()
    if len(eligible_users) == 0:
        raise ValueError("No synthetic users have enough ratings for the requested profile size.")
    if target_user is None:
        target_user = int(rng.choice(eligible_users))
    elif target_user not in set(map(int, eligible_users)):
        raise ValueError(f"target_user={target_user} does not have at least {n_known} ratings.")

    history = ratings[ratings["customer_id"] == target_user]
    known_rows = history.sample(n=n_known, random_state=int(rng.integers(0, np.iinfo(np.int32).max)))
    known_rows = known_rows.merge(movie_titles, on="movie_id", how="left")

    imdb_ratings = known_rows[["title", "year"]].copy()
    imdb_values = known_rows["rating"].to_numpy(dtype=np.float32) * 2.0
    noisy_mask = rng.random(len(imdb_values)) < rating_noise_probability
    imdb_values[noisy_mask] += rng.choice([-2, -1, 1, 2], size=int(noisy_mask.sum()))
    imdb_ratings["rating"] = np.clip(np.rint(imdb_values), 1, 10).astype("int8")
    imdb_ratings["user"] = profile_label
    imdb_ratings = imdb_ratings[["title", "year", "user", "rating"]].reset_index(drop=True)
    return SyntheticProfile(target_user=int(target_user), profile_label=profile_label, imdb_ratings=imdb_ratings)
