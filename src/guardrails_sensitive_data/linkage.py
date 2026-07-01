"""Probabilistic record linkage attack from IMDb ratings to Netflix users."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .imdb import MatchConfig, match_imdb_to_netflix
from .netflix_io import iter_combined_ratings, netflix_paths


@dataclass(frozen=True)
class LinkageConfig:
    rating_sigma: float = 0.75
    missing_log_penalty: float = -0.75
    min_matches: int = 2
    top_n: int = 50


def prepare_linkage_facts(matched_ratings: pd.DataFrame) -> pd.DataFrame:
    """Keep matched IMDb rows and collapse duplicate Netflix movie ids."""

    facts = matched_ratings.dropna(subset=["netflix_movie_id"]).copy()
    if facts.empty:
        return facts

    facts["netflix_movie_id"] = facts["netflix_movie_id"].astype("int32")
    facts = facts.sort_values(["netflix_movie_id", "match_score"], ascending=[True, False])
    collapsed = (
        facts.groupby("netflix_movie_id", as_index=False)
        .agg(
            imdb_title=("imdb_title", "first"),
            netflix_title=("netflix_title", "first"),
            imdb_rating=("imdb_rating", "mean"),
            expected_netflix_rating=("expected_netflix_rating", "mean"),
            rating_low=("rating_low", "min"),
            rating_high=("rating_high", "max"),
            match_method=("match_method", "first"),
            match_score=("match_score", "max"),
        )
        .sort_values(["match_score", "netflix_movie_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return collapsed


def score_netflix_candidates(
    facts: pd.DataFrame,
    data_dir: Path,
    config: LinkageConfig | None = None,
) -> pd.DataFrame:
    """Score Netflix customers against matched external IMDb rating facts."""

    config = config or LinkageConfig()
    if facts.empty:
        raise ValueError("No matched IMDb/Netflix facts are available for linkage.")

    paths = netflix_paths(data_dir)
    fact_lookup = {
        int(row.netflix_movie_id): {
            "expected": float(row.expected_netflix_rating),
            "low": int(row.rating_low),
            "high": int(row.rating_high),
        }
        for row in facts.itertuples(index=False)
    }
    movie_ids = set(fact_lookup)
    n_facts = len(movie_ids)

    log_score: defaultdict[int, float] = defaultdict(float)
    squared_error: defaultdict[int, float] = defaultdict(float)
    abs_error: defaultdict[int, float] = defaultdict(float)
    match_count: defaultdict[int, int] = defaultdict(int)
    exact_count: defaultdict[int, int] = defaultdict(int)

    for movie_id, customer_id, rating, _date_text in iter_combined_ratings(
        paths.combined_files,
        movie_ids=movie_ids,
    ):
        fact = fact_lookup[movie_id]
        diff = float(rating) - fact["expected"]
        log_score[customer_id] += -0.5 * (diff / config.rating_sigma) ** 2
        squared_error[customer_id] += diff * diff
        abs_error[customer_id] += abs(diff)
        match_count[customer_id] += 1
        if fact["low"] <= rating <= fact["high"]:
            exact_count[customer_id] += 1

    rows: list[dict[str, object]] = []
    for customer_id, matches in match_count.items():
        if matches < config.min_matches:
            continue
        missing = n_facts - matches
        score = log_score[customer_id] + missing * config.missing_log_penalty
        rows.append(
            {
                "customer_id": customer_id,
                "log_score": score,
                "matched_facts": matches,
                "total_facts": n_facts,
                "coverage": matches / n_facts,
                "exact_rating_matches": exact_count[customer_id],
                "exact_rating_rate": exact_count[customer_id] / matches,
                "mean_abs_error": abs_error[customer_id] / matches,
                "rmse_to_imdb_profile": float(np.sqrt(squared_error[customer_id] / matches)),
            }
        )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values(
        ["log_score", "matched_facts", "exact_rating_rate"],
        ascending=[False, False, False],
    ).head(config.top_n)
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    return candidates.reset_index(drop=True)


def run_linkage_attack(
    imdb_ratings: pd.DataFrame,
    movie_titles: pd.DataFrame,
    data_dir: Path,
    user: str | None = None,
    match_config: MatchConfig | None = None,
    linkage_config: LinkageConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run title matching plus probabilistic candidate scoring."""

    ratings = imdb_ratings.copy()
    if user:
        ratings = ratings[ratings["user"].astype(str).str.casefold() == user.casefold()]
        if ratings.empty:
            raise ValueError(f"No IMDb ratings found for user: {user}")

    matched = match_imdb_to_netflix(ratings, movie_titles, config=match_config)
    facts = prepare_linkage_facts(matched)
    candidates = score_netflix_candidates(facts, data_dir, config=linkage_config)
    return matched, facts, candidates
