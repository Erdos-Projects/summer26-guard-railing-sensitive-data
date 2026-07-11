"""Probabilistic record linkage attack from IMDb ratings to Netflix users."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import MatchConfig, iter_combined_ratings, make_synthetic_imdb_profile, match_imdb_to_netflix, netflix_paths


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


def _score_candidate_observations(
    facts: pd.DataFrame,
    observations: Iterable[tuple[int, int, int]],
    config: LinkageConfig,
) -> pd.DataFrame:
    config = config or LinkageConfig()
    if facts.empty:
        raise ValueError("No matched IMDb/Netflix facts are available for linkage.")

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

    for movie_id, customer_id, rating in observations:
        if movie_id not in movie_ids:
            continue
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


def score_candidate_ratings_frame(
    facts: pd.DataFrame,
    ratings: pd.DataFrame,
    config: LinkageConfig | None = None,
) -> pd.DataFrame:
    """Score candidate customers from an in-memory ratings frame."""

    config = config or LinkageConfig()
    required = {"movie_id", "customer_id", "rating"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Ratings frame missing columns: {sorted(missing)}")

    observations = (
        (int(row.movie_id), int(row.customer_id), int(row.rating))
        for row in ratings[["movie_id", "customer_id", "rating"]].itertuples(index=False)
    )
    return _score_candidate_observations(facts, observations, config)


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
    movie_ids = set(facts["netflix_movie_id"].dropna().astype(int))
    observations = (
        (movie_id, customer_id, rating)
        for movie_id, customer_id, rating, _date_text in iter_combined_ratings(
            paths.combined_files,
            movie_ids=movie_ids,
        )
    )
    return _score_candidate_observations(facts, observations, config)


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


def run_planted_linkage_benchmark(
    ratings: pd.DataFrame,
    movie_titles: pd.DataFrame,
    *,
    n_profiles: int = 25,
    n_known: int = 18,
    rating_noise_probability: float = 0.15,
    seed: int = 333,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Measure whether planted external profiles recover their source users."""

    if n_profiles <= 0 or n_known <= 0 or top_n <= 0:
        raise ValueError("n_profiles, n_known, and top_n must be positive.")

    rng = np.random.default_rng(seed)
    unique_users = ratings["customer_id"].nunique()
    rank_config = LinkageConfig(min_matches=max(2, min(n_known, 4)), top_n=unique_users)
    public_config = LinkageConfig(min_matches=rank_config.min_matches, top_n=top_n)

    trial_rows: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []
    for profile_index in range(1, n_profiles + 1):
        profile = make_synthetic_imdb_profile(
            ratings,
            movie_titles,
            n_known=n_known,
            profile_label=f"synthetic_profile_{profile_index:03d}",
            rating_noise_probability=rating_noise_probability,
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
        matched = match_imdb_to_netflix(profile.imdb_ratings, movie_titles, config=MatchConfig(fuzzy=False))
        facts = prepare_linkage_facts(matched)
        ranked = score_candidate_ratings_frame(facts, ratings, config=rank_config)

        if ranked.empty:
            rank = np.nan
            top_score_gap = np.nan
        else:
            matches = ranked[ranked["customer_id"] == profile.target_user]
            rank = float(matches["rank"].iloc[0]) if not matches.empty else np.nan
            top_score = float(ranked.iloc[0]["log_score"])
            true_score = float(matches["log_score"].iloc[0]) if not matches.empty else np.nan
            top_score_gap = top_score - true_score if not np.isnan(true_score) else np.nan

        public_candidates = score_candidate_ratings_frame(facts, ratings, config=public_config)
        public_candidates.insert(0, "profile_label", profile.profile_label)
        public_candidates.insert(1, "target_user", profile.target_user)
        candidate_frames.append(public_candidates)
        trial_rows.append(
            {
                "profile_label": profile.profile_label,
                "target_user": profile.target_user,
                "known_facts": len(facts),
                "rank": rank,
                "top_1": rank == 1,
                "top_5": rank <= 5 if not np.isnan(rank) else False,
                "top_10": rank <= 10 if not np.isnan(rank) else False,
                "top_score_gap": top_score_gap,
            }
        )

    trials = pd.DataFrame(trial_rows)
    summary = pd.DataFrame(
        [
            {
                "n_profiles": len(trials),
                "n_known": n_known,
                "rating_noise_probability": rating_noise_probability,
                "mean_rank": float(trials["rank"].mean()),
                "median_rank": float(trials["rank"].median()),
                "top_1_rate": float(trials["top_1"].mean() * 100),
                "top_5_rate": float(trials["top_5"].mean() * 100),
                "top_10_rate": float(trials["top_10"].mean() * 100),
            }
        ]
    )
    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    return trials, summary, candidates
