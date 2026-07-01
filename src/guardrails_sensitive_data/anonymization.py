"""Anonymized release definitions and privacy-risk metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReleaseData:
    name: str
    data: pd.DataFrame
    knowledge_cols: tuple[str, ...]
    rating_col: str | None
    description: str


def normalize_rating_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common notebook column names."""

    data = frame.copy()
    if "customer_id" not in data.columns and "user_id" in data.columns:
        data = data.rename(columns={"user_id": "customer_id"})
    required = {"customer_id", "movie_id", "rating"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Ratings frame missing columns: {sorted(missing)}")
    return data


def add_time_and_rating_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = normalize_rating_frame(frame)
    if "date" in data.columns:
        dates = pd.to_datetime(data["date"], errors="coerce")
        data["month"] = dates.dt.to_period("M").astype("string")
        data["year"] = dates.dt.year.astype("Int64")
    elif "month" not in data.columns:
        data["month"] = "unknown"
        data["year"] = pd.Series([pd.NA] * len(data), dtype="Int64")

    data["rating_bucket"] = np.select(
        [data["rating"] <= 2, data["rating"] >= 4],
        ["disliked", "liked"],
        default="neutral",
    )
    data["rating_bucket_midpoint"] = np.select(
        [data["rating_bucket"] == "disliked", data["rating_bucket"] == "liked"],
        [1.5, 4.5],
        default=3.0,
    ).astype("float32")
    return data


def add_rating_noise(frame: pd.DataFrame, seed: int, flip_probability: float = 0.25) -> pd.DataFrame:
    data = frame.copy()
    rng = np.random.default_rng(seed)
    noise = rng.choice(
        [-1, 0, 1],
        size=len(data),
        p=[flip_probability / 2, 1 - flip_probability, flip_probability / 2],
    )
    data["rating_noisy"] = np.clip(data["rating"].to_numpy(dtype=np.int16) + noise, 1, 5).astype("int8")
    return data


def suppress_low_k_facts(frame: pd.DataFrame, cols: tuple[str, ...], k: int) -> pd.DataFrame:
    data = frame.copy()
    group_sizes = (
        data[list(cols) + ["customer_id"]]
        .drop_duplicates()
        .groupby(list(cols), dropna=False)["customer_id"]
        .nunique()
        .rename("_equivalence_class_size")
        .reset_index()
    )
    merged = data.merge(group_sizes, on=list(cols), how="left")
    filtered = merged[merged["_equivalence_class_size"] >= k].drop(columns="_equivalence_class_size")
    return filtered.copy()


def make_releases(
    frame: pd.DataFrame,
    seed: int = 333,
    rare_movie_min_users: int = 500,
    k_suppression: int = 5,
) -> list[ReleaseData]:
    """Build the anonymized release variants used throughout the project."""

    base = add_time_and_rating_features(frame)
    noisy = add_rating_noise(base, seed=seed)

    movie_counts = base.groupby("movie_id")["customer_id"].nunique()
    common_movie_ids = set(movie_counts[movie_counts >= rare_movie_min_users].index.astype(int))
    no_rare = base[base["movie_id"].isin(common_movie_ids)].copy()

    suppressed_cols = ("movie_id", "rating", "month")
    suppressed = suppress_low_k_facts(base, suppressed_cols, k=k_suppression)

    return [
        ReleaseData(
            name="original_movie_rating_month",
            data=base.copy(),
            knowledge_cols=("movie_id", "rating", "month"),
            rating_col="rating",
            description="Original-style release with movie, exact rating, and rating month.",
        ),
        ReleaseData(
            name="remove_month",
            data=base.copy(),
            knowledge_cols=("movie_id", "rating"),
            rating_col="rating",
            description="Dates are removed; exact ratings remain.",
        ),
        ReleaseData(
            name="year_only",
            data=base.copy(),
            knowledge_cols=("movie_id", "rating", "year"),
            rating_col="rating",
            description="Exact dates are generalized from month to year.",
        ),
        ReleaseData(
            name="coarsen_rating",
            data=base.copy(),
            knowledge_cols=("movie_id", "rating_bucket", "month"),
            rating_col="rating_bucket_midpoint",
            description="Ratings are grouped into disliked, neutral, and liked.",
        ),
        ReleaseData(
            name="remove_month_and_coarsen_rating",
            data=base.copy(),
            knowledge_cols=("movie_id", "rating_bucket"),
            rating_col="rating_bucket_midpoint",
            description="Dates are removed and ratings are grouped.",
        ),
        ReleaseData(
            name="add_rating_noise",
            data=noisy,
            knowledge_cols=("movie_id", "rating_noisy", "month"),
            rating_col="rating_noisy",
            description="Exact ratings are perturbed by -1, 0, or +1 with clipping.",
        ),
        ReleaseData(
            name=f"remove_rare_movies_min_{rare_movie_min_users}",
            data=no_rare,
            knowledge_cols=("movie_id", "rating", "month"),
            rating_col="rating",
            description="Movies below a minimum user-count threshold are removed.",
        ),
        ReleaseData(
            name=f"k_suppress_movie_rating_month_k{k_suppression}",
            data=suppressed,
            knowledge_cols=suppressed_cols,
            rating_col="rating",
            description="Facts with fewer than k distinct users are suppressed.",
        ),
        ReleaseData(
            name="movie_only",
            data=base.copy(),
            knowledge_cols=("movie_id",),
            rating_col=None,
            description="Only movie ids are released; rating prediction labels are absent.",
        ),
    ]


def fact_k_anonymity(frame: pd.DataFrame, cols: Iterable[str]) -> dict[str, float]:
    """Summarize k-anonymity over released fact equivalence classes."""

    cols = tuple(cols)
    if frame.empty:
        return {
            "fact_count": 0,
            "min_k": np.nan,
            "median_k": np.nan,
            "mean_k": np.nan,
            "pct_unique_facts": np.nan,
        }

    small = frame[list(cols) + ["customer_id"]].drop_duplicates()
    sizes = small.groupby(list(cols), dropna=False)["customer_id"].nunique()
    return {
        "fact_count": float(len(sizes)),
        "min_k": float(sizes.min()),
        "median_k": float(sizes.median()),
        "mean_k": float(sizes.mean()),
        "pct_unique_facts": float((sizes == 1).mean() * 100),
    }


def build_fact_index(frame: pd.DataFrame, cols: tuple[str, ...]) -> dict[object, set[int]]:
    small = frame[list(cols) + ["customer_id"]].drop_duplicates()
    return small.groupby(list(cols), dropna=False)["customer_id"].agg(lambda values: set(map(int, values))).to_dict()


def row_key(row: pd.Series, cols: tuple[str, ...]) -> object:
    values = tuple(row[col] for col in cols)
    return values[0] if len(values) == 1 else values


def candidate_count_for_rows(
    known_rows: pd.DataFrame,
    cols: tuple[str, ...],
    fact_index: dict[object, set[int]],
) -> int:
    candidates: set[int] | None = None
    for _, row in known_rows.iterrows():
        matches = fact_index.get(row_key(row, cols), set())
        candidates = set(matches) if candidates is None else candidates & matches
    return 0 if candidates is None else len(candidates)


def evaluate_release_linkage_risk(
    release: ReleaseData,
    n_known_values: tuple[int, ...] = (1, 2, 3),
    trials: int = 300,
    seed: int = 333,
) -> pd.DataFrame:
    """Estimate candidate-set sizes for sampled linkage attacks."""

    data = release.data
    cols = release.knowledge_cols
    if data.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    fact_index = build_fact_index(data, cols)
    user_counts = data["customer_id"].value_counts()
    user_histories = {int(user_id): group for user_id, group in data.groupby("customer_id", sort=False)}

    rows: list[dict[str, object]] = []
    for n_known in n_known_values:
        eligible_users = user_counts[user_counts >= n_known].index.to_numpy()
        if len(eligible_users) == 0:
            continue
        for _ in range(trials):
            target_user = int(rng.choice(eligible_users))
            history = user_histories[target_user]
            sample_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            known_rows = history.sample(n=n_known, random_state=sample_seed)
            candidate_count = candidate_count_for_rows(known_rows, cols, fact_index)
            rows.append(
                {
                    "release_name": release.name,
                    "knowledge_cols": " + ".join(cols),
                    "n_known": n_known,
                    "target_user": target_user,
                    "candidate_count": candidate_count,
                    "unique": candidate_count == 1,
                    "surety_pct": 100 / candidate_count if candidate_count > 0 else 0,
                }
            )
    return pd.DataFrame(rows)


def summarize_linkage_trials(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return pd.DataFrame()
    return (
        trials.groupby(["release_name", "knowledge_cols", "n_known"], as_index=False)
        .agg(
            avg_candidate_size=("candidate_count", "mean"),
            median_candidate_size=("candidate_count", "median"),
            pct_unique=("unique", lambda values: values.mean() * 100),
            pct_5_or_less=("candidate_count", lambda values: (values <= 5).mean() * 100),
            pct_10_or_less=("candidate_count", lambda values: (values <= 10).mean() * 100),
            avg_surety_pct=("surety_pct", "mean"),
            median_surety_pct=("surety_pct", "median"),
            trials=("candidate_count", "size"),
        )
        .sort_values(["n_known", "median_candidate_size", "avg_candidate_size"])
        .reset_index(drop=True)
    )


def evaluate_releases(
    frame: pd.DataFrame,
    n_known_values: tuple[int, ...] = (1, 2, 3),
    trials: int = 300,
    seed: int = 333,
    rare_movie_min_users: int = 500,
    k_suppression: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate k-anonymity and sampled linkage risk for all releases."""

    releases = make_releases(
        frame,
        seed=seed,
        rare_movie_min_users=rare_movie_min_users,
        k_suppression=k_suppression,
    )
    k_rows: list[dict[str, object]] = []
    trial_frames: list[pd.DataFrame] = []

    for release in releases:
        k_summary = fact_k_anonymity(release.data, release.knowledge_cols)
        k_rows.append(
            {
                "release_name": release.name,
                "knowledge_cols": " + ".join(release.knowledge_cols),
                "rows_released": len(release.data),
                "rating_col": release.rating_col,
                "description": release.description,
                **k_summary,
            }
        )
        trial_frames.append(
            evaluate_release_linkage_risk(
                release,
                n_known_values=n_known_values,
                trials=trials,
                seed=seed,
            )
        )

    k_summary_frame = pd.DataFrame(k_rows)
    trial_frame = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
    return k_summary_frame, trial_frame, summarize_linkage_trials(trial_frame)
