"""Fast rating-prediction baselines for utility evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .anonymization import make_releases, normalize_rating_frame
from .netflix_io import iter_combined_ratings, netflix_paths, read_probe


@dataclass
class BiasRecommender:
    global_mean: float
    user_bias: np.ndarray
    movie_bias: np.ndarray
    user_index: pd.Index
    movie_index: pd.Index

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        data = normalize_rating_frame(frame)
        user_ids = self.user_index.get_indexer(data["customer_id"])
        movie_ids = self.movie_index.get_indexer(data["movie_id"])
        predictions = np.full(len(data), self.global_mean, dtype=np.float32)

        known_users = user_ids >= 0
        known_movies = movie_ids >= 0
        predictions[known_users] += self.user_bias[user_ids[known_users]]
        predictions[known_movies] += self.movie_bias[movie_ids[known_movies]]
        return np.clip(predictions, 1.0, 5.0)


def fit_bias_recommender(
    frame: pd.DataFrame,
    rating_col: str = "rating",
    epochs: int = 8,
    regularization: float = 10.0,
) -> BiasRecommender:
    """Fit a regularized user/movie bias model."""

    data = normalize_rating_frame(frame).dropna(subset=[rating_col]).copy()
    if data.empty:
        raise ValueError("Cannot fit recommender on an empty training set.")

    user_index = pd.Index(data["customer_id"].unique())
    movie_index = pd.Index(data["movie_id"].unique())
    users = user_index.get_indexer(data["customer_id"]).astype(np.int32)
    movies = movie_index.get_indexer(data["movie_id"]).astype(np.int32)
    ratings = data[rating_col].to_numpy(dtype=np.float32)

    global_mean = float(ratings.mean())
    user_bias = np.zeros(len(user_index), dtype=np.float32)
    movie_bias = np.zeros(len(movie_index), dtype=np.float32)
    user_count = np.bincount(users, minlength=len(user_index)).astype(np.float32)
    movie_count = np.bincount(movies, minlength=len(movie_index)).astype(np.float32)

    for _epoch in range(epochs):
        residual_for_users = ratings - global_mean - movie_bias[movies]
        user_sum = np.bincount(users, weights=residual_for_users, minlength=len(user_index)).astype(np.float32)
        user_bias = user_sum / (user_count + regularization)

        residual_for_movies = ratings - global_mean - user_bias[users]
        movie_sum = np.bincount(movies, weights=residual_for_movies, minlength=len(movie_index)).astype(np.float32)
        movie_bias = movie_sum / (movie_count + regularization)

    return BiasRecommender(
        global_mean=global_mean,
        user_bias=user_bias,
        movie_bias=movie_bias,
        user_index=user_index,
        movie_index=movie_index,
    )


def rmse(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    actual_array = np.asarray(actual, dtype=np.float32)
    predicted_array = np.asarray(predicted, dtype=np.float32)
    return float(np.sqrt(np.mean((actual_array - predicted_array) ** 2)))


def mae(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    actual_array = np.asarray(actual, dtype=np.float32)
    predicted_array = np.asarray(predicted, dtype=np.float32)
    return float(np.mean(np.abs(actual_array - predicted_array)))


def train_test_split_ratings(
    frame: pd.DataFrame,
    test_fraction: float = 0.2,
    seed: int = 333,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = normalize_rating_frame(frame)
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    mask = rng.random(len(data)) < test_fraction
    train = data.loc[~mask].reset_index(drop=True)
    test = data.loc[mask].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty partition.")
    return train, test


def compare_release_rmse(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int = 333,
    epochs: int = 8,
    regularization: float = 10.0,
    rare_movie_min_users: int = 500,
    k_suppression: int = 5,
) -> pd.DataFrame:
    """Train the same recommender on each release and evaluate true test RMSE."""

    releases = make_releases(
        train,
        seed=seed,
        rare_movie_min_users=rare_movie_min_users,
        k_suppression=k_suppression,
    )
    rows: list[dict[str, object]] = []
    actual = normalize_rating_frame(test)["rating"].to_numpy(dtype=np.float32)

    for release in releases:
        if release.rating_col is None:
            rows.append(
                {
                    "release_name": release.name,
                    "rating_col": None,
                    "train_rows": len(release.data),
                    "test_rows": len(test),
                    "rmse": np.nan,
                    "mae": np.nan,
                    "status": "skipped_no_rating_labels",
                }
            )
            continue

        train_release = release.data.dropna(subset=[release.rating_col])
        if train_release.empty:
            rows.append(
                {
                    "release_name": release.name,
                    "rating_col": release.rating_col,
                    "train_rows": 0,
                    "test_rows": len(test),
                    "rmse": np.nan,
                    "mae": np.nan,
                    "status": "skipped_empty_training_set",
                }
            )
            continue

        model = fit_bias_recommender(
            train_release,
            rating_col=release.rating_col,
            epochs=epochs,
            regularization=regularization,
        )
        predicted = model.predict(test)
        rows.append(
            {
                "release_name": release.name,
                "rating_col": release.rating_col,
                "train_rows": len(train_release),
                "test_rows": len(test),
                "global_mean": model.global_mean,
                "rmse": rmse(actual, predicted),
                "mae": mae(actual, predicted),
                "status": "ok",
            }
        )

    return pd.DataFrame(rows).sort_values(["status", "rmse"], na_position="last").reset_index(drop=True)


def load_official_probe_holdout(
    data_dir: Path,
    max_train_rows: int | None = None,
    max_probe_rows: int | None = None,
    max_scan_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train rows and the official probe ratings by scanning combined files."""

    paths = netflix_paths(data_dir)
    probe_pairs = read_probe(paths.probe_file, max_rows=max_probe_rows)
    probe_keys = set(zip(probe_pairs["movie_id"].astype(int), probe_pairs["customer_id"].astype(int)))

    train_rows: list[tuple[int, int, int, str]] = []
    probe_rows: list[tuple[int, int, int, str]] = []
    scanned = 0

    for movie_id, customer_id, rating, date_text in iter_combined_ratings(paths.combined_files):
        scanned += 1
        key = (movie_id, customer_id)
        if key in probe_keys and (max_probe_rows is None or len(probe_rows) < max_probe_rows):
            probe_rows.append((movie_id, customer_id, rating, date_text))
        elif max_train_rows is None or len(train_rows) < max_train_rows:
            train_rows.append((movie_id, customer_id, rating, date_text))

        if max_scan_rows is not None and scanned >= max_scan_rows:
            break
        if max_train_rows is not None and len(train_rows) >= max_train_rows and max_probe_rows is not None and len(probe_rows) >= max_probe_rows:
            break

    def to_frame(rows: list[tuple[int, int, int, str]]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["movie_id", "customer_id", "rating", "date"]).astype(
            {"movie_id": "int32", "customer_id": "int32", "rating": "int8"}
        )

    train = to_frame(train_rows)
    probe = to_frame(probe_rows)
    if train.empty or probe.empty:
        raise ValueError("Official probe holdout load produced an empty train or probe set.")
    return train, probe
