"""Fast rating-prediction baselines for utility evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .anonymization import make_releases, normalize_rating_frame
from .data import iter_combined_ratings, netflix_paths, read_probe


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


def sampled_ranking_metrics(
    model: BiasRecommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    k: int = 10,
    negatives_per_user: int = 50,
    max_users: int = 500,
    seed: int = 333,
) -> dict[str, float]:
    """Estimate hit-rate and NDCG with sampled negative movies."""

    if k <= 0 or negatives_per_user <= 0 or max_users <= 0:
        raise ValueError("k, negatives_per_user, and max_users must be positive.")

    train_data = normalize_rating_frame(train)
    test_data = normalize_rating_frame(test)
    positives = test_data[test_data["rating"] >= 4]
    if positives.empty:
        return {
            "ranking_users": 0.0,
            "ranking_k": float(k),
            "ranking_negatives_per_user": float(negatives_per_user),
            "hit_rate_at_k": np.nan,
            "ndcg_at_k": np.nan,
        }

    rng = np.random.default_rng(seed)
    known_movies = model.movie_index.to_numpy(dtype=np.int32)
    known_movie_set = set(map(int, known_movies))
    known_user_set = set(map(int, model.user_index.to_numpy(dtype=np.int32)))
    train_seen = train_data.groupby("customer_id")["movie_id"].agg(lambda values: set(map(int, values))).to_dict()
    positive_groups = dict(tuple(positives.groupby("customer_id", sort=False)))
    users = np.array([int(user) for user in positive_groups if int(user) in known_user_set], dtype=np.int32)
    rng.shuffle(users)

    hits: list[float] = []
    ndcgs: list[float] = []
    for user_id in users[:max_users]:
        user_positives = positive_groups[int(user_id)]
        candidate_positive_movies = [
            int(movie_id)
            for movie_id in user_positives["movie_id"].to_numpy()
            if int(movie_id) in known_movie_set
        ]
        if not candidate_positive_movies:
            continue

        positive_movie = int(rng.choice(candidate_positive_movies))
        unavailable = set(train_seen.get(int(user_id), set()))
        unavailable.update(map(int, user_positives["movie_id"].to_numpy()))
        negative_pool = np.array([movie_id for movie_id in known_movies if int(movie_id) not in unavailable], dtype=np.int32)
        if len(negative_pool) == 0:
            continue

        sample_size = min(negatives_per_user, len(negative_pool))
        negatives = rng.choice(negative_pool, size=sample_size, replace=False)
        eval_movies = np.concatenate([[positive_movie], negatives.astype(np.int32)])
        eval_frame = pd.DataFrame(
            {
                "customer_id": np.full(len(eval_movies), int(user_id), dtype=np.int32),
                "movie_id": eval_movies.astype(np.int32),
                "rating": np.zeros(len(eval_movies), dtype=np.int8),
            }
        )
        scores = model.predict(eval_frame)
        positive_score = float(scores[0])
        rank = int(1 + np.sum(scores[1:] > positive_score))
        hits.append(float(rank <= k))
        ndcgs.append(float(1 / np.log2(rank + 1)) if rank <= k else 0.0)

    if not hits:
        return {
            "ranking_users": 0.0,
            "ranking_k": float(k),
            "ranking_negatives_per_user": float(negatives_per_user),
            "hit_rate_at_k": np.nan,
            "ndcg_at_k": np.nan,
        }

    return {
        "ranking_users": float(len(hits)),
        "ranking_k": float(k),
        "ranking_negatives_per_user": float(negatives_per_user),
        "hit_rate_at_k": float(np.mean(hits) * 100),
        "ndcg_at_k": float(np.mean(ndcgs)),
    }


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
    ranking_k: int = 10,
    ranking_negatives_per_user: int = 50,
    ranking_max_users: int = 500,
) -> pd.DataFrame:
    """Train the same recommender on each release and evaluate utility."""

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
                    "ranking_users": 0,
                    "ranking_k": ranking_k,
                    "ranking_negatives_per_user": ranking_negatives_per_user,
                    "hit_rate_at_k": np.nan,
                    "ndcg_at_k": np.nan,
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
                    "ranking_users": 0,
                    "ranking_k": ranking_k,
                    "ranking_negatives_per_user": ranking_negatives_per_user,
                    "hit_rate_at_k": np.nan,
                    "ndcg_at_k": np.nan,
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
        ranking = sampled_ranking_metrics(
            model,
            train_release,
            test,
            k=ranking_k,
            negatives_per_user=ranking_negatives_per_user,
            max_users=ranking_max_users,
            seed=seed,
        )
        rows.append(
            {
                "release_name": release.name,
                "rating_col": release.rating_col,
                "train_rows": len(train_release),
                "test_rows": len(test),
                "global_mean": model.global_mean,
                "rmse": rmse(actual, predicted),
                "mae": mae(actual, predicted),
                **ranking,
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
