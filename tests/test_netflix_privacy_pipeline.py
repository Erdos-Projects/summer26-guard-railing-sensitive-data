from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardrails_sensitive_data.anonymization import evaluate_releases
from guardrails_sensitive_data.imdb import extract_imdb_user_ratings, match_imdb_to_netflix
from guardrails_sensitive_data.linkage import LinkageConfig, run_linkage_attack
from guardrails_sensitive_data.netflix_io import read_movie_titles, read_netflix_ratings
from guardrails_sensitive_data.recommender import compare_release_rmse, train_test_split_ratings


class NetflixPrivacyPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        (self.data_dir / "movie_titles.csv").write_text(
            "id,year,title\n"
            "1,1999,The Matrix\n"
            "2,2000,Small Drama\n"
            "3,2001,Movie, With Comma\n",
            encoding="latin1",
        )
        (self.data_dir / "combined_data_1.txt").write_text(
            "1:\n"
            "10,5,2005-01-01\n"
            "20,1,2005-01-02\n"
            "30,5,2005-01-03\n"
            "2:\n"
            "10,2,2005-02-01\n"
            "20,5,2005-02-02\n"
            "30,3,2005-02-03\n"
            "3:\n"
            "10,4,2005-03-01\n"
            "20,2,2005-03-02\n",
            encoding="latin1",
        )
        for part in range(2, 5):
            (self.data_dir / f"combined_data_{part}.txt").write_text("", encoding="latin1")
        (self.data_dir / "probe.txt").write_text("1:\n10\n", encoding="latin1")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_movie_titles_preserve_commas(self) -> None:
        titles = read_movie_titles(self.data_dir / "movie_titles.csv")
        self.assertEqual(titles.loc[titles["movie_id"] == 3, "title"].item(), "Movie, With Comma")

    def test_read_netflix_ratings_filter(self) -> None:
        ratings = read_netflix_ratings(self.data_dir, movie_ids={1}, include_date=True)
        self.assertEqual(len(ratings), 3)
        self.assertEqual(set(ratings["movie_id"]), {1})

    def test_imdb_title_matching(self) -> None:
        imdb = pd.DataFrame(
            [
                {"title": "Matrix", "year": 1999, "user": "demo", "rating": 10},
                {"title": "Small Drama", "year": 2000, "user": "demo", "rating": 4},
            ]
        )
        matched = match_imdb_to_netflix(imdb, read_movie_titles(self.data_dir / "movie_titles.csv"))
        self.assertEqual(set(matched["netflix_movie_id"].dropna().astype(int)), {1, 2})

    def test_probabilistic_linkage_ranks_best_customer(self) -> None:
        imdb = pd.DataFrame(
            [
                {"title": "The Matrix", "year": 1999, "user": "demo", "rating": 10},
                {"title": "Small Drama", "year": 2000, "user": "demo", "rating": 4},
            ]
        )
        titles = read_movie_titles(self.data_dir / "movie_titles.csv")
        _matched, facts, candidates = run_linkage_attack(
            imdb,
            titles,
            self.data_dir,
            user="demo",
            linkage_config=LinkageConfig(min_matches=2, top_n=5),
        )
        self.assertEqual(len(facts), 2)
        self.assertFalse(candidates.empty)
        self.assertEqual(int(candidates.iloc[0]["customer_id"]), 10)

    def test_anonymization_evaluation(self) -> None:
        ratings = read_netflix_ratings(self.data_dir, include_date=True)
        k_summary, trials, trial_summary = evaluate_releases(
            ratings,
            n_known_values=(1, 2),
            trials=3,
            rare_movie_min_users=1,
            k_suppression=1,
        )
        self.assertFalse(k_summary.empty)
        self.assertFalse(trials.empty)
        self.assertFalse(trial_summary.empty)
        self.assertIn("remove_month", set(k_summary["release_name"]))

    def test_rmse_comparison(self) -> None:
        ratings = read_netflix_ratings(self.data_dir, include_date=True)
        train, test = train_test_split_ratings(ratings, test_fraction=0.25, seed=1)
        summary = compare_release_rmse(
            train,
            test,
            rare_movie_min_users=1,
            k_suppression=1,
            epochs=2,
        )
        self.assertIn("original_movie_rating_month", set(summary["release_name"]))
        ok_rows = summary[summary["status"] == "ok"]
        self.assertTrue((ok_rows["rmse"] >= 0).all())

    def test_extract_imdb_json_rating_rows(self) -> None:
        html = """
        <script id="__NEXT_DATA__" type="application/json">
        {"props":{"pageProps":{"items":[{"titleText":{"text":"The Matrix"},"releaseYear":{"year":1999},"userRating":{"value":10}}]}}}
        </script>
        """
        rows = extract_imdb_user_ratings(html, username="demo")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0]["title"], "The Matrix")


if __name__ == "__main__":
    unittest.main()
