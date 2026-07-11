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

from guardrails_sensitive_data.anonymization import add_rating_noise, add_time_and_rating_features, evaluate_releases
from guardrails_sensitive_data.data import (
    extract_imdb_profile_year,
    extract_imdb_earliest_rating_year,
    extract_imdb_user_ratings,
    extract_imdb_user_ratings_with_beautifulsoup,
    imdb_identifier_to_user_label,
    imdb_ratings_url_candidates,
    make_synthetic_movie_titles,
    make_synthetic_netflix_ratings,
    match_imdb_to_netflix,
    read_movie_titles,
    read_netflix_ratings,
    scrape_imdb_user_ratings,
)
from guardrails_sensitive_data.linkage import LinkageConfig, run_linkage_attack, run_planted_linkage_benchmark
from guardrails_sensitive_data.reporting import build_report
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

    def test_imdb_url_helpers(self) -> None:
        url = "https://www.imdb.com/user/p.example123/ratings?ref_=demo"
        self.assertEqual(imdb_identifier_to_user_label(url), "p.example123")
        self.assertEqual(
            imdb_ratings_url_candidates(url),
            ("https://www.imdb.com/user/p.example123/ratings/?sort=num_votes%2Cdesc",),
        )
        tracked_url = "https://www.imdb.com/user/p.qa2bgjpgs56q2o66ojrr2kvy6y?ref_=tt_ururv_c_1_uname/ratings"
        self.assertEqual(
            imdb_ratings_url_candidates(tracked_url),
            ("https://www.imdb.com/user/p.qa2bgjpgs56q2o66ojrr2kvy6y/ratings/?sort=num_votes%2Cdesc",),
        )
        candidates = imdb_ratings_url_candidates("demo_user")
        self.assertIn("https://www.imdb.com/user/demo_user/ratings/?sort=num_votes%2Cdesc", candidates)

    def test_extract_imdb_ratings_from_json_payload(self) -> None:
        html = """
        <script id="__NEXT_DATA__" type="application/json">
        {
          "props": {
            "pageProps": {
              "createdDate": "2021-04-01",
              "items": [
                {
                  "title": {"titleText": {"text": "The Matrix"}, "releaseYear": {"year": 1999}},
                  "userRating": {"rating": 10}
                },
                {
                  "titleText": {"text": "New Movie"},
                  "releaseYear": {"year": 2024},
                  "userRating": 7
                }
              ]
            }
          }
        }
        </script>
        """
        ratings = extract_imdb_user_ratings(html, username="json_user")
        self.assertEqual(set(ratings["title"]), {"The Matrix", "New Movie"})
        self.assertEqual(int(ratings.loc[ratings["title"] == "The Matrix", "year"].item()), 1999)
        self.assertEqual(float(ratings.loc[ratings["title"] == "The Matrix", "rating"].item()), 10.0)
        self.assertEqual(extract_imdb_profile_year(html), 2021)

    def test_extract_imdb_ratings_from_html_markup(self) -> None:
        html = """
        <ol>
          <li class="ipc-metadata-list-summary-item">
            <span>Rated on Feb 03, 2015</span>
            <h4 class="ipc-title__text">1. Small Drama</h4>
            <ul><li class="ipc-inline-list__item">2000</li></ul>
            <span data-testid="ratingGroup--other-user-rating">
              <span class="ipc-rating-star--rating">4</span>
            </span>
          </li>
        </ol>
        <span>Member since June 2004</span>
        """
        ratings = extract_imdb_user_ratings(html, username="html_user")
        self.assertEqual(len(ratings), 1)
        self.assertEqual(ratings.iloc[0]["title"], "Small Drama")
        self.assertEqual(int(ratings.iloc[0]["year"]), 2000)
        self.assertEqual(float(ratings.iloc[0]["rating"]), 4.0)
        self.assertEqual(extract_imdb_profile_year(html), 2004)
        self.assertEqual(extract_imdb_earliest_rating_year(html), 2015)

    def test_beautifulsoup_parser_or_install_hint(self) -> None:
        html = """
        <li class="ipc-metadata-list-summary-item">
          <span>Rated on Jan 02, 2019</span>
          <h4 class="ipc-title__text">1. Browser Movie</h4>
          <li class="ipc-inline-list__item">2001</li>
          <span data-testid="ratingGroup--other-user-rating">
            <span class="ipc-rating-star--rating">8</span>
          </span>
        </li>
        """
        try:
            ratings = extract_imdb_user_ratings_with_beautifulsoup(html, username="browser_user")
        except ValueError as exc:
            self.assertIn(".[scrape]", str(exc))
            return

        self.assertEqual(len(ratings), 1)
        self.assertEqual(ratings.iloc[0]["title"], "Browser Movie")
        self.assertEqual(int(ratings.iloc[0]["year"]), 2001)
        self.assertEqual(float(ratings.iloc[0]["rating"]), 8.0)

    def test_scrape_imdb_user_ratings_uses_fetcher(self) -> None:
        html = """
        <script id="__NEXT_DATA__" type="application/json">
        {"props": {"pageProps": {"dateCreated": "2022-01-01", "items": [
          {"titleText": {"text": "The Matrix"}, "releaseYear": {"year": 1999}, "userRating": 9}
        ]}}}
        </script>
        """

        seen_urls: list[str] = []

        def fake_fetch(url: str) -> str:
            seen_urls.append(url)
            return html

        result = scrape_imdb_user_ratings(
            "https://www.imdb.com/user/p.example123/ratings",
            username="live_user",
            max_pages=1,
            sleep_seconds=0,
            fetcher=fake_fetch,
        )
        self.assertEqual(len(result.ratings), 1)
        self.assertEqual(result.profile_created_year, 2022)
        self.assertEqual(result.ratings.iloc[0]["user"], "live_user")
        self.assertIn("sort=num_votes%2Cdesc", seen_urls[0])

    def test_scrape_imdb_user_ratings_rejects_bad_fetch_method(self) -> None:
        with self.assertRaisesRegex(ValueError, "fetch_method"):
            scrape_imdb_user_ratings("demo", fetch_method="spaceship")

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
        self.assertIn("target_user_hash", trials.columns)
        self.assertNotIn("target_user", trials.columns)

    def test_time_features_from_existing_month(self) -> None:
        frame = pd.DataFrame(
            [
                {"customer_id": 1, "movie_id": 10, "rating": 4, "month": "2005-02"},
                {"customer_id": 2, "movie_id": 10, "rating": 3, "month": pd.NA},
            ]
        )
        features = add_time_and_rating_features(frame)
        self.assertIn("year", features.columns)
        self.assertEqual(int(features.loc[0, "year"]), 2005)
        self.assertTrue(pd.isna(features.loc[1, "year"]))

    def test_rating_noise_rejects_invalid_probability(self) -> None:
        ratings = read_netflix_ratings(self.data_dir, include_date=True)
        with self.assertRaises(ValueError):
            add_rating_noise(ratings, seed=1, flip_probability=1.5)

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
        self.assertIn("hit_rate_at_k", summary.columns)
        self.assertIn("ndcg_at_k", summary.columns)

    def test_synthetic_planted_attack_and_report(self) -> None:
        movie_titles = make_synthetic_movie_titles(n_movies=35, seed=7)
        ratings = make_synthetic_netflix_ratings(
            n_users=40,
            n_movies=35,
            mean_ratings_per_user=14,
            seed=7,
        )
        trials, planted_summary, candidates = run_planted_linkage_benchmark(
            ratings,
            movie_titles,
            n_profiles=3,
            n_known=8,
            rating_noise_probability=0.0,
            seed=7,
            top_n=5,
        )
        self.assertEqual(len(trials), 3)
        self.assertFalse(planted_summary.empty)
        self.assertFalse(candidates.empty)
        self.assertGreaterEqual(float(planted_summary.iloc[0]["top_5_rate"]), 0)

        k_summary, privacy_trials, privacy_summary = evaluate_releases(
            ratings,
            n_known_values=(1, 2, 3),
            trials=2,
            rare_movie_min_users=1,
            k_suppression=1,
        )
        train, test = train_test_split_ratings(ratings, test_fraction=0.25, seed=7)
        utility = compare_release_rmse(
            train,
            test,
            rare_movie_min_users=1,
            k_suppression=1,
            epochs=2,
            ranking_max_users=5,
            ranking_negatives_per_user=5,
        )
        with tempfile.TemporaryDirectory() as report_dir_text:
            report_dir = Path(report_dir_text)
            k_summary.to_csv(report_dir / "synthetic_privacy_k_anonymity_summary.csv", index=False)
            privacy_trials.to_csv(report_dir / "synthetic_privacy_linkage_trials.csv", index=False)
            privacy_summary.to_csv(report_dir / "synthetic_privacy_linkage_summary.csv", index=False)
            utility.to_csv(report_dir / "synthetic_utility_rmse_summary.csv", index=False)
            planted_summary.to_csv(report_dir / "synthetic_planted_attack_summary.csv", index=False)
            outputs = build_report(report_dir, prefix="synthetic_", title="Synthetic Test Report")
            self.assertTrue(outputs.report_path.exists())
            self.assertTrue(outputs.frontier_path.exists())


if __name__ == "__main__":
    unittest.main()
