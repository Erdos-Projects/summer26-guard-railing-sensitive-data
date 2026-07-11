"""Command line interface for the Netflix privacy project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from .anonymization import evaluate_releases, redact_customer_ids
from .data import (
    MatchConfig,
    imdb_identifier_to_user_label,
    make_synthetic_movie_titles,
    make_synthetic_netflix_ratings,
    netflix_paths,
    read_imdb_ratings_csv,
    read_movie_titles,
    read_netflix_ratings,
    scrape_imdb_user_ratings,
    verify_netflix_files,
)
from .linkage import LinkageConfig, run_linkage_attack, run_planted_linkage_benchmark
from .recommender import compare_release_rmse, load_official_probe_holdout, train_test_split_ratings
from .reporting import build_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NETFLIX_DIR = PROJECT_ROOT / "data" / "netflix"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"
DEFAULT_IMDB_CSV = PROJECT_ROOT / "notebooks" / "imdb_data.csv"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _looks_like_url(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.strip().lower()
    return lowered.startswith(("http://", "https://", "www.", "imdb.com/"))


def _safe_output_label(value: str | None) -> str:
    label = value or "all_users"
    label = imdb_identifier_to_user_label(label) if _looks_like_url(label) else label
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "imdb_profile"


def _load_imdb_ratings_for_linkage(args: argparse.Namespace):
    requested_identifier = args.ratings_url or args.user

    cached = None
    if args.imdb_csv.exists():
        cached = read_imdb_ratings_csv(args.imdb_csv)
    else:
        _warn(f"IMDb cache not found: {args.imdb_csv}")

    if cached is not None:
        if args.user and not _looks_like_url(args.user):
            user_rows = cached[cached["user"].astype(str).str.casefold() == args.user.casefold()]
            if not user_rows.empty:
                print(f"IMDb source: cached CSV {args.imdb_csv} ({len(user_rows):,} rows for {args.user!r}).")
                return cached, args.user, False
            _warn(f"No cached IMDb rows found for {args.user!r}.")
        elif not requested_identifier:
            user_count = cached["user"].nunique()
            print(f"IMDb source: cached CSV {args.imdb_csv} ({len(cached):,} rows across {user_count:,} users).")
            return cached, None, False

    if not requested_identifier:
        raise ValueError("No IMDb ratings are available. Provide --user or --ratings-url to attempt a live scrape.")
    if args.no_scrape:
        raise ValueError(f"No cached IMDb rows for {requested_identifier!r}, and --no-scrape was set.")

    label = args.user if args.user and not _looks_like_url(args.user) else imdb_identifier_to_user_label(requested_identifier)
    _warn(f"Attempting live IMDb scrape for {requested_identifier!r}.")
    result = scrape_imdb_user_ratings(
        requested_identifier,
        username=label,
        max_pages=args.imdb_max_pages,
        sleep_seconds=args.imdb_sleep_seconds,
        fetch_method=args.imdb_fetch_method,
        browser_headless=not args.imdb_browser_headed,
        browser_wait_seconds=args.imdb_browser_wait_seconds,
    )
    ratings = result.ratings
    print(
        "IMDb source: live scrape "
        f"{result.source_url} ({len(ratings):,} rows from {result.pages_fetched:,} page(s))."
    )

    if result.profile_created_year is None:
        if result.earliest_rating_year is not None and result.earliest_rating_year > 2006:
            _warn(
                "IMDb profile creation year was not detected; earliest scraped rating activity "
                f"is from {result.earliest_rating_year}, after the Netflix Prize data window ended in 2006. "
                "Running linkage anyway."
            )
        else:
            _warn("IMDb profile creation year was not detected; profile-age warning is unavailable.")
    elif result.profile_created_year > 2006:
        _warn(
            f"IMDb profile appears to have been created in {result.profile_created_year}, "
            "after the Netflix Prize data window ended in 2006. Running linkage anyway."
        )
    else:
        print(f"IMDb profile creation year: {result.profile_created_year}")
    if result.earliest_rating_year is not None:
        print(f"Earliest scraped IMDb rating activity year: {result.earliest_rating_year}")

    years = ratings["year"].dropna().astype(int)
    if not years.empty:
        after_2006 = int((years > 2006).sum())
        through_2006 = int((years <= 2006).sum())
        if after_2006:
            _warn(f"{after_2006:,} scraped ratings are for titles after 2006 and cannot appear in Netflix Prize.")
        if through_2006 == 0:
            _warn("No scraped rated titles are dated 2006 or earlier; linkage may produce no Netflix matches.")
    for warning in result.warnings:
        _warn(warning)

    return ratings, label, True


def cmd_verify_data(args: argparse.Namespace) -> int:
    report = verify_netflix_files(args.data_dir, require_probe=args.require_probe)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def cmd_linkage_attack(args: argparse.Namespace) -> int:
    output_dir = ensure_directory(args.output_dir)
    paths = netflix_paths(args.data_dir)
    imdb_ratings, selected_user, scraped_live = _load_imdb_ratings_for_linkage(args)
    movie_titles = read_movie_titles(paths.movie_titles_file)

    matched, facts, candidates = run_linkage_attack(
        imdb_ratings=imdb_ratings,
        movie_titles=movie_titles,
        data_dir=args.data_dir,
        user=selected_user,
        match_config=MatchConfig(
            fuzzy=not args.no_fuzzy,
            fuzzy_threshold=args.fuzzy_threshold,
            fuzzy_margin=args.fuzzy_margin,
        ),
        linkage_config=LinkageConfig(
            rating_sigma=args.rating_sigma,
            missing_log_penalty=args.missing_log_penalty,
            min_matches=args.min_matches,
            top_n=args.top_n,
        ),
    )

    prefix = f"linkage_{_safe_output_label(selected_user)}" if selected_user else "linkage_all_users"
    matched_path = output_dir / f"{prefix}_matched_titles.csv"
    facts_path = output_dir / f"{prefix}_facts.csv"
    candidates_path = output_dir / f"{prefix}_candidates.csv"
    candidate_output = candidates if args.unsafe_include_customer_ids else redact_customer_ids(candidates)
    if scraped_live:
        scraped_path = output_dir / f"{prefix}_scraped_imdb_ratings.csv"
        imdb_ratings.to_csv(scraped_path, index=False)
        print(f"Wrote scraped IMDb ratings to {scraped_path}")
    matched.to_csv(matched_path, index=False)
    facts.to_csv(facts_path, index=False)
    candidate_output.to_csv(candidates_path, index=False)

    print(f"Matched {len(facts):,} unique Netflix titles from {len(matched):,} IMDb rows.")
    print(f"Wrote candidates to {candidates_path}")
    if not candidate_output.empty:
        print(candidate_output.head(min(10, len(candidate_output))).to_string(index=False))
    return 0


def cmd_privacy_eval(args: argparse.Namespace) -> int:
    output_dir = ensure_directory(args.output_dir)
    frame = read_netflix_ratings(args.data_dir, max_rows=args.max_rows, include_date=True)
    k_summary, trials, trial_summary = evaluate_releases(
        frame,
        n_known_values=tuple(args.n_known),
        trials=args.trials,
        seed=args.seed,
        rare_movie_min_users=args.rare_movie_min_users,
        k_suppression=args.k_suppression,
    )

    k_path = output_dir / "privacy_k_anonymity_summary.csv"
    trials_path = output_dir / "privacy_linkage_trials.csv"
    summary_path = output_dir / "privacy_linkage_summary.csv"
    k_summary.to_csv(k_path, index=False)
    trials.to_csv(trials_path, index=False)
    trial_summary.to_csv(summary_path, index=False)

    print(f"Loaded {len(frame):,} ratings.")
    print(f"Wrote k-anonymity summary to {k_path}")
    print(f"Wrote linkage-risk summary to {summary_path}")
    if not trial_summary.empty:
        print(trial_summary.head(15).to_string(index=False))
    return 0


def cmd_rmse_eval(args: argparse.Namespace) -> int:
    output_dir = ensure_directory(args.output_dir)
    if args.holdout == "probe":
        train, test = load_official_probe_holdout(
            args.data_dir,
            max_train_rows=args.max_train_rows,
            max_probe_rows=args.max_probe_rows,
            max_scan_rows=args.max_scan_rows,
        )
    else:
        frame = read_netflix_ratings(args.data_dir, max_rows=args.max_rows, include_date=True)
        train, test = train_test_split_ratings(frame, test_fraction=args.test_fraction, seed=args.seed)

    summary = compare_release_rmse(
        train,
        test,
        seed=args.seed,
        epochs=args.epochs,
        regularization=args.regularization,
        rare_movie_min_users=args.rare_movie_min_users,
        k_suppression=args.k_suppression,
        ranking_k=args.ranking_k,
        ranking_negatives_per_user=args.ranking_negatives,
        ranking_max_users=args.ranking_max_users,
    )
    output = output_dir / "utility_rmse_summary.csv"
    summary.to_csv(output, index=False)
    print(f"Train rows: {len(train):,}; test rows: {len(test):,}")
    print(f"Wrote RMSE summary to {output}")
    print(summary.to_string(index=False))
    return 0


def _write_experiment_outputs(
    output_dir: Path,
    *,
    prefix: str,
    k_summary,
    trials,
    trial_summary,
    utility_summary,
) -> None:
    k_summary.to_csv(output_dir / f"{prefix}privacy_k_anonymity_summary.csv", index=False)
    trials.to_csv(output_dir / f"{prefix}privacy_linkage_trials.csv", index=False)
    trial_summary.to_csv(output_dir / f"{prefix}privacy_linkage_summary.csv", index=False)
    utility_summary.to_csv(output_dir / f"{prefix}utility_rmse_summary.csv", index=False)


def cmd_run_synthetic_demo(args: argparse.Namespace) -> int:
    output_dir = ensure_directory(args.output_dir)
    prefix = "synthetic_"
    movie_titles = make_synthetic_movie_titles(n_movies=args.synthetic_movies, seed=args.seed)
    ratings = make_synthetic_netflix_ratings(
        n_users=args.synthetic_users,
        n_movies=args.synthetic_movies,
        mean_ratings_per_user=args.synthetic_mean_ratings,
        seed=args.seed,
    )
    movie_titles.to_csv(output_dir / f"{prefix}movie_titles.csv", index=False)
    ratings.to_csv(output_dir / f"{prefix}ratings.csv", index=False)

    k_summary, trials, trial_summary = evaluate_releases(
        ratings,
        n_known_values=(1, 2, 3),
        trials=args.trials,
        seed=args.seed,
        rare_movie_min_users=args.rare_movie_min_users,
        k_suppression=args.k_suppression,
    )
    train, test = train_test_split_ratings(ratings, test_fraction=0.2, seed=args.seed)
    utility_summary = compare_release_rmse(
        train,
        test,
        seed=args.seed,
        epochs=args.epochs,
        regularization=args.regularization,
        rare_movie_min_users=args.rare_movie_min_users,
        k_suppression=args.k_suppression,
        ranking_k=args.ranking_k,
        ranking_negatives_per_user=args.ranking_negatives,
        ranking_max_users=args.ranking_max_users,
    )
    _write_experiment_outputs(
        output_dir,
        prefix=prefix,
        k_summary=k_summary,
        trials=trials,
        trial_summary=trial_summary,
        utility_summary=utility_summary,
    )

    planted_trials, planted_summary, planted_candidates = run_planted_linkage_benchmark(
        ratings,
        movie_titles,
        n_profiles=args.synthetic_profiles,
        n_known=args.synthetic_known_facts,
        rating_noise_probability=args.synthetic_rating_noise,
        seed=args.seed,
    )
    planted_trials = redact_customer_ids(planted_trials, column="target_user", hashed_column="target_user_hash")
    planted_candidates = redact_customer_ids(planted_candidates, column="target_user", hashed_column="target_user_hash")
    planted_candidates = redact_customer_ids(planted_candidates, column="customer_id", hashed_column="candidate_hash")
    planted_trials.to_csv(output_dir / f"{prefix}planted_attack_trials.csv", index=False)
    planted_summary.to_csv(output_dir / f"{prefix}planted_attack_summary.csv", index=False)
    planted_candidates.to_csv(output_dir / f"{prefix}planted_attack_candidates.csv", index=False)

    outputs = build_report(
        output_dir,
        prefix=prefix,
        n_known=3,
        title="Synthetic Netflix Privacy-Utility Audit",
    )
    print(f"Wrote synthetic ratings to {output_dir / f'{prefix}ratings.csv'}")
    print(f"Wrote synthetic privacy/utility report to {outputs.report_path}")
    print(utility_summary.to_string(index=False))
    print(planted_summary.to_string(index=False))
    return 0


def cmd_run_demo(args: argparse.Namespace) -> int:
    if args.synthetic:
        return cmd_run_synthetic_demo(args)

    privacy_args = argparse.Namespace(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        n_known=[1, 2, 3],
        trials=args.trials,
        seed=args.seed,
        rare_movie_min_users=args.rare_movie_min_users,
        k_suppression=args.k_suppression,
    )
    rmse_args = argparse.Namespace(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        holdout="random",
        max_rows=args.max_rows,
        max_train_rows=None,
        max_probe_rows=None,
        max_scan_rows=None,
        test_fraction=0.2,
        seed=args.seed,
        epochs=args.epochs,
        regularization=args.regularization,
        rare_movie_min_users=args.rare_movie_min_users,
        k_suppression=args.k_suppression,
        ranking_k=args.ranking_k,
        ranking_negatives=args.ranking_negatives,
        ranking_max_users=args.ranking_max_users,
    )
    cmd_privacy_eval(privacy_args)
    cmd_rmse_eval(rmse_args)
    if args.imdb_csv.exists():
        linkage_args = argparse.Namespace(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            imdb_csv=args.imdb_csv,
            user=args.user,
            ratings_url=None,
            no_scrape=False,
            imdb_max_pages=5,
            imdb_sleep_seconds=1.0,
            imdb_fetch_method="auto",
            imdb_browser_headed=False,
            imdb_browser_wait_seconds=15.0,
            no_fuzzy=False,
            fuzzy_threshold=0.93,
            fuzzy_margin=0.03,
            rating_sigma=0.75,
            missing_log_penalty=-0.75,
            min_matches=2,
            top_n=25,
            unsafe_include_customer_ids=False,
        )
        cmd_linkage_attack(linkage_args)
    outputs = build_report(args.output_dir, n_known=3)
    print(f"Wrote privacy/utility report to {outputs.report_path}")
    return 0


def cmd_build_report(args: argparse.Namespace) -> int:
    output_dir = ensure_directory(args.output_dir)
    outputs = build_report(
        output_dir,
        prefix=args.prefix,
        n_known=args.n_known,
        title=args.title,
    )
    print(f"Wrote report to {outputs.report_path}")
    print(f"Wrote frontier table to {outputs.frontier_path}")
    if outputs.plot_path is not None:
        print(f"Wrote frontier plot to {outputs.plot_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Netflix Prize privacy/utility experiments")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_NETFLIX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-data", help="Validate local Netflix Prize files")
    verify.add_argument("--require-probe", action="store_true")
    verify.set_defaults(func=cmd_verify_data)

    linkage = subparsers.add_parser("linkage-attack", help="Run IMDb-to-Netflix probabilistic linkage")
    linkage.add_argument("--imdb-csv", type=Path, default=DEFAULT_IMDB_CSV)
    linkage.add_argument("--user", help="Restrict to a single IMDb user, user id, or profile URL")
    linkage.add_argument("--ratings-url", help="IMDb profile/ratings URL to scrape if the cached user is unavailable")
    linkage.add_argument("--no-scrape", action="store_true", help="Disable live IMDb scraping on cache misses")
    linkage.add_argument("--imdb-max-pages", type=_positive_int, default=5, help="Maximum IMDb ratings pages to fetch")
    linkage.add_argument("--imdb-sleep-seconds", type=float, default=1.0, help="Delay between IMDb page fetches")
    linkage.add_argument(
        "--imdb-fetch-method",
        choices=["auto", "http", "browser"],
        default="auto",
        help="Use plain HTTP, Selenium browser rendering, or HTTP with browser fallback",
    )
    linkage.add_argument(
        "--imdb-browser-headed",
        action="store_true",
        help="Show the Selenium browser window instead of using headless Chrome",
    )
    linkage.add_argument(
        "--imdb-browser-wait-seconds",
        type=float,
        default=15.0,
        help="Seconds to wait for IMDb ratings to render in Selenium",
    )
    linkage.add_argument("--no-fuzzy", action="store_true")
    linkage.add_argument("--fuzzy-threshold", type=float, default=0.93)
    linkage.add_argument("--fuzzy-margin", type=float, default=0.03)
    linkage.add_argument("--rating-sigma", type=float, default=0.75)
    linkage.add_argument("--missing-log-penalty", type=float, default=-0.75)
    linkage.add_argument("--min-matches", type=_positive_int, default=2)
    linkage.add_argument("--top-n", type=_positive_int, default=50)
    linkage.add_argument(
        "--unsafe-include-customer-ids",
        action="store_true",
        help="Write raw Netflix customer ids instead of public-safe hashes.",
    )
    linkage.set_defaults(func=cmd_linkage_attack)

    privacy = subparsers.add_parser("privacy-eval", help="Evaluate anonymized releases")
    privacy.add_argument("--max-rows", type=_positive_int, default=1_000_000)
    privacy.add_argument("--n-known", type=_positive_int, nargs="+", default=[1, 2, 3])
    privacy.add_argument("--trials", type=_positive_int, default=300)
    privacy.add_argument("--seed", type=int, default=333)
    privacy.add_argument("--rare-movie-min-users", type=_positive_int, default=500)
    privacy.add_argument("--k-suppression", type=_positive_int, default=5)
    privacy.set_defaults(func=cmd_privacy_eval)

    rmse_eval = subparsers.add_parser("rmse-eval", help="Compare downstream rating-prediction RMSE")
    rmse_eval.add_argument("--holdout", choices=["random", "probe"], default="random")
    rmse_eval.add_argument("--max-rows", type=_positive_int, default=1_000_000)
    rmse_eval.add_argument("--max-train-rows", type=_positive_int)
    rmse_eval.add_argument("--max-probe-rows", type=_positive_int)
    rmse_eval.add_argument("--max-scan-rows", type=_positive_int)
    rmse_eval.add_argument("--test-fraction", type=float, default=0.2)
    rmse_eval.add_argument("--seed", type=int, default=333)
    rmse_eval.add_argument("--epochs", type=_positive_int, default=8)
    rmse_eval.add_argument("--regularization", type=float, default=10.0)
    rmse_eval.add_argument("--rare-movie-min-users", type=_positive_int, default=500)
    rmse_eval.add_argument("--k-suppression", type=_positive_int, default=5)
    rmse_eval.add_argument("--ranking-k", type=_positive_int, default=10)
    rmse_eval.add_argument("--ranking-negatives", type=_positive_int, default=50)
    rmse_eval.add_argument("--ranking-max-users", type=_positive_int, default=500)
    rmse_eval.set_defaults(func=cmd_rmse_eval)

    demo = subparsers.add_parser("run-demo", help="Run linkage, privacy, and RMSE on manageable samples")
    demo.add_argument("--synthetic", action="store_true", help="Run a public-safe synthetic demo instead of Netflix files")
    demo.add_argument("--max-rows", type=_positive_int, default=200_000)
    demo.add_argument("--trials", type=_positive_int, default=50)
    demo.add_argument("--seed", type=int, default=333)
    demo.add_argument("--epochs", type=_positive_int, default=6)
    demo.add_argument("--regularization", type=float, default=10.0)
    demo.add_argument("--rare-movie-min-users", type=_positive_int, default=100)
    demo.add_argument("--k-suppression", type=_positive_int, default=5)
    demo.add_argument("--imdb-csv", type=Path, default=DEFAULT_IMDB_CSV)
    demo.add_argument("--user", default="planktonrules")
    demo.add_argument("--ranking-k", type=_positive_int, default=10)
    demo.add_argument("--ranking-negatives", type=_positive_int, default=50)
    demo.add_argument("--ranking-max-users", type=_positive_int, default=500)
    demo.add_argument("--synthetic-users", type=_positive_int, default=600)
    demo.add_argument("--synthetic-movies", type=_positive_int, default=160)
    demo.add_argument("--synthetic-mean-ratings", type=_positive_int, default=45)
    demo.add_argument("--synthetic-profiles", type=_positive_int, default=25)
    demo.add_argument("--synthetic-known-facts", type=_positive_int, default=18)
    demo.add_argument("--synthetic-rating-noise", type=float, default=0.15)
    demo.set_defaults(func=cmd_run_demo)

    report = subparsers.add_parser("build-report", help="Build a Markdown report and privacy-utility frontier plot")
    report.add_argument("--prefix", default="", help="File prefix for summary CSVs, e.g. synthetic_")
    report.add_argument("--n-known", type=_positive_int, default=3)
    report.add_argument("--title", default="Netflix Prize Privacy-Utility Audit")
    report.set_defaults(func=cmd_build_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
