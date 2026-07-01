"""Command line interface for the Netflix privacy project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .anonymization import evaluate_releases
from .imdb import MatchConfig, read_imdb_ratings_csv, scrape_imdb_user_ratings
from .linkage import LinkageConfig, run_linkage_attack
from .netflix_io import (
    download_file,
    netflix_paths,
    read_movie_titles,
    read_netflix_ratings,
    unpack_archive,
    verify_netflix_files,
)
from .paths import DEFAULT_IMDB_CSV, DEFAULT_NETFLIX_DIR, DEFAULT_OUTPUT_DIR, ensure_directory
from .recommender import compare_release_rmse, load_official_probe_holdout, train_test_split_ratings


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def cmd_verify_data(args: argparse.Namespace) -> int:
    report = verify_netflix_files(args.data_dir, require_probe=args.require_probe)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def cmd_download_netflix(args: argparse.Namespace) -> int:
    ensure_directory(args.data_dir)
    archive_path = args.archive
    if args.url:
        destination = args.destination or (args.data_dir / Path(args.url).name)
        archive_path = download_file(args.url, destination)
        print(f"Downloaded archive to {archive_path}")

    if archive_path:
        unpack_archive(archive_path, args.data_dir)
        print(f"Unpacked archive into {args.data_dir}")

    report = verify_netflix_files(args.data_dir, require_probe=False)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def cmd_scrape_imdb(args: argparse.Namespace) -> int:
    output = args.output or (DEFAULT_OUTPUT_DIR / f"imdb_{args.user or 'ratings'}.csv")
    ensure_directory(output.parent)
    frame = scrape_imdb_user_ratings(
        args.ratings_url or args.user,
        username=args.user,
        sleep_seconds=args.sleep_seconds,
    )
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame):,} IMDb ratings to {output}")
    return 0


def cmd_linkage_attack(args: argparse.Namespace) -> int:
    output_dir = ensure_directory(args.output_dir)
    paths = netflix_paths(args.data_dir)
    imdb_ratings = read_imdb_ratings_csv(args.imdb_csv)
    movie_titles = read_movie_titles(paths.movie_titles_file)

    matched, facts, candidates = run_linkage_attack(
        imdb_ratings=imdb_ratings,
        movie_titles=movie_titles,
        data_dir=args.data_dir,
        user=args.user,
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

    prefix = f"linkage_{args.user}" if args.user else "linkage_all_users"
    matched_path = output_dir / f"{prefix}_matched_titles.csv"
    facts_path = output_dir / f"{prefix}_facts.csv"
    candidates_path = output_dir / f"{prefix}_candidates.csv"
    matched.to_csv(matched_path, index=False)
    facts.to_csv(facts_path, index=False)
    candidates.to_csv(candidates_path, index=False)

    print(f"Matched {len(facts):,} unique Netflix titles from {len(matched):,} IMDb rows.")
    print(f"Wrote candidates to {candidates_path}")
    if not candidates.empty:
        print(candidates.head(min(10, len(candidates))).to_string(index=False))
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
    )
    output = output_dir / "utility_rmse_summary.csv"
    summary.to_csv(output, index=False)
    print(f"Train rows: {len(train):,}; test rows: {len(test):,}")
    print(f"Wrote RMSE summary to {output}")
    print(summary.to_string(index=False))
    return 0


def cmd_run_demo(args: argparse.Namespace) -> int:
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
    )
    cmd_privacy_eval(privacy_args)
    cmd_rmse_eval(rmse_args)
    if args.imdb_csv.exists():
        linkage_args = argparse.Namespace(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            imdb_csv=args.imdb_csv,
            user=args.user,
            no_fuzzy=False,
            fuzzy_threshold=0.93,
            fuzzy_margin=0.03,
            rating_sigma=0.75,
            missing_log_penalty=-0.75,
            min_matches=2,
            top_n=25,
        )
        cmd_linkage_attack(linkage_args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Netflix Prize privacy/utility experiments")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_NETFLIX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-data", help="Validate local Netflix Prize files")
    verify.add_argument("--require-probe", action="store_true")
    verify.set_defaults(func=cmd_verify_data)

    download = subparsers.add_parser("download-netflix", help="Download/unpack an explicitly supplied Netflix archive")
    download.add_argument("--url", help="Authorized archive URL")
    download.add_argument("--archive", type=Path, help="Local zip/tar archive to unpack")
    download.add_argument("--destination", type=Path, help="Download destination path")
    download.set_defaults(func=cmd_download_netflix)

    scrape = subparsers.add_parser("scrape-imdb", help="Fetch a public IMDb user's ratings page")
    scrape.add_argument("--user", required=True, help="IMDb username/user id used for output labeling")
    scrape.add_argument("--ratings-url", help="Full IMDb /ratings URL; preferred when known")
    scrape.add_argument("--sleep-seconds", type=float, default=1.0)
    scrape.add_argument("--output", type=Path)
    scrape.set_defaults(func=cmd_scrape_imdb)

    linkage = subparsers.add_parser("linkage-attack", help="Run IMDb-to-Netflix probabilistic linkage")
    linkage.add_argument("--imdb-csv", type=Path, default=DEFAULT_IMDB_CSV)
    linkage.add_argument("--user", help="Restrict to a single IMDb user, e.g. planktonrules")
    linkage.add_argument("--no-fuzzy", action="store_true")
    linkage.add_argument("--fuzzy-threshold", type=float, default=0.93)
    linkage.add_argument("--fuzzy-margin", type=float, default=0.03)
    linkage.add_argument("--rating-sigma", type=float, default=0.75)
    linkage.add_argument("--missing-log-penalty", type=float, default=-0.75)
    linkage.add_argument("--min-matches", type=_positive_int, default=2)
    linkage.add_argument("--top-n", type=_positive_int, default=50)
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
    rmse_eval.set_defaults(func=cmd_rmse_eval)

    demo = subparsers.add_parser("run-demo", help="Run linkage, privacy, and RMSE on manageable samples")
    demo.add_argument("--max-rows", type=_positive_int, default=200_000)
    demo.add_argument("--trials", type=_positive_int, default=50)
    demo.add_argument("--seed", type=int, default=333)
    demo.add_argument("--epochs", type=_positive_int, default=6)
    demo.add_argument("--regularization", type=float, default=10.0)
    demo.add_argument("--rare-movie-min-users", type=_positive_int, default=100)
    demo.add_argument("--k-suppression", type=_positive_int, default=5)
    demo.add_argument("--imdb-csv", type=Path, default=DEFAULT_IMDB_CSV)
    demo.add_argument("--user", default="planktonrules")
    demo.set_defaults(func=cmd_run_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
