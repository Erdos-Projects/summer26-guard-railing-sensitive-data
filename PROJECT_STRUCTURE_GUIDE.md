# Project Structure Guide

This guide reflects the simplified codebase. The maintained package now has a
small number of source files, each with a clear job.

## Mental Model

The project asks:

> How do simple anonymization choices change re-identification risk and
> recommender utility for sparse movie ratings?

The core workflow is:

1. Load real Netflix Prize data or generate synthetic Netflix-like data.
2. Build anonymized release variants.
3. Measure privacy risk with k-anonymity, sampled linkage trials, and an
   IMDb-style linkage attack.
4. Measure utility with a fast recommender baseline.
5. Write a public-safe report.

## Maintained Source Files

### `src/guardrails_sensitive_data/data.py`

Data access and data preparation.

Contains:

- `NetflixPaths`: dataclass with conventional Netflix file locations.
- `MatchConfig`: dataclass for title matching settings.
- `SyntheticProfile`: dataclass for a planted synthetic external profile.
- `ImdbScrapeResult`: dataclass describing a runtime IMDb scrape attempt.

Netflix helpers:

- `netflix_paths(data_dir)`: returns all conventional Netflix paths.
- `missing_netflix_files(data_dir, require_probe=False)`: lists missing files.
- `verify_netflix_files(data_dir, require_probe=False)`: returns a small
  validation report.
- `iter_combined_ratings(...)`: streams rows from `combined_data_*.txt`.
- `read_netflix_ratings(...)`: reads Netflix ratings into a DataFrame.
- `read_movie_titles(path)`: reads movie titles while preserving commas.
- `read_probe(path, max_rows=None)`: reads official probe pairs.

IMDb/title-matching helpers:

- `normalize_title(value)`: normalizes titles for conservative matching.
- `parse_year(value)`: extracts a four-digit year.
- `imdb_rating_to_netflix_expected(value)`: maps IMDb 1-10 ratings to Netflix
  1-5 expected ratings.
- `imdb_rating_bounds(value)`: returns plausible integer Netflix rating bounds.
- `preference_bucket_from_netflix_rating(value)`: returns `liked`, `neutral`,
  or `disliked`.
- `read_imdb_ratings_csv(path)`: reads cached IMDb ratings.
- `empty_imdb_ratings_frame()`: returns an empty frame with the maintained IMDb
  ratings schema.
- `imdb_identifier_to_user_label(identifier)`: turns a username, user id, or
  profile URL into the local profile label used in outputs.
- `imdb_ratings_url_candidates(identifier)`: builds likely IMDb ratings URLs
  from a username, user id, or profile URL.
- `extract_imdb_user_ratings(page_html, username="imdb_profile")`: parses
  user-rating rows from IMDb HTML/JSON payloads.
- `extract_imdb_user_ratings_with_beautifulsoup(...)`: Selenium-friendly
  BeautifulSoup parser that mirrors the original notebook scraping selectors.
- `extract_imdb_profile_year(page_html)`: best-effort detection of profile
  creation/join year for Netflix-era warnings.
- `extract_imdb_earliest_rating_year(page_html)`: best-effort detection of the
  earliest visible `Rated on ...` year when profile creation date is unavailable.
- `find_next_imdb_ratings_url(page_html, current_url)`: finds a next-page link
  when IMDb exposes one as an anchor.
- `scrape_imdb_user_ratings(...)`: best-effort runtime scrape of a public IMDb
  ratings profile. It can use `fetch_method="http"`, `"browser"`, or `"auto"`;
  auto tries HTTP first and then falls back to Selenium/BeautifulSoup. It
  returns `ImdbScrapeResult` and raises a clear `ValueError` when
  fetching/parsing fails.
- `match_imdb_to_netflix(imdb_ratings, movie_titles, config=None)`: attaches
  Netflix movie ids to IMDb rows.

Synthetic demo helpers:

- `make_synthetic_movie_titles(...)`: creates synthetic movie metadata.
- `make_synthetic_netflix_ratings(...)`: creates synthetic sparse ratings.
- `make_synthetic_imdb_profile(...)`: creates a planted IMDb-like profile from
  one synthetic user.

Older notebooks may still import `guardrails_sensitive_data.imdb`,
`guardrails_sensitive_data.netflix_io`, or `guardrails_sensitive_data.synthetic`.
Those names are compatibility aliases to `data.py`; new code should import from
`data.py` directly.

### `src/guardrails_sensitive_data/anonymization.py`

Blue-team releases and privacy-risk summaries.

Contains:

- `ReleaseData`: dataclass describing one release variant.
- `stable_id_hash(...)` and `redact_customer_ids(...)`: public-safe id hashing.
- `normalize_rating_frame(frame)`: checks and standardizes rating columns.
- `add_time_and_rating_features(frame)`: adds month/year and rating buckets.
- `add_rating_noise(frame, seed, flip_probability=0.25)`: creates noisy
  ratings.
- `suppress_low_k_facts(frame, cols, k)`: drops low-k fact combinations.
- `make_releases(...)`: creates all release variants.
- `fact_k_anonymity(frame, cols)`: summarizes equivalence-class sizes.
- `candidate_count_for_rows(...)`: computes candidate-set intersection size.
- `evaluate_release_linkage_risk(...)`: sampled k-known-fact attack trials.
- `summarize_linkage_trials(trials)`: aggregates trial rows.
- `evaluate_releases(...)`: full privacy evaluation for all releases.

Add new anonymization defenses here, usually inside `make_releases`.

### `src/guardrails_sensitive_data/linkage.py`

Red-team linkage attacks.

Contains:

- `LinkageConfig`: dataclass for probabilistic scoring settings.
- `prepare_linkage_facts(matched_ratings)`: converts matched IMDb rows into
  one fact per Netflix movie id.
- `score_candidate_ratings_frame(...)`: scores candidates in an in-memory
  ratings DataFrame.
- `score_netflix_candidates(...)`: streams real Netflix files and scores
  candidate users.
- `run_linkage_attack(...)`: full IMDb-to-Netflix attack once the external IMDb
  ratings have been loaded.
- `run_planted_linkage_benchmark(...)`: synthetic ground-truth benchmark that
  checks whether planted external profiles recover their source users.

Add new red-team record-linkage logic here.

### `src/guardrails_sensitive_data/recommender.py`

Utility evaluation.

Contains:

- `BiasRecommender`: fast regularized user/movie bias model.
- `fit_bias_recommender(...)`: fits the bias model.
- `rmse(...)` and `mae(...)`: rating prediction metrics.
- `sampled_ranking_metrics(...)`: sampled hit-rate@k and NDCG@k.
- `train_test_split_ratings(...)`: random train/test split.
- `compare_release_rmse(...)`: trains/evaluates the recommender on each
  release variant.
- `load_official_probe_holdout(...)`: reconstructs the official probe holdout.

Add new recommender utility metrics here.

### `src/guardrails_sensitive_data/reporting.py`

Portfolio/report output.

Contains:

- `ReportOutputs`: dataclass with report, frontier CSV, and optional plot path.
- `build_frontier_table(...)`: merges privacy and utility summaries.
- `write_frontier_plot(...)`: creates RMSE-versus-linkage-risk scatter plot.
- `build_report(...)`: writes the Markdown report and frontier artifacts.

Add new report sections here.

### `src/guardrails_sensitive_data/cli.py`

Command-line entry point.

Maintained commands:

- `verify-data`
- `linkage-attack`
- `privacy-eval`
- `rmse-eval`
- `run-demo`
- `build-report`

Important command functions:

- `cmd_verify_data(args)`: validates local Netflix files.
- `cmd_linkage_attack(args)`: loads cached IMDb rows or scrapes a public IMDb
  profile on cache miss, runs linkage, emits scrape/profile-age warnings, and
  redacts candidates by default. Browser scraping controls live here as CLI
  flags: `--imdb-fetch-method`, `--imdb-browser-headed`, and
  `--imdb-browser-wait-seconds`.
- `cmd_privacy_eval(args)`: writes privacy summary CSVs.
- `cmd_rmse_eval(args)`: writes utility summary CSV.
- `cmd_run_synthetic_demo(args)`: public-safe one-command demo.
- `cmd_run_demo(args)`: small real-data or synthetic workflow.
- `cmd_build_report(args)`: rebuilds report artifacts from CSV outputs.

Add new CLI commands here only after the underlying logic exists in another
module.

### `src/guardrails_sensitive_data/__init__.py`

Package metadata plus compatibility aliases for old notebook imports.

### `src/guardrails_sensitive_data/__main__.py`

Allows:

```bash
python -m guardrails_sensitive_data
```

## Notebooks

The notebooks are retained for exploration and presentation. They are not the
maintained execution path.

- `00_eda.ipynb`: early exploratory analysis; code-only scratchwork.
- `01_synthetic_netflix_generator.ipynb`: SDV synthetic data prototype.
- `02_ml_deanonymization_attacks.ipynb`: older scikit-learn attack prototype.
  Its ideas are useful background, but the simplified package no longer exposes
  a separate ML attack command.
- `03_als_downstream_and_anonymization_feedback.ipynb`: PySpark ALS utility
  prototype.
- `04_empirical_privacy_utility_study.ipynb`: main report-style notebook.
- `imdb_data.csv`: cached IMDb ratings used by `linkage-attack`.
- `imdb_scrape_test.ipynb`: historical scraping experiment.
- `old/`: historical Adult dataset, Polars, scraping, and early Netflix
  experiments.

## Common Workflows

### Public-safe synthetic demo

```bash
python main.py run-demo --synthetic
```

Source path:

- `cli.py`
- `data.py`
- `anonymization.py`
- `linkage.py`
- `recommender.py`
- `reporting.py`

### Real-data privacy evaluation

```bash
python main.py privacy-eval --max-rows 1000000 --trials 300
```

Source path:

- `data.py` reads ratings.
- `anonymization.py` builds releases and privacy metrics.
- `cli.py` writes CSV outputs.

### IMDb linkage attack

```bash
python main.py linkage-attack --user planktonrules
python main.py linkage-attack --user new_reviewer --ratings-url "https://www.imdb.com/user/p.example/ratings"
python main.py linkage-attack --user "https://www.imdb.com/user/p.example/ratings" --imdb-fetch-method browser --imdb-browser-headed
```

Source path:

- `data.py` reads the cached IMDb CSV or performs a best-effort runtime profile
  scrape, then matches titles. Runtime scraping supports a lightweight HTTP
  parser and an optional Selenium/BeautifulSoup browser fallback.
- `linkage.py` scores candidate Netflix users.
- `anonymization.py` redacts candidate ids for public-safe output.

### Utility evaluation

```bash
python main.py rmse-eval --max-rows 1000000
```

Source path:

- `data.py` reads ratings or probe pairs.
- `recommender.py` evaluates release utility.

### Report generation

```bash
python main.py build-report
python main.py build-report --prefix synthetic_
```

Source path:

- `reporting.py` reads CSV summaries and writes report artifacts.

## Where To Add Future Work

- New data readers or synthetic data knobs: `data.py`.
- New release variants or privacy metrics: `anonymization.py`.
- New linkage attacks: `linkage.py`.
- New recommender utility metrics: `recommender.py`.
- New report sections: `reporting.py`.
- New user-facing commands: `cli.py`.
