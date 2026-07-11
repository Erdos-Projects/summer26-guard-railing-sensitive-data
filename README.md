# Netflix Prize Privacy-Utility Audit

This project studies a focused question:

> How much re-identification risk is created by sparse movie/rating/date facts,
> and how much recommender utility survives simple anonymization defenses?

Concretely, we are able to achieve the following:

1. Load Netflix Prize data or generate synthetic Netflix-like data.
2. Build anonymized release variants.
3. Measure k-anonymity and sampled linkage risk.
4. Run an IMDb-style probabilistic linkage attack.
5. Compare downstream recommender utility.
6. Build a public-safe Markdown report and privacy-utility frontier plot.

## Quick Demo

Run the public-safe demo without the restricted Netflix Prize files:

```bash
python main.py run-demo --synthetic
```

This writes synthetic data summaries, a planted linkage benchmark, a
privacy-utility frontier, and:

```text
reports/synthetic_privacy_utility_report.md
```

## Source Layout

The maintained package lives in `src/guardrails_sensitive_data/`:

- `data.py`: Netflix file readers, IMDb cache/runtime profile loading, title
  matching, and synthetic Netflix-like data generation.
- `anonymization.py`: release variants, k-anonymity summaries, sampled
  linkage-risk evaluation, and public-safe id hashing.
- `linkage.py`: probabilistic IMDb-to-Netflix linkage attack and planted
  synthetic linkage benchmark.
- `recommender.py`: fast bias-model recommender plus RMSE, MAE, hit-rate, and
  NDCG utility metrics.
- `reporting.py`: privacy-utility frontier table, plot, and Markdown report.
- `cli.py`: command-line orchestration.
- `__main__.py` and `__init__.py`: package entry points and compatibility
  aliases for older notebooks.

The older `imdb`, `netflix_io`, and `synthetic` module names are aliased to
`data.py` for notebook compatibility, but new code should import from
`guardrails_sensitive_data.data`.

For a more detailed map, see:

```text
PROJECT_STRUCTURE_GUIDE.md
```

## Ethical Scope

This repository is for privacy-risk measurement and defense evaluation. Do not
publish raw candidate Netflix customer ids, and do not contact or identify
candidate real-world users.

Detailed CLI outputs use hashed identifiers by default. The
`--unsafe-include-customer-ids` option exists only for private local debugging.

## Data

The Netflix Prize data is not redistributed here. If you have authorized access,
place the official files in `data/netflix/`:

- `combined_data_1.txt`
- `combined_data_2.txt`
- `combined_data_3.txt`
- `combined_data_4.txt`
- `movie_titles.csv` or `movie_titles.txt`
- `probe.txt` for official holdout evaluation

Verify local files:

```bash
python main.py verify-data --require-probe
```

The cached IMDb ratings file used by default is:

```text
notebooks/imdb_data.csv
```

`linkage-attack` uses that CSV first. If you pass a username/user id/profile URL
that is not in the cache, the CLI attempts a best-effort live scrape of the
public IMDb ratings page at runtime. By default it tries lightweight HTTP first,
then falls back to a Selenium-rendered Chrome page when the optional scrape
dependencies are installed. IMDb can block automated traffic or change markup,
so scraped runs print explicit success/failure messages and warnings when
profile age cannot be detected, when profile creation or earliest visible rating
activity appears to be after 2006, or when scraped rated titles fall outside the
Netflix Prize data window.

For fully reproducible cache-only runs, add `--no-scrape`.

## Environment

Recommended Python: 3.12.

Install the lightweight CLI/package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For the notebook experiments with SDV, scikit-learn, and PySpark:

```bash
python -m pip install -e ".[ml]"
```

For runtime IMDb scraping with Selenium and BeautifulSoup:

```bash
python -m pip install -e ".[scrape]"
```

Conda users can also use:

```bash
conda env create -f environment.yml
conda activate erdos_project_environment
```

## CLI Commands

```bash
python main.py --help
```

Maintained commands:

- `verify-data`: check local Netflix Prize files.
- `linkage-attack`: run IMDb-to-Netflix probabilistic linkage from cached IMDb
  rows or a live public profile scrape.
- `privacy-eval`: evaluate anonymized releases with k-anonymity and sampled
  candidate-set sizes.
- `rmse-eval`: compare recommender utility across release variants.
- `run-demo`: run a small end-to-end demo; use `--synthetic` for public-safe
  data.
- `build-report`: build or rebuild the Markdown report and frontier artifacts
  from existing CSV outputs.

Examples:

```bash
python main.py linkage-attack --user planktonrules --top-n 50
python main.py linkage-attack --user new_reviewer --ratings-url "https://www.imdb.com/user/p.example/ratings"
python main.py linkage-attack --user "https://www.imdb.com/user/p.example/ratings" --imdb-max-pages 2
python main.py linkage-attack --user "https://www.imdb.com/user/p.example/ratings" --imdb-fetch-method browser --imdb-browser-headed
python main.py privacy-eval --max-rows 1000000 --trials 300
python main.py rmse-eval --max-rows 1000000
python main.py run-demo --synthetic
python main.py build-report --prefix synthetic_ --title "Synthetic Netflix Privacy-Utility Audit"
```

For the official probe holdout:

```bash
python main.py rmse-eval --holdout probe --max-train-rows 5000000 --max-probe-rows 100000
```

## Release Variants

The blue-team releases currently evaluated are:

- original movie + exact rating + month
- remove month
- generalize month to year
- coarsen rating into disliked / neutral / liked
- remove month and coarsen rating
- add bounded rating noise
- remove rare movies
- suppress low-k movie/rating/month facts
- movie only

`movie_only` is useful for privacy comparison but skipped for RMSE because it
does not release rating labels.

## Notebooks

The notebooks are retained as research/prototype material. The most important
one is:

```text
notebooks/04_empirical_privacy_utility_study.ipynb
```

The maintained execution path is the CLI and package code. Prefer adding new
reusable logic to `src/guardrails_sensitive_data/` and using notebooks for
explanation and plots.

## Public Output Hygiene

`reports/` is for generated local artifacts. Aggregated summaries and synthetic
reports are safe to share; detailed attack outputs should be redacted or
regenerated from synthetic data before publication.

Local Netflix Prize data, generated reports, bytecode, and package build
artifacts are ignored by `.gitignore`.

## Method References

The main methodological anchors are:

- Narayanan and Shmatikov, "Robust De-anonymization of Large Sparse Datasets"
  for the Netflix linkage threat model.
- Sweeney's k-anonymity model for fact suppression/generalization.
- Regularized bias recommenders and matrix-factorization-style utility metrics
  for downstream recommendation quality.
- SDV and Spark ALS remain notebook-only extensions, not part of the simplified
  core package.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover
```

The tests use tiny synthetic Netflix files and do not require the full dataset.
GitHub Actions runs the same suite on Python 3.12.
