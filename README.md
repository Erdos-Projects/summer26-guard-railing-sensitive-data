# Netflix Prize Privacy Red-Team / Blue-Team Study

This project studies the privacy-utility tradeoff in the Netflix Prize dataset.
It now has a concrete direction:

1. Use public IMDb user ratings as auxiliary information.
2. Match those ratings to Netflix Prize movie ids.
3. Run a probabilistic record-linkage attack against anonymous Netflix customer
   ids.
4. Create anonymized Netflix releases by removing/generalizing dates,
   coarsening ratings, adding noise, suppressing low-k facts, and removing rare
   movies.
5. Add ML red-team audits with sparse nearest-neighbor linkage and
   membership-inference classifiers.
6. Generate synthetic releases with SDV single-table synthesizers.
7. Compare privacy risk with k-anonymity, sampled linkage attacks, and ML
   attacks.
8. Compare utility with recommender baselines, including the package bias model
   and notebook PySpark ALS experiments.

The maintained package implementation lives in `src/guardrails_sensitive_data`.
The active notebooks in `notebooks/` are empirical studies and prototypes; the
old notebooks in `notebooks/old/` are historical exploration.

## Ethical Scope

This repository is for a data privacy course/project. Run linkage attacks only
on public IMDb profiles used for demonstration, and do not publish or contact
candidate real-world identities. The purpose is to quantify privacy risk and
evaluate defenses.

## Data

The Netflix Prize data is not redistributed here. Place the official files in
`data/netflix/`:

- `combined_data_1.txt`
- `combined_data_2.txt`
- `combined_data_3.txt`
- `combined_data_4.txt`
- `movie_titles.csv` or `movie_titles.txt`
- `probe.txt` for official holdout evaluation

This workspace already has the Netflix files locally, so normal commands can
run without downloading anything.

Verify the local data:

```bash
python main.py verify-data --require-probe
```

If you have an authorized archive URL or local archive:

```bash
python main.py download-netflix --url "https://example.com/authorized/netflix.zip"
python main.py download-netflix --archive /path/to/netflix.zip
```

## Environment

The recommended environment is Python 3.12. The ML notebooks use current
compatible major versions of scikit-learn, SDV, and PySpark:

- `scikit-learn>=1.9,<2.0`
- `sdv>=1.37,<2.0`
- `pyspark[sql]>=4.1,<4.2`
- `pyarrow>=15.0`

PySpark 4.1 requires Java 17 or later with `JAVA_HOME` set. On macOS, for
example:

```bash
brew install openjdk@17
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
```

Using conda is the easiest path because `environment.yml` installs the package
plus the notebook ML dependencies:

```bash
conda env create -f environment.yml
conda activate erdos_project_environment
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ml]"
```

For the lightweight CLI-only install, omit the ML extra:

```bash
python -m pip install -e .
```

The CLI works from the repository root even before installation:

```bash
python main.py --help
```

After installation, the same commands are available as:

```bash
netflix-privacy --help
```

## Notebooks

Active notebooks:

- `notebooks/01_synthetic_netflix_generator.ipynb` uses SDV
  `GaussianCopulaSynthesizer` by default, with simple switches for CTGAN and
  TVAE.
- `notebooks/02_ml_deanonymization_attacks.ipynb` uses scikit-learn sparse
  vectors, `NearestNeighbors`, and `LogisticRegression` for red-team audits.
- `notebooks/03_als_downstream_and_anonymization_feedback.ipynb` uses PySpark
  MLlib ALS and item-factor diagnostics to guide targeted anonymization noise.
- `notebooks/04_empirical_privacy_utility_study.ipynb` is the cohesive
  report-style study that combines probabilistic linkage, anonymization,
  nearest-neighbor attacks, membership inference, synthetic data, downstream
  utility, plots, citations, and takeaways.
- final_pipeline.ipynb contains the final anonymization pipeline with the final privacy-utility curves.

Run notebooks from the repository root so relative imports and data paths line
up:

```bash
jupyter lab
```

## IMDb Ratings

The cached exploratory file `notebooks/imdb_data.csv` includes ratings for
`planktonrules` and other public IMDb users. Use that for reproducible runs:

```bash
python main.py linkage-attack --user planktonrules
```

To fetch a public IMDb ratings page into a CSV, supply a user id or, preferably,
the full `/ratings` URL:

```bash
python main.py scrape-imdb \
  --user planktonrules \
  --ratings-url "https://www.imdb.com/user/urXXXXXXXX/ratings" \
  --output reports/imdb_planktonrules.csv
```

IMDb markup changes often and may require login or JavaScript. Cached CSVs are
the reliable research path.

## Experiments

Run the probabilistic linkage attack:

```bash
python main.py linkage-attack --user planktonrules --top-n 50
```

Outputs:

- `reports/linkage_planktonrules_matched_titles.csv`
- `reports/linkage_planktonrules_facts.csv`
- `reports/linkage_planktonrules_candidates.csv`

Evaluate anonymization defenses:

```bash
python main.py privacy-eval --max-rows 1000000 --trials 300
```

Outputs:

- `reports/privacy_k_anonymity_summary.csv`
- `reports/privacy_linkage_trials.csv`
- `reports/privacy_linkage_summary.csv`

Compare downstream RMSE:

```bash
python main.py rmse-eval --max-rows 1000000
```

For the official probe holdout, use:

```bash
python main.py rmse-eval --holdout probe --max-train-rows 5000000 --max-probe-rows 100000
```

The default RMSE path uses a random holdout from a sampled subset, which is much
faster and is enough to compare releases consistently.

Run a small end-to-end demo:

```bash
python main.py run-demo --max-rows 200000 --trials 50
```

For the full empirical report, open:

```bash
jupyter lab notebooks/04_empirical_privacy_utility_study.ipynb
```

The report notebook has runtime knobs at the top, including `MAX_ROWS`,
`PRIVACY_TRIALS`, `PROFILE_USERS`, and `SYNTH_TRAIN_ROWS`.

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

`movie_only` is useful for privacy comparison but is skipped for RMSE because it
does not release rating labels.

## Method References

The notebooks cite the full reference list. The main methodological anchors are:

- Narayanan and Shmatikov, "Robust De-anonymization of Large Sparse Datasets"
  for the Netflix linkage threat model.
- Sweeney's k-anonymity model for fact suppression/generalization.
- scikit-learn nearest-neighbor and logistic-regression models for ML red-team
  audits.
- SDV, Gaussian Copula, CTGAN, and TVAE for plug-and-play synthetic tabular
  data generation.
- Spark MLlib ALS and matrix factorization for the stronger downstream
  recommender baseline.
- Membership-inference attacks for user-presence risk.

## Tests

```bash
python -m unittest discover
```

The tests use tiny synthetic Netflix files and do not require the full dataset.
