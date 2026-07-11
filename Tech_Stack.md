# Tech Stack

This project is now a Python data pipeline rather than a notebook-only
experiment.

## Runtime

- Python 3.12+
- pandas and NumPy for data processing
- Matplotlib for report plots
- Optional notebook stack: Jupyter, Seaborn, SDV, scikit-learn, and PySpark

## Project Interface

- `python main.py ...` works from the repository root.
- After installing with `pip install -e .`, the same CLI is available as
  `netflix-privacy ...`.
- `python -m unittest discover` is the supported test command, so the test suite
  does not require extra dev dependencies.

## Data Layout

- `data/netflix/`: local Netflix Prize files. These are intentionally ignored
  by git because the dataset license restricts redistribution.
- `notebooks/imdb_data.csv`: cached IMDb ratings collected during exploration.
- `reports/`: generated experiment outputs.

## Modeling

The default utility model is a regularized user/movie bias recommender. It is
much faster than the exploratory Spark/ALS notebook and is sufficient for
comparing anonymized releases against the original release.

## Maintained Source Modules

- `data.py`
- `anonymization.py`
- `linkage.py`
- `recommender.py`
- `reporting.py`
- `cli.py`
