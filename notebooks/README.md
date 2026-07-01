# Notebooks

Active notebook prototypes:

- `01_synthetic_netflix_generator.ipynb` builds a first-pass synthetic Netflix
  release generator and evaluates distribution, utility, and privacy sanity
  checks.
- `02_ml_deanonymization_attacks.ipynb` adds nearest-neighbor profile matching
  and user-level membership-inference attacks using scikit-learn.
- `03_als_downstream_and_anonymization_feedback.ipynb` adds a PySpark ALS
  downstream recommender baseline and uses latent factor diagnostics to guide
  non-isotropic anonymization noise.
- `04_empirical_privacy_utility_study.ipynb` combines the project pieces into a
  report-style empirical study with narrative, plots, privacy attacks, synthetic
  data, anonymization variants, recommender utility, and takeaways.

Optional notebook ML dependencies:

```bash
python -m pip install -e ".[ml]"
```

Equivalent direct install:

```bash
python -m pip install scikit-learn pyspark sdv
```

PySpark also needs a local Java runtime available on `PATH`.

The notebooks in `old/` are historical exploration. The maintained package path
is the CLI in `src/guardrails_sensitive_data`:

```bash
python main.py verify-data --require-probe
python main.py linkage-attack --user planktonrules
python main.py privacy-eval --max-rows 1000000
python main.py rmse-eval --max-rows 1000000
```

Use notebooks for presentation and visualization, but prefer adding reusable
logic to the package.

## Citations and Algorithms

- Netflix de-anonymization / sparse ratings linkage: Narayanan and Shmatikov,
  "Robust De-anonymization of Large Sparse Datasets" (2008),
  https://arxiv.org/abs/cs/0610105.
- k-anonymity and suppression/generalization: Sweeney, "k-anonymity: a model
  for protecting privacy" (2002), https://doi.org/10.1142/S0218488502001648.
- scikit-learn nearest neighbors, logistic regression, and metrics:
  Pedregosa et al., "Scikit-learn: Machine Learning in Python" (2011),
  https://arxiv.org/abs/1201.0490; API docs:
  https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
  and https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.
- Membership inference: Shokri et al., "Membership Inference Attacks Against
  Machine Learning Models" (2017), https://arxiv.org/abs/1610.05820.
- Matrix factorization and ALS recommenders: Koren, Bell, and Volinsky,
  "Matrix Factorization Techniques for Recommender Systems" (2009),
  https://doi.org/10.1109/MC.2009.263; Spark collaborative filtering docs:
  https://spark.apache.org/docs/latest/ml-collaborative-filtering.html.
- Spark MLlib: Meng et al., "MLlib: Machine Learning in Apache Spark" (2015),
  https://arxiv.org/abs/1505.06807.
- SDV / plug-and-play synthetic data: Synthetic Data Vault docs,
  https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers;
  Gaussian Copula docs,
  https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/gaussiancopulasynthesizer.
- CTGAN: Xu et al., "Modeling Tabular data using Conditional GAN" (2019),
  https://arxiv.org/abs/1907.00503; SDV CTGAN docs:
  https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/ctgansynthesizer.
- TVAE: Xu et al. discuss variational autoencoder tabular synthesis in the
  same CTGAN paper; SDV TVAE docs:
  https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/tvaesynthesizer.
- Differentially private marginal synthetic data / Private-PGM and MST:
  McKenna, Miklau, and Sheldon, "Winning the NIST Contest: A scalable and
  general approach to differentially private synthetic data" (2021),
  https://arxiv.org/abs/2108.04978.
