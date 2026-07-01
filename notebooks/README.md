# Notebooks

These notebooks are legacy exploratory work. The maintained project path is the
CLI in `src/guardrails_sensitive_data`:

```bash
python main.py verify-data --require-probe
python main.py linkage-attack --user planktonrules
python main.py privacy-eval --max-rows 1000000
python main.py rmse-eval --max-rows 1000000
```

Use notebooks for presentation and visualization, but prefer adding reusable
logic to the package.
