# Reproducible forecast-only core

`forecast_core.py` separates the current no-intercept `lh_fit` calculation from the notebook's global variables. It accepts explicitly supplied annual data and returns prediction and score tables. It does not change the frozen 2027 protocol. It is a retrospective research calculation; input prices and forecasts are not established point-in-time merely by passing validation. Keep the input and output.

## Environment and tests

Use Python 3.10 or newer, NumPy and pandas. In a fresh local virtual environment, install `pip install -r requirements-forecast-core.txt`. Kaggle's Python image normally supplies NumPy and pandas already. In the directory containing both Python files, run the synthetic, offline checks with:

```
python -m unittest -v test_forecast_core
```

These checks do not require historical data or network access. A local missing-package error is an environment issue, not a notebook run failure.

## One-command comparison

Prepare a **CSV** with one annual row for each available year. Required columns are `year`, `as_of` (exactly YYYY-05-31), `target_filed` (ISO date after June 30), `target_q2_usd`, `prior_q2_usd`, `price_ratio`, `forecast_mean_p12_pct`. Values used for target, prior revenue, and price must match the intended cohort; for the phosphate-adjusted experiment the training rows need the historical phosphate adjustment already applied. This script does not calculate that adjustment or collect raw inputs. Do not change recent test rows to adjusted values unless the existing experiment calls for it.

Where individually verified, optional `prior_filed`, `price_available` and `forecast_available` ISO dates are checked against the row's cutoff; if any of those columns is supplied, every value must be present. If dates are omitted, no claim about their availability can be made. Historical forecast issue time, source revisions, and archive completeness remain unresolved. Forecast probability is not rain volume.

```
python forecast_core.py --input /private/path/annual_features.csv --output /private/path/forecast_results --test-start 2022 --test-end 2026
```

The command writes prediction and score CSVs. It stops with a clear error if a requested year is absent, a required value is missing, there are duplicate years, a provided input availability date is later than the cutoff, or a fold has fewer than six earlier filed target values. All five variants use identical training and evaluation years, and training-only scaling: the original price-only and no-intercept forecast models, a price-only half-intercept model, and forecast models with half and full training-mean log-residual intercepts. Strengths are fixed choices, not tuned within the test years. Results from already inspected years are exploratory. This runner reproduces the no-intercept forecast formula and tests sensitivities; it is not a full replacement for source ingestion, phosphate calculation, the archived notebook's many research analyses, or their release/vintage audits.
