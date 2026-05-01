# VinDatathon 2026 Revenue and COGS Forecasting

This repository contains the forecasting solution for the VinDatathon 2026 Vietnamese fashion e-commerce task. The goal is to predict daily `Revenue` and `COGS` for `2023-01-01` through `2024-07-01` from historical business, transaction, analytical, and operational data.

## Project Layout

```text
notebooks/final.ipynb      Clean Kaggle submission notebook
src/forecasting/           Reusable pipeline package
scripts/train.py           Train full v3 pipeline and write artifacts
scripts/predict.py         Rebuild submission from trained artifacts
scripts/evaluate.py        Print CV and submission sanity report
scripts/tune.py            Optional calibration candidate writer
```

The required dataset files are:

```text
sales.csv
sample_submission.csv
promotions.csv
web_traffic.csv
inventory.csv
orders.csv
customers.csv             optional for the model
```

Other master and transaction files may be present, but the v3 solution uses the files above.

## Solution Summary

The solution is a tabular time-series ensemble. Instead of recursively forecasting daily sales, it builds deterministic calendar, holiday, promotion, and business features for both the training dates and the hidden test horizon, then trains direct regressors for daily log `Revenue` and log `COGS`.

The main modeling choices are:

- `Revenue` and `COGS` are modeled separately on `log(value)`.
- Training uses sample weights that emphasize the high-signal 2014-2018 era and downweight weaker regimes.
- The base ensemble combines Ridge, LightGBM, and CatBoost.
- Quarter-specialist LightGBM and CatBoost models are trained by boosting sample weights for Q1-Q4 and composing predictions by the test date quarter.
- Final predictions use manual blending and calibration from the v3 notebook:
  - `alpha = 0.60`
  - `q_boost = 3.0`
  - `Revenue calibration = 1.32`
  - `COGS calibration = 1.36`

This design performed better than the earlier heavier architecture because the test period is a fixed known horizon and the strongest signal comes from calendar seasonality, promotion windows, Tet effects, and historical regime weighting.

## Feature Architecture

`src/forecasting/features.py` implements the feature layer used by the final solution:

- Calendar: year, month, day, day of week, day of year, quarter, week, month edges.
- Cyclical encodings: yearly, weekly, and monthly Fourier features.
- Regime and trend: pre-2019, 2019, post-2019 flags and trend interactions.
- Vietnamese retail calendar: Tet proximity windows, fixed holidays, Black Friday, double-day events.
- Promotion windows: spring sale, mid-year sale, fall launch, year-end, and odd-year campaign proxies.
- Odd/even-year interactions: month and quarter effects that were important in the v3 notebook.
- Operational signal: monthly order-volume pattern from `orders.csv`.

The feature builder creates one unified frame over train and test dates so the same columns are used for fitting and submission.

## Local Setup

This project uses Python 3.12 with `uv`.

```bash
uv sync
```

By default, the code reads data from `dataset/`. To use another location, set:

```bash
export DATATHON_DATA_DIR=/path/to/competition/files
```

## Running the Pipeline

Train all models, write artifacts, and create `submission.csv`:

```bash
uv run python scripts/train.py
```

Rebuild `submission.csv` from existing artifacts:

```bash
uv run python scripts/predict.py
```

Print the cross-validation and submission sanity report:

```bash
uv run python scripts/evaluate.py
```

Optionally write calibrated candidate submissions from trained artifacts:

```bash
uv run python scripts/tune.py
```

## Kaggle / Judge Notebook

Open [`notebooks/final.ipynb`](notebooks/final.ipynb) on Kaggle. It is standalone and expects the competition dataset at:

```text
/kaggle/input/competitions/datathon-2026-round-1
```

The notebook:

1. Loads the required CSV files.
2. Builds the same features as `src/forecasting`.
3. Trains Ridge, LightGBM, CatBoost, and quarter specialists.
4. Blends and calibrates predictions.
5. Writes `submission.csv`.

## Validation

The repository includes a rolling validation scheme from the v3 notebook:

| Fold | Train end | Validation range |
|---|---:|---|
| `A_primary` | 2021-12-31 | 2022-01-01 to 2022-12-31 |
| `B_stability` | 2020-12-31 | 2021-01-01 to 2021-12-31 |
| `C_horizon` | 2021-06-30 | 2021-07-01 to 2022-06-30 |

The CV metric is MAE on linear-scale `Revenue` after training LightGBM on log-transformed targets. The final submission is additionally checked for row count, date range, positivity, and `COGS / Revenue` sanity.

## Development Checks

```bash
uv run ruff check src/ scripts/
uv run pytest
```

There may be no local tests in a fresh competition workspace; in that case, the most important smoke check is running `scripts/train.py` on the provided dataset and verifying the generated `submission.csv`.
