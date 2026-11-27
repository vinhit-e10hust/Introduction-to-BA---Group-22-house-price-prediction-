# Group 22 — Modeling & Evaluation → Deployment Handoff

**Project:** House Price Prediction for a Real Estate Listing Platform  
**Source:** `main` of [Group 22 repository](https://github.com/vinhit-e10hust/Introduction-to-BA---Group-22-house-price-prediction-)  
**Recipient:** Deployment / App Development member  
**Scope:** Deploy the trained **historical Ames Housing** model as a working demonstration; do not claim applicability to current property values without further validation.

## 1. What has been completed

- Data Engineering: processed Ames data and documented synthetic features.
- Data Analysis: EDA notebook, report, and figures.
- Modeling: preprocessing, training, evaluation, and prediction source code in `src/models/`.
- **Selected model (from local evaluation supplied by the team):** XGBoost regressor with USD target, 5-fold CV; held-out 2010 sales: **175 properties**, **RMSE $26,695**, **MAE $15,851**, **MAPE 9.22%**, **R² 0.8893**. Nominal 90% prediction interval coverage observed: **89.14%**, average interval width **$72,499**. These are historical evaluation results, **not live production metrics**.

**Important availability distinction:** Source code is on `main`, but `models/house_price.joblib` and `reports/modeling/` were **not found in the inspected GitHub tree**. They were generated locally by the training run. The model owner must transfer the artifact and any reports to the Deployment member through an agreed, trusted channel before inference can run on another machine.

## 2. File map

| Path | Function |
|---|---|
| `src/models/features.py` | Required input columns (`LISTING_INPUTS`), feature engineering, categorical/numeric preprocessing |
| `src/models/predict.py` | CLI and `predict_prices(bundle, dataframe)` inference function |
| `src/models/train.py` | Training and report/artifact generation |
| `src/models/evaluation.py` | Evaluation metrics, residual analysis and holdout reports |
| `src/models/split.py` | Historical development/calibration/holdout splitting |
| `src/data/common.py` | CSV loading that preserves special Ames `NA` categories |
| `data/processed/test_cleaned.csv` | 1,459 unlabeled Ames rows for a smoke test (not an evaluation set) |
| `models/house_price.joblib` | **Locally created trained bundle; must be transferred** |
| `reports/modeling/model_metadata.json` | **Locally created** model config, input feature list and calibration details |
| `reports/modeling/holdout_metrics.csv` | **Locally created** held-out evaluation metrics |

## 3. Setup (Windows PowerShell)

From a clean clone, in the repository root:

```powershell
git clone https://github.com/vinhit-e10hust/Introduction-to-BA---Group-22-house-price-prediction-.git
cd Introduction-to-BA---Group-22-house-price-prediction-
git switch main
git pull --ff-only origin main
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install pandas numpy scipy scikit-learn==1.6.1 xgboost==3.0.1 joblib matplotlib
```

Use the exact Python version and lockfile provided by the Modeling member if available. Versions above reflect versions used in the local test; they are not a complete, validated deployment lockfile. The app framework (for example Streamlit) should be added to a reproducible requirements file by the Deployment member.

Copy the **trusted** trained artifact to `models/house_price.joblib` before proceeding. Never load an untrusted `.joblib`/pickle: deserialization can execute code.

## 4. Smoke-test inference before building the app

Run from the repository root (use the same Python interpreter used to install dependencies):

```powershell
.\.venv\Scripts\python.exe -m src.models.predict `
    data/processed/test_cleaned.csv `
    --model models/house_price.joblib `
    --output test_predictions.csv

(Import-Csv .\test_predictions.csv).Count
Import-Csv .\test_predictions.csv | Select-Object -First 5 | Format-Table
```

**Expected:** 1,459 rows, columns `Id` (when present), `PredictedSalePrice`, `LowerPrice`, `UpperPrice`, `PredictedUSDPerSquareMeter`. Prices are **USD**, living area for the per-square-meter estimate derives from `GrLivArea` in **square feet**. The test file has no `SalePrice`; successful inference is not a substitute for evaluation.

If the artifact is absent, coordinate with Modeling to transfer it. Avoid immediately running the full training grid on every deployment environment.

## 5. App integration contract

The trained artifact is a **dict**, not a plain estimator:

```python
from pathlib import Path
import joblib
import pandas as pd
from src.models.predict import predict_prices
from src.models.features import LISTING_INPUTS

bundle = joblib.load(Path('models/house_price.joblib'))  # trusted team artifact only
# `row` must contain all the same Ames-format fields as LISTING_INPUTS.
# Construct from a validated form/CSV, preserving categorical values and numeric types.
input_df = pd.DataFrame([row])
result = predict_prices(bundle, input_df)
print(result.to_dict(orient='records')[0])
```

- Required schema: `LISTING_INPUTS` from `src/models/features.py` and the artifact's `input_columns` metadata. **Do not assume a minimal 4–5-field form will work**: the current model expects dozens of Ames-style fields.
- The selected model **does not require** `SalePrice`, `YrSold`, `DaysOnMarket` or simulated distances as deployment inputs.
- Keep the exact Ames categorical values, including literal `NA` meaning an absent amenity for certain columns. Treat missing and unknown values consistently with the `read_housing()` rules.
- `GrLivArea` must be positive to calculate meaningful USD/m². Check all required fields and numeric ranges before calling inference.
- `LowerPrice` and `UpperPrice` are calibrated prediction bounds, **not guarantees** or formal appraisals.
- `features.py` uses `REFERENCE_YEAR = 2010` to derive house age. This is suitable for the historical demo, **not current-year valuation** without reviewing and retraining the model.

## 6. Recommended minimum demo scope

**Stage A: functional demo first**
1. App loads the trusted model once and displays its name/version.
2. User uploads an Ames-format CSV (or selects one existing record from the supplied test file).
3. Validate the required columns and show clear errors for wrong/missing inputs.
4. Predict and show price USD, calibrated interval, price USD/m², and selected property information.
5. Optionally provide CSV download of predictions and a simple chart.

**Stage B: richer UI, only if feasible**
- A manual listing form may be built, but it requires supplying/mapping *all* expected model inputs and explicit defaults for optional properties; do not silently invent missing attributes.
- Add contextual warnings about historical 2006–2010 Ames data and synthetic features.

A lightweight **Streamlit** application is a reasonable implementation choice; FastAPI is an alternative if a separate frontend/backend is required. These are recommendations, not claims that deployment code already exists.

## 7. Monitoring / logging handoff

For each demo prediction, log a request ID, timestamp, model version, input validation outcome, missing-field counts, processing time, and prediction summary. Avoid storing sensitive user-entered details unnecessarily.

Without an observed future sale price, monitoring cannot compute live MAE/RMSE. Until labels arrive, monitor request volume, errors, latency, input ranges, missingness, and distribution drift relative to training. When true sale prices become available, add labeled performance monitoring by relevant property cohorts.

## 8. Acceptance checklist

- [ ] `main` contains `src/models/` and the deployment app code has its own branch/PR.
- [ ] Trained `.joblib` and model metadata transferred securely; hashes/version recorded.
- [ ] Requirements and supported Python versions documented.
- [ ] New environment can load model and predict all 1,459 Kaggle test rows.
- [ ] App validates input schema, units, missing values and unknown categories.
- [ ] Display includes predicted USD, bounds, and meaningful errors.
- [ ] No leakage-only fields required in app input.
- [ ] Historical dataset limitations are clearly disclosed.
- [ ] App screenshots/demo link, deployment instructions and a basic monitoring/logging plan committed.
- [ ] Responsible member creates PR into `main` after review.

## 9. Suggested owner workflow

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c feature/deployment
# implement and test app, documentation and configuration
# git add only intended deployment files, then commit and push
```

**Questions to confirm with Modeling before publishing a demo:** exact model file/checksum, Python/package versions, final validation metrics, which UI inputs can safely be collected, and ownership of retraining when moving beyond historical Ames data.
