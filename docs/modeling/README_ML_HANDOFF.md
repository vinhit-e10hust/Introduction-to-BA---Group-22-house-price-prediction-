# Group 22 — Data Engineering & Data Analysis Handoff to ML Engineering

**Project:** House Price Prediction for a Real Estate Listing Platform  
**Source branch:** `main` (contains merged DE and DA work)  
**Target:** Predict residential property sale price (`SalePrice`, USD) using the Ames Housing dataset.

## 1. Business problem and modeling objective

Develop a reproducible supervised regression pipeline to estimate house sale prices from available property characteristics. Compare candidate regression models using a fair train/validation strategy; communicate error metrics and limitations. Interpret model performance in the context of practical price estimation, not causal inference.

## 2. Repository and inputs

Repository: https://github.com/vinhit-e10hust/Introduction-to-BA---Group-22-house-price-prediction-

| File | Purpose |
|---|---|
| `data/processed/train_cleaned.csv` | **Primary labeled modeling dataset:** 1,460 rows, 89 columns including `SalePrice` |
| `data/processed/test_cleaned.csv` | Kaggle unlabeled test set: 1,459 rows, 88 columns; **no `SalePrice`** |
| `data/processed/train_extended.csv` | Pre-cleaning integrated train export; DA found it byte-identical to `train_cleaned.csv` in the current snapshot |
| `docs/data_engineering/data_dictionary.xlsx` | Definitions and constraints of generated features |
| `docs/data_engineering/synthetic_assumptions.md` | Generation formulas, synthetic status and inference-time caveats |
| `docs/data_engineering/cleaning_report.md` | Data cleaning choices and limitations |
| `docs/data_analysis/eda/DA_EDA_Report.md` | EDA conclusions, associations and outlier sensitivity |
| `src/eda/eda_tasks_1_6.ipynb` | Reproducible analysis and figures |
| `src/eda/eda_tables_viewer.ipynb` | EDA result tables/snapshots |
| `reports/figures/eda/` | Exported figures |
| `reports/data_quality/` | Profiling, missing values and IQR reports |

### Quick start (Windows PowerShell, from the repository root)

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
python -m pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter ipykernel
python -c "import pandas as pd; d=pd.read_csv('data/processed/train_cleaned.csv'); print(d.shape, 'SalePrice' in d.columns)"
```

Expected check: `(1460, 89) True`. If `git switch` fails because of uncommitted modifications, save/stash/commit those changes first; do not discard them blindly.

## 3. Key DA findings to consider

The following numbers are **descriptive results reported by the DA** on the 1,460 labeled Ames records, not model feature-importance estimates or causal effects:

- Median sale price: **$163,000**; mean: **$180,921**; sale-price distribution positively skewed (skewness ≈ **1.88**).
- `OverallQual` versus `SalePrice`: Pearson **r ≈ 0.791**; Spearman **ρ ≈ 0.810**.
- `GrLivArea` versus `SalePrice`: Pearson **r ≈ 0.709**; Spearman **ρ ≈ 0.731**.
- Neighborhood median prices vary. Compare neighborhood groups with their sample counts (small groups may be unstable).
- Pairs with correlated information include `GarageCars`–`GarageArea` (**r ≈ 0.882**) and `GrLivArea`–`TotRmsAbvGrd` (**r ≈ 0.825**). Correlation alone does **not** require dropping either feature; test impacts through cross-validation and model choice.
- IQR flags: `SalePrice` 61, `GrLivArea` 31, `LotArea` 69, `TotalBsmtSF` 61, `GarageArea` 21. Do not remove all flagged rows automatically. DA found that excluding SalePrice flags changes mean price and area-price correlation materially.
- Reported missing cells in labeled train data: **357** (notably `LotFrontage`: 259; `GarageYrBlt`: 81; `MasVnrType` and `MasVnrArea`: 8 each; `Electrical`: 1). These were deliberately retained for model-specific preprocessing.

See `docs/data_analysis/eda/DA_EDA_Report.md` for complete analysis, qualifications and plots.

## 4. Critical preprocessing constraints

1. **Separate `SalePrice` before preprocessing.** Treat `Id` as a key, not a predictive variable.
2. **Avoid leakage:** split labeled data into training and validation (or use K-fold cross-validation) *before* fitting imputers, encoders, scalers, feature selectors or other learned transformations. Fit them only within the training folds using a `Pipeline`/`ColumnTransformer`.
3. **Preserve the meaning of literal `NA` in Ames data.** For features such as `Alley`, `PoolQC`, `GarageType` and basement-related categorical fields, `NA` can indicate *absence*, not unknown. The DE `read_housing()` logic in `src/data/common.py` documents the distinction. Avoid generic `pd.read_csv()` defaults if these labels are needed; inspect categorical missingness carefully.
4. **Do not mistake identifier-like or numeric-coded categories for continuous quantities.** E.g., `MSSubClass` is a coded property type; `OverallQual` is ordinal; `Neighborhood` is nominal.
5. **Outliers are analysis flags**, not established data errors. Evaluate robust transforms such as `log1p(SalePrice)` and/or sensitivity analyses, with correct inverse transform when reporting price-space errors.
6. **Apply exactly the same fitted preprocessing steps** to validation, holdout, Kaggle test, and deployment input.

## 5. Synthetic and transaction-time feature policy

Eight generated/derived columns are: `DistanceToSchoolKm`, `DistanceToHospitalKm`, `DistanceToTransitKm`, `RenovationFlag`, `YearsSinceRenovation`, `DaysOnMarket`, `SimulatedInterestRatePct`, `SimulatedPriceIndex`.

- These distance features are **simulated**, not real geocoded property distances. In the current DA report they have near-zero correlations with price. Do not claim they quantify real-world amenity effects.
- `DaysOnMarket` is not available when a house is first listed. **Exclude it from an initial-listing valuation model.**
- `YrSold` and any variables derived from sale year may be unavailable for a new listing and can leak post-listing information. **Exclude `YrSold` from the initial-listing model**, or clearly define a retrospective prediction task if using it.
- `YearsSinceRenovation` currently uses `YrSold`; for prospective use, reconstruct it from a known valuation date and available renovation year, or exclude it.
- `SimulatedInterestRatePct` and `SimulatedPriceIndex` are deterministic hypothetical functions of `YrSold`, **not observed market indicators**. Exclude from the deployable model or replace with legitimate as-of-valuation-date data.
- `RenovationFlag` derives from `YearRemodAdd > YearBuilt`; it is potentially usable if those construction/remodel dates are available at inference.
- Even for simulated distances, include them in a deployable model **only if the app can generate consistently defined inputs for unseen properties**; otherwise exclude them. If you compare an experimental model using synthetic inputs, report that evaluation separately.
- Also review *all other* original Ames fields for availability at prediction time (e.g., sale transaction terms such as `SaleType` and `SaleCondition`) according to the intended application.

### Suggested candidate exclusions for an initial-listing baseline

```python
UNAVAILABLE_AT_LISTING = [
    "Id", "YrSold", "MoSold", "DaysOnMarket",
    "YearsSinceRenovation", "SimulatedInterestRatePct",
    "SimulatedPriceIndex",
]
```

This is **not an exhaustive drop list**. Determine whether `SaleType`, `SaleCondition`, synthetic distances and other fields will actually exist at inference time. Make that decision explicit in the final input schema.

## 6. Modeling workflow and suggested techniques

1. **Define the deployment input schema** and the candidate feature set before training; record any exclusions.
2. **Exploratory baseline:** `DummyRegressor` (e.g., predict median/mean) and simple linear models.
3. **Preprocessing:** `SimpleImputer` with appropriate categorical absence handling; `OneHotEncoder(handle_unknown='ignore')` for nominal categories; scale numeric inputs when appropriate for the chosen model.
4. **Candidate models:** Linear Regression, Ridge/Lasso/Elastic Net, Random Forest, Gradient Boosting (and/or another justified boosting approach if installed).
5. **Evaluation:** consistent K-fold CV or a holdout protocol. Report **MAE, RMSE, R²** on appropriate held-out predictions; if using log-target training, inverse-transform predictions for price-space metrics. MAPE is optional and should be interpreted carefully.
6. **Tuning:** tune hyperparameters within CV, and reserve an untouched final evaluation set if using one.
7. **Diagnostics:** residual plots, errors by `Neighborhood` and price bands, signs of overfitting, effect of outlier treatment and synthetic-feature inclusion.
8. **Reproducibility:** fix random seeds, save fitted preprocessing+model as a single artifact, document library versions and exact train/validation protocol.

## 7. Expected ML deliverables

Suggested paths (not yet claimed as existing):

- `src/models/` — preprocessing, training, evaluation and inference scripts.
- `notebooks/model_comparison.ipynb` — experiments and analyses (optional).
- `reports/modeling/model_comparison.csv` — model/CV scores and selected settings.
- `reports/modeling/model_evaluation.md` — MAE, RMSE, R², leakage policy, residual and subgroup diagnostics, limitations.
- `models/` — serialized production pipeline, if the team agrees to track it via Git/LFS or another storage location.
- `docs/modeling/README_MLOPS_HANDOFF.md` — feature schema, preprocessing, artifact loading, expected prediction input/output.

## 8. Git handoff workflow

After DE and DA have been merged to `main`, the ML engineer should branch from the latest `main`:

```powershell
git clone https://github.com/vinhit-e10hust/Introduction-to-BA---Group-22-house-price-prediction-.git
cd Introduction-to-BA---Group-22-house-price-prediction-
git switch main
git pull --ff-only origin main
git switch -c feature/modeling
```

If the repository is already cloned locally, **skip `git clone` and `cd`**; just update `main` and create the feature branch. Commit ML files and open a Pull Request to `main` for review.

## 9. Acceptance checklist

- [ ] Data shape and target verified, `Id` excluded from model features.
- [ ] No leakage: preprocessing fitted exclusively on training folds.
- [ ] Ames absence categories handled correctly.
- [ ] Selected feature list is available at intended prediction time.
- [ ] Baseline and at least one stronger candidate trained and evaluated consistently.
- [ ] Cross-validation and MAE/RMSE/R² documented.
- [ ] Error and outlier sensitivity analyses completed.
- [ ] Input schema and inference procedure handed to deployment team.
- [ ] Reproduction commands and dependency versions documented.

**Source of findings:** `docs/data_analysis/eda/DA_EDA_Report.md`, `docs/data_engineering/cleaning_report.md`, `docs/data_engineering/synthetic_assumptions.md`. Numerical results here are inherited from the DA/DE reports, not independently recomputed by this handoff document.
