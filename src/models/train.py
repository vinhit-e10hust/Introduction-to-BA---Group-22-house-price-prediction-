import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

from src.data.common import HOLDOUT_YEAR, SALES_SPLIT_SEED, historical_split_ids, read_housing
from src.models.evaluation import (
    area_weighting_diagnostics,
    cohort_metrics,
    cv_scores,
    error_metrics,
    holdout_predictions,
    plot_residuals,
    vif_table,
)
from src.models.features import (
    DISTANCE_COLUMNS,
    LISTING_INPUTS,
    make_pipeline,
)
from src.models.predict import MODEL_PATH, point_predictions


SEED = SALES_SPLIT_SEED
FOLDS = 5
INTERVAL_COVERAGE = .90
DEFAULT_DATA = Path("data/processed/train_cleaned.csv")
DEFAULT_REPORT = Path("reports/modeling")


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: BaseEstimator
    grid: dict
    scale: bool = False
    log_target: bool = True


def candidates(threads):
    ridge_grid = {
        "model__regressor__alpha": [1, 5, 15, 50],
        "select__percentile": [75, 100],
    }
    boosted_trees = XGBRegressor(
        n_estimators=480, tree_method="hist", objective="reg:squarederror",
        subsample=.85, colsample_bytree=.75, reg_lambda=3,
        random_state=SEED, n_jobs=threads,
    )
    boosting_grid = {
        "model__regressor__max_depth": [2, 4],
        "model__regressor__learning_rate": [.025, .055],
        "select__percentile": [75, 100],
    }
    return [
        Candidate("Median baseline", DummyRegressor(strategy="median"), {}, log_target=False),
        Candidate("Linear Regression", LinearRegression(tol=1e-5), {}, scale=True),
        Candidate("Linear Regression (USD target)", LinearRegression(tol=1e-5), {},
                  scale=True, log_target=False),
        Candidate("Ridge", Ridge(), ridge_grid, scale=True),
        Candidate("Ridge (USD target)", Ridge(), ridge_grid, scale=True, log_target=False),
        Candidate("Lasso", Lasso(max_iter=20_000), {
            "model__regressor__alpha": [.0002, .0005, .001, .002],
        }, scale=True),
        Candidate("Lasso (USD target)", Lasso(max_iter=20_000), {
            "model__regressor__alpha": [20, 60, 150, 350, 800],
        }, scale=True, log_target=False),
        Candidate("Random Forest", RandomForestRegressor(
            n_estimators=240, random_state=SEED, n_jobs=threads
        ), {
            "model__regressor__min_samples_leaf": [1, 3],
            "model__regressor__max_features": [.55, .9],
        }),
        Candidate("XGBoost", boosted_trees, boosting_grid),
        Candidate("XGBoost (USD target)", boosted_trees, boosting_grid, log_target=False),
    ]


def fit_candidate(candidate, development, folds, *, include_amenities=False, grid=None):
    pipeline = make_pipeline(
        candidate.estimator, scale=candidate.scale, log_target=candidate.log_target,
        include_amenities=include_amenities,
    )
    search = GridSearchCV(
        pipeline, candidate.grid if grid is None else grid, cv=folds,
        scoring=cv_scores, refit="rmse_usd", error_score="raise",
    )
    input_columns = LISTING_INPUTS + (DISTANCE_COLUMNS if include_amenities else ())
    return search.fit(development.loc[:, list(input_columns)], development["SalePrice"])


def comparison_row(candidate, search, feature_set, holdout=None):
    results = search.cv_results_
    index = search.best_index_
    row = {
        "model": candidate.name,
        "feature_set": feature_set,
        "target_scale": "log1p" if candidate.log_target else "USD",
        "mean_cv_rmse_usd": -results["mean_test_rmse_usd"][index],
        "std_cv_rmse_usd": results["std_test_rmse_usd"][index],
        "mean_cv_mae_usd": -results["mean_test_mae_usd"][index],
        "mean_cv_mape_pct": -results["mean_test_mape_pct"][index],
        "mean_cv_r2": results["mean_test_r2"][index],
        "parameters": json.dumps(search.best_params_, sort_keys=True),
    }
    if holdout is not None:
        price = point_predictions(search.best_estimator_, holdout)
        holdout_scores = error_metrics(holdout["SalePrice"], price)
        row.update({f"holdout_{key}": value for key, value in holdout_scores.items()})
    return row


def load_sales(path):
    data = read_housing(path)
    if not data["Id"].is_unique:
        raise ValueError("Expected unique property IDs.")
    if not data["SalePrice"].gt(0).all() or not data["YrSold"].notna().all():
        raise ValueError("Expected positive sale prices and sale years.")
    year_columns = data.columns.intersection(["YearBuilt", "YearRemodAdd", "GarageYrBlt"])
    future_years = data[year_columns].gt(data["YrSold"], axis=0)
    data[year_columns] = data[year_columns].mask(future_years)
    return data, {column: int(count) for column, count in future_years.sum().items()}


def calibrate_interval(estimator, calibration):
    estimate = point_predictions(estimator, calibration)
    scores = np.abs(np.log1p(calibration["SalePrice"]) - np.log1p(estimate)).to_numpy()
    rank = int(np.ceil((len(scores) + 1) * INTERVAL_COVERAGE))
    if rank > len(scores):
        raise ValueError("Calibration split is too small for the prediction interval.")
    return float(np.partition(scores, rank - 1)[rank - 1])


def train(data_path, report_dir, artifact_path, threads):
    data, masked_years = load_sales(data_path)
    split_reference = data
    if data_path.resolve() == DEFAULT_DATA.resolve():
        split_reference = read_housing(data_path.with_name("train_extended.csv"))
    development_ids, calibration_ids = historical_split_ids(split_reference)
    sales_by_id = data.set_index("Id", drop=False)
    development = sales_by_id.loc[development_ids.intersection(sales_by_id.index)]
    calibration = sales_by_id.loc[calibration_ids.intersection(sales_by_id.index)]
    holdout = data.loc[data["YrSold"] == HOLDOUT_YEAR]
    if holdout.empty:
        raise ValueError(f"Expected held-out sales from {HOLDOUT_YEAR}.")
    folds = list(KFold(FOLDS, shuffle=True, random_state=SEED).split(development))

    runs = [
        (candidate, fit_candidate(candidate, development, folds))
        for candidate in candidates(threads)
    ]
    runs.sort(key=lambda run: run[1].best_score_, reverse=True)
    winner, selected = runs[0]
    comparison = [
        comparison_row(candidate, search, "listing", holdout)
        for candidate, search in runs
    ]
    with_amenities = fit_candidate(
        winner, development, folds, include_amenities=True,
        grid={key: [value] for key, value in selected.best_params_.items()},
    )
    comparison.append(comparison_row(winner, with_amenities, "listing + simulated amenities"))

    estimator = selected.best_estimator_
    outlier_diagnostics = area_weighting_diagnostics(
        estimator, development.loc[:, list(LISTING_INPUTS)], development["SalePrice"], folds,
    )
    log_radius = calibrate_interval(estimator, calibration)
    metadata = {
        "model": winner.name,
        "parameters": selected.best_params_,
        "target_scale": "log1p" if winner.log_target else "USD",
        "input_columns": LISTING_INPUTS,
        "encoded_feature_count": len(estimator["select"].scores_),
        "selected_feature_count": int(estimator["select"].get_support().sum()),
        "development_rows": len(development),
        "calibration_rows": len(calibration),
        "holdout_rows": len(holdout),
        "holdout_year": HOLDOUT_YEAR,
        "cv_folds": FOLDS,
        "random_seed": SEED,
        "nominal_interval_coverage": INTERVAL_COVERAGE,
        "log_radius": log_radius,
        "post_sale_years_masked": masked_years,
    }
    bundle = {**metadata, "estimator": estimator}
    predictions = holdout_predictions(bundle, holdout, development)
    metrics = cohort_metrics(predictions)

    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison).to_csv(report_dir / "model_comparison.csv", index=False)
    outlier_diagnostics.to_csv(report_dir / "outlier_sensitivity.csv", index=False)
    metrics.to_csv(report_dir / "holdout_metrics.csv", index=False)
    predictions.to_csv(report_dir / "holdout_predictions.csv", index=False)
    vif_table(development).to_csv(report_dir / "vif.csv", index=False)
    (report_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    plot_residuals(predictions, report_dir / "residuals.png")
    joblib.dump(bundle, artifact_path, compress=3)

    headline = metrics.loc[metrics["category"].eq("overall")].iloc[0]
    print(f"Selected model: {winner.name}")
    print(f"{HOLDOUT_YEAR} holdout RMSE: USD {headline['rmse_usd']:,.0f}")
    print(f"{HOLDOUT_YEAR} holdout MAE: USD {headline['mae_usd']:,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Ames house-price models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--artifact", type=Path, default=MODEL_PATH)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    train(args.data, args.reports, args.artifact, args.threads)


if __name__ == "__main__":
    main()
