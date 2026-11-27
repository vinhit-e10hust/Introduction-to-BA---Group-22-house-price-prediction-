import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)

from src.models.features import SQFT_TO_SQM
from src.models.predict import point_predictions, predict_prices


def error_metrics(actual, estimate):
    return {
        "rmse_usd": root_mean_squared_error(actual, estimate),
        "mae_usd": mean_absolute_error(actual, estimate),
        "mape_pct": 100 * mean_absolute_percentage_error(actual, estimate),
        "r2": r2_score(actual, estimate) if len(actual) > 1 else np.nan,
    }


def cv_scores(estimator, data, actual):
    scores = error_metrics(actual, point_predictions(estimator, data))
    return {key: -value if key != "r2" else value for key, value in scores.items()}


def vif_table(development):
    fields = [
        "OverallQual", "YearBuilt", "YearRemodAdd", "GrLivArea", "TotalBsmtSF",
        "1stFlrSF", "TotRmsAbvGrd", "GarageCars", "GarageArea", "LotArea",
    ]
    data = development[fields].dropna()
    standardized = (data - data.mean()) / data.std(ddof=0)
    rows = []
    for field in fields:
        y = standardized[field].to_numpy()
        x = standardized.drop(columns=field).to_numpy()
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        unexplained_share = np.mean((y - x @ coefficients) ** 2)
        rows.append({
            "feature": field,
            "vif": 1 / unexplained_share if unexplained_share > 1e-12 else np.inf,
            "complete_rows": len(data),
        })
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def area_outlier_flags(data, reference):
    fields = ["LotArea", "GrLivArea"]
    quartiles = reference[fields].quantile([.25, .75])
    q1, q3 = quartiles.loc[.25], quartiles.loc[.75]
    iqr = q3 - q1
    return (
        data[fields].lt(q1 - 1.5 * iqr)
        | data[fields].gt(q3 + 1.5 * iqr)
    ).any(axis=1)


def area_weighting_diagnostics(estimator, inputs, target, folds):
    rows = []
    for fold, (train_indices, validation_indices) in enumerate(folds, start=1):
        train_x, train_y = inputs.iloc[train_indices], target.iloc[train_indices]
        val_x, val_y = inputs.iloc[validation_indices], target.iloc[validation_indices]
        train_flags = area_outlier_flags(train_x, train_x)
        val_flags = area_outlier_flags(val_x, train_x)
        training_policies = {
            "uniform": {},
            "half_weight_area_outliers": {
                "model__sample_weight": np.where(train_flags, 0.5, 1.0),
            },
        }
        for policy, fit_params in training_policies.items():
            fitted = clone(estimator).fit(train_x, train_y, **fit_params)
            train_price = point_predictions(fitted, train_x)
            val_price = point_predictions(fitted, val_x)
            val_abs_error = (val_y - val_price).abs()
            val_metrics = error_metrics(val_y, val_price)
            rows.append({
                "fold": fold,
                "training_policy": policy,
                "training_rows": len(train_y),
                "training_area_flags": int(train_flags.sum()),
                "training_rmse_usd": root_mean_squared_error(train_y, train_price),
                "validation_rows": len(val_y),
                "validation_area_flags": int(val_flags.sum()),
                "validation_flagged_mae_usd": val_abs_error.loc[val_flags].mean(),
                "validation_within_10_pct_rows": int((10 * val_abs_error <= val_y).sum()),
                **{f"validation_{key}": value for key, value in val_metrics.items()},
            })
    return pd.DataFrame(rows)


def holdout_predictions(bundle, holdout, development):
    predictions = predict_prices(bundle, holdout)
    predictions["SalePrice"] = holdout["SalePrice"]
    predictions["ResidualUSD"] = predictions["PredictedSalePrice"] - holdout["SalePrice"]
    predictions["Neighborhood"] = holdout["Neighborhood"]
    predictions["AreaIQRFlag"] = area_outlier_flags(holdout, development)
    predictions["ObservedUSDPerSquareMeter"] = (
        holdout["SalePrice"] / (holdout["GrLivArea"] * SQFT_TO_SQM)
    )
    predictions["IntervalCovered"] = predictions["SalePrice"].between(
        predictions["LowerPrice"], predictions["UpperPrice"]
    )
    predictions["PriceBandUSD"] = pd.cut(
        predictions["SalePrice"], [0, 125_000, 175_000, 250_000, np.inf],
        labels=["<=125k", "125k–175k", "175k–250k", ">250k"],
    )
    return predictions


def cohort_metrics(predictions):
    rows = []
    cohorts = [
        ("overall", "all", predictions),
        ("area_iqr", "flagged", predictions.loc[predictions["AreaIQRFlag"]]),
        ("area_iqr", "unflagged", predictions.loc[~predictions["AreaIQRFlag"]]),
    ]
    for field in ("Neighborhood", "PriceBandUSD"):
        cohorts.extend(
            (field, label, group)
            for label, group in predictions.groupby(field, observed=True)
        )
    for category, name, group in cohorts:
        if group.empty:
            continue
        rows.append({
            "category": category,
            "group": name,
            "n": len(group),
            **error_metrics(group["SalePrice"], group["PredictedSalePrice"]),
            "mean_residual_usd": group["ResidualUSD"].mean(),
            "interval_coverage_pct": 100 * group["IntervalCovered"].mean(),
            "mean_interval_width_usd": (group["UpperPrice"] - group["LowerPrice"]).mean(),
        })
    return pd.DataFrame(rows)


def plot_residuals(predictions, path):
    fig, (scatter, histogram) = plt.subplots(1, 2, figsize=(9.5, 3.6), layout="constrained")
    estimate = predictions["PredictedSalePrice"] / 1_000
    residual = predictions["ResidualUSD"] / 1_000
    scatter.scatter(estimate, residual, s=15, color="black", alpha=.48)
    scatter.axhline(0, color="black", linestyle="--", linewidth=1)
    scatter.set(xlabel="Estimated sale price (thousand USD)",
                ylabel="Estimate − actual (thousand USD)", title="Residuals by estimate")
    histogram.hist(residual, bins=18, facecolor="white", edgecolor="black")
    histogram.axvline(0, color="black", linestyle="--", linewidth=1)
    histogram.set(xlabel="Estimate − actual (thousand USD)", ylabel="Homes",
                  title="Residual distribution")
    fig.savefig(path, dpi=160)
    plt.close(fig)
