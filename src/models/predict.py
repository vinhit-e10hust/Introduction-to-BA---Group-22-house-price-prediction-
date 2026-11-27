import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.common import read_housing
from src.models.features import SQFT_TO_SQM


MODEL_PATH = Path("models/house_price.joblib")


def point_predictions(estimator, data):
    return np.maximum(np.asarray(estimator.predict(data), dtype=float), 0)


def prediction_bounds(price, log_radius):
    center = np.log1p(price)
    return np.maximum(np.expm1(center - log_radius), 0), np.expm1(center + log_radius)


def predict_prices(bundle, data):
    price = point_predictions(bundle["estimator"], data)
    lower, upper = prediction_bounds(price, bundle["log_radius"])
    area_m2 = data["GrLivArea"].where(data["GrLivArea"].gt(0)) * SQFT_TO_SQM
    predictions = pd.DataFrame({
        "PredictedSalePrice": price,
        "LowerPrice": lower,
        "UpperPrice": upper,
        "PredictedUSDPerSquareMeter": price / area_m2,
    }, index=data.index)
    if "Id" in data:
        predictions.insert(0, "Id", data["Id"])
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Estimate home prices from Ames-format inputs.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    predictions = predict_prices(bundle, read_housing(args.input))
    if args.output is None:
        print(predictions.to_csv(index=False), end="")
    else:
        predictions.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
