
import pandas as pd

HOLDOUT_YEAR = 2010
SALES_SPLIT_SEED = 42
CALIBRATION_YEAR = 2009


def historical_split_ids(data):
    """Split labeled Ames sales into development and calibration IDs."""
    required = {"Id", "YrSold"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if data["Id"].duplicated().any():
        raise ValueError("Property IDs must be unique.")

    development = data.loc[
        data["YrSold"] < CALIBRATION_YEAR, "Id"
    ]

    calibration = data.loc[
        data["YrSold"] == CALIBRATION_YEAR, "Id"
    ]

    if development.empty or calibration.empty:
        raise ValueError("Development or calibration split is empty.")

    if set(development) & set(calibration):
        raise ValueError("Development and calibration overlap.")

    return pd.Index(development), pd.Index(calibration)
