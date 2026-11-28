"""Run: .\\.venv\\Scripts\\python.exe -m src.app_deploy.scripts.build_contract"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.common import UNKNOWN_CATEGORICAL, read_housing
from src.models.features import CATEGORICAL_INPUTS, LISTING_INPUTS, NUMERIC_INPUTS

TRAIN_DATA = Path("data/processed/train_cleaned.csv")
OUTPUT = Path("src/app_deploy/contract.json")
MUST_BE_POSITIVE = ("GrLivArea",)


def py(value):
    return value.item() if hasattr(value, "item") else value


def num_stats(series):
    seen = series.dropna()
    q = seen.quantile([.01, .25, .5, .75, .99])
    return {
        "dtype": str(series.dtype),
        "min": float(seen.min()),
        "max": float(seen.max()),
        "p01": float(q[.01]),
        "q1": float(q[.25]),
        "median": float(q[.5]),
        "q3": float(q[.75]),
        "p99": float(q[.99]),
        "mean": round(float(seen.mean()), 4),
        "std": round(float(seen.std()), 4),
        "mode": float(seen.mode().iloc[0]),
        "missing_count": int(series.isna().sum()),
        "missing_pct": round(100 * float(series.isna().mean()), 4),
        "required": not series.isna().any(),
        "must_be_positive": series.name in MUST_BE_POSITIVE,
    }


def cat_stats(series):
    seen = series.dropna()
    allowed = sorted(py(value) for value in seen.unique())
    share = seen.value_counts(normalize=True) * 100
    return {
        "dtype": str(series.dtype),
        "allowed": allowed,
        "n_allowed": len(allowed),
        "mode": py(seen.mode().iloc[0]),
        "frequency_pct": {str(k): round(float(v), 4) for k, v in share.items()},
        "missing_count": int(series.isna().sum()),
        "missing_pct": round(100 * float(series.isna().mean()), 4),
        "required": not series.isna().any(),
        "na_means_unknown": series.name in UNKNOWN_CATEGORICAL,
    }


def main():
    data = read_housing(TRAIN_DATA)
    contract = {
        "contract_version": "1.0",
        "generated_on": date.today().isoformat(),
        "source_data": TRAIN_DATA.as_posix(),
        "source_rows": len(data),
        "column_source": "src.models.features.LISTING_INPUTS",
        "n_columns": len(LISTING_INPUTS),
        "columns": list(LISTING_INPUTS),
        "n_num_cols": len(NUMERIC_INPUTS),
        "n_cat_cols": len(CATEGORICAL_INPUTS),
        "numeric_dtype_categoricals": [
            name for name in CATEGORICAL_INPUTS
            if pd.api.types.is_numeric_dtype(data[name])
        ],
        "must_be_positive": list(MUST_BE_POSITIVE),
        "num_cols": {name: num_stats(data[name]) for name in NUMERIC_INPUTS},
        "cat_cols": {name: cat_stats(data[name]) for name in CATEGORICAL_INPUTS},
    }
    OUTPUT.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    specs = {**contract["num_cols"], **contract["cat_cols"]}
    print(f"{OUTPUT}: {contract['n_columns']} columns, "
          f"{sum(s['required'] for s in specs.values())} required")


if __name__ == "__main__":
    main()
