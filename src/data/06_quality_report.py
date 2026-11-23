
from pathlib import Path
import pandas as pd
import numpy as np

INPUT = Path("data/processed")
OUTPUT = Path("reports/data_quality")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Preserve "NA" as a legitimate category
def read_data(path):
    return pd.read_csv(
        path,
        keep_default_na=False,
        na_values=[""]
    )

def missing_report(df, split):
    missing = df.isna().sum()

    report = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_pct": (
            missing.values / len(df) * 100
        ).round(2)
    })

    report = report[
        report["missing_count"] > 0
    ].sort_values(
        "missing_count", ascending=False
    )

    report.to_csv(
        OUTPUT / f"{split}_missing_report.csv",
        index=False
    )

    return report

def outlier_report(df):
    columns = [
        "GrLivArea",
        "LotArea",
        "TotalBsmtSF",
        "GarageArea",
        "SalePrice"
    ]

    records = []

    for col in columns:
        if col not in df.columns:
            continue

        values = pd.to_numeric(
            df[col], errors="coerce"
        ).dropna()

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (values < lower) | (values > upper)

        records.append({
            "column": col,
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
            "outlier_count": int(mask.sum()),
            "outlier_pct": round(
                mask.mean() * 100, 2
            ),
        })

    return pd.DataFrame(records)

for split in ["train", "test"]:
    before = read_data(
        INPUT / f"{split}_extended.csv"
    )
    after = read_data(
        INPUT / f"{split}_cleaned.csv"
    )

    print(f"\n========== {split.upper()} ==========")
    print("Rows before:", len(before))
    print("Rows after:", len(after))
    print("Missing before:", before.isna().sum().sum())
    print("Missing after:", after.isna().sum().sum())

    print("\nMissing value report:")
    print(missing_report(after, split).to_string(index=False))

    outliers = outlier_report(after)

    outliers.to_csv(
        OUTPUT / f"{split}_outlier_report.csv",
        index=False
    )

    print("\nOutlier report:")
    print(outliers.to_string(index=False))

print("\nData quality reports generated successfully!")
