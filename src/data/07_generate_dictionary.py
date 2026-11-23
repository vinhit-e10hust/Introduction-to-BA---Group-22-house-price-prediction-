
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/synthetic/contextual_features.csv")
OUTPUT_DIR = Path("docs/data_engineering")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

metadata = {
    "DistanceToSchoolKm": (
        "float", "km",
        "Simulated neighborhood accessibility; "
        "max(0.1, Normal(5 - 4*a, 0.7))."
    ),
    "DistanceToHospitalKm": (
        "float", "km",
        "Simulated neighborhood accessibility; "
        "max(0.1, Normal(8 - 5*a, 1.1))."
    ),
    "DistanceToTransitKm": (
        "float", "km",
        "Simulated neighborhood accessibility; "
        "max(0.1, Normal(4 - 3*a, 0.6))."
    ),
    "RenovationFlag": (
        "integer", "binary",
        "1 if YearRemodAdd > YearBuilt, else 0."
    ),
    "YearsSinceRenovation": (
        "integer", "years",
        "max(0, YrSold - YearRemodAdd)."
    ),
    "DaysOnMarket": (
        "integer", "days",
        "max(1, round(Normal("
        "95 - 16*a - 8*RenovationFlag "
        "+ 0.25*YearsSinceRenovation, 14)))."
    ),
    "SimulatedInterestRatePct": (
        "float", "%",
        "6.5 - 0.2 * (YrSold - 2006)."
    ),
    "SimulatedPriceIndex": (
        "float", "index",
        "100 * 1.02 ** (YrSold - 2006)."
    )
}

rows = []

for column, (dtype, unit, logic) in metadata.items():
    if column not in df.columns:
        raise ValueError(f"Missing synthetic column: {column}")

    rows.append({
        "Column Name": column,
        "Data Type": dtype,
        "Unit": unit,
        "Valid Range / Constraint": {
            "DistanceToSchoolKm": ">= 0.1",
            "DistanceToHospitalKm": ">= 0.1",
            "DistanceToTransitKm": ">= 0.1",
            "RenovationFlag": "{0, 1}",
            "YearsSinceRenovation": ">= 0",
            "DaysOnMarket": ">= 1",
            "SimulatedInterestRatePct": "> 0 in current dataset",
            "SimulatedPriceIndex": "> 0"
        }[column],
        "Observed Min": df[column].min(),
        "Observed Max": df[column].max(),
        "Missing Count": int(df[column].isna().sum()),
        "Generation Logic": logic,
        "Source": "Synthetic / Derived"
    })

dictionary = pd.DataFrame(rows)

output_path = OUTPUT_DIR / "data_dictionary.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    dictionary.to_excel(
        writer,
        sheet_name="Synthetic Features",
        index=False
    )

    worksheet = writer.sheets["Synthetic Features"]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    widths = {
        "A": 28, "B": 16, "C": 12,
        "D": 32, "E": 18, "F": 18,
        "G": 17, "H": 90, "I": 22
    }

    for col, width in widths.items():
        worksheet.column_dimensions[col].width = width

print("Data Dictionary generated:", output_path)
print(dictionary[[
    "Column Name", "Observed Min", "Observed Max"
]].to_string(index=False))
