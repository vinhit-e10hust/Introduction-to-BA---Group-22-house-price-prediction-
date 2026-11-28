"""
@Thang

layer 2: Validate inputs
Input:
    - dataframe (df) returned by read_housing() (this is the unseen data, could include multiple rows)
    - contract.json (output of layer 1)
Process:
    -1 check schema: if not enough columns then break to 5
    -2 check type: all num_cols have numeric dtype, vice versa with cat_cols
    -3 check value range: required not NaN, some columns required value >0, cat_cols value belongs to allowed set
    -4 check others: value out of [min,max] in contract.json, lack optional cols, strange cols**
    -5 combine results of checks: return Valid_report object
Output:
    - ValidReport object **


---------notes
** strange cols: could be either in these 3 cases:
    - columns that have been pruned during data engineering
    - the target column
    - columns that have not appeared in the test_cleaned before.

** returns a report, never raise error.
** This layer does not write/update to the input df.

** ValidReport
    - ...
** Issue   
    - ...

"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Issue:
    level: str
    code: str
    column: str
    rows: list[int]


@dataclass(frozen=True)
class ValidReport:
    is_valid: bool
    errors: list[Issue]
    warnings: list[Issue]
    n_rows: int
    n_missing_cells: int


def hits(mask):
    return mask.index[mask.fillna(False)].tolist()


def validate(df, contract):
    errors, warnings = [], []
    num, cat = contract["num_cols"], contract["cat_cols"]
    numeric_cat = set(contract["numeric_dtype_categoricals"])

    def report():
        return ValidReport(
            not errors, errors, warnings, len(df), int(df.isna().sum().sum())
        )

    for name in contract["columns"]:
        if name not in df.columns:
            errors.append(Issue("error", "1_MISSING_COLUMN", name, []))
    if errors:
        return report()

    for name in num:
        if not pd.api.types.is_numeric_dtype(df[name]):
            errors.append(Issue("error", "2_BAD_TYPE_NUM", name, []))
    for name in cat:
        if name not in numeric_cat and pd.api.types.is_numeric_dtype(df[name]):
            errors.append(Issue("error", "2_BAD_TYPE_CAT", name, []))

    for name, spec in {**num, **cat}.items():
        blank = hits(df[name].isna())
        if not blank:
            continue
        if spec["required"]:
            errors.append(Issue("error", "3_REQUIRED_MISSING", name, blank))
        else:
            warnings.append(Issue("warning", "4_OPTIONAL_MISSING", name, blank))

    for name in contract["must_be_positive"]:
        if pd.api.types.is_numeric_dtype(df[name]):
            bad = hits(~df[name].gt(0))
            if bad:
                errors.append(Issue("error", "3_NON_POSITIVE", name, bad))

    for name, spec in cat.items():
        unseen = hits(df[name].notna() & ~df[name].isin(spec["allowed"]))
        if unseen:
            errors.append(Issue("error", "3_UNSEEN_CATEGORY", name, unseen))

    for name, spec in num.items():
        if not pd.api.types.is_numeric_dtype(df[name]):
            continue
        out = hits(df[name].lt(spec["min"]) | df[name].gt(spec["max"]))
        if out:
            warnings.append(Issue("warning", "4_OUT_OF_RANGE", name, out))

    for name in df.columns:
        if name not in contract["columns"] and name != "Id":
            warnings.append(Issue("warning", "4_STRANGE_COLUMN", name, []))

    return report()


if __name__ == "__main__":
    import sys

    from src.app_deploy.contract import load_contract
    from src.data.common import read_housing

    result = validate(read_housing(sys.argv[1]), load_contract())
    print(f"n_rows={result.n_rows} n_missing_cells={result.n_missing_cells}"
          f" is_valid={result.is_valid}"
          f" errors={len(result.errors)} warnings={len(result.warnings)}")
    for issue in result.errors + result.warnings:
        print(f"{issue.code:20} {issue.column:14} rows={issue.rows}")
