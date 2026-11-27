import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


REFERENCE_YEAR = 2010
SQFT_TO_SQM = 0.09290304
DISTANCE_COLUMNS = (
    "DistanceToSchoolKm", "DistanceToHospitalKm", "DistanceToTransitKm",
)
NUMERIC_INPUTS = (
    "LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt",
    "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
    "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "BsmtFullBath",
    "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
    "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea",
    "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
    "PoolArea",
)
CATEGORICAL_INPUTS = (
    "MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour",
    "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
    "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",
    "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual",
    "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature",
)
LISTING_INPUTS = (*NUMERIC_INPUTS, *CATEGORICAL_INPUTS)
ENGINEERED_NUMERIC = (
    *[column for column in NUMERIC_INPUTS if column != "YearBuilt"],
    "HouseAgeAt2010", "RenovationFlag",
)


def listing_features(data, include_amenities=False):
    input_columns = LISTING_INPUTS + (DISTANCE_COLUMNS if include_amenities else ())
    missing_columns = sorted(set(input_columns) - set(data.columns))
    if missing_columns:
        raise ValueError("Missing input columns: " + ", ".join(missing_columns))
    features = data.loc[:, list(input_columns)].copy()
    features["MSSubClass"] = features["MSSubClass"].astype("Int64").astype("string")
    categorical = features.loc[:, list(CATEGORICAL_INPUTS)].astype(object)
    features[list(CATEGORICAL_INPUTS)] = categorical.where(categorical.notna(), np.nan)

    basement_fields = [
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
    ]
    basement_labels = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
    basement_absent = features[basement_labels].eq("NA").all(axis=1)
    basement_measured = features[basement_fields].gt(0).any(axis=1)
    features.loc[basement_absent & ~basement_measured, basement_fields] = 0

    veneer_type = features["MasVnrType"]
    veneer_area = features["MasVnrArea"]
    both_missing = veneer_type.isna() & veneer_area.isna()
    recorded_material = veneer_type.notna() & ~veneer_type.isin(["None", "Unknown"])
    measured_veneer = veneer_area.gt(0).fillna(False)
    features["MasVnrPresent"] = (recorded_material | measured_veneer).astype("int8")
    features.loc[both_missing, "MasVnrType"] = "None"
    features.loc[both_missing, "MasVnrArea"] = 0

    built = features.pop("YearBuilt")
    remodeled = features["YearRemodAdd"]
    features["HouseAgeAt2010"] = REFERENCE_YEAR - built
    features["RenovationFlag"] = remodeled.gt(built).astype(float).where(
        remodeled.notna() & built.notna()
    )
    if include_amenities:
        distances = features.loc[:, list(DISTANCE_COLUMNS)]
        features["AmenityProximityScore"] = (1 / (1 + distances)).mean(
            axis=1, skipna=False
        )
    return features


def make_pipeline(regressor, *, scale=False, log_target=True, include_amenities=False):
    numeric_columns = list(ENGINEERED_NUMERIC)
    if include_amenities:
        numeric_columns.extend((*DISTANCE_COLUMNS, "AmenityProximityScore"))
    numeric_steps = [
        ("imputer", SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    categories = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(
            handle_unknown="ignore", min_frequency=4, sparse_output=False
        )),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric_columns),
        ("category", categories, list(CATEGORICAL_INPUTS)),
        ("veneer_presence", "passthrough", ["MasVnrPresent"]),
    ])
    target_model = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p if log_target else None,
        inverse_func=np.expm1 if log_target else None,
    )
    return Pipeline([
        ("features", FunctionTransformer(
            listing_features, kw_args={"include_amenities": include_amenities}
        )),
        ("preprocess", preprocessing),
        ("select", SelectPercentile(f_regression, percentile=100)),
        ("model", target_model),
    ])
