from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "house-prices-advanced-regression-techniques"
SYN = ROOT/'data'/'synthetic'
OUT = ROOT/'data'/'processed'
REPORT = ROOT/'reports'/'data_quality'

NUMERIC_COLUMNS = {
    'Id','MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond',
    'YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2',
    'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF',
    'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',
    'BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt',
    'GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',
    '3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold','SalePrice',
    'DistanceToSchoolKm','DistanceToHospitalKm','DistanceToTransitKm',
    'RenovationFlag','YearsSinceRenovation','DaysOnMarket',
    'SimulatedInterestRatePct','SimulatedPriceIndex'
}
# Categorical columns where literal NA truly denotes unknown, not 'absent'.
UNKNOWN_CATEGORICAL = {'MSZoning','Utilities','Exterior1st','Exterior2nd',
    'Electrical','KitchenQual','Functional','SaleType','MasVnrType'}

def read_housing(path):
    # Preserve valid Ames absence categories (NA = no garage/pool/etc),
    # but treat NA as missing in numeric and genuinely unknown fields.
    df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    for c in df.columns.intersection(list(NUMERIC_COLUMNS)):
        df[c] = pd.to_numeric(df[c].replace('NA',float('nan')), errors='coerce')
    for c in df.columns.intersection(list(UNKNOWN_CATEGORICAL)):
        df[c] = df[c].replace('NA',pd.NA)
    return df

def init_dirs():
    for d in (RAW, SYN, OUT, REPORT, ROOT/'docs'):
        d.mkdir(parents=True, exist_ok=True)
