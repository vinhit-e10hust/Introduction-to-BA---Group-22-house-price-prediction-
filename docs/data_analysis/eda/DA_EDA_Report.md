# DA EDA Report — Ames House Prices

## Scope

Main data: `data/processed/train_cleaned.csv`, 1,460 labeled Ames, Iowa sales. The derived result tables are available as snapshots in `notebooks/eda_tables_viewer.ipynb`; the Vietnamese and technical field dictionaries are available as CSV files. Target: `SalePrice` (USD). The work is descriptive and tests associations, not causation or current-market generalization. Original rows are retained; outlier sensitivity uses explicit comparison cohorts.

## Task 1 — Data understanding

The dataset contains **1,460 rows × 89 columns**, with 46 numeric-storage and 43 text-storage columns. Storage type is not semantic type: `OverallQual` is ordinal, `MSSubClass` is a coded property category, and `Id` is a key, not a predictor. The full feature classification and numerical/categorical summary tables are in `notebooks/eda_tables_viewer.ipynb`. There are **357 missing cells**. Top gaps: LotFrontage 259, GarageYrBlt 81, MasVnrType and MasVnrArea 8 each, Electrical 1.

`SalePrice`: mean $180,921, median $163,000, standard deviation $79,443, range $34,900–$755,000.

## File recheck — extended vs cleaned

SHA-256: cleaned `bcb896ed6faffa77e6fbf56c85f98006e150f784b1ac1bf322242dab5bddcd89`; extended `bcb896ed6faffa77e6fbf56c85f98006e150f784b1ac1bf322242dab5bddcd89`. Byte-identical: **True**. Shapes (1460, 89) and (1460, 89); same column order True; columns only in cleaned `[]`; only in extended `[]`; same Id set True; total aligned cell differences **0**. So for the files present in this workspace, both schema and values match. The cleaner strips surrounding whitespace from text and asserts ranges; it does not impute, remove rows or drop columns. Current exports therefore cannot demonstrate a cleaning effect. The full cleaning comparison snapshot is included in `notebooks/eda_tables_viewer.ipynb`.

## Task 2 — Univariate analysis

SalePrice skewness **1.88**, GrLivArea **1.37**, LotArea **12.21**. The positive skew and long right tails make median and IQR useful alongside means and standard deviations. Numeric histograms, KDEs, and boxplots are in `figures/univariate_numeric.png`; Neighborhood sale counts are in `figures/neighborhood_counts.png`, and price distributions are in `figures/saleprice_by_neighborhood.png`. The descriptive summary tables are in `notebooks/eda_tables_viewer.ipynb`. Highest median neighborhoods: NridgHt ($315,000, n=77), NoRidge ($301,500, n=41), StoneBr ($278,000, n=25). Lowest: MeadowV ($88,000, n=17), IDOTRR ($103,000, n=37), BrDale ($106,000, n=16). Blueste has only two sales; its median is reported with the count and treated as a descriptive observation.

## Task 3 — Bivariate price analysis

`GrLivArea` vs `SalePrice`: Pearson r **0.709**, Spearman ρ **0.731** (n=1460); larger living area is associated with higher price, but substantial variation remains. `OverallQual` has Pearson r=0.791 and Spearman rho=0.810 with price; median price is $50,150 at score 1 (n=2) vs $432,390 at score 10 (n=18), a strong increasing pattern. House age at sale is `YrSold - YearBuilt`; the youngest age band has median $219,355 (n=434) compared with $123,250 for the 100+ year band (n=28). Newer houses have higher sale prices in this sample; the age-band comparison does not separate age from quality, renovation, or neighborhood. Neighborhood price distributions vary; medians, means, quartiles, and sample counts are in `notebooks/eda_tables_viewer.ipynb`. Figures: `figures/price_relationships.png`. Pearson/Spearman coefficients and p-values for selected features are also in that notebook. These are observational relationships, not causal effects.

## Task 4 — Multivariate analysis

Pearson/Spearman numeric predictor matrices are included in the table viewer notebook; heatmap: `figures/pearson_heatmap.png`. The table below lists predictor pairs crossing |r| or |ρ| ≥ 0.60. DA decision: retain the overlapping variables in the descriptive analysis. Pairwise correlation documents shared information but is not, by itself, the selection criterion for removing a feature.

| feature_1                | feature_2                |   pearson_r |   spearman_rho |    n |    pearson_p |   spearman_p |
|:-------------------------|:-------------------------|------------:|---------------:|-----:|-------------:|-------------:|
| YrSold                   | SimulatedInterestRatePct |   -1        |      -1        | 1460 | 0            | 0            |
| SimulatedInterestRatePct | SimulatedPriceIndex      |   -0.999933 |      -1        | 1460 | 0            | 0            |
| YrSold                   | SimulatedPriceIndex      |    0.999933 |       1        | 1460 | 0            | 0            |
| YearRemodAdd             | YearsSinceRenovation     |   -0.997931 |      -0.990916 | 1460 | 0            | 0            |
| GarageCars               | GarageArea               |    0.882475 |       0.853317 | 1460 | 0            | 0            |
| YearBuilt                | GarageYrBlt              |    0.825667 |       0.890546 | 1379 | 0            | 0            |
| GrLivArea                | TotRmsAbvGrd             |    0.825489 |       0.827874 | 1460 | 0            | 0            |
| TotalBsmtSF              | 1stFlrSF                 |    0.81953  |       0.829292 | 1460 | 0            | 0            |
| 2ndFlrSF                 | GrLivArea                |    0.687501 |       0.642857 | 1460 | 7.72478e-205 | 5.07955e-171 |
| BedroomAbvGr             | TotRmsAbvGrd             |    0.67662  |       0.667822 | 1460 | 4.82044e-196 | 3.32041e-189 |
| BsmtFinSF1               | BsmtFullBath             |    0.649212 |       0.674175 | 1460 | 1.73594e-175 | 4.05226e-194 |
| GarageYrBlt              | YearsSinceRenovation     |   -0.642637 |      -0.726835 | 1379 | 1.73899e-161 | 6.25568e-227 |

`notebooks/eda_tables_viewer.ipynb` includes exploratory VIFs for selected numeric predictors using complete cases. The highest values are GarageCars (5.33), GrLivArea (5.24), and GarageArea (4.98) in this selected set. GarageCars–GarageArea represent garage capacity and area; GrLivArea–TotRmsAbvGrd represent living space and room count; YearBuilt–GarageYrBlt represent house and garage age; TotalBsmtSF–1stFlrSF represent related building areas. YrSold and the synthetic macro columns are nearly perfectly related because the macro fields were generated from sale year; YearsSinceRenovation is nearly inverse to YearRemodAdd by construction. The DA report treats these as overlapping or derived measurements and does not count them as independent market evidence. The VIF values are exploratory and cover the selected numeric features only.

## Task 5 — Missingness and outliers

IQR flags (1.5×IQR): SalePrice 61, GrLivArea 31, LotArea 69, TotalBsmtSF 61, GarageArea 21. These are statistical flags, not proof of data error. All rows remain in the main analysis.

| Cohort | n | Median SalePrice | Mean SalePrice | Pearson GrLivArea–price | Spearman GrLivArea–price |
|---|---:|---:|---:|---:|---:|
| All observations | 1460 | $163,000 | $180,921 | 0.709 | 0.731 |
| Exclude SalePrice IQR flags | 1399 | $159,500 | $170,237 | 0.661 | 0.706 |
| Exclude LotArea IQR flags | 1391 | $160,000 | $178,136 | 0.723 | 0.726 |
| Exclude SalePrice/LotArea/GrLivArea flags | 1328 | $157,000 | $168,135 | 0.690 | 0.697 |

DA decision: retain all 1,460 valid sales in the primary analysis; do not remove observations solely because they are IQR flags. No record has been confirmed as a data-entry error. Excluding 61 SalePrice flags lowers the mean by $10,684 (from $180,921 to $170,237), the median by $3,500, and Pearson area–price r from 0.709 to 0.661. Excluding LotArea flags alone raises r to 0.723; excluding the SalePrice, LotArea, and GrLivArea flags together yields r=0.690. The full sample remains the primary result; exclusion cohorts are presented only as sensitivity checks. Thresholds, flags, and row keys are included in the table viewer notebook; visualization in `figures/outlier_sensitivity.png`. Missingness summaries and missing-versus-observed price comparisons are included in the table viewer notebook. For example, median price is $172,400 when LotFrontage is missing vs $159,500 when observed; for GarageYrBlt it is $100,000 vs $167,500. DA decision: retain missing values without imputation or row deletion in this EDA and report missing-versus-observed groups separately. The price gaps are descriptive and do not establish that missingness causes price differences.

## Task 6 — Synthetic features

The six requested fields and two simulated macro fields are summarized in `notebooks/eda_tables_viewer.ipynb`; the first six appear in `figures/synthetic_features_price.png`. Distance/price correlations are close to zero here: school 0.029, hospital -0.000, transit -0.003. RenovationFlag=0 has median $170,000 (n=764) vs flag=1 $155,000 (n=696); this unadjusted comparison may reflect age, quality, and neighborhood. YearsSinceRenovation has r=-0.509, DaysOnMarket r=-0.151 with SalePrice in this constructed sample. Distances are simulated from neighborhood accessibility; DaysOnMarket also uses simulated access/renovation logic. RenovationFlag derives from construction/remodel years; YearsSinceRenovation uses historical sale year. DA decision: synthetic variables remain a separate Task 6 description and are excluded from conclusions about real-world price drivers. The reported associations describe generated values only, not measured amenity effects or causal price impacts.

## Power BI handoff

DA dashboard decisions: `data/processed/train_cleaned.csv` is the sole property fact source; the byte-identical extended file is not appended and no cleaning-comparison page is included. The five pages are (1) Overview and data quality, (2) distributions, (3) price relationships, (4) correlation and outlier sensitivity, and (5) synthetic features. The overview reports 1,460 sales and 357 missing cells; the sensitivity page keeps all valid rows by default and displays exclusions only as comparisons. Module 7 monitoring is outside this DA EDA deliverable.
