# Synthetic contextual features — simulation assumptions

All contextual values are **simulated** rather than observed/geocoded Ames property attributes. All random outputs use NumPy seed 42, and houses are ordered by ID. No SalePrice is used to generate features.

| Field | Type | Unit | Valid range | Logic / assumption | Available at initial listing? |
|---|---|---|---|---|---|
| DistanceToSchoolKm | float | km | >=0.1 | Normal(5 - 4 * neighborhood accessibility, 0.7), lower-capped at 0.1 | Synthetic estimate only |
| DistanceToHospitalKm | float | km | >=0.1 | Normal(8 - 5 * accessibility, 1.1), lower-capped | Synthetic estimate only |
| DistanceToTransitKm | float | km | >=0.1 | Normal(4 - 3 * accessibility, 0.6), lower-capped | Synthetic estimate only |
| RenovationFlag | int | binary | 0 or 1 | Derived from YearRemodAdd > YearBuilt | Yes if history provided |
| YearsSinceRenovation | int | years | >=0 | max(0, YrSold - YearRemodAdd); historical proxy | Use valuation year instead for new listing |
| DaysOnMarket | int | days | >=1 | Normal(95-16*accessibility-8*renovated+0.25*years_since,14), capped and rounded | **No**, exclude from initial-listing model |
| SimulatedInterestRatePct | float | percentage points | 5.7 to 6.5 on 2006–2010 data | 6.5 - 0.2*(YrSold-2006), NOT historical real rates | Replace with real valuation-date indicator |
| SimulatedPriceIndex | float | index (2006=100) | 100 to 108.24 for 2006–2010 | 100*1.02^(YrSold-2006), hypothetical growth | Replace with real valuation-date indicator |

**Important:** Features like `YrSold` are historical transaction attributes. The model deployed for new listings should use fields available at valuation time and should not assume knowledge of a future sale date. Synthetic measurements must be clearly identified as simulated in reports and in the demo.
