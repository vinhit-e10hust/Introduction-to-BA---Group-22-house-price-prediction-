from common import *
import numpy as np
init_dirs()
train=read_housing(RAW/'train.csv').drop(columns=['SalePrice'])
test=read_housing(RAW/'test.csv')
base=pd.concat([train,test],ignore_index=True).sort_values('Id').reset_index(drop=True)
assert base.Id.is_unique
rng=np.random.default_rng(42)
# Neighborhood-specific latent accessibility index: simulated, not geocoded.
areas=sorted(base.Neighborhood.unique())
accessibility=dict(zip(areas,rng.uniform(.2,.9,len(areas))))
a=base.Neighborhood.map(accessibility).to_numpy(dtype=float)

def distance(upper,coef,noise):
    return np.maximum(.1, rng.normal(upper-coef*a,noise)).round(2)

year_built=base.YearBuilt.astype(int)
remodel=base.YearRemodAdd.astype(int)
sale_year=base.YrSold.astype(int)
# Historical records can have remodeling dates after recorded sale year;
# such inconsistencies are reported rather than silently changing source values.
renovated=(remodel>year_built).astype(int)
renovation_age=np.maximum(0,sale_year-remodel)
market_days=np.maximum(1,np.rint(rng.normal(95-16*a-8*renovated+0.25*renovation_age,14))).astype(int)
synthetic=pd.DataFrame({
    'Id':base.Id.astype(int),
    'DistanceToSchoolKm':distance(5,4,.7),
    'DistanceToHospitalKm':distance(8,5,1.1),
    'DistanceToTransitKm':distance(4,3,.6),
    'RenovationFlag':renovated,
    'YearsSinceRenovation':renovation_age,
    'DaysOnMarket':market_days,
    'SimulatedInterestRatePct':(6.5-.2*(sale_year-2006)).round(2),
    'SimulatedPriceIndex':(100*1.02**(sale_year-2006)).round(2),
})
assert synthetic.Id.is_unique and len(synthetic)==len(base)
synthetic.to_csv(SYN/'contextual_features.csv',index=False)
print('Synthetic:',synthetic.shape)
