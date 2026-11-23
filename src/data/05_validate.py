from common import *
init_dirs()
for split in ('train','test'):
    raw=read_housing(RAW/f'{split}.csv')
    cleaned=read_housing(OUT/f'{split}_cleaned.csv')
    assert len(raw)==len(cleaned)
    assert cleaned.Id.is_unique
    assert raw.Id.equals(cleaned.Id)
    if split=='train':
        assert raw.SalePrice.equals(cleaned.SalePrice)
    else: assert 'SalePrice' not in cleaned.columns
    assert cleaned[['DistanceToSchoolKm','DistanceToHospitalKm','DistanceToTransitKm']].notna().all().all()
    assert cleaned[['DistanceToSchoolKm','DistanceToHospitalKm','DistanceToTransitKm']].ge(0).all().all()
train=read_housing(OUT/'train_cleaned.csv')
test=read_housing(OUT/'test_cleaned.csv')
assert set(train.columns)-{'SalePrice'}==set(test.columns)
print('PASS: joins, row counts, IDs, target, schema, synthetic distances')
