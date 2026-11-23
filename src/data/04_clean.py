from common import *
init_dirs()
# Conservative, deterministic cleaning only. Statistical preprocessing belongs
# in sklearn Pipeline fitted only inside training/CV folds.
for split in ('train','test'):
    d=read_housing(OUT/f'{split}_extended.csv')
    before=d.isna().sum().sum()
    assert d.Id.is_unique
    for c in d.select_dtypes(include=['object', 'str']).columns:
        d[c]=d[c].str.strip()
    for c in ['LotArea','GrLivArea','DistanceToSchoolKm','DistanceToHospitalKm','DistanceToTransitKm','DaysOnMarket']:
        assert d[c].dropna().ge(0).all(), f'Negative {c} in {split}'
    # Do NOT treat literal 'NA' as missing. Do NOT impute LotFrontage here.
    d.to_csv(OUT/f'{split}_cleaned.csv',index=False)
    print(f'{split}: {len(d)} rows, missing {before} -> {d.isna().sum().sum()}')
