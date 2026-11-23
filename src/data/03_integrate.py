from common import *
init_dirs()
s=read_housing(SYN/'contextual_features.csv')
assert s.Id.is_unique
for split in ('train','test'):
    d=read_housing(RAW/f'{split}.csv')
    joined=d.merge(s,on='Id',how='left',validate='one_to_one',indicator=True)
    assert len(joined)==len(d) and joined._merge.eq('both').all()
    joined.drop(columns='_merge').to_csv(OUT/f'{split}_extended.csv',index=False)
    print(split,joined.shape)
