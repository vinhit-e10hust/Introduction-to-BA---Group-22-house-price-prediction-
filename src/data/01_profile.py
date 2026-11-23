from common import *
init_dirs()
for split in ('train','test'):
    d=read_housing(RAW/f'{split}.csv')
    profile=pd.DataFrame({'column':d.columns,'dtype':[str(x) for x in d.dtypes], 'missing_count':d.isna().sum().to_numpy(),'missing_pct':(100*d.isna().mean()).round(2).to_numpy(), 'unique_values':d.nunique(dropna=True).to_numpy()})
    profile.to_csv(REPORT/f'{split}_profile.csv',index=False)
    print(f'{split}: rows={len(d)}, columns={len(d.columns)}, missing_cells={d.isna().sum().sum()}, duplicate_IDs={d.Id.duplicated().sum()}')
    print(profile.sort_values('missing_count',ascending=False).head(10).to_string(index=False))
