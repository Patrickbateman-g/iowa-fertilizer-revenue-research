from pathlib import Path
from types import SimpleNamespace
import datetime as dt
import math
import numpy as np
import pandas as pd

source=Path(__file__).with_name('analysis_decoupled_forecast.py').read_text(encoding='utf-8')
ns={}
exec(source.split('DECOUPLED_FORECAST_RESULTS=')[0],ns)
ns['lh_date']=dt.date.fromisoformat
def fit(train,row,features):
    x=train[features].to_numpy(float)
    z=(x-x.mean(axis=0))/x.std(axis=0)
    test=(np.array([getattr(row,k) for k in features])-x.mean(axis=0))/x.std(axis=0)
    response=np.log(train.target_q2_usd.to_numpy()/(train.prior_q2_usd.to_numpy()*train.price_ratio.to_numpy()))
    beta=np.linalg.solve(z.T@z+np.eye(len(features)),z.T@response)
    return float(row.prior_q2_usd*row.price_ratio)*math.exp(float(test@beta))
ns['lh_fit']=fit
rows=[]
for y in range(2016,2026):
    rows.append({'year':y,'as_of':f'{y}-05-31','target_filed':f'{y}-08-01',
                 'target_q2_usd':(110+y-2016)*1e6,'prior_q2_usd':100e6,
                 'price_ratio':1.0,'forecast_mean_p12_pct':15+y-2016,
                 'fieldwork_complete':True,'fieldwork_mean_days':5.0})
df=pd.DataFrame(rows)
test=SimpleNamespace(year=2025,as_of='2025-05-31',prior_q2_usd=100e6,
                     price_ratio=1.0,forecast_mean_p12_pct=24.0)
p1,t1=ns['predict_forecast_only'](df,test)
changed=df.copy(); changed.fieldwork_complete=False; changed.fieldwork_mean_days=np.nan
p2,t2=ns['predict_forecast_only'](changed,test)
assert p1==p2
pd.testing.assert_frame_equal(t1,t2)
assert 2025 not in t1.year.to_list()
late=df.copy(); late.loc[late.year.eq(2024),'target_filed']='2025-06-01'
_,t3=ns['predict_forecast_only'](late,test)
assert 2024 not in t3.year.to_list()
duplicate=pd.concat([df,df.iloc[[0]]],ignore_index=True)
try: ns['forecast_only_rows'](duplicate)
except ValueError as e: assert 'Duplicate' in str(e)
else: raise AssertionError('Duplicate year was accepted')
bad=df.copy(); bad.loc[0,'forecast_mean_p12_pct']=np.nan
try: ns['forecast_only_rows'](bad)
except ValueError as e: assert 'Non-finite' in str(e)
else: raise AssertionError('Missing forecast was accepted')
assert ns['_check_no_fieldwork_dependency'](df)
print('Synthetic invariance, target-cutoff, duplicate, and missing-forecast checks passed.')
