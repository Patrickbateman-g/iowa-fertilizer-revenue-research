from pathlib import Path
from urllib.parse import urlsplit
import datetime as dt
import math
import numpy as np
import pandas as pd


def further_2025_sensitivities():
    v2=V2_RESULTS['predictions'].copy()
    prior=PHOSPHATE_FORECAST_2025_TESTS['predictions'].copy()
    vols={}
    for v in EXTRA_POINTS['q1_volume']:
        y=int(v['year'])
        if y in vols: raise ValueError('Duplicate volume year')
        for k in ('current_q1_tons','previous_q1_tons'):
            if not np.isfinite(float(v[k])) or float(v[k]) <= 0: raise ValueError('Invalid Q1 volume')
        if v['units'] != 'thousand product tons': raise ValueError('Volume unit mismatch')
        if not (dt.date(y,3,31)<lh_date(v['current_filed'])<=dt.date(y,5,31)
                and dt.date(y-1,3,31)<lh_date(v['previous_filed'])<=dt.date(y-1,5,31)):
            raise ValueError('Q1 volume release not available before cutoff')
        for k in ('current_source','previous_source'):
            u=urlsplit(v[k])
            if u.scheme!='https' or u.hostname!='www.sec.gov' or not u.path.startswith('/Archives/edgar/data/1324404/'):
                raise ValueError('Q1 source mismatch')
        vols[y]=v
    if not set(range(2016,2027)).issubset(vols): raise ValueError('Missing volume observations')

    def pick_v2(y,model):
        t=v2[(v2.year==y)&(v2.cohort=='price_ratio')&(v2.model==model)]
        if len(t)!=1: raise ValueError(f'Missing/duplicate V2 {model} for {y}')
        r=t.iloc[0]
        if not all(np.isfinite(float(r[k])) and float(r[k])>0 for k in ('target_usd','prediction_usd')):
            raise ValueError('Invalid V2 result')
        return r
    def pick_old(y,model):
        t=prior[(prior.year==y)&(prior.candidate==model)]
        if len(t)!=1: raise ValueError(f'Missing/duplicate previous candidate {model} for {y}')
        return t.iloc[0]
    rows=[]
    for y in range(2016,2027):
        price=pick_v2(y,'price_ratio'); q1=pick_v2(y,'q1_ratio')
        if not np.isclose(price.target_usd,q1.target_usd,rtol=1e-10,atol=1): raise ValueError('Target mismatch')
        ratio=float(vols[y]['current_q1_tons'])/float(vols[y]['previous_q1_tons'])
        p=float(price.prediction_usd); q=float(q1.prediction_usd); actual=float(price.target_usd)
        predictions={'price_only':p, 'price_sqrt_volume':p*math.sqrt(ratio),
                     'q1_revenue_ratio':q, 'geometric_price_q1_revenue':math.sqrt(p*q)}
        if y>=2022:
            old=pick_old(y,'adjusted_sqrt_q1_volume')
            full=pick_old(y,'adjusted_full')
            half=pick_old(y,'adjusted_half_log_correction')
            oldprice=pick_old(y,'price_only')
            if not np.isclose(old.actual_usd,actual,rtol=1e-10,atol=1): raise AssertionError('Actual mismatch')
            if not np.isclose(oldprice.prediction_usd,p,rtol=1e-10,atol=1): raise AssertionError('Price baseline mismatch')
            rebuilt=float(full.prediction_usd)*math.sqrt(ratio)
            if not np.isclose(old.prediction_usd,rebuilt,rtol=1e-10,atol=1): raise AssertionError('Existing adjusted model mismatch')
            predictions['previous_weather_sqrt_volume']=rebuilt
            predictions['half_weather_sqrt_volume']=float(half.prediction_usd)*math.sqrt(ratio)
        for name,forecast in predictions.items():
            rows.append(dict(year=y,candidate=name,actual_usd=actual,prediction_usd=forecast,
                             signed_error_usd=forecast-actual,absolute_error_usd=abs(forecast-actual)))
    pred=pd.DataFrame(rows)
    groups=[('2016-2021',range(2016,2022)),('2022-2026',range(2022,2027)),
            ('2022-2026 excluding 2025',[2022,2023,2024,2026])]
    scores=[]
    for group,years in groups:
        part=pred[pred.year.isin(years)]
        required=len(years)
        for name,d in part.groupby('candidate'):
            if len(d)!=required: raise ValueError('Incomplete comparison cohort')
            scores.append(dict(period=group,candidate=name,years=required,
                               mae_musd=float(d.absolute_error_usd.mean()/1e6)))
    scores=pd.DataFrame(scores)
    annual=pred.pivot(index='year',columns='candidate',values='absolute_error_usd')/1e6
    current=pred[pred.year.eq(2025)].copy()
    r=vols[2025]
    row=FIELDWORK_WINDOW_RESULTS['six']['data']; row=row[row.year.eq(2025)]
    if len(row)!=1: raise ValueError('Missing 2025 model row')
    prior_q2=float(row.iloc[0].prior_q2_usd)
    print('2025 previous Q2 revenue (USD m):',round(prior_q2/1e6,2),
          '| Q1 shipment ratio:',round(float(r['current_q1_tons'])/float(r['previous_q1_tons']),4))
    print('2025 predicted revenue and signed errors (USD millions):')
    display(current.assign(prediction_m=current.prediction_usd/1e6,
                           signed_error_m=current.signed_error_usd/1e6)
            [['candidate','prediction_m','signed_error_m']].round(2))
    print('Matched mean absolute errors (USD millions):')
    display(scores.round(2))
    print('Year-by-year absolute errors (USD millions; missing older forecast variants are intentional):')
    display(annual.round(2))
    report='''# Further 2025 exploratory sensitivity checks

This reproduces the prior phosphate-adjusted weather plus square-root Q1-volume prediction, and compares fixed variants for 2022–2026. A reduced weather correction and no-weather sensitivity each retain the same square-root Q1-volume ratio. A separate geometric average combines the price-only and Q1-revenue-ratio forecasts: Q1 revenue already reflects prices and volumes and is not an independent volume estimate. Q1-volume disclosures must be filed by each historical May 31 cutoff. The V2 price and Q1-revenue forecasts reuse the previously validated as-of data and their tests; exact target and baseline agreement is enforced.

The simple price-plus-volume and geometric price-plus-Q1 comparisons also run in 2016–2021 to probe earlier performance. The weather-plus-volume model does not have enough comparable historical training rows for that earlier period. All years have already been inspected and candidates were suggested after looking at those outcomes; the small matched evaluation is not an untouched test and candidate selection may exaggerate improvements. A year excluded from a summary is not a holdout when it informed model choices. Q1 company sales reflect both prices and volume and comparisons can be distorted by changes in the previous year. Company-wide shipments and sales are not Iowa corn purchases. Forecast probability is not observed precipitation. Historical price and forecast vintage, geographical mismatch, and changing business conditions limit conclusions. These diagnostics do not establish causation or alter the frozen 2027 test.
'''
    out=Path('/kaggle/working/iowa_forecast_further_2025'); out.mkdir(parents=True,exist_ok=True)
    pred.to_csv(out/'predictions_private.csv',index=False)
    scores.to_csv(out/'scores_private.csv',index=False)
    annual.reset_index().to_csv(out/'annual_errors_private.csv',index=False)
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    print(report)
    return {'predictions':pred,'scores':scores,'annual':annual,'report':report}

FURTHER_2025_RESULTS=further_2025_sensitivities()
