from pathlib import Path
from collections import defaultdict
import math
import numpy as np
import pandas as pd

MD_STATIONS=('KAMW','KCID','KSUX')
MD_YEARS=tuple(range(2010,2027))
MD_OUT=Path('/kaggle/working/iowa_missing_forecast_days')

def md_day_map(records, station, year):
    expected=list(pd.date_range(f'{year}-04-01',f'{year}-05-20',freq='D',tz='UTC'))
    vals=defaultdict(set); seen=defaultdict(int); nulls=defaultdict(int); invalid=defaultdict(int)
    for row in records:
        if not isinstance(row,dict): raise ValueError('Unexpected non-dict in selected ledger')
        runtime=_audit_utc(row.get('runtime')); valid=_audit_utc(row.get('ftime'))
        if runtime not in expected or valid is None or valid!=runtime+pd.Timedelta(hours=24):
            raise ValueError('Selected row outside configured window')
        seen[runtime]+=1
        if row.get('station')!=station or row.get('model')!='GFS':
            invalid[runtime]+=1; continue
        value=row.get('p12')
        if value is None or value in ('','M','null'):
            nulls[runtime]+=1; continue
        if isinstance(value,bool): invalid[runtime]+=1; continue
        try: v=float(value)
        except (TypeError,ValueError): invalid[runtime]+=1; continue
        if not math.isfinite(v) or not 0<=v<=100: invalid[runtime]+=1; continue
        vals[runtime].add(v)
    result=[]
    for day in expected:
        choices=vals.get(day,set())
        if len(choices)>1: state='conflicting_values'; v=None
        elif len(choices)==1: state='valid'; v=next(iter(choices))
        elif invalid.get(day,0): state='invalid_only'; v=None
        elif nulls.get(day,0): state='null_only'; v=None
        else: state='absent_in_selected_ledger'; v=None
        result.append({'year':year,'station':station,'runtime_utc':day.isoformat(),
                       'date':day.date().isoformat(),'state':state,'p12_pct':v,
                       'selected_rows':seen.get(day,0),'null_rows':nulls.get(day,0),
                       'invalid_rows':invalid.get(day,0),'distinct_values':len(choices)})
    return result

def md_build_daily(audit):
    if not isinstance(audit,dict) or not {'ledger','manifest'}<=set(audit):
        raise ValueError('Missing input audit ledger and manifest')
    man=audit['manifest'].copy()
    if man.duplicated(['year','station']).any(): raise ValueError('Duplicate audit manifest')
    observed={}
    for row in audit['ledger']:
        key=(int(row['year']),str(row['station']))
        if 'original_source_record' not in row: raise ValueError('Unexpected non-dict selected row')
        if _audit_hash(row['original_source_record'])!=row['original_record_sha256']:
            raise AssertionError('Selected source record checksum mismatch')
        observed.setdefault(key,[]).append(row['original_source_record'])
    daily=[]; stats=[]
    for y in MD_YEARS:
        for station in MD_STATIONS:
            item=man[(man.year.eq(y)) & man.station.eq(station)]
            if len(item)!=1: raise ValueError('Expected one manifest per station-year')
            m=item.iloc[0]
            if m.status!='retrieved':
                stats.append({'year':y,'station':station,'source_status':str(m.status),'available':False})
                continue
            raw=observed.get((y,station),[])
            _, replay=audit_forecast_rows(raw,station,y)
            for k in ('valid_days','duplicate_rows','conflicting_days','invalid_rows'):
                if int(replay[k])!=int(m[k]): raise AssertionError('Saved manifest failed replay: '+k)
            if replay['station_mean_p12_pct'] is not None and not np.isclose(replay['station_mean_p12_pct'],float(m.station_mean_p12_pct),rtol=0,atol=1e-10):
                raise AssertionError('Station mean failed replay')
            rows=md_day_map(raw,station,y); daily.extend(rows)
            stats.append({'year':y,'station':station,'source_status':'retrieved','available':True,
                'valid_days':replay['valid_days'],'missing_or_unusable_days':50-replay['valid_days'],
                'duplicate_rows':replay['duplicate_rows'],'conflicting_days':replay['conflicting_days'],
                'invalid_rows':replay['invalid_rows'],'usable_original':replay['usable'],
                'station_mean_p12_pct':replay['station_mean_p12_pct'],
                'missing_dates':','.join(r['date'] for r in rows if r['state']!='valid')})
    return pd.DataFrame(daily),pd.DataFrame(stats)

def md_annual_features(daily, stats):
    rows=[]
    for y in MD_YEARS:
        g=stats[stats.year.eq(y)]
        if len(g)!=3 or not g.available.fillna(False).all():
            rows.append({'year':y,'source_complete':False,'common_days':np.nan,'strict_50':False,
                         'original_pct':np.nan,'common_pct':np.nan}); continue
        d=daily[daily.year.eq(y)]
        wide=d.pivot(index='date',columns='station',values='p12_pct').reindex(columns=MD_STATIONS)
        common=wide.dropna()
        original=float(g.station_mean_p12_pct.mean()) if g.usable_original.all() else np.nan
        common_pct=float(common.to_numpy().mean()) if g.usable_original.all() and len(common)>=40 else np.nan
        strict=bool(g.usable_original.all() and len(common)==50)
        rows.append({'year':y,'source_complete':True,'common_days':len(common),'strict_50':strict,
                     'original_pct':original,'common_pct':common_pct,
                     'common_minus_original_pct_points':common_pct-original if np.isfinite(original) and np.isfinite(common_pct) else np.nan})
    return pd.DataFrame(rows)

def md_forecast_comparisons(annual, dataset, segments, saved):
    data=dataset.copy()
    if data.year.duplicated().any(): raise ValueError('Duplicate forecast dataset year')
    data=data.merge(annual,on='year',how='left',validate='one_to_one')
    if data.original_pct.isna().any(): raise ValueError('Missing original replay on existing data')
    if not np.allclose(data.original_pct,data.forecast_mean_p12_pct,atol=1e-5,rtol=0):
        raise AssertionError('Rebuilt annual feature differs from original')
    segments={int(x['year']):x for x in segments}
    if set(segments)!=set(range(2009,2016)): raise ValueError('Incomplete phosphate records')
    for idx,row in data[data.year.between(2010,2015)].iterrows():
        y=int(row.year)
        if lh_date(segments[y-1]['filed_date'])>lh_date(str(row.as_of)[:10]) or lh_date(segments[y]['filed_date'])>lh_date(str(row.target_filed)[:10]):
            raise ValueError('Phosphate record chronology error')
        data.loc[idx,'target_q2_usd']-=float(segments[y]['phosphate_q2_usd'])
        data.loc[idx,'prior_q2_usd']-=float(segments[y-1]['phosphate_q2_usd'])
    records=[]; omissions=[]
    for mode in ('common','strict_50'):
        eligibility=data.common_pct.notna() if mode=='common' else data.strict_50.fillna(False)
        eligible=data[eligibility].copy()
        for y in range(2022,2027):
            match=eligible[eligible.year.eq(y)]
            if len(match)!=1:
                omissions.append({'mode':mode,'year':y,'reason':'test year does not meet coverage'}); continue
            row=next(match.itertuples()); cutoff=str(row.as_of)[:10]
            train=eligible[(eligible.year<y)&(eligible.target_filed.astype(str).str[:10]<=cutoff)].copy()
            if len(train)<6:
                omissions.append({'mode':mode,'year':y,'reason':'fewer than six matched training years'}); continue
            actual=float(row.target_q2_usd)
            for variant,feature in [('original','original_pct'),('common_day','common_pct')]:
                prediction=lh_fit(train,row,[feature])
                records.append({'cohort':mode,'year':y,'variant':variant,'training_years':len(train),
                    'training_year_list':','.join(map(str,train.year.astype(int))),
                    'prediction_usd':prediction,'actual_usd':actual,
                    'signed_error_usd':prediction-actual,'absolute_error_usd':abs(prediction-actual)})
    pred=pd.DataFrame(records,columns=['cohort','year','variant','training_years','training_year_list','prediction_usd','actual_usd','signed_error_usd','absolute_error_usd'])
    if pred.empty: scores=pd.DataFrame(columns=['cohort','variant','years','mae_musd'])
    else: scores=pred.groupby(['cohort','variant'],as_index=False).agg(years=('year','size'),mae_musd=('absolute_error_usd',lambda x:x.mean()/1e6))
    original=pred[(pred.cohort=='common')&(pred.variant=='original')]
    for row in original.itertuples():
        savedrow=saved[(saved.year.eq(row.year))&(saved.variant.eq('full_adjusted'))]
        if len(savedrow)!=1: raise ValueError('Saved full adjusted reference missing')
        prior=dataset[(dataset.year<row.year)&(dataset.target_filed.astype(str).str[:10]<=f'{row.year}-05-31')]
        if row.training_year_list==','.join(map(str,prior.sort_values('year').year.astype(int))) and not np.isclose(row.prediction_usd,float(savedrow.prediction_usd.iloc[0]),rtol=1e-10,atol=1e-3):
            raise AssertionError('Saved adjusted reference mismatch')
    return pred,scores,pd.DataFrame(omissions,columns=['mode','year','reason'])

def missing_forecast_days_research(audit, dataset, segments, saved):
    daily,stats=md_build_daily(audit)
    annual=md_annual_features(daily,stats)
    pred,scores,omitted=md_forecast_comparisons(annual,dataset,segments,saved)
    MD_OUT.mkdir(parents=True,exist_ok=True)
    for name,frame in [('daily_status_private.csv',daily),('station_coverage_private.csv',stats),
                       ('annual_features_private.csv',annual),('predictions_private.csv',pred),
                       ('scores_private.csv',scores),('omitted_private.csv',omitted)]: frame.to_csv(MD_OUT/name,index=False)
    print('Captured station-years:',int(stats.available.fillna(False).sum()),'/',len(stats))
    print('Station-years with fewer than 50 usable forecast days:',int((stats.valid_days.fillna(0)<50).sum()))
    print('Days with a non-valid state in retrieved selected records:',int((daily.state!='valid').sum()))
    print('Years with all 50 days valid at all three stations:',int(annual.strict_50.fillna(False).sum()),'/',len(annual))
    print('Annual original vs within-year common days (percentage points):')
    display(annual.round(6))
    print('Station-years with unavailable/short/conflicting/invalid days:')
    display(stats[(~stats.available.fillna(False)) | (stats.valid_days.fillna(0)<50) | (stats.conflicting_days.fillna(0)>0) | (stats.invalid_rows.fillna(0)>0)])
    print('Same-cohort model mean absolute errors (USD millions):'); display(scores.round(3))
    print('Omitted model test years:'); display(omitted)
    report='''This is a retrospective audit of saved selected forecast rows, not a fresh holdout. For each year common days are intersected among all three stations within that year; common-day means use the same dates/stations. Strict coverage requires 50 valid days at every station. The original rule accepts >=40 days per station and disallows invalid rows; conflicting values are excluded, duplicates counted once, and nulls omitted. Model variants in each comparison use exactly the same test and training rows. The precipitation values are forecasts of probability, not rainfall amounts. Browser preselection cannot prove source completeness, and archive revisions and historical forecast publication times remain unverified. No 2027 protocol changes. All outputs.
'''
    (MD_OUT/'REPORT.txt').write_text(report,encoding='utf-8'); print(report)
    return {'daily':daily,'station_coverage':stats,'annual':annual,'predictions':pred,'scores':scores,'omitted':omitted,'report':report}

MISSING_DAYS_RESULTS=missing_forecast_days_research(FORECAST_INPUT_AUDIT,DECOUPLED_FORECAST_RESULTS['data'],PHOSPHATE_SEGMENT,PHOSPHATE_FORECAST_RESULTS['predictions'])
