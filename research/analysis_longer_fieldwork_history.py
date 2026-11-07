from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import datetime as dt
import json
import math
import numpy as np
import pandas as pd

LH_YEARS = tuple(range(2010, 2016))
LH_STATIONS = ('KAMW', 'KCID', 'KSUX')
LH_OUTPUT = Path('/kaggle/working/iowa_long_history')

def lh_date(value):
    if not isinstance(value, str):
        raise ValueError('Date must be ISO YYYY-MM-DD')
    date=dt.date.fromisoformat(value)
    if date.isoformat()!=value: raise ValueError('Invalid ISO date')
    return date

def lh_expected(year):
    end=dt.date(year,5,20)
    sunday=end-dt.timedelta(days=(end.weekday()+1)%7)
    return [sunday-dt.timedelta(days=7*i) for i in range(5,-1,-1)]

def lh_url(url, group):
    u=urlsplit(url if isinstance(url,str) else '')
    host=u.hostname or ''
    if u.scheme!='https': raise ValueError('Require HTTPS source')
    if group=='price':
        allowed=host=='bls.gov' or host.endswith('.bls.gov')
    elif group=='fieldwork':
        allowed=(host=='usda.gov' or host.endswith('.usda.gov') or host in (
            'iowaagriculture.gov','publications.iowa.gov','news.utcrops.com',
            'k923.fm','whoradio.iheart.com','thegrundyregister.com','www.thegrundyregister.com','northiowatoday.com',
            'downloads.usda.library.cornell.edu','www.farmprogress.com','files01.core.ac.uk'))
    else:
        allowed=host=='mesonet.agron.iastate.edu' and u.path=='/cgi-bin/request/mos.py'
    if not allowed: raise ValueError(f'Unexpected source host for {group}: {host}')
    return u

def lh_forecasts(records):
    expected={(y,s) for y in LH_YEARS for s in LH_STATIONS}; found={}; coverage=[]
    required={'year','station','expected_days','valid_days','duplicate_rows','conflicting_days',
              'invalid_rows','station_mean_p12_pct','source_url','status'}
    for r in records:
        if not isinstance(r,dict) or not required.issubset(r): raise ValueError('Incomplete station summary')
        y,s=r['year'],r['station']
        if (y,s) not in expected or (y,s) in found: raise ValueError('Unexpected or duplicate station-year')
        for k in ('expected_days','valid_days','duplicate_rows','conflicting_days','invalid_rows'):
            if isinstance(r[k],bool) or not isinstance(r[k],int) or r[k]<0: raise ValueError('Invalid coverage count')
        if r['expected_days']!=50 or r['valid_days']+r['conflicting_days']>50: raise ValueError('Invalid 50-day window')
        u=lh_url(r['source_url'],'forecast'); q=parse_qs(u.query)
        wanted={'station':s,'model':'GFS','sts':f'{y}-04-01T00:00Z','ets':f'{y}-05-21T00:00Z','format':'json'}
        if any(q.get(k)!=[v] for k,v in wanted.items()): raise ValueError('Wrong forecast source parameters')
        mean=r['station_mean_p12_pct']
        if mean is not None and (isinstance(mean,bool) or not isinstance(mean,(int,float)) or not np.isfinite(mean) or not 0<=mean<=100): raise ValueError('Invalid forecast probability')
        if (r['valid_days']==0)!=(mean is None): raise ValueError('Forecast mean/count inconsistency')
        usable=(r['status']=='ok' and r['valid_days']>=40 and r['invalid_rows']==0 and mean is not None)
        found[(y,s)]={**r,'usable':usable}
    for y,s in sorted(expected): coverage.append(found.get((y,s),{'year':y,'station':s,'status':'missing','usable':False}))
    annual=[]
    for y in LH_YEARS:
        three=[found.get((y,s)) for s in LH_STATIONS]
        ok=all(x is not None and x['usable'] for x in three)
        annual.append({'year':y,'forecast_complete':ok,'forecast_mean_p12_pct':float(np.mean([x['station_mean_p12_pct'] for x in three])) if ok else np.nan})
    return pd.DataFrame(annual),pd.DataFrame(coverage)

def lh_fieldwork(records):
    found={}; cov=[]
    for r in records:
        needed={'year','week_end','release_date','days_suitable','source_url'}
        if not isinstance(r,dict) or not needed.issubset(r): raise ValueError('Incomplete weekly report')
        y=r['year']; end=lh_date(r['week_end'])
        if y not in LH_YEARS or end not in lh_expected(y) or (y,end) in found: raise ValueError('Unexpected or duplicate week')
        value=r['days_suitable']; pub=lh_date(r['release_date'])
        if not end<pub<=dt.date(y,5,31): raise ValueError('Fieldwork document after cutoff or before week ended')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<=value<=7: raise ValueError('Invalid days suitable')
        lh_url(r['source_url'],'fieldwork'); found[(y,end)]={**r,'usable':True}
    annual=[]
    for y in LH_YEARS:
        expected=lh_expected(y); selected=[found.get((y,d)) for d in expected]
        for d,item in zip(expected,selected): cov.append(item if item is not None else {'year':y,'week_end':d.isoformat(),'usable':False,'status':'missing'})
        ok=all(x is not None for x in selected)
        annual.append({'year':y,'fieldwork_complete':ok,'weeks_present':sum(x is not None for x in selected),
                       'fieldwork_mean_days':float(np.mean([x['days_suitable'] for x in selected])) if ok else np.nan})
    return pd.DataFrame(annual),pd.DataFrame(cov)

def lh_prices(records):
    found={}; required={'year','current_april','previous_april','current_release_date','previous_release_date','current_source_url','previous_source_url'}
    for r in records:
        if not isinstance(r,dict) or not required.issubset(r): raise ValueError('Incomplete BLS price row')
        y=r['year']
        if y not in LH_YEARS or y in found: raise ValueError('Unexpected or duplicate BLS year')
        now=lh_date(r['current_release_date']); prior=lh_date(r['previous_release_date'])
        if not (dt.date(y,4,30)<now<=dt.date(y,5,31) and dt.date(y-1,8,31)<prior<dt.date(y,1,1)):
            raise ValueError('Price source dates unavailable by cutoff')
        lh_url(r['current_source_url'],'price'); lh_url(r['previous_source_url'],'price')
        a,b=r['current_april'],r['previous_april']
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0 for v in (a,b)): raise ValueError('Invalid BLS index')
        found[y]={**r,'price_ratio':float(a)/float(b),'price_complete':True}
    return pd.DataFrame([found.get(y,{'year':y,'price_complete':False}) for y in LH_YEARS])

def lh_original_revenue(snapshot):
    raw=snapshot.get('revenue') if isinstance(snapshot,dict) else None
    if not isinstance(raw,list): raise ValueError('Missing originally reported revenue snapshot')
    rev=pd.DataFrame(raw)
    if not {'year','q2_usd','q2_filed','q2_accession'}.issubset(rev.columns) or rev.year.duplicated().any(): raise ValueError('Invalid revenue coverage')
    return rev.set_index('year')

def lh_fit(train,row,features):
    baseline=float(row.prior_q2_usd*row.price_ratio)
    if not features: return baseline
    x=train[features].to_numpy(float); mu=x.mean(axis=0); sd=x.std(axis=0,ddof=0)
    if not np.isfinite(x).all() or not np.isfinite(sd).all() or (sd<=1e-12).any(): raise ValueError('No training feature variation')
    z=(x-mu)/sd; test=(np.array([getattr(row,c) for c in features],float)-mu)/sd
    response=np.log(train.target_q2_usd.to_numpy(float)/(train.prior_q2_usd.to_numpy(float)*train.price_ratio.to_numpy(float)))
    beta=np.linalg.solve(z.T@z+np.eye(len(features)),z.T@response)
    exponent=float(test@beta)
    if not np.isfinite(exponent) or abs(exponent)>700: raise ValueError('Invalid prediction')
    return baseline*math.exp(exponent)

def long_history_research(window_results,snapshot,forecasts,fieldwork,prices):
    fa,fc=lh_forecasts(forecasts); wa,wc=lh_fieldwork(fieldwork); pa=lh_prices(prices)
    available=fa.merge(wa,on='year',validate='one_to_one').merge(pa,on='year',validate='one_to_one')
    for col in ('forecast_complete','fieldwork_complete','price_complete'): available[col]=available[col].fillna(False).astype(bool)
    rev=lh_original_revenue(snapshot); rows=[]
    for x in available.itertuples():
        if not (x.forecast_complete and x.fieldwork_complete and x.price_complete): continue
        y=int(x.year)
        if y not in rev.index or y-1 not in rev.index: continue
        actual,prior=rev.loc[y],rev.loc[y-1]
        cutoff=f'{y}-05-31'; filed=str(actual.q2_filed)[:10]; previous_filed=str(prior.q2_filed)[:10]
        if not (f'{y}-06-30'<=filed<=f'{y+1}-05-31' and previous_filed<=cutoff): raise ValueError('Revenue chronology error')
        if float(actual.q2_usd)<=0 or float(prior.q2_usd)<=0 or not str(actual.q2_accession).strip() or not str(prior.q2_accession).strip(): raise ValueError('Invalid revenue')
        rows.append({'year':y,'as_of':cutoff,'target_filed':filed,'target_q2_usd':float(actual.q2_usd),
                     'prior_q2_usd':float(prior.q2_usd),'price_ratio':float(x.price_ratio),
                     'forecast_mean_p12_pct':float(x.forecast_mean_p12_pct),'fieldwork_mean_days':float(x.fieldwork_mean_days)})
    columns=['year','as_of','target_filed','target_q2_usd','prior_q2_usd','price_ratio','forecast_mean_p12_pct','fieldwork_mean_days']
    history=pd.DataFrame(rows,columns=columns)
    original=window_results['six']['data'].copy()
    if original.year.duplicated().any() or set(original.year)&set(LH_YEARS): raise ValueError('Original cohort mismatch')
    augmented=pd.concat([history,original[columns]],ignore_index=True).sort_values('year')
    existing=window_results['six']['predictions']
    features={'price_only':[],'price_forecast':['forecast_mean_p12_pct'],
              'price_forecast_fieldwork':['forecast_mean_p12_pct','fieldwork_mean_days']}
    output=[]
    for y in range(2022,2027):
        matches=original[original.year.eq(y)]
        if len(matches)!=1: raise ValueError(f'Missing original test year {y}')
        row=next(matches.itertuples()); cutoff=str(row.as_of)[:10]
        if cutoff!=f'{y}-05-31' or str(row.target_filed)[:10]<=cutoff: raise ValueError('Test cutoff mismatch')
        for label,df in (('original',original),('extended',augmented)):
            train=df[(df.year<y)&(df.target_filed.astype(str).str[:10]<=cutoff)]
            if len(train)<6: raise ValueError('Less than six eligible training years')
            for model,cols in features.items():
                pred=lh_fit(train,row,cols); actual=float(row.target_q2_usd)
                output.append({'year':y,'training':label,'model':model,'train_years':len(train),
                               'earlier_years':','.join(map(str,train.year.astype(int))),
                               'prediction_usd':pred,'actual_usd':actual,'absolute_error_usd':abs(pred-actual)})
                if label=='original':
                    prior=existing[(existing.year==y)&(existing.model==model)]
                    if len(prior)!=1 or not np.isclose(pred,float(prior.prediction_usd.iloc[0]),rtol=1e-10,atol=1e-3):
                        raise AssertionError('Original-model reproduction differs from existing notebook')
    pred=pd.DataFrame(output)
    score=pred.groupby(['model','training'],as_index=False).agg(years=('year','size'),mae_usd=('absolute_error_usd','mean'))
    paired=pred.pivot(index=['year','model'],columns='training',values='absolute_error_usd').reset_index()
    paired['extended_minus_original_usd']=paired.extended-paired.original
    summary=paired.groupby('model',as_index=False).agg(years=('year','size'),extended_wins=('extended_minus_original_usd',lambda x:int((x<0).sum())))
    cov=available[['year','forecast_complete','fieldwork_complete','price_complete','weeks_present']].copy()
    cov['used']=cov.year.isin(set(history.year))
    LH_OUTPUT.mkdir(parents=True,exist_ok=True)
    for name,frame in [('forecast_coverage_private.csv',fc),('fieldwork_coverage_private.csv',wc),('price_sources_private.csv',pa),('coverage_private.csv',cov),('historical_features_private.csv',history),('predictions_private.csv',pred),('scores_private.csv',score),('paired_private.csv',paired),('wins_private.csv',summary)]: frame.to_csv(LH_OUTPUT/name,index=False)
    p22=pred[(pred.year==2022)&(pred.model=='price_forecast_fieldwork')].set_index('training')
    report=f'''# Older-training-year exploratory test

Earlier years with complete Iowa GFS MOS forecast, six-week fieldwork, originally reported revenue and BLS price sources: {', '.join(map(str,history.year.astype(int))) if len(history) else 'none'}. The original and extended models use the same 2022–2026 prediction rows, same price-only baseline, training-only standardization, penalty 1 and no intercept. The original model is required to reproduce the existing saved predictions exactly. In 2022, original six-week prediction error was ${float(p22.loc['original','absolute_error_usd'])/1e6:,.1f}m, extended error ${float(p22.loc['extended','absolute_error_usd'])/1e6:,.1f}m. Full matched scores, per-year results, coverage and source metadata are in this folder.

Older nitrogen price ratios are reconstructed from original BLS current-April initial releases and the prior April's four-month-revised value published the previous September; they are not ALFRED vintages. Only earlier target results filed by each simulated cutoff are used. Forecast aggregates are manually transcribed archive summaries and may have been revised; exact historic product-publication timing has not been independently verified. Fieldwork records may use a later report's previous-week value or attributed reproductions; record their individual dates and possible revisions. The notebook validates the metadata and arithmetic, not the accuracy of the transcription or external source contents.

The historical tests through 2026 were already inspected; this is exploratory, not a new untouched holdout. More training years do not establish causality or guarantee better future performance. No frozen 2027 prediction or protocol was altered.
'''
    (LH_OUTPUT/'REPORT.md').write_text(report,encoding='utf8')
    print(report)
    display(cov)
    display(score.assign(mae_musd=lambda x:x.mae_usd/1e6).drop(columns='mae_usd').round(2))
    display(summary)
    display((paired.query("model == 'price_forecast_fieldwork'").set_index('year')[['original','extended']]/1e6).round(1))
    return {'history':history,'coverage':cov,'predictions':pred,'scores':score,'paired':paired,'summary':summary,'report':report}

LONG_HISTORY_RESULTS=long_history_research(FIELDWORK_WINDOW_RESULTS,SNAPSHOT_JSON,HISTORICAL_FORECASTS,HISTORICAL_FIELDWORK,HISTORICAL_PRICES)
