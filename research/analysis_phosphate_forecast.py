from pathlib import Path
from urllib.parse import urlsplit
import datetime as dt
import numpy as np
import pandas as pd


def forecast_phosphate_comparison():
    features=['forecast_mean_p12_pct']
    oldhistory=LONG_HISTORY_RESULTS['history'].copy()
    add=MISSING_FIELDWORK_RESULTS['additional_years'].copy()
    alternate=MISSING_FIELDWORK_SENSITIVITY['additional_years'].copy()
    orig=FIELDWORK_WINDOW_RESULTS['six']['data'].copy()
    cols=list(oldhistory.columns)
    if set(add.year.astype(int)) != {2010,2011,2013}: raise ValueError('Missing added years')
    independent=[c for c in cols if c!='fieldwork_mean_days']
    pd.testing.assert_frame_equal(add[independent].sort_values('year').reset_index(drop=True),
                                  alternate[independent].sort_values('year').reset_index(drop=True))
    prev=pd.concat([oldhistory,orig[cols]],ignore_index=True).sort_values('year')
    full=pd.concat([prev,add[cols]],ignore_index=True).sort_values('year')
    if prev.year.duplicated().any() or full.year.duplicated().any(): raise ValueError('Duplicate cohort year')
    if not set(range(2010,2016)).issubset(set(full.year.astype(int))): raise ValueError('Incomplete older cohort')

    records={}
    for r in PHOSPHATE_SEGMENT:
        if not isinstance(r,dict) or not {'year','phosphate_q2_usd','filed_date','source_url','basis'}<=set(r):
            raise ValueError('Missing segment provenance')
        y=r['year']; value=r['phosphate_q2_usd']; url=urlsplit(r['source_url']); filed=lh_date(r['filed_date'])
        if isinstance(y,bool) or not isinstance(y,int) or y not in range(2009,2016) or y in records: raise ValueError('Invalid segment year')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0: raise ValueError('Invalid phosphate amount')
        if not dt.date(y,6,30)<filed<=dt.date(y+1,5,31): raise ValueError('Invalid segment filing date')
        if url.scheme!='https' or url.hostname!='www.sec.gov' or not url.path.startswith('/Archives/edgar/data/1324404/') or not str(r['basis']).strip(): raise ValueError('Invalid segment source')
        records[y]=dict(r)
    if set(records)!=set(range(2009,2016)) or records[2015]['phosphate_q2_usd']!=0: raise ValueError('Incomplete segment series')

    def adjust(data):
        adjusted=data.copy()
        for idx,row in adjusted[adjusted.year.between(2010,2015)].iterrows():
            y=int(row.year); current=records[y]; prior=records[y-1]
            if lh_date(prior['filed_date'])>lh_date(str(row.as_of)[:10]): raise ValueError('Prior segment unavailable at row cutoff')
            if lh_date(current['filed_date'])>lh_date(str(row.target_filed)[:10]): raise ValueError('Target segment later than revenue filing')
            target=float(row.target_q2_usd)-current['phosphate_q2_usd']
            base=float(row.prior_q2_usd)-prior['phosphate_q2_usd']
            if min(target,base)<=0: raise ValueError('Segment exceeds sales')
            adjusted.loc[idx,['target_q2_usd','prior_q2_usd']]=[target,base]
        return adjusted
    prev_adjusted=adjust(prev); full_adjusted=adjust(full)
    variants={'prior_unadjusted':prev,'prior_adjusted':prev_adjusted,
              'full_unadjusted':full,'full_adjusted':full_adjusted}
    output=[]
    for y in range(2022,2027):
        match=orig[orig.year.eq(y)]
        if len(match)!=1: raise ValueError('Missing test row')
        row=next(match.itertuples()); cutoff=str(row.as_of)[:10]
        if cutoff!=f'{y}-05-31' or str(row.target_filed)[:10]<=cutoff: raise ValueError('Bad test cutoff')
        for label,data in variants.items():
            train=data[(data.year<y)&(data.target_filed.astype(str).str[:10]<=cutoff)].copy()
            if len(train)<6: raise ValueError('Too few training rows')
            if 'adjusted' in label and not label.endswith('unadjusted'):
                for trainyear in train.year.astype(int):
                    if trainyear in range(2010,2016) and lh_date(records[trainyear]['filed_date'])>lh_date(cutoff): raise ValueError('Segment target unavailable')
            prediction=lh_fit(train,row,features)
            if label.endswith('unadjusted'):
                if label=='prior_unadjusted':
                    ref=LONG_HISTORY_RESULTS['predictions']; train_label='extended'
                else:
                    ref=MISSING_FIELDWORK_RESULTS['predictions']; train_label='new'
                ref=ref[(ref.year==y)&(ref.model=='price_forecast')&(ref.training==train_label)]
                if len(ref)!=1 or not np.isclose(prediction,float(ref.prediction_usd.iloc[0]),rtol=1e-10,atol=1e-3): raise AssertionError('Prior prediction mismatch')
            actual=float(row.target_q2_usd)
            output.append({'year':y,'variant':label,'training_years':len(train),
                           'training_year_list':','.join(map(str,train.year.astype(int))),
                           'actual_usd':actual,'prediction_usd':prediction,'signed_error_usd':prediction-actual,
                           'absolute_error_usd':abs(prediction-actual)})
    pred=pd.DataFrame(output)
    scores=pred.groupby('variant',as_index=False).agg(years=('year','size'),mae_usd=('absolute_error_usd','mean'))
    annual=pred.pivot(index='year',columns='variant',values='absolute_error_usd')
    for cohort in ('prior','full'):
        if f'{cohort}_adjusted' not in annual.columns or f'{cohort}_unadjusted' not in annual.columns: raise ValueError('Missing comparison')
    report='''# Phosphate-adjusted simpler forecast model: exploratory comparison

The one-feature forecast model uses the same 2022–2026 test rows and original company-wide test targets, the same price-only benchmark, forecast feature, training-only scaling and penalty as before. The phosphate adjustment changes the training Q2 target and previous Q2 baseline for historical years 2010–2015 only, using the original dated CF segment releases. Both the shorter prior-history cohort and the full-history cohort are compared with and without that adjustment, on matched years. Existing unadjusted predictions must be reproduced exactly. The two 2013 fieldwork alternatives agree on every variable used here, so this model is invariant to that fieldwork discrepancy.

Past outcomes informed this and earlier research; these five years are not an untouched holdout. The nitrogen-only historical training figures are only an approximation to current company-wide revenue, not a restatement of test revenue. Historic forecast-vintage uncertainty and older BLS price reconstructions remain limitations. More historical years and additional model comparisons increase selection risk; results do not establish causation or reliable future improvement. The frozen 2027 protocol is unchanged.
'''
    out=Path('/kaggle/working/iowa_phosphate_forecast'); out.mkdir(parents=True,exist_ok=True)
    pred.to_csv(out/'predictions_private.csv',index=False)
    scores.to_csv(out/'scores_private.csv',index=False)
    annual.reset_index().to_csv(out/'annual_errors_private.csv',index=False)
    (out/'REPORT.md').write_text(report,encoding='utf8')
    print(report)
    display(scores.assign(mae_musd=lambda d:d.mae_usd/1e6).drop(columns='mae_usd').round(2))
    display((annual/1e6).round(1))
    display(pred.assign(actual_m=lambda d:d.actual_usd/1e6,prediction_m=lambda d:d.prediction_usd/1e6,signed_error_m=lambda d:d.signed_error_usd/1e6)[['year','variant','training_years','actual_m','prediction_m','signed_error_m']].round(2))
    return {'predictions':pred,'scores':scores,'annual':annual,'report':report}

PHOSPHATE_FORECAST_RESULTS=forecast_phosphate_comparison()
