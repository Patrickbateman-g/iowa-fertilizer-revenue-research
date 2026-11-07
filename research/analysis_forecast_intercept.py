from pathlib import Path
import datetime as dt
import math
import numpy as np
import pandas as pd


def correction_predict(train, row, shrinkage=0.0):
    if shrinkage not in (0.0, 0.5, 1.0):
        raise ValueError('Unexpected intercept shrinkage')
    x=train['forecast_mean_p12_pct'].to_numpy(dtype=float)
    means=x.mean(); sd=x.std(ddof=0)
    if len(x)<6 or not np.isfinite(x).all() or not np.isfinite(sd) or sd<=1e-12:
        raise ValueError('Insufficient usable training variation')
    target=train['target_q2_usd'].to_numpy(dtype=float)
    baseline=train['prior_q2_usd'].to_numpy(dtype=float)*train['price_ratio'].to_numpy(dtype=float)
    row_baseline=float(row.prior_q2_usd)*float(row.price_ratio)
    feature=float(row.forecast_mean_p12_pct)
    if (not np.isfinite(target).all() or not np.isfinite(baseline).all() or
        (target<=0).any() or (baseline<=0).any() or not np.isfinite(row_baseline) or
        row_baseline<=0 or not np.isfinite(feature)):
        raise ValueError('Invalid revenue, price or forecast')
    residual=np.log(target/baseline)
    z=(x-means)/sd
    beta=float(z@residual/(z@z+1.0))
    intercept=float(shrinkage*residual.mean())
    exponent=intercept+(feature-means)/sd*beta
    if not np.isfinite(exponent) or abs(exponent)>700:
        raise ValueError('Invalid prediction exponent')
    return row_baseline*math.exp(exponent),intercept,beta


def adjusted_training_inputs(data, segments):
    required={'year','as_of','target_filed','target_q2_usd','prior_q2_usd',
              'price_ratio','forecast_mean_p12_pct'}
    if not required.issubset(data.columns) or data.year.duplicated().any():
        raise ValueError('Incomplete dataset or duplicate year')
    lookup={}
    for segment in segments:
        y=int(segment['year'])
        if y in lookup: raise ValueError('Duplicate phosphate year')
        amount=float(segment['phosphate_q2_usd'])
        if not np.isfinite(amount) or amount<0: raise ValueError('Invalid segment')
        lookup[y]=segment
    if set(lookup)!=set(range(2009,2016)): raise ValueError('Missing segment year')
    result=data.copy().sort_values('year').reset_index(drop=True)
    for idx,row in result.iterrows():
        y=int(row.year)
        if str(row.as_of)[:10]!=f'{y}-05-31' or str(row.target_filed)[:10]<=f'{y}-06-30':
            raise ValueError('Invalid prediction or target filing date')
        if y in range(2010,2016):
            prior,current=lookup[y-1],lookup[y]
            if str(prior['filed_date'])[:10]>str(row.as_of)[:10] or str(current['filed_date'])[:10]>str(row.target_filed)[:10]:
                raise ValueError('Segment chronology mismatch')
            result.loc[idx,'target_q2_usd']-=float(current['phosphate_q2_usd'])
            result.loc[idx,'prior_q2_usd']-=float(prior['phosphate_q2_usd'])
    if (result[['target_q2_usd','prior_q2_usd','price_ratio']]<=0).any().any():
        raise ValueError('Nonpositive adjusted revenue or price')
    return result


def intercept_comparison(data, segments, older_prior_years, saved_predictions=None):
    adjusted=adjusted_training_inputs(data,segments)
    prior_set=set(int(y) for y in older_prior_years)
    if not prior_set.issubset(set(range(2010,2016))): raise ValueError('Unexpected older cohort')
    result=[]; omissions=[]
    for cohort in ('full','prior'):
        rows=adjusted if cohort=='full' else adjusted[(adjusted.year>=2016)|adjusted.year.isin(prior_set)]
        for year in range(2016,2027):
            test=rows[rows.year.eq(year)]
            if len(test)!=1:
                omissions.append({'cohort':cohort,'year':year,'reason':'test year unavailable'}); continue
            row=next(test.itertuples()); cutoff=f'{year}-05-31'
            train=rows[(rows.year<year)&(rows.target_filed.astype(str).str[:10]<=cutoff)].sort_values('year')
            if len(train)<6:
                omissions.append({'cohort':cohort,'year':year,'reason':'fewer than six eligible training years'}); continue
            try:
                preds={f'alpha_{w:g}':correction_predict(train,row,w) for w in (0.0,0.5,1.0)}
            except ValueError as exc:
                omissions.append({'cohort':cohort,'year':year,'reason':str(exc)}); continue
            actual=float(row.target_q2_usd)
            if year in range(2022,2027) and saved_predictions is not None:
                ref=saved_predictions[(saved_predictions.year.eq(year)) & (saved_predictions.variant.eq(f'{cohort}_adjusted'))]
                if len(ref)!=1 or not np.isclose(preds['alpha_0'][0],float(ref.prediction_usd.iloc[0]),rtol=1e-10,atol=1e-3):
                    raise AssertionError('Existing adjusted prediction not reproduced')
            preds['price_only']=(float(row.prior_q2_usd*row.price_ratio),0.0,0.0)
            baseline=float(row.prior_q2_usd*row.price_ratio)
            preds['price_correction_only']=(baseline*math.exp(preds['alpha_0.5'][1]),preds['alpha_0.5'][1],0.0)
            for variant,(prediction,intercept,beta) in preds.items():
                result.append({'cohort':cohort,'year':year,'period':'2016-2021' if year<=2021 else '2022-2026',
                    'variant':variant,'training_years':len(train),'training_year_list':','.join(map(str,train.year.astype(int))),
                    'actual_usd':actual,'prediction_usd':prediction,'signed_error_usd':prediction-actual,
                    'absolute_error_usd':abs(prediction-actual),'intercept_log':intercept,'weather_coefficient':beta})
    pred=pd.DataFrame(result)
    if pred.empty: raise ValueError('No valid comparison years')
    scores=pred.groupby(['cohort','period','variant'],as_index=False).agg(years=('year','size'),mae_musd=('absolute_error_usd',lambda x:x.mean()/1e6))
    for key,group in pred.groupby(['cohort','year']):
        if len(group)!=5 or group.training_year_list.nunique()!=1 or group.actual_usd.nunique()!=1:
            raise AssertionError(f'Unmatched comparison: {key}')
    return {'predictions':pred,'scores':scores,'omitted':pd.DataFrame(omissions,columns=['cohort','year','reason'])}


def save_intercept_comparison(result, output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    for k in ('predictions','scores','omitted'):
        result[k].to_csv(out/(k+'_private.csv'),index=False)
    report=('Exploratory expanding-window comparison. Alpha_0 is the original zero-intercept fit; '
        'alpha_0.5 adds half the earlier training mean log ratio of revenue to the price baseline; '
        'alpha_1 adds its full mean, as a sensitivity. The weather slope uses the same centered '
        'training features and ridge penalty 1 in all fits. Price_only and price_correction_only '
        'are controls. The same '
        'training and test years are used within each comparison. Prior and full history show '
        'training-cohort sensitivity. Full-history 2016-2021 and both cohorts in 2022-2026 are '
        'reported, with any omitted years. Historical data and model choices were already '
        'examined, so these are not new holdouts. Historical weather publication and vintage '
        'remain unverified; older prices are reconstructed and phosphate-adjusted revenue is '
        'approximate. A correction is not automatically adopted based on this small sample. '
        'The frozen 2027 protocol remains unchanged.\n')
    (out/'REPORT.txt').write_text(report,encoding='utf-8')
    return report
