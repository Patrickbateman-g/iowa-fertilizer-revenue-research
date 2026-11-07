from pathlib import Path
import math
import numpy as np
import pandas as pd


def diagnose_phosphate_forecast_2025():
    orig=FIELDWORK_WINDOW_RESULTS['six']['data'].copy()
    old=LONG_HISTORY_RESULTS['history'].copy()
    cols=list(old.columns)
    data=pd.concat([old,orig[cols]],ignore_index=True).sort_values('year').reset_index(drop=True)
    if data.year.duplicated().any(): raise ValueError('Duplicate historical year')
    records={int(r['year']):r for r in PHOSPHATE_SEGMENT}
    if set(records)!=set(range(2009,2016)): raise ValueError('Incomplete phosphate sources')
    adjusted=data.copy()
    for idx, row in adjusted[adjusted.year.between(2010,2015)].iterrows():
        y=int(row.year); current=records[y]; prior=records[y-1]
        if lh_date(prior['filed_date'])>lh_date(str(row.as_of)[:10]) or lh_date(current['filed_date'])>lh_date(str(row.target_filed)[:10]):
            raise ValueError('Segment data was not available at stated cutoff')
        target=float(row.target_q2_usd)-float(current['phosphate_q2_usd'])
        base=float(row.prior_q2_usd)-float(prior['phosphate_q2_usd'])
        if min(target,base)<=0: raise ValueError('Invalid segment adjustment')
        adjusted.loc[idx,['target_q2_usd','prior_q2_usd']]=[target,base]
    rowmatches=orig[orig.year.eq(2025)]
    if len(rowmatches)!=1: raise ValueError('2025 test year missing')
    row=next(rowmatches.itertuples()); cutoff=str(row.as_of)[:10]
    if cutoff!='2025-05-31' or str(row.target_filed)[:10] <= cutoff: raise ValueError('Invalid cutoff')

    def train_from(frame):
        train=frame[(frame.year<2025)&(frame.target_filed.astype(str).str[:10]<=cutoff)].copy()
        if len(train)<6 or (train.year>=2025).any(): raise ValueError('Invalid training cohort')
        return train
    def fit(train):
        x=train.forecast_mean_p12_pct.to_numpy(float)
        target=np.log(train.target_q2_usd.to_numpy(float)/(train.prior_q2_usd.to_numpy(float)*train.price_ratio.to_numpy(float)))
        mean=float(np.mean(x)); sd=float(np.std(x,ddof=0))
        if not np.isfinite(x).all() or not np.isfinite(target).all() or not np.isfinite(sd) or sd<=1e-12:
            raise ValueError('Invalid features or targets')
        z=(x-mean)/sd; beta=float((z@target)/(z@z+1.0))
        ztest=(float(row.forecast_mean_p12_pct)-mean)/sd; term=beta*ztest
        baseline=float(row.prior_q2_usd*row.price_ratio)
        prediction=baseline*math.exp(term)
        if not np.isclose(prediction,lh_fit(train,row,['forecast_mean_p12_pct']),rtol=1e-10,atol=1e-3):
            raise AssertionError('Original fitting does not reproduce')
        return {'mu':mean,'sd':sd,'minimum':float(x.min()),'maximum':float(x.max()),'z':ztest,
                'beta':beta,'log_adjustment':term,'multiplier':math.exp(term),'baseline':baseline,
                'prediction':prediction,'log_targets':target,'features':x}
    train_unadjusted=train_from(data); train_adjusted=train_from(adjusted)
    original=fit(train_unadjusted); adj=fit(train_adjusted)
    prev=PHOSPHATE_FORECAST_RESULTS['predictions']
    def check(label,result):
        r=prev[(prev.year==2025)&(prev.variant==label)]
        if len(r)!=1 or not np.isclose(result['prediction'],float(r.prediction_usd.iloc[0]),rtol=1e-10,atol=1e-3):
            raise AssertionError('Saved 2025 result is not reproduced')
    check('prior_unadjusted',original);check('prior_adjusted',adj)
    actual=float(row.target_q2_usd)
    if not np.isclose(actual,float(prev[prev.year.eq(2025)].actual_usd.iloc[0])): raise AssertionError('Target mismatch')
    profiles=[]
    for label,train,f in [('unadjusted',train_unadjusted,original),('phosphate_adjusted',train_adjusted,adj)]:
        profiles.append({'variant':label,'training_years':len(train),'forecast_p12_pct':float(row.forecast_mean_p12_pct),
                         'training_mean_pct':f['mu'],'training_sd_pct':f['sd'],'training_min_pct':f['minimum'],
                         'training_max_pct':f['maximum'],'forecast_z':f['z'],'coefficient':f['beta'],
                         'forecast_log_adjustment':f['log_adjustment'],'forecast_multiplier':f['multiplier'],
                         'baseline_usd':f['baseline'],'prediction_usd':f['prediction'],'actual_usd':actual,
                         'signed_error_usd':f['prediction']-actual,'absolute_error_usd':abs(f['prediction']-actual),
                         'training_year_list':','.join(map(str,train.year.astype(int)))})
    profiles=pd.DataFrame(profiles)
    train_adjusted['adjusted_log_residual']=adj['log_targets']
    influence=[]
    for y in train_adjusted.year.astype(int):
        loo=train_adjusted[train_adjusted.year.ne(y)]
        shifted=fit(loo)['prediction']
        influence.append({'removed_year':y,'prediction_without_year_usd':shifted,
                          'change_if_removed_usd':shifted-adj['prediction']})
    influence=pd.DataFrame(influence)
    out=Path('/kaggle/working/iowa_phosphate_forecast_2025');out.mkdir(parents=True,exist_ok=True)
    profiles.to_csv(out/'diagnostic_private.csv',index=False)
    influence.to_csv(out/'leave_one_training_year_out_private.csv',index=False)
    train_adjusted.to_csv(out/'training_rows_private.csv',index=False)
    report=f'''# Diagnosing the 2025 phosphate-adjusted forecast

The fixed price-based prediction was ${adj['baseline']/1e6:,.1f}m, actual sales ${actual/1e6:,.1f}m. The precipitation probability forecast was {float(row.forecast_mean_p12_pct):.2f}%, versus earlier training mean {adj['mu']:.2f}% and range {adj['minimum']:.2f}–{adj['maximum']:.2f}% (z={adj['z']:.2f}). The fitted coefficient was {adj['beta']:+.4f}, and it changed the price-based estimate by a factor of {adj['multiplier']:.4f}, producing ${adj['prediction']/1e6:,.1f}m and an absolute error of ${abs(adj['prediction']-actual)/1e6:,.1f}m. The unadjusted model prediction was ${original['prediction']/1e6:,.1f}m. Earlier training data are restricted to targets filed by May 31, 2025; the original old and adjusted predictions must both reproduce exactly.

Leave-one-training-year-out results are retrospective influence diagnostics, not independent tests or causation. Prediction movement reflects refitting the coefficient and scaling; fitted relationships may be unstable in this small sample. The forecast is the mean probability of measurable precipitation, not total rain or purchases. CF revenue and the price baseline are company-wide/national rather than Iowa-specific. 2025 outcomes were already inspected, so no evaluated approach is an untouched test. Historical source revisions and forecast-vintage uncertainty continue to apply. The frozen 2027 protocol is unchanged.
'''
    (out/'REPORT.md').write_text(report,encoding='utf8')
    print(report)
    display(profiles.assign(baseline_m=lambda d:d.baseline_usd/1e6,prediction_m=lambda d:d.prediction_usd/1e6,actual_m=lambda d:d.actual_usd/1e6,signed_error_m=lambda d:d.signed_error_usd/1e6,absolute_error_m=lambda d:d.absolute_error_usd/1e6).drop(columns=['baseline_usd','prediction_usd','actual_usd','signed_error_usd','absolute_error_usd']).round(4))
    display(influence.assign(prediction_without_year_m=lambda d:d.prediction_without_year_usd/1e6,change_if_removed_m=lambda d:d.change_if_removed_usd/1e6).drop(columns=['prediction_without_year_usd','change_if_removed_usd']).sort_values('change_if_removed_m').round(2))
    display(train_adjusted[['year','forecast_mean_p12_pct','adjusted_log_residual']].round(4))
    return {'profiles':profiles,'influence':influence,'train':train_adjusted,'report':report}

PHOSPHATE_FORECAST_2025_DIAGNOSIS=diagnose_phosphate_forecast_2025()


from pathlib import Path
from urllib.parse import urlsplit
import datetime as dt
import math
import numpy as np
import pandas as pd


def forecast_phosphate_2025_tests():
    historical = LONG_HISTORY_RESULTS['history'].copy()
    original = FIELDWORK_WINDOW_RESULTS['six']['data'].copy()
    columns = list(historical.columns)
    data = pd.concat([historical, original[columns]], ignore_index=True).sort_values('year')
    if data.year.duplicated().any() or set(historical.year.astype(int)) != {2012, 2014, 2015}:
        raise ValueError('Unexpected prior-history training cohort')
    segments = {}
    for entry in PHOSPHATE_SEGMENT:
        if not isinstance(entry, dict) or not {'year','phosphate_q2_usd','filed_date','source_url','basis'} <= set(entry):
            raise ValueError('Incomplete historical segment record')
        year = entry['year']
        value = entry['phosphate_q2_usd']
        date = lh_date(entry['filed_date'])
        url = urlsplit(entry['source_url'])
        if (isinstance(year, bool) or not isinstance(year, int) or year in segments or year not in range(2009,2016)
            or isinstance(value,bool) or not isinstance(value,(float,int)) or not np.isfinite(value) or value < 0
            or not dt.date(year,6,30) < date <= dt.date(year+1,5,31)
            or url.scheme != 'https' or url.hostname != 'www.sec.gov'
            or not url.path.startswith('/Archives/edgar/data/1324404/') or not str(entry['basis']).strip()):
            raise ValueError('Invalid historical segment record')
        segments[year] = entry
    if set(segments) != set(range(2009,2016)) or segments[2015]['phosphate_q2_usd'] != 0:
        raise ValueError('Incomplete historical segment series')

    adjusted = data.copy()
    for idx, row in adjusted[adjusted.year.between(2010,2015)].iterrows():
        year = int(row.year)
        prior, current = segments[year-1], segments[year]
        if lh_date(prior['filed_date']) > lh_date(str(row.as_of)[:10]):
            raise ValueError('Prior segment unavailable by that year cutoff')
        if lh_date(current['filed_date']) > lh_date(str(row.target_filed)[:10]):
            raise ValueError('Segment source after originally filed target')
        target = float(row.target_q2_usd) - float(current['phosphate_q2_usd'])
        baseline = float(row.prior_q2_usd) - float(prior['phosphate_q2_usd'])
        if min(target,baseline) <= 0: raise ValueError('Invalid adjusted revenue')
        adjusted.loc[idx,['target_q2_usd','prior_q2_usd']] = [target, baseline]

    volumes = {}
    q1rows = EXTRA_POINTS.get('q1_volume') if isinstance(EXTRA_POINTS,dict) else None
    if not isinstance(q1rows,list): raise ValueError('Missing Q1 volume records')
    for entry in q1rows:
        required={'year','current_q1_tons','previous_q1_tons','current_filed','previous_filed',
                  'current_accession','previous_accession','units','current_source','previous_source'}
        if not isinstance(entry,dict) or not required <= set(entry): raise ValueError('Incomplete Q1 volume record')
        y=entry['year']
        if isinstance(y,bool) or not isinstance(y,int) or y in volumes: raise ValueError('Duplicate or invalid volume year')
        for key in ('current_q1_tons','previous_q1_tons'):
            v=entry[key]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0:
                raise ValueError('Invalid shipment volume')
        if entry['units']!='thousand product tons': raise ValueError('Wrong shipment units')
        if not (dt.date(y,3,31)<lh_date(entry['current_filed'])<=dt.date(y,5,31)
                and dt.date(y-1,3,31)<lh_date(entry['previous_filed'])<=dt.date(y-1,5,31)):
            raise ValueError('Q1 shipment disclosure was unavailable')
        for kind in ('current','previous'):
            u=urlsplit(entry[kind+'_source'])
            if u.scheme!='https' or u.hostname!='www.sec.gov' or not u.path.startswith('/Archives/edgar/data/1324404/') or not str(entry[kind+'_accession']).strip():
                raise ValueError('Invalid Q1 volume provenance')
        volumes[y]=entry
    if not set(range(2022,2027)).issubset(volumes): raise ValueError('Incomplete Q1 volume evaluation cohort')

    predictions=[]; fold_diagnostics=[]
    for year in range(2022,2027):
        matching=original[original.year.eq(year)]
        if len(matching)!=1: raise ValueError('Missing or duplicate test row')
        test=next(matching.itertuples()); cutoff=str(test.as_of)[:10]
        if cutoff!=f'{year}-05-31' or str(test.target_filed)[:10] <= cutoff:
            raise ValueError('Invalid test cutoff')
        train=adjusted[(adjusted.year < year) & (adjusted.target_filed.astype(str).str[:10] <= cutoff)].copy()
        if len(train)<6 or any(lh_date(segments[int(y)]['filed_date']) > lh_date(cutoff)
                              for y in train.year if int(y) in range(2010,2016)):
            raise ValueError('Insufficient eligible training data')
        x=train.forecast_mean_p12_pct.to_numpy(float)
        if not np.isfinite(x).all(): raise ValueError('Invalid training forecast')
        mean=float(x.mean()); sd=float(x.std(ddof=0))
        if sd<=1e-12 or not np.isfinite(sd): raise ValueError('No forecast variation')
        z=(x-mean)/sd
        log_response=np.log(train.target_q2_usd.to_numpy(float)/(train.prior_q2_usd.to_numpy(float)*train.price_ratio.to_numpy(float)))
        beta=float((z@log_response)/(z@z+1.0))
        current=float(test.forecast_mean_p12_pct)
        baseline=float(test.prior_q2_usd*test.price_ratio)
        ztest=(current-mean)/sd
        full=baseline*math.exp(beta*ztest)
        expected=PHOSPHATE_FORECAST_RESULTS['predictions']
        expected=expected[(expected.year==year)&(expected.variant=='prior_adjusted')]
        if len(expected)!=1 or not np.isclose(full,float(expected.prediction_usd.iloc[0]),rtol=1e-10,atol=1e-3):
            raise AssertionError('Best phosphate-adjusted model was not reproduced')
        if not np.isclose(full,lh_fit(train,test,['forecast_mean_p12_pct']),rtol=1e-10,atol=1e-3):
            raise AssertionError('Single-feature fit disagrees with original model')
        low,high=float(x.min()),float(x.max())
        capped=float(np.clip(current,low,high))
        predictions_by_candidate={
            'adjusted_full':full,
            'adjusted_half_log_correction':baseline*math.exp(beta*ztest*0.5),
            'adjusted_training_range_clip':baseline*math.exp(beta*(capped-mean)/sd),
            'adjusted_sqrt_q1_volume':full*math.sqrt(float(volumes[year]['current_q1_tons'])/float(volumes[year]['previous_q1_tons'])),
            'price_only':baseline,
        }
        prior=PHOSPHATE_FORECAST_RESULTS['predictions']
        prior=prior[(prior.year==year)&(prior.variant=='prior_unadjusted')]
        if len(prior)!=1: raise ValueError('Missing unadjusted forecast')
        predictions_by_candidate['unadjusted_forecast']=float(prior.prediction_usd.iloc[0])
        actual=float(test.target_q2_usd)
        for label,pred in predictions_by_candidate.items():
            predictions.append({'year':year,'candidate':label,'train_years':len(train),'actual_usd':actual,
                                'prediction_usd':pred,'signed_error_usd':pred-actual,'absolute_error_usd':abs(pred-actual)})
        fold_diagnostics.append({'year':year,'train_years':len(train),'feature_pct':current,'training_mean_pct':mean,
            'training_sd_pct':sd,'training_min_pct':low,'training_max_pct':high,'standardized_value':ztest,
            'ridge_coefficient':beta,'log_adjustment':beta*ztest,'price_baseline_usd':baseline})
    pred=pd.DataFrame(predictions)
    diag=pd.DataFrame(fold_diagnostics)
    score=[]
    for omitted in (None,2025):
        group=pred if omitted is None else pred[pred.year.ne(omitted)]
        reference=group[group.candidate.eq('adjusted_full')].set_index('year').absolute_error_usd
        for name,part in group.groupby('candidate'):
            if len(part)!=len(reference): raise ValueError('Incomplete matched folds')
            score.append({'exclude_year':str(omitted) if omitted else 'none','candidate':name,
                          'years':len(part),'mae_usd':float(part.absolute_error_usd.mean()),
                          'wins_vs_adjusted_full':int((part.set_index('year').absolute_error_usd < reference).sum())})
    scores=pd.DataFrame(score)
    annual=pred.pivot(index='year',columns='candidate',values='absolute_error_usd')
    out=Path('/kaggle/working/iowa_phosphate_forecast_2025_sensitivities')
    out.mkdir(parents=True,exist_ok=True)
    for name,frame in [('predictions_private.csv',pred),('diagnostics_private.csv',diag),('scores_private.csv',scores),('annual_errors_private.csv',annual.reset_index())]:
        frame.to_csv(out/name,index=False)
    report='''# Phosphate-adjusted 2025 forecast: exploratory sensitivities

The one-feature forecast model and these fixed alternatives use identical historical May 31 prediction rows and phosphate-adjusted training rows from the prior study. Full-model predictions must reproduce that study exactly. Feature scaling, the coefficient, and the clipping bounds use only earlier training data for each simulated cutoff; test-year targets are used only for evaluation. The forecast sensitivities halve the fitted log correction or clamp the forecast feature to the training range. A third fixed candidate scales the adjusted forecast by the square root of the ratio of current to previous company Q1 product shipments; Q1 disclosures are checked against May 31, and shipment volumes represent company-wide sales rather than Iowa farmer orders. The price-only and original unadjusted forecasts are matched references.

All choices were considered after the years were inspected; no result here is an independent future test. The 2025-omitted summary is descriptive, not a fresh holdout. The full prior forecast fit lacks enough eligible earlier training rows for a like-for-like 2016–2021 comparison, so these tests do not establish performance in the earlier period. Older forecast archives, fieldwork-selected cohorts, and reconstructed price/segment history retain their prior comparability and vintage limitations. Weather probability is a local proxy, while CF revenue is company-wide. No 2027 protocol or prediction is changed.
'''
    (out/'REPORT.md').write_text(report,encoding='utf8')
    print(report)
    display(scores.assign(mae_musd=lambda d:d.mae_usd/1e6).drop(columns='mae_usd').round(2))
    display((annual/1e6).round(2))
    display(diag.round(4))
    display(pred[pred.year.eq(2025)].assign(actual_m=lambda d:d.actual_usd/1e6,prediction_m=lambda d:d.prediction_usd/1e6,signed_error_m=lambda d:d.signed_error_usd/1e6)[['candidate','actual_m','prediction_m','signed_error_m']].round(2))
    return {'predictions':pred,'diagnostics':diag,'scores':scores,'annual':annual,'report':report}

PHOSPHATE_FORECAST_2025_TESTS = forecast_phosphate_2025_tests()
