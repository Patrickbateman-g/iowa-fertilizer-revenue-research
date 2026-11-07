from itertools import permutations
from pathlib import Path
import math
import numpy as np
import pandas as pd


def diagnose_extra_training_years():
    cols=list(LONG_HISTORY_RESULTS['history'].columns)
    old_years=LONG_HISTORY_RESULTS['history'].copy()
    extra=MISSING_FIELDWORK_RESULTS['additional_years'].copy()
    orig=FIELDWORK_WINDOW_RESULTS['six']['data'].copy()
    assert set(extra.year.astype(int)) == {2010,2011,2013}
    old=pd.concat([old_years,orig[cols]],ignore_index=True).sort_values('year')
    if old.year.duplicated().any(): raise ValueError('Duplicate year')
    all_data=pd.concat([old,extra[cols]],ignore_index=True).sort_values('year')
    if all_data.year.duplicated().any(): raise ValueError('Duplicate year')
    fields=['forecast_mean_p12_pct','fieldwork_mean_days']
    predictions=MISSING_FIELDWORK_RESULTS['predictions']
    effects=[]; feature_rows=[]; year_rows=[]; training_rows=[]

    def evaluate(train,row):
        x=train[fields].to_numpy(dtype=float)
        mu=x.mean(axis=0); sd=x.std(axis=0,ddof=0)
        if not np.isfinite(x).all() or not np.isfinite(sd).all() or (sd<=1e-12).any(): raise ValueError('Invalid training features')
        z=(x-mu)/sd
        target=np.log(train.target_q2_usd.to_numpy(float)/(train.prior_q2_usd.to_numpy(float)*train.price_ratio.to_numpy(float)))
        beta=np.linalg.solve(z.T@z+np.eye(2),z.T@target)
        point=np.array([getattr(row,c) for c in fields],dtype=float)
        test_z=(point-mu)/sd; terms=beta*test_z
        baseline=float(row.prior_q2_usd*row.price_ratio)
        prediction=baseline*math.exp(float(terms.sum()))
        if not np.isclose(prediction,lh_fit(train,row,fields),rtol=1e-12,atol=1e-3): raise AssertionError('Reconstruction mismatch')
        return dict(mu=mu,sd=sd,test_z=test_z,beta=beta,terms=terms,baseline=baseline,prediction=prediction,
                    correlation=float(np.corrcoef(x.T)[0,1]),log_target_mean=float(np.mean(target)))

    for year in (2023,2026):
        row=next(orig[orig.year.eq(year)].itertuples()); cutoff=str(row.as_of)[:10]
        def allowed(frame): return frame[(frame.year<year)&(frame.target_filed.astype(str).str[:10]<=cutoff)].copy()
        base=allowed(old); added=allowed(extra); full=allowed(all_data)
        if set(added.year.astype(int)) != {2010,2011,2013}: raise ValueError('Unexpected additions')
        old_fit=evaluate(base,row); new_fit=evaluate(full,row)
        for label,fit in [('prior',old_fit),('new',new_fit)]:
            ref=predictions[(predictions.year==year)&(predictions.model=='price_forecast_fieldwork')&(predictions.training==label)]
            if len(ref)!=1 or not np.isclose(fit['prediction'],float(ref.prediction_usd.iloc[0]),rtol=1e-10,atol=1e-3): raise AssertionError('Prior results differ')
            actual=float(row.target_q2_usd)
            effects.append(dict(year=year,training=label,training_years=len(base) if label=='prior' else len(full),
                actual_usd=actual,baseline_usd=fit['baseline'],prediction_usd=fit['prediction'],
                signed_error_usd=fit['prediction']-actual,absolute_error_usd=abs(fit['prediction']-actual),
                adjustment_pct=(fit['prediction']/fit['baseline']-1)*100,forecast_fieldwork_train_correlation=fit['correlation']))
            for j,feature in enumerate(fields):
                feature_rows.append(dict(year=year,training=label,feature=feature,value=float(getattr(row,feature)),
                    training_mean=fit['mu'][j],training_std=fit['sd'][j],test_z=fit['test_z'][j],
                    ridge_coefficient=fit['beta'][j],log_contribution=fit['terms'][j]))
        total={y:0. for y in (2010,2011,2013)}
        for order in permutations(total):
            selected=[]; before=old_fit['prediction']
            for addition in order:
                selected.append(addition)
                train=pd.concat([base,added[added.year.isin(selected)]],ignore_index=True)
                after=evaluate(train,row)['prediction']
                total[addition]+=(after-before)/6.
                before=after
            if not np.isclose(before,new_fit['prediction'],rtol=1e-10,atol=1e-3): raise AssertionError('Order endpoint mismatch')
        if not np.isclose(sum(total.values()),new_fit['prediction']-old_fit['prediction'],rtol=1e-10,atol=1e-3): raise AssertionError('Contributions do not sum')
        for y,value in total.items(): year_rows.append(dict(year=year,added_year=y,average_prediction_change_usd=value))
        for r in added.itertuples():
            training_rows.append(dict(prediction_year=year,training_year=int(r.year),
                forecast_mean_p12_pct=r.forecast_mean_p12_pct,fieldwork_mean_days=r.fieldwork_mean_days,
                log_revenue_over_price_baseline=math.log(r.target_q2_usd/(r.prior_q2_usd*r.price_ratio))))
    effects=pd.DataFrame(effects); feats=pd.DataFrame(feature_rows); years=pd.DataFrame(year_rows); added_profiles=pd.DataFrame(training_rows)
    out=Path('/kaggle/working/iowa_older_year_diagnostic'); out.mkdir(parents=True,exist_ok=True)
    for name,df in [('predictions_private.csv',effects),('feature_terms_private.csv',feats),('year_contributions_private.csv',years),('added_year_profiles_private.csv',added_profiles)]: df.to_csv(out/name,index=False)
    report='''# Why adding the older years changed predictions

This decomposition exactly reconstructs the historical ridge model. Adding old rows changes the fitted coefficients and training means and standard deviations; the same fixed test-year values interact with these refitted quantities. Feature terms are on a log scale, and their change is a mechanical decomposition rather than a causal statement. The contribution of each added year is the average prediction change over all six orders of adding the three years; it is allocation-dependent and not an independent causal effect. Training labels are filtered to those filed by each May 31 cutoff, but several older fieldwork observations come from a current archive with unknown historical revisions. All outcomes through 2026 have been inspected and these analyses are exploratory. CF net sales cover more than Iowa corn. Frozen 2027 protocol unchanged.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    print(report)
    display(effects.assign(actual_m=lambda x:x.actual_usd/1e6,baseline_m=lambda x:x.baseline_usd/1e6,prediction_m=lambda x:x.prediction_usd/1e6,signed_error_m=lambda x:x.signed_error_usd/1e6,absolute_error_m=lambda x:x.absolute_error_usd/1e6).drop(columns=['actual_usd','baseline_usd','prediction_usd','signed_error_usd','absolute_error_usd']).round(3))
    display(feats.round(4))
    display(years.assign(average_change_m=lambda x:x.average_prediction_change_usd/1e6).drop(columns='average_prediction_change_usd').round(2))
    display(added_profiles.drop(columns='prediction_year').drop_duplicates().round(4))
    return {'predictions':effects,'features':feats,'year_contributions':years,'added_profiles':added_profiles,'report':report}

OLDER_YEAR_DIAGNOSTIC=diagnose_extra_training_years()


from pathlib import Path
import numpy as np
import pandas as pd


def recency_weighted_fit(train, row, features, old_weight=0.5):
    weights=np.where(train.year.isin([2010, 2011, 2013]),old_weight,1.0)
    weights=weights/weights.mean()
    x=train[features].to_numpy(float)
    mu=np.average(x,axis=0,weights=weights)
    sd=np.sqrt(np.average((x-mu)**2,axis=0,weights=weights))
    if not np.isfinite(x).all() or not np.isfinite(sd).all() or (sd<=1e-12).any():
        raise ValueError('Invalid training features')
    z=(x-mu)/sd
    log_resid=np.log(train.target_q2_usd.to_numpy(float)/(train.prior_q2_usd.to_numpy(float)*train.price_ratio.to_numpy(float)))
    beta=np.linalg.solve(z.T@(weights[:,None]*z)+np.eye(len(features)),z.T@(weights*log_resid))
    test=(np.asarray([getattr(row,c) for c in features],float)-mu)/sd
    return float(row.prior_q2_usd*row.price_ratio)*np.exp(float(test@beta))


def recency_sensitivity(recovered, tag):
    if tag not in ('main','2013_table'): raise ValueError('Invalid tag')
    orig=FIELDWORK_WINDOW_RESULTS['six']['data'].copy()
    old_history=LONG_HISTORY_RESULTS['history'].copy()
    added=recovered['additional_years'].copy()
    columns=list(old_history.columns)
    old=pd.concat([old_history,orig[columns]],ignore_index=True).sort_values('year')
    all_years=pd.concat([old,added],ignore_index=True).sort_values('year')
    if old.year.duplicated().any() or all_years.year.duplicated().any(): raise ValueError('Duplicate years')
    if set(added.year)!={2010,2011,2013}: raise ValueError('Missing expected years')
    features=['forecast_mean_p12_pct','fieldwork_mean_days']
    saved=recovered['predictions']
    records=[]
    for y in range(2022,2027):
        row=next(orig[orig.year.eq(y)].itertuples()); cutoff=str(row.as_of)[:10]
        if cutoff != f'{y}-05-31' or str(row.target_filed)[:10]<=cutoff: raise ValueError('Cutoff invalid')
        oldtrain=old[(old.year<y)&(old.target_filed.astype(str).str[:10]<=cutoff)]
        train=all_years[(all_years.year<y)&(all_years.target_filed.astype(str).str[:10]<=cutoff)]
        oldpred=lh_fit(oldtrain,row,features)
        full=lh_fit(train,row,features)
        price=lh_fit(train,row,[])
        forecast=lh_fit(train,row,['forecast_mean_p12_pct'])
        oldforecast=lh_fit(oldtrain,row,['forecast_mean_p12_pct'])
        halfweight=recency_weighted_fit(train,row,features)
        halfadjust=np.sqrt(full*price)
        fullcheck=recency_weighted_fit(train,row,features,old_weight=1.0)
        if not np.isclose(full,fullcheck,rtol=1e-12,atol=1e-3): raise AssertionError('Equal weights fail to reproduce original fit')
        for label,estimate in [('prior_long',oldpred),('all_full',full),('all_half_weight_added',halfweight),
                               ('all_half_adjust',halfadjust),('all_forecast_only',forecast),('prior_forecast_only',oldforecast),('price_only',price)]:
            if label in ('prior_long','all_full'):
                previous=saved[(saved.year==y)&(saved.model=='price_forecast_fieldwork')&(saved.training==('prior' if label=='prior_long' else 'new'))]
                if len(previous)!=1 or not np.isclose(estimate,float(previous.prediction_usd.iloc[0]),rtol=1e-10,atol=1e-3):
                    raise AssertionError('Saved prediction mismatch')
            actual=float(row.target_q2_usd)
            records.append({'year':y,'candidate':label,'train_years':len(oldtrain) if label in ('prior_long','prior_forecast_only') else len(train),
                            'actual_usd':actual,'prediction_usd':estimate,'signed_error_usd':estimate-actual,
                            'absolute_error_usd':abs(estimate-actual)})
    pred=pd.DataFrame(records)
    scores=pred.groupby('candidate',as_index=False).agg(years=('year','size'),mae_usd=('absolute_error_usd','mean'))
    base=pred[pred.candidate.eq('all_full')].set_index('year').absolute_error_usd
    scores['wins_vs_all_full']=[int((g.set_index('year').absolute_error_usd < base).sum()) for _,g in pred.groupby('candidate')]
    output=Path('/kaggle/working/iowa_recency_sensitivity')/tag
    output.mkdir(parents=True,exist_ok=True)
    pred.to_csv(output/'predictions_private.csv',index=False)
    scores.to_csv(output/'scores_private.csv',index=False)
    report='''The two fixed variants half-weight the three reconstructed years or halve the complete fitted log correction. All years use earlier filed labels and identical test rows. Scaling and coefficients are fit only on the training fold; weights are normalized to mean one before applying the original ridge penalty. Historical sources may be revised. Both choices are post-hoc exploratory sensitivities based on inspected years, not evidence of future improvement. The 2027 protocol remains unchanged.'''
    (output/'REPORT.md').write_text(report,encoding='utf8')
    print('2013 source:',tag,report)
    display(scores.assign(mae_musd=lambda d:d.mae_usd/1e6).drop(columns='mae_usd').round(2))
    display((pred.pivot(index='year',columns='candidate',values='absolute_error_usd')/1e6).round(1))
    display((pred[pred.year.isin([2023,2026])].pivot(index='year',columns='candidate',values='signed_error_usd')/1e6).round(1))
    return {'predictions':pred,'scores':scores}

RECENCY_SENSITIVITY=recency_sensitivity(MISSING_FIELDWORK_RESULTS,'main')
RECENCY_SENSITIVITY_2013_TABLE=recency_sensitivity(MISSING_FIELDWORK_SENSITIVITY,'2013_table')


from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
import pandas as pd


def phosphate_sensitivity(segments, result, output_subdir):
    if output_subdir not in ('iowa_phosphate_sensitivity','iowa_phosphate_alt_fieldwork'):
        raise ValueError('Invalid output directory')
    if not isinstance(segments, list): raise ValueError('Expected segment records')
    records = {}
    for r in segments:
        needed={'year','phosphate_q2_usd','filed_date','source_url','basis'}
        if not isinstance(r,dict) or not needed.issubset(r): raise ValueError('Incomplete segment provenance')
        y=r['year']; value=r['phosphate_q2_usd']; date=lh_date(r['filed_date']); url=urlsplit(r['source_url'])
        if isinstance(y,bool) or not isinstance(y,int) or y not in range(2009,2016) or y in records:
            raise ValueError('Unexpected or duplicate segment year')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0:
            raise ValueError('Invalid phosphate sales')
        if not dt.date(y,6,30) < date <= dt.date(y+1,5,31): raise ValueError('Invalid filing date')
        if url.scheme!='https' or url.hostname!='www.sec.gov' or not url.path.startswith('/Archives/edgar/data/1324404/'):
            raise ValueError('Expected CF SEC filing')
        if not isinstance(r['basis'],str) or not r['basis'].strip(): raise ValueError('Missing segment basis')
        records[y]=dict(r)
    if set(records) != set(range(2009,2016)): raise ValueError('Missing segment records; cannot safely subtract')
    if records[2015]['phosphate_q2_usd'] != 0: raise ValueError('Expected no disposed phosphate segment in 2015')
    history=LONG_HISTORY_RESULTS['history'].copy()
    additional=result['additional_years'].copy()
    original=FIELDWORK_WINDOW_RESULTS['six']['data'].copy()
    cols=list(history.columns)
    full=pd.concat([history,additional,original[cols]],ignore_index=True).sort_values('year')
    if full.year.duplicated().any() or not set(range(2010,2016)).issubset(set(full.year)):
        raise ValueError('Expected complete, unique historical cohort')
    adjusted=full.copy()
    changes=[]
    for y in range(2010,2016):
        i=adjusted.index[adjusted.year.eq(y)]
        if len(i)!=1: raise ValueError('Missing historical year')
        i=i[0]; prior, current=records[y-1],records[y]
        row=adjusted.loc[i]; cutoff=lh_date(str(row.as_of)[:10]); target_filed=lh_date(str(row.target_filed)[:10])
        if lh_date(prior['filed_date'])>cutoff: raise ValueError('Prior segment not available at historical cutoff')
        if lh_date(current['filed_date'])>target_filed: raise ValueError('Segment source later than recorded target filing')
        new_target=float(row.target_q2_usd)-float(current['phosphate_q2_usd'])
        new_prior=float(row.prior_q2_usd)-float(prior['phosphate_q2_usd'])
        if min(new_target,new_prior)<=0: raise ValueError('Segment exceeds total sales')
        adjusted.loc[i,['target_q2_usd','prior_q2_usd']]=[new_target,new_prior]
        changes.append({'year':y,'target_segment_filed':current['filed_date'],'prior_segment_filed':prior['filed_date'],
                        'target_phosphate_usd':current['phosphate_q2_usd'],'prior_phosphate_usd':prior['phosphate_q2_usd'],
                        'target_source':current['source_url'],'prior_source':prior['source_url']})
    features=['forecast_mean_p12_pct','fieldwork_mean_days']
    predictions=[]
    for year in range(2022,2027):
        test=original[original.year.eq(year)]
        if len(test)!=1: raise ValueError('Missing test year')
        row=next(test.itertuples()); cutoff=str(row.as_of)[:10]
        if cutoff!=f'{year}-05-31' or str(row.target_filed)[:10]<=cutoff: raise ValueError('Test cutoff invalid')
        for label, data in [('unadjusted',full),('segment_adjusted',adjusted)]:
            train=data[(data.year<year)&(data.target_filed.astype(str).str[:10]<=cutoff)]
            if any(lh_date(records[int(y)]['filed_date'])>lh_date(cutoff) for y in train.year if int(y) in range(2010,2016)):
                raise ValueError('Segment target not yet known at evaluation')
            prediction=lh_fit(train,row,features)
            if label=='unadjusted':
                reference=result['predictions']
                ref=reference[(reference.year==year)&(reference.model=='price_forecast_fieldwork')&(reference.training=='new')]
                if len(ref)!=1 or not np.isclose(prediction,float(ref.prediction_usd.iloc[0]),rtol=1e-10,atol=1e-3):
                    raise AssertionError('Original full-history prediction differs from previous notebook result')
            predictions.append({'year':year,'variant':label,'training_years':len(train),
                                'actual_usd':float(row.target_q2_usd),'prediction_usd':prediction,
                                'absolute_error_usd':abs(prediction-float(row.target_q2_usd))})
    pred=pd.DataFrame(predictions)
    scores=pred.groupby('variant',as_index=False).agg(years=('year','size'),mae_usd=('absolute_error_usd','mean'))
    paired=pred.pivot(index='year',columns='variant',values='absolute_error_usd')
    report='''# Phosphate-segment comparability: post-hoc sensitivity

The added older years precede the 2014 sale of CF's phosphate business. For 2010–2015 training rows only, compare originally reported company-wide Q2 sales with historical total sales minus the separately reported phosphate segment, subtracting phosphate from both the target and prior-year base. Retain the same price/weather features, test cutoffs, full company-wide test targets, training-only standardization and ridge penalty. Reproduce the unadjusted full-history predictions exactly before reporting a comparison.

The old segment data are sourced from dated CF filings and checked for availability. Some 2010–2013 fieldwork records are reconstructed from a current, potentially revised IEM series. The prior forecast-vintage and price-reconstruction caveats also apply. Nitrogen-only historical sales are an approximation to present-day company-wide business, not a restatement of the test target. A phosphate exclusion was proposed after looking at past results, so this is exploratory. 2022–2026 are not an untouched test. It cannot demonstrate that phosphate caused errors or establish a predictive improvement. The frozen 2027 protocol is unchanged.
'''
    out=Path('/kaggle/working')/output_subdir; out.mkdir(parents=True,exist_ok=True)
    for name,frame in [('segment_provenance_private.csv',pd.DataFrame(records.values())),('training_adjustments_private.csv',pd.DataFrame(changes)),('predictions_private.csv',pred),('scores_private.csv',scores),('paired_private.csv',paired.reset_index())]:
        frame.to_csv(out/name,index=False)
    (out/'REPORT.md').write_text(report,encoding='utf8')
    print(report)
    display(scores.assign(mae_musd=lambda d:d.mae_usd/1e6).drop(columns='mae_usd').round(2))
    display((paired/1e6).round(1))
    return {'scores':scores,'predictions':pred,'paired':paired,'report':report}

PHOSPHATE_RESULTS=phosphate_sensitivity(PHOSPHATE_SEGMENT,MISSING_FIELDWORK_RESULTS,'iowa_phosphate_sensitivity')
PHOSPHATE_ALT_FIELDWORK_RESULTS=phosphate_sensitivity(PHOSPHATE_SEGMENT,MISSING_FIELDWORK_SENSITIVITY,'iowa_phosphate_alt_fieldwork')
