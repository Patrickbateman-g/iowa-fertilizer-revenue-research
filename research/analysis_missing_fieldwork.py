from pathlib import Path
import numpy as np
import pandas as pd


def missing_fieldwork_research(reported, archived, output_subdir='iowa_missing_fieldwork'):
    old_history = LONG_HISTORY_RESULTS['history'].copy()
    old_predictions = LONG_HISTORY_RESULTS['predictions']
    combined = HISTORICAL_FIELDWORK + reported
    fa, _ = lh_forecasts(HISTORICAL_FORECASTS)
    wa, wc = lh_fieldwork(combined)
    expected_url = 'https://mesonet.agron.iastate.edu/plotting/auto/plot/127/state:IA::short_desc:FD::w:last::threshold:50::cmap:tab20b::_r:43::dpi:100.csv'
    seen = set()
    for r in archived:
        y=int(r['year']); end=lh_date(r['week_end']); prev=lh_date(r['previous_week_end'])
        if end not in lh_expected(y) or (end-prev).days != 7 or (y,end) in seen:
            raise ValueError('Invalid or duplicate archive week')
        seen.add((y,end))
        if r['source_url'] != expected_url or lh_date(r['retrieved_date']) <= dt.date(y,5,31):
            raise ValueError('Archive source or retrieval date invalid')
        before,after=float(r['cumulative_previous']),float(r['cumulative_current'])
        value=after-before
        if not np.isfinite(value) or not 0 <= value <= 7: raise ValueError('Invalid difference')
        mask=(wc.year==y)&(wc.week_end==end.isoformat())
        if mask.sum()!=1 or bool(wc.loc[mask,'usable'].iloc[0]): raise ValueError('Archive week conflicts with an existing observation')
        wc.loc[mask,'days_suitable']=round(value,6)
        wc.loc[mask,'source_url']=r['source_url']
        wc.loc[mask,'release_date']=None
        wc.loc[mask,'basis']='later_archive_difference_not_point_in_time'
        wc.loc[mask,'retrieved_date']=r['retrieved_date']
        wc.loc[mask,'usable']=True
    for y in LH_YEARS:
        selection=wc[(wc.year==y)&(wc.usable==True)]
        complete=len(selection)==6
        wa.loc[wa.year==y,'weeks_present']=len(selection)
        wa.loc[wa.year==y,'fieldwork_complete']=complete
        wa.loc[wa.year==y,'fieldwork_mean_days']=float(selection.days_suitable.mean()) if complete else np.nan
    pa = lh_prices(HISTORICAL_PRICES)
    avail = fa.merge(wa, on='year', validate='one_to_one').merge(pa, on='year', validate='one_to_one')
    rev = lh_original_revenue(SNAPSHOT_JSON)
    rows = []
    previously = set(old_history.year.astype(int))
    for x in avail.itertuples():
        y = int(x.year)
        if y in previously or not (x.forecast_complete and x.fieldwork_complete and x.price_complete):
            continue
        if y not in rev.index or y-1 not in rev.index: continue
        now, prior = rev.loc[y], rev.loc[y-1]
        cutoff = f'{y}-05-31'; filed = str(now.q2_filed)[:10]; prior_filed = str(prior.q2_filed)[:10]
        if not (f'{y}-06-30' <= filed <= f'{y+1}-05-31' and prior_filed <= cutoff):
            raise ValueError('Revenue chronology invalid')
        if float(now.q2_usd) <= 0 or float(prior.q2_usd) <= 0 or not str(now.q2_accession).strip() or not str(prior.q2_accession).strip():
            raise ValueError('Invalid revenue')
        rows.append({'year':y, 'as_of':cutoff, 'target_filed':filed,
                     'target_q2_usd':float(now.q2_usd), 'prior_q2_usd':float(prior.q2_usd),
                     'price_ratio':float(x.price_ratio), 'forecast_mean_p12_pct':float(x.forecast_mean_p12_pct),
                     'fieldwork_mean_days':float(x.fieldwork_mean_days)})
    extra = pd.DataFrame(rows, columns=old_history.columns)
    orig = FIELDWORK_WINDOW_RESULTS['six']['data'].copy()
    cols = list(old_history.columns)
    old = pd.concat([old_history, orig[cols]], ignore_index=True).sort_values('year')
    new = pd.concat([old, extra], ignore_index=True).sort_values('year')
    if old.year.duplicated().any() or new.year.duplicated().any(): raise ValueError('Duplicate training years')
    features = {'price_forecast': ['forecast_mean_p12_pct'],
                'price_forecast_fieldwork': ['forecast_mean_p12_pct', 'fieldwork_mean_days']}
    predictions=[]
    for y in range(2022, 2027):
        test = orig[orig.year.eq(y)]
        if len(test) != 1: raise ValueError('Missing evaluation year')
        row = next(test.itertuples()); cutoff=str(row.as_of)[:10]
        if cutoff != f'{y}-05-31' or str(row.target_filed)[:10] <= cutoff: raise ValueError('Test cutoff invalid')
        for label, data in [('prior',old),('new',new)]:
            train = data[(data.year < y) & (data.target_filed.astype(str).str[:10] <= cutoff)]
            for model, fields in features.items():
                pred=lh_fit(train,row,fields); actual=float(row.target_q2_usd)
                if label=='prior':
                    oldp=old_predictions[(old_predictions.year==y)&(old_predictions.model==model)&(old_predictions.training=='extended')]
                    if len(oldp)!=1 or not np.isclose(pred,float(oldp.prediction_usd.iloc[0]),rtol=1e-10,atol=1e-3):
                        raise AssertionError('Prior long-history result could not be reproduced')
                predictions.append({'year':y,'model':model,'training':label,'train_years':len(train),
                                    'earlier_years':','.join(map(str,train.year.astype(int))),
                                    'actual_usd':actual,'prediction_usd':pred,'absolute_error_usd':abs(pred-actual)})
    pred=pd.DataFrame(predictions)
    scores=pred.groupby(['model','training'],as_index=False).agg(years=('year','size'),mae_usd=('absolute_error_usd','mean'))
    paired=pred.pivot(index=['model','year'],columns='training',values='absolute_error_usd').reset_index()
    paired['new_minus_prior_usd']=paired['new']-paired['prior']
    if output_subdir not in ('iowa_missing_fieldwork','iowa_missing_fieldwork_sensitivity'): raise ValueError('Invalid output directory')
    output=Path('/kaggle/working') / output_subdir; output.mkdir(parents=True,exist_ok=True)
    for name,df in [('coverage_private.csv',wc),('additional_years_private.csv',extra),('predictions_private.csv',pred),('scores_private.csv',scores),('paired_private.csv',paired)]: df.to_csv(output/name,index=False)
    report = f'''# Additional Iowa fieldwork records: exploratory comparison

New complete earlier years: {', '.join(map(str, extra.year.astype(int))) if len(extra) else 'none'}. Same 2022–2026 evaluation years, training-only standardization, features and model as previous long-history comparison; prior predictions must be reproduced exactly. Only previous targets filed by the cutoff are used. Some missing fieldwork values are reconstructed by differencing the current IEM Quick Stats cumulative series. They may have been revised and cannot be claimed to represent data known at the original cutoff. No report date is invented for them. The 2013 May 5 report narrative and USDA state stories say 2.3 days; an Iowa report table says 2.0, so a sensitivity analysis is necessary. Weekly source links, retrieval/release dates, and caveats appear in the notebook and coverage table. Forecast archive availability and revisions, BLS price reconstructions, and previously documented source limitations continue to apply.

All test years had already been examined. This is exploratory, not a new independent test or proof of causality; the prospective 2027 protocol remains unchanged. Company-wide revenue is not Iowa-specific.
'''
    (output/'REPORT.md').write_text(report,encoding='utf8')
    print(report)
    display(avail[['year','weeks_present','fieldwork_complete']])
    display(scores.assign(mae_musd=lambda d:d.mae_usd/1e6).drop(columns='mae_usd').round(2))
    display((paired.query("model == 'price_forecast_fieldwork'").set_index('year')[['prior','new']]/1e6).round(1))
    return {'additional_years':extra, 'coverage':wc, 'predictions':pred, 'scores':scores, 'paired':paired, 'report':report}

MISSING_FIELDWORK_RESULTS = missing_fieldwork_research(NEW_FIELDWORK_REPORTED, NEW_FIELDWORK_ARCHIVE)

_alternate_reported = [dict(r) for r in NEW_FIELDWORK_REPORTED]
for _r in _alternate_reported:
    if _r['year']==2013 and _r['week_end']=='2013-05-05':
        _r.update(days_suitable=2.0, release_date='2013-05-06',
                  source_url='https://publications.iowa.gov/14585/1/Vol08__05_06_13.pdf',
                  basis='conflicting_original_report_table_sensitivity')
MISSING_FIELDWORK_SENSITIVITY = missing_fieldwork_research(_alternate_reported, NEW_FIELDWORK_ARCHIVE, 'iowa_missing_fieldwork_sensitivity')
