from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd

DECOUPLED_COLUMNS = ['year','as_of','target_filed','target_q2_usd',
                     'prior_q2_usd','price_ratio','forecast_mean_p12_pct']


def forecast_only_rows(frame):
    if not set(DECOUPLED_COLUMNS).issubset(frame.columns):
        raise ValueError('Missing forecast, price, revenue, or date field')
    data = frame[DECOUPLED_COLUMNS].copy()
    if data.year.duplicated().any():
        raise ValueError('Duplicate year')
    for name in ('target_q2_usd','prior_q2_usd','price_ratio','forecast_mean_p12_pct'):
        data[name] = pd.to_numeric(data[name], errors='raise')
        if not np.isfinite(data[name]).all(): raise ValueError(f'Non-finite {name}')
    if (data[['target_q2_usd','prior_q2_usd','price_ratio']] <= 0).any().any():
        raise ValueError('Invalid nonpositive revenue or price ratio')
    if not data.forecast_mean_p12_pct.between(0,100).all():
        raise ValueError('Invalid precipitation probability')
    for row in data.itertuples():
        y = int(row.year)
        if y != row.year or str(row.as_of)[:10] != f'{y}-05-31': raise ValueError('Bad cutoff')
        if lh_date(str(row.target_filed)[:10]) <= dt.date(y,6,30): raise ValueError('Bad target filing')
    return data.sort_values('year').reset_index(drop=True)


def build_older_forecast_only(forecasts, prices, snapshot):
    fa, forecast_coverage = lh_forecasts(forecasts)
    pa = lh_prices(prices)
    available = fa.merge(pa,on='year',how='outer',validate='one_to_one')
    for col in ('forecast_complete','price_complete'):
        available[col] = available[col].fillna(False).astype(bool)
    rev = lh_original_revenue(snapshot)
    rows, statuses = [], []
    for row in available.sort_values('year').itertuples():
        y = int(row.year)
        revenue_ok = y in rev.index and y-1 in rev.index
        complete = bool(row.forecast_complete and row.price_complete and revenue_ok)
        statuses.append(dict(year=y,forecast_complete=bool(row.forecast_complete),
                             price_complete=bool(row.price_complete),revenue_available=revenue_ok,
                             forecast_only_eligible=complete))
        if not complete: continue
        actual, prior = rev.loc[y], rev.loc[y-1]
        cutoff = dt.date(y,5,31)
        filed = lh_date(str(actual.q2_filed)[:10])
        previous_filed = lh_date(str(prior.q2_filed)[:10])
        if not (dt.date(y,6,30)<filed<=dt.date(y+1,5,31) and previous_filed<=cutoff):
            raise ValueError('Revenue filing unavailable at stated cutoff')
        if not str(actual.q2_accession).strip() or not str(prior.q2_accession).strip():
            raise ValueError('Missing source accession')
        rows.append(dict(year=y,as_of=cutoff.isoformat(),target_filed=filed.isoformat(),
                         target_q2_usd=float(actual.q2_usd),prior_q2_usd=float(prior.q2_usd),
                         price_ratio=float(row.price_ratio),forecast_mean_p12_pct=float(row.forecast_mean_p12_pct)))
    frame = pd.DataFrame(rows, columns=DECOUPLED_COLUMNS)
    return forecast_only_rows(frame), pd.DataFrame(statuses), forecast_coverage


def predict_forecast_only(training_rows, test_row):
    data=forecast_only_rows(training_rows)
    cutoff=str(test_row.as_of)[:10]
    train=data[(data.year<int(test_row.year)) & (data.target_filed.astype(str).str[:10]<=cutoff)].copy()
    if len(train)<6: raise ValueError('Insufficient earlier training observations')
    return lh_fit(train,test_row,['forecast_mean_p12_pct']),train


def _check_no_fieldwork_dependency(frame):
    base=forecast_only_rows(frame)
    changed=frame.copy()
    changed['fieldwork_complete']=False
    changed['fieldwork_mean_days']=np.nan
    pd.testing.assert_frame_equal(base,forecast_only_rows(changed))
    if len(base)>=7:
        row=next(base.tail(1).itertuples())
        p1,t1=predict_forecast_only(frame,row)
        p2,t2=predict_forecast_only(changed,row)
        pd.testing.assert_frame_equal(t1,t2)
        if not np.isclose(p1,p2,rtol=1e-12,atol=1e-3):
            raise AssertionError('Forecast changed when fieldwork was removed')
    return True


def decoupled_forecast_research():
    old, old_coverage, _ = build_older_forecast_only(HISTORICAL_FORECASTS,HISTORICAL_PRICES,SNAPSHOT_JSON)
    recent_raw = FORECAST_RESULTS['data'].copy()
    if 'price_as_of' not in recent_raw: raise ValueError('Missing recent price vintage')
    for row in recent_raw.itertuples():
        if str(row.price_as_of)[:10] != f'{int(row.year)}-05-31':
            raise ValueError('Recent price vintage mismatch')
    recent = forecast_only_rows(recent_raw)
    _check_no_fieldwork_dependency(recent_raw)
    if set(old.year) & set(recent.year): raise ValueError('Duplicate old/recent years')
    standalone = forecast_only_rows(pd.concat([old,recent],ignore_index=True))

    old_matched = forecast_only_rows(LONG_HISTORY_RESULTS['history'])
    prior_recent = forecast_only_rows(FIELDWORK_WINDOW_RESULTS['six']['data'])
    recovered = forecast_only_rows(MISSING_FIELDWORK_RESULTS['additional_years'])
    matched_prior = forecast_only_rows(pd.concat([old_matched,prior_recent],ignore_index=True))
    matched_full = forecast_only_rows(pd.concat([matched_prior,recovered],ignore_index=True))

    segments = {int(item['year']):item for item in PHOSPHATE_SEGMENT}
    if len(segments)!=len(PHOSPHATE_SEGMENT) or set(segments)!=set(range(2009,2016)):
        raise ValueError('Missing or duplicate phosphate segments')
    def adjust(frame):
        result = frame.copy()
        for idx, row in result[result.year.between(2010,2015)].iterrows():
            y=int(row.year); current=segments[y]; prior=segments[y-1]
            if lh_date(prior['filed_date'])>lh_date(str(row.as_of)[:10]) or lh_date(current['filed_date'])>lh_date(str(row.target_filed)[:10]):
                raise ValueError('Phosphate segment unavailable at cutoff')
            values=[float(row.target_q2_usd)-float(current['phosphate_q2_usd']),
                    float(row.prior_q2_usd)-float(prior['phosphate_q2_usd'])]
            if not all(np.isfinite(v) and v>0 for v in values): raise ValueError('Invalid phosphate adjustment')
            result.loc[idx,['target_q2_usd','prior_q2_usd']]=values
        return result
    cohorts = {'prior_unadjusted':matched_prior,'prior_adjusted':adjust(matched_prior),
               'full_unadjusted':matched_full,'full_adjusted':adjust(matched_full),
               'decoupled_unadjusted':standalone,'decoupled_adjusted':adjust(standalone)}
    reference = PHOSPHATE_FORECAST_RESULTS['predictions']
    outrows=[]
    for y in range(2022,2027):
        test = prior_recent[prior_recent.year.eq(y)]
        if len(test)!=1: raise ValueError('Missing matched test year')
        row=next(test.itertuples()); cutoff=str(row.as_of)[:10]
        if cutoff != f'{y}-05-31' or str(row.target_filed)[:10] <= cutoff: raise ValueError('Test cutoff mismatch')
        for label,data in cohorts.items():
            prediction,train=predict_forecast_only(data,row)
            if label.endswith('_adjusted'):
                for ty in train.year.astype(int):
                    if ty in range(2010,2016) and lh_date(segments[ty]['filed_date'])>lh_date(cutoff):
                        raise ValueError('Training segment filed after forecast cutoff')
            if label in ('prior_unadjusted','prior_adjusted','full_unadjusted','full_adjusted'):
                check=reference[(reference.year.eq(y)) & (reference.variant.eq(label))]
                if len(check)!=1 or not np.isclose(prediction,float(check.prediction_usd.iloc[0]),rtol=1e-10,atol=1e-3):
                    raise AssertionError('Saved matched benchmark failed to reproduce')
            actual=float(row.target_q2_usd)
            outrows.append(dict(year=y,variant=label,training_years=len(train),
                                training_year_list=','.join(map(str,train.year.astype(int))),
                                prediction_usd=prediction,actual_usd=actual,
                                signed_error_usd=prediction-actual,absolute_error_usd=abs(prediction-actual)))
    pred=pd.DataFrame(outrows)
    scores=pred.groupby('variant',as_index=False).agg(years=('year','size'),mae_usd=('absolute_error_usd','mean'))
    annual=pred.pivot(index='year',columns='variant',values='absolute_error_usd')
    prior_years=set(matched_prior.year.astype(int)); full_years=set(matched_full.year.astype(int)); free_years=set(standalone.year.astype(int))
    common=matched_full.merge(standalone,on='year',suffixes=('_full','_free'),validate='one_to_one')
    for col in ('target_q2_usd','prior_q2_usd','price_ratio','forecast_mean_p12_pct'):
        if not np.allclose(common[col+'_full'],common[col+'_free'],rtol=1e-10,atol=1e-3):
            raise AssertionError(f'Decoupled feature differs on shared rows: {col}')
    for col in ('as_of','target_filed'):
        if not common[col+'_full'].astype(str).str[:10].equals(common[col+'_free'].astype(str).str[:10]):
            raise AssertionError(f'Decoupled date differs on shared rows: {col}')
    newly_prior=sorted(free_years-prior_years); newly_full=sorted(free_years-full_years)
    absent_full=sorted(full_years-free_years)
    if not newly_full and not absent_full:
        for variant in ('adjusted','unadjusted'):
            a=pred[pred.variant.eq('full_'+variant)].sort_values('year').prediction_usd.to_numpy()
            b=pred[pred.variant.eq('decoupled_'+variant)].sort_values('year').prediction_usd.to_numpy()
            if not np.allclose(a,b,rtol=1e-10,atol=1e-3): raise AssertionError('Identical cohorts must reproduce predictions')
    report=f'''# Forecast-only fieldwork decoupling (exploratory)

The standalone training cohort is built from complete forecast, price and revenue records without requiring fieldwork, including both older 2010–2015 and newer 2016–2026 records. Test rows remain the same matched 2022–2026 rows, and fits use only earlier filed targets with training-only scaling. Existing prior/full, adjusted/unadjusted predictions must reproduce before comparing. Years eligible in the decoupled cohort relative to the earlier prior cohort: {newly_prior}; additional years relative to the previously recovered full cohort: {newly_full}; years in full but absent from the decoupled cohort: {absent_full}. Some formerly missing fieldwork years were already restored to the full comparison, so decoupling does not necessarily add new data or improve scores. Full and independent cohorts with identical year and feature sets must produce identical predictions.

This is retrospective, not a fresh holdout; all 2022–2026 results were inspected. Dated sources are validated as in the earlier notebook, but manually recorded historic forecast values, forecast issuance times, older price reconstructions, and revisions remain unverified. The forecast feature is a probability rather than rainfall, fieldwork or farmer purchases. Results concern total company revenue. The frozen 2027 protocol is unchanged. Coverage, per-year errors and results stay on Kaggle.
'''
    output=Path('/kaggle/working/iowa_decoupled_forecast'); output.mkdir(parents=True,exist_ok=True)
    for filename,frame in [('historical_coverage_private.csv',old_coverage),('decoupled_rows_private.csv',standalone),('predictions_private.csv',pred),('scores_private.csv',scores),('annual_errors_private.csv',annual.reset_index())]:
        frame.to_csv(output/filename,index=False)
    (output/'REPORT.md').write_text(report,encoding='utf8')
    print(report)
    display(old_coverage)
    display(scores.assign(mae_musd=lambda d:d.mae_usd/1e6).drop(columns='mae_usd').round(2))
    display((annual/1e6).round(2))
    return {'historical_coverage':old_coverage,'data':standalone,'predictions':pred,'scores':scores,'annual':annual,'report':report}


DECOUPLED_FORECAST_RESULTS=decoupled_forecast_research()
