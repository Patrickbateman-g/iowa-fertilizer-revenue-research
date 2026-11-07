from __future__ import annotations
import datetime as dt
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

REPORT_MODELS = ('price_only', 'price_weather', 'price_shipments', 'price_weather_shipments')
REPORT_LABELS = {'price_only':'Price', 'price_weather':'Price + weather',
                 'price_shipments':'Price + shipments', 'price_weather_shipments':'Price + both'}


def _text(value):
    return str(value).replace('|', r'\|').replace('\n', ' ')


def _money(value, signed=False):
    if pd.isna(value):
        return '—'
    return f'{float(value)/1e6:+,.1f}' if signed else f'{float(value)/1e6:,.1f}'


def _display_period(period, predictions):
    selected = predictions[predictions.year.le(2021) if period == 'through_2021' else predictions.year.ge(2022)]
    years = sorted(int(y) for y in selected.year.unique())
    if not years:
        return period.replace('_', ' ')
    if len(years) == 1:
        return str(years[0])
    if years == list(range(years[0], years[-1] + 1)):
        return f'{years[0]}–{years[-1]}'
    return ', '.join(map(str, years))


def _load_result_rows(register, study, kind):
    frame=register[(register.study==study) & (register.result_kind==kind)]
    records=[]
    for row in frame.itertuples():
        try:
            rec=json.loads(row.result_row_json)
        except (ValueError,TypeError) as exc:
            raise ValueError(f'Invalid registered result in {study}') from exc
        if not isinstance(rec,dict): raise ValueError('Registered result is not an object')
        records.append({**rec, "_register_source": row.result_source})
    return records


def _load_score_rows(register, study):
    return _load_result_rows(register, study, 'reported_score')


def _unique_research_row(register, study, source_suffix, year, kind, model=None):
    rows = register[(register.study == study) & (register.result_kind == kind)]
    found = []
    for row in rows.itertuples():
        source = str(row.result_source)
        if not (source == source_suffix or source.endswith('/' + source_suffix)):
            continue
        try:
            value = json.loads(row.result_row_json)
        except (TypeError, ValueError):
            continue
        if (isinstance(value, dict) and value.get('year') == year and
                (model is None or value.get('model') == model)):
            found.append((value, source))
    return found[0] if len(found) == 1 else None


def _nonnegative_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if np.isfinite(value) and value >= 0 else None


def _audit_results(comparison, ranges):
    needed={'predictions','scores','period_scores','incremental'}
    if not isinstance(comparison,dict) or not needed.issubset(comparison):
        raise ValueError('Incomplete incremental comparison')
    p=comparison['predictions'].copy()
    required={'year','model','decision_date','target_filed','prediction_usd','actual_usd','signed_error_usd','absolute_error_usd'}
    if not isinstance(p,pd.DataFrame) or p.empty or not required.issubset(p):
        raise ValueError('Incomplete prediction table')
    if p.duplicated(['year','model']).any(): raise ValueError('Duplicate prediction')
    for col in ('prediction_usd','actual_usd','signed_error_usd','absolute_error_usd'):
        p[col]=pd.to_numeric(p[col],errors='raise')
        if not np.isfinite(p[col]).all(): raise ValueError('Missing or non-finite prediction result')
    if not np.allclose(p.prediction_usd-p.actual_usd,p.signed_error_usd,rtol=1e-10,atol=.001):
        raise ValueError('Signed error mismatch')
    if not np.allclose(abs(p.signed_error_usd),p.absolute_error_usd,rtol=1e-10,atol=.001):
        raise ValueError('Absolute error mismatch')
    for y,g in p.groupby('year'):
        if set(g.model)!=set(REPORT_MODELS) or g.decision_date.nunique()!=1 or g.target_filed.nunique()!=1:
            raise ValueError(f'Unmatched models or decision dates for {y}')
        if not np.allclose(g.actual_usd,g.actual_usd.iloc[0],rtol=0,atol=.001):
            raise ValueError('Unequal annual outcomes')
    periods=comparison['period_scores']
    if not {'period','model','years','mae_usd'}.issubset(periods):
        raise ValueError('Missing period scores')
    computed=(p.assign(period=np.where(p.year<=2021,'through_2021','2022_onward'))
                .groupby(['period','model'],as_index=False)
                .agg(years=('year','size'),mae_usd=('absolute_error_usd','mean')))
    joined=computed.merge(periods[['period','model','years','mae_usd']],on=['period','model'],
                          validate='one_to_one',suffixes=('_expected','_reported'),how='outer',indicator=True)
    if not joined._merge.eq('both').all() or not (joined.years_expected==joined.years_reported).all() or not np.allclose(joined.mae_usd_expected,joined.mae_usd_reported,atol=.001,rtol=1e-10):
        raise ValueError('Period scores do not match annual results')
    if not isinstance(ranges,pd.DataFrame) or not {'year','model','decision_date','target_filed','prediction_usd','actual_usd','lower_usd','upper_usd','contains_actual','calibration_count'}.issubset(ranges):
        raise ValueError('Incomplete range table')
    if ranges.duplicated(['year','model']).any(): raise ValueError('Duplicate range')
    paired=p.merge(ranges[['year','model','decision_date','target_filed','prediction_usd','actual_usd','lower_usd','upper_usd','calibration_count']],
                   on=['year','model'],how='outer',indicator=True,validate='one_to_one',suffixes=('','_range'))
    if (not paired._merge.eq('both').all() or
        not np.allclose(paired.prediction_usd,paired.prediction_usd_range,rtol=1e-10,atol=.001) or
        not np.allclose(paired.actual_usd,paired.actual_usd_range,rtol=1e-10,atol=.001) or
        not paired.decision_date.eq(paired.decision_date_range).all() or
        not paired.target_filed.eq(paired.target_filed_range).all()):
        raise ValueError('Ranges and predictions do not match')
    return p, periods, paired


def _audit_range_derivation(predictions, ranges, calibration_predictions):
    fields={'year','model','decision_date','target_filed','prediction_usd','actual_usd'}
    if not isinstance(calibration_predictions,pd.DataFrame) or calibration_predictions.empty or not fields.issubset(calibration_predictions):
        raise ValueError('Missing calibration prediction history')
    q=calibration_predictions.copy()
    years=pd.to_numeric(q.year,errors='raise')
    if years.isna().any() or not np.isfinite(years).all() or (years!=np.floor(years)).any():
        raise ValueError('Invalid calibration year')
    q['year']=years.astype(int)
    if q.duplicated(['year','model']).any() or any(m not in REPORT_MODELS for m in q.model):
        raise ValueError('Invalid or duplicate calibration model/year')
    for col in ('prediction_usd','actual_usd'):
        q[col]=pd.to_numeric(q[col],errors='raise')
    if not np.isfinite(q.prediction_usd).all() or (q.prediction_usd<=0).any() or not np.isfinite(q.actual_usd.dropna()).all() or (q.actual_usd.dropna()<=0).any():
        raise ValueError('Invalid calibration prediction or outcome')
    def day(value):
        if not isinstance(value,str): raise ValueError('Invalid calibration date')
        try: parsed=dt.date.fromisoformat(value)
        except ValueError as exc: raise ValueError('Invalid calibration date') from exc
        if parsed.isoformat()!=value: raise ValueError('Invalid calibration date')
        return parsed
    for row in q.itertuples(index=False):
        if day(row.decision_date)!=dt.date(row.year,5,31): raise ValueError('Invalid calibration decision date')
        if pd.isna(row.actual_usd):
            if pd.notna(row.target_filed) and row.target_filed!='': raise ValueError('Filing without calibration outcome')
        elif day(row.target_filed)<=dt.date(row.year,6,30):
            raise ValueError('Invalid calibration outcome filing date')
    for _,group in q.groupby('year'):
        if (set(group.model)!=set(REPORT_MODELS) or group.decision_date.nunique()!=1 or
            group.actual_usd.isna().nunique()!=1 or group.target_filed.fillna('').nunique()!=1 or
            (group.actual_usd.notna().all() and not np.allclose(group.actual_usd,group.actual_usd.iloc[0],rtol=0,atol=.001))):
            raise ValueError('Calibration history must have matched models and outcomes')
    matched=predictions.merge(q[list(fields)],on=['year','model'],how='left',validate='one_to_one',suffixes=('','_history'),indicator=True)
    if (not matched._merge.eq('both').all() or
        not np.allclose(matched.prediction_usd,matched.prediction_usd_history,rtol=1e-10,atol=.001) or
        not np.allclose(matched.actual_usd,matched.actual_usd_history,rtol=1e-10,atol=.001) or
        not matched.decision_date.eq(matched.decision_date_history).all() or
        not matched.target_filed.eq(matched.target_filed_history).all()):
        raise ValueError('Calibration history does not match report predictions')
    need={'calibration_years','calibration_count','nominal_coverage','rank','absolute_error_radius_usd'}
    if not need.issubset(ranges): raise ValueError('Missing range derivation details')
    def same_number(actual, expected):
        if expected is None: return pd.isna(actual)
        try: return bool(np.isfinite(float(actual)) and math.isclose(float(actual),expected,rel_tol=1e-10,abs_tol=.001))
        except (TypeError,ValueError): return False
    for row in ranges.itertuples(index=False):
        earlier=q[(q.model==row.model)&(q.year<row.year)&q.actual_usd.notna()].copy()
        earlier=earlier[earlier.target_filed<=row.decision_date].sort_values('year')
        n=len(earlier); rank=math.ceil((n+1)*.8)
        expected_years=','.join(map(str,earlier.year))
        recorded_years='' if pd.isna(row.calibration_years) else row.calibration_years
        if (recorded_years!=expected_years or isinstance(row.calibration_count,bool) or
            row.calibration_count!=n or isinstance(row.rank,bool) or row.rank!=rank or
            not same_number(row.nominal_coverage,.8)):
            raise ValueError('Range calibration years, count or rank mismatch')
        radius=float(np.sort(abs(earlier.prediction_usd-earlier.actual_usd).to_numpy())[rank-1]) if rank<=n else None
        lower=max(0.0,float(row.prediction_usd)-radius) if radius is not None else None
        upper=float(row.prediction_usd)+radius if radius is not None else None
        if not all((same_number(row.absolute_error_radius_usd,radius),same_number(row.lower_usd,lower),same_number(row.upper_usd,upper))):
            raise ValueError('Range bounds or radius do not match earlier errors')


def build_research_report(comparison, ranges, coverage_by_period, replay_report, register, calibration_context=False, calibration_predictions=None):
    p,periods,paired=_audit_results(comparison,ranges)
    if calibration_context and calibration_predictions is None:
        raise ValueError('Calibration context requires earlier prediction history')
    if (not calibration_context and calibration_predictions is not None and
        isinstance(calibration_predictions,pd.DataFrame) and {'year','model'}.issubset(calibration_predictions) and
        set(map(tuple,calibration_predictions[['year','model']].to_numpy())) !=
        set(map(tuple,p[['year','model']].to_numpy()))):
        raise ValueError('Pre-window calibration requires calibration context')
    _audit_range_derivation(p,ranges,p if calibration_predictions is None else calibration_predictions)
    if not isinstance(coverage_by_period,pd.DataFrame) or not {'period','model','evaluated_ranges','covered','empirical_coverage'}.issubset(coverage_by_period):
        raise ValueError('Missing empirical coverage')
    if not isinstance(replay_report,(list,tuple)) or not replay_report:
        raise ValueError('Missing replay report')
    rr=pd.DataFrame(replay_report)
    if not {'decision_date','series','status'}.issubset(rr): raise ValueError('Incomplete replay report')
    if not isinstance(register,pd.DataFrame) or register.empty or not {'study','result_kind','result_row_json','result_source'}.issubset(register):
        raise ValueError('Missing experiment register')
    if coverage_by_period.duplicated(['period','model']).any(): raise ValueError('Duplicate coverage row')
    known_periods={'2016-2021':(2016,2021),'2022-2026':(2022,2026)}
    for r in coverage_by_period.itertuples(index=False):
        if r.period not in known_periods or r.model not in REPORT_LABELS:
            raise ValueError('Unknown coverage period or model')
        lo,hi=known_periods[r.period]
        g=ranges[(ranges.year>=lo)&(ranges.year<=hi)&(ranges.model==r.model)]
        evaluated=g[g.contains_actual.notna()]
        covered=int(evaluated.contains_actual.sum()) if len(evaluated) else 0
        if len(evaluated)!=int(r.evaluated_ranges) or covered!=int(r.covered):
            raise ValueError('Coverage counts do not match annual ranges')
        expected=covered/len(evaluated) if len(evaluated) else np.nan
        if not ((pd.isna(expected) and pd.isna(r.empirical_coverage)) or np.isclose(expected,float(r.empirical_coverage))):
            raise ValueError('Coverage rate does not match annual ranges')
    if set(coverage_by_period.model)!=set(REPORT_MODELS) or set(coverage_by_period.period)!=set(known_periods) or len(coverage_by_period)!=8:
        raise ValueError('Incomplete coverage model/period rows')
    for r in ranges.itertuples(index=False):
        finite=pd.notna(r.lower_usd) and pd.notna(r.upper_usd)
        if finite and (r.lower_usd > r.prediction_usd or r.upper_usd < r.prediction_usd or r.lower_usd > r.upper_usd):
            raise ValueError('Invalid range bounds')
        expected=(bool(r.lower_usd<=r.actual_usd<=r.upper_usd) if finite and pd.notna(r.actual_usd) else None)
        if (expected is None and pd.notna(r.contains_actual)) or (expected is not None and (pd.isna(r.contains_actual) or bool(r.contains_actual)!=expected)):
            raise ValueError('Coverage indicator does not match range bounds')
    lines=['# Iowa corn-season fertilizer revenue: exploratory research', '',
           '## Question and scope', '',
           'On May 31, can Iowa precipitation forecasts and CF Industries Q1 shipments improve a price-based estimate of CF’s Q2 company-wide net sales? The selected P12 feature is the forecast probability of at least 0.01 inch of liquid-equivalent precipitation during lead hours 12–24; it is not observed rain, total rainfall, or fertilizer purchases. The company sells to markets well beyond Iowa corn. This is a nowcast after part of Q2 has elapsed; correlation does not establish causation.', '',
           '## Information timeline', '',
           'The decision date is May 31. The price baseline uses prior-Q2 sales and the current/prior April nitrogen price-index ratio. The weather feature summarizes forecasts initialized from April 1 through May 20, a window of up to 50 dates, requiring at least 40 available dates per station; the shipment factor uses Q1 company product tons filed by the cutoff. The weather regression uses earlier Q2 outcomes filed by the cutoff and training-only scaling. A future target is assessed after its Q2 filing. Historical feature availability and actual publication times require the replay audit below; a date in a snapshot alone is not proof.', '',
           '## Matched comparison', '',
           'Each year has the same decision date and target for all four methods. Weather uses the original centered, no-intercept ridge correction. Shipments use the fixed square root of the current/prior Q1 shipment ratio; the both model applies both effects. The shipment multiplier was explored after examining historical outcomes.', '',
           'Mean absolute revenue error in USD millions; negative Δ versus price means the model improved on price alone.', '',
           '| Period | Model | Years | Mean absolute error | Δ versus price |', '|---|---|---:|---:|---:|']
    for period in ('through_2021','2022_onward'):
        subset=periods[periods.period==period]
        if subset.empty: continue
        if set(subset.model)!=set(REPORT_MODELS): raise ValueError('Incomplete period cohort')
        baseline=float(subset[subset.model=='price_only'].mae_usd.iloc[0])
        for model in REPORT_MODELS:
            row=subset[subset.model==model].iloc[0]
            lines.append(f"| {_display_period(period,p)} | {REPORT_LABELS[model]} | {int(row.years)} | {_money(row.mae_usd)} | {_money(float(row.mae_usd)-baseline,True)} |")
    lines += ['', 'Year-by-year values in USD millions. Each model cell is **prediction (signed error) [lower, upper]**; signed error is prediction minus actual. A dash means a range could not yet be calculated.', '',
              '| Year | Actual | Price | Price + weather | Price + shipments | Price + both |', '|---:|---:|---:|---:|---:|---:|']
    for year,g in paired.groupby('year',sort=True):
        cells=[]
        for model in REPORT_MODELS:
            item=g[g.model==model].iloc[0]
            interval='—' if pd.isna(item.lower_usd) or pd.isna(item.upper_usd) else f'[{_money(item.lower_usd)}, {_money(item.upper_usd)}]'
            cells.append(f'{_money(item.prediction_usd)} ({_money(item.signed_error_usd,True)}) {interval}')
        lines.append('| '+str(int(year))+' | '+_money(g.actual_usd.iloc[0])+' | '+' | '.join(cells)+' |')
    lines += ['', '## Uncertainty and observed coverage', '',
              ('Ranges use the same model’s earlier out-of-sample absolute errors whose Q2 outcomes had been filed by the decision date. The 80% finite rank is ceil((n+1) × 0.8); a finite range requires at least four prior errors. Lower bounds are clipped at zero. Missing ranges and unavailable outcomes are excluded from the coverage denominator, rather than counted as failures. ' + ('Eligible matching pre-window folds from 2016 onward may provide additional calibration errors; the separate calibration inventory lists unavailable or insufficient earlier folds. Their provenance is audited separately.' if calibration_context else 'Calibration uses only prior prediction rows included in the selected reporting window; changing the first requested year can change or remove a range even if earlier data exist.')), '',
              '| Period | Model | Covered / evaluated | Observed coverage |', '|---|---|---:|---:|']
    for r in coverage_by_period.sort_values(['period','model']).itertuples(index=False):
        if r.model not in REPORT_LABELS: raise ValueError('Unexpected coverage model')
        if int(r.covered)>int(r.evaluated_ranges) or int(r.covered)<0: raise ValueError('Invalid coverage count')
        lo, hi=known_periods[r.period]
        available=p[p.year.between(lo,hi)]
        if available.empty:
            continue
        label=_display_period('through_2021' if hi==2021 else '2022_onward',available)
        rate='—' if pd.isna(r.empirical_coverage) else f'{float(r.empirical_coverage)*100:.1f}%'
        lines.append(f'| {_text(label)} | {REPORT_LABELS[r.model]} | {int(r.covered)} / {int(r.evaluated_ranges)} | {rate} |')
    lines += ['', 'The series is very short, model and feature choices were informed by the same historical outcomes, and errors may change over time. The nominal 80% is a construction target, not a reliable calibration or a coverage guarantee. Actual past availability and out-of-sample execution of the archived predictions are not proven merely by calculating a range.', '',
              '## Recorded negative and mixed findings', '',
              'Labels such as holdout_2026 describe the original plan; those outcomes have since been inspected and are no longer untouched holdouts.', '']
    for period in ('through_2021','2022_onward'):
        group=periods[periods.period==period].set_index('model')
        if len(group)!=4: continue
        base=float(group.loc['price_only','mae_usd'])
        worse=[(REPORT_LABELS[m],float(group.loc[m,'mae_usd'])) for m in REPORT_MODELS[1:] if float(group.loc[m,'mae_usd'])>base]
        if worse:
            comparisons='; '.join(f'{name}: USD {v/1e6:,.1f}m' for name,v in worse)
            lines.append(f'- In {_display_period(period,p)}, the price-only error was USD {base/1e6:,.1f}m; these additions had higher average errors on the same years: {comparisons}.')
    original = _load_score_rows(register, 'iowa_corn_revenue')
    original_predictions = _load_result_rows(register, 'iowa_corn_revenue', 'derived_from_prediction_rows')
    groups = sorted({(r['_register_source'], r.get('stage')) for r in original
                     if isinstance(r.get('stage'), str) and r.get('model') in ('last_year','ridge_prior','ridge_weather')})
    for source, stage in groups:
        score_rows = [r for r in original if r['_register_source']==source and r.get('stage')==stage]
        prediction_source = source.rsplit('/',1)[0] + '/predictions_private.csv'
        models = ('last_year','ridge_prior','ridge_weather')
        checked = {}; years = None
        for model in models:
            sr = [r for r in score_rows if r.get('model')==model]
            pr = [r for r in original_predictions if r['_register_source']==prediction_source
                  and r.get('stage')==stage and r.get('model')==model]
            if len(sr)!=1 or len(pr)!=1: break
            score, pred = sr[0], pr[0]
            try:
                raw_years = pred.get('years')
                if not isinstance(raw_years,str) or not raw_years: break
                year_list = [int(v) for v in raw_years.split(',')]
                if (any(str(y)!=v for y,v in zip(year_list,raw_years.split(','))) or
                        year_list!=sorted(set(year_list))): break
                count = len(year_list)
                if (isinstance(score.get('n_years'),bool) or score.get('n_years')!=count or
                        pred.get('rows')!=count or pred.get('error_count')!=count or
                        pred.get('missing_or_invalid_error_count')!=0 or
                        not np.isfinite(float(score['mae_usd'])) or float(score['mae_usd'])<0 or
                        not np.isclose(float(score['mae_usd']),float(pred['mean_absolute_error']),rtol=1e-10,atol=.01)):
                    break
                if years is not None and year_list!=years: break
                years=year_list;checked[model]=float(score['mae_usd'])
            except (ValueError,TypeError,KeyError,OverflowError): break
        if len(checked)!=3:
            lines.append(f'- Original observed-weather scores for {_text(stage)} from `{_text(source)}` lack unique, matching year-level prediction summaries; inspect the retained source artifacts.')
            continue
        w,b,r=(checked[m] for m in ('ridge_weather','last_year','ridge_prior'))
        if w>=b or w>=r:
            label=', '.join(map(str,years))
            lines.append(f'- Original observed-weather ridge ({_text(stage)}; years {label}): USD {w/1e6:,.1f}m average error versus previous-Q2 USD {b/1e6:,.1f}m and revenue-only ridge USD {r/1e6:,.1f}m; it did not beat both comparisons. Scores: `{_text(source)}`; matched-year summaries: `{_text(prediction_source)}`.')
    volume_scores = _load_score_rows(register,'iowa_v4')
    volume_diagnostics = _load_result_rows(register,'iowa_v4','diagnostic_row')
    for source in sorted({r['_register_source'] for r in volume_scores if r.get('period')=='2016-2021'}):
        score_rows=[r for r in volume_scores if r['_register_source']==source and r.get('period')=='2016-2021']
        diagnostic_source=source.rsplit('/',1)[0]+'/yearly_diagnostics_private.csv'
        details=[r for r in volume_diagnostics if r['_register_source']==diagnostic_source and r.get('period')=='2016-2021']
        models=('base','full','damped'); result={}; years=[]
        try:
            for row in details:
                year=row.get('year')
                if isinstance(year,bool) or not isinstance(year,(int,float)) or not float(year).is_integer() or not 2016<=int(year)<=2021:
                    raise ValueError('Invalid diagnostic year')
                years.append(int(year))
            if not years or len(set(years))!=len(years): raise ValueError('Missing or duplicate diagnostic year')
            for model in models:
                matching=[r for r in score_rows if r.get('model')==model]
                if len(matching)!=1: raise ValueError('Missing or duplicate score')
                score=matching[0]
                if isinstance(score.get('years'),bool) or score.get('years')!=len(years): raise ValueError('Different year count')
                errors=[]
                for row in details:
                    actual=_nonnegative_number(row.get('actual_usd'))
                    prediction=_nonnegative_number(row.get(model+'_prediction_usd'))
                    error=_nonnegative_number(row.get(model+'_absolute_error_usd'))
                    if actual is None or prediction is None or error is None or not np.isclose(abs(prediction-actual),error,rtol=1e-10,atol=.01):
                        raise ValueError('Diagnostic error mismatch')
                    errors.append(error)
                mae=float(score['mae_usd'])
                if not np.isfinite(mae) or mae<0 or not np.isclose(mae,float(np.mean(errors)),rtol=1e-10,atol=.01):
                    raise ValueError('Score mismatch')
                result[model]=mae
        except (ValueError,TypeError,KeyError,OverflowError):
            result={}
        if len(result)!=3:
            lines.append(f'- Earlier shipment scores from `{_text(source)}` lack unique, matching annual diagnostic rows; inspect the retained source artifacts.')
            continue
        label=', '.join(map(str,sorted(years)))
        vals='; '.join(f'{model} USD {result[model]/1e6:,.1f}m' for model in models)
        lines.append(f'- Earlier shipment investigation, matched years {label}: {vals}. Scores: `{_text(source)}`; annual diagnostics: `{_text(diagnostic_source)}`. These are reported separately from the four-model comparison above.')
    int_scores=_load_score_rows(register,'iowa_intercept_research')
    pick=lambda variant:[r for r in int_scores if r.get('cohort')=='full' and r.get('period')=='2022-2026' and r.get('variant')==variant and r.get('mae_musd') is not None]
    zero,half=pick('alpha_0'),pick('alpha_0.5')
    if len(zero)==1 and len(half)==1 and float(half[0]['mae_musd'])>float(zero[0]['mae_musd']):
        half_source=_text(half[0]['_register_source'])
        lines.append(f"- The separate full-history intercept sensitivity had higher average error in 2022–2026 with the half correction (USD {float(half[0]['mae_musd']):,.1f}m) than without it (USD {float(zero[0]['mae_musd']):,.1f}m). See `{half_source}`.")
    fw = _unique_research_row(register, 'iowa_fieldwork_2022',
                              'iowa_fieldwork_2022/annual_private.csv', 2022,
                              'annual_error_row')
    fw_ok = False
    if fw:
        d, source = fw
        actual = _nonnegative_number(d.get('actual_usd'))
        pairs = [('price_forecast_fieldwork', 'full_six_week_absolute_error_usd'),
                 ('one_week', 'one_week_absolute_error_usd'),
                 ('price_forecast', 'forecast_only_absolute_error_usd')]
        vals = [(_nonnegative_number(d.get(pred)), _nonnegative_number(d.get(err)))
                for pred, err in pairs]
        fw_ok = actual is not None and all(p is not None and e is not None and
                  np.isclose(abs(p-actual), e, rtol=1e-10, atol=.01) for p,e in vals)
        if fw_ok:
            lines.append(f"- In the 2022 fieldwork investigation, absolute errors were USD {vals[0][1]/1e6:,.1f}m for the six-week model, {vals[1][1]/1e6:,.1f}m for the matched one-week model, and {vals[2][1]/1e6:,.1f}m for forecast-only. Source: `{_text(source)}`. These are retrospective sensitivities.")
    if not fw_ok:
        lines.append('- The 2022 fieldwork investigation cannot be summarized by year from an unambiguous checked annual-error row in the register; inspect the retained source artifact.')
    for year in (2023, 2026):
        item = _unique_research_row(register, 'iowa_missing_fieldwork',
                                    'iowa_missing_fieldwork/paired_private.csv', year,
                                    'paired_comparison', 'price_forecast_fieldwork')
        ok = False
        if item:
            d, source = item
            prior, new = (_nonnegative_number(d.get(k)) for k in ('prior','new'))
            delta = d.get('new_minus_prior_usd')
            ok = (prior is not None and new is not None and not isinstance(delta, bool)
                  and isinstance(delta, (int,float)) and np.isfinite(delta)
                  and np.isclose(new-prior, float(delta), rtol=1e-10, atol=.01))
            if ok:
                lines.append(f'- In {year}, the longer-history fieldwork comparison had USD {prior/1e6:,.1f}m absolute error with the prior history and USD {new/1e6:,.1f}m after adding older years; the change was USD {(new-prior)/1e6:+,.1f}m. Source: `{_text(source)}`.')
        if not ok:
            lines.append(f'- The {year} longer-history fieldwork comparison lacks an unambiguous checked paired row in the register; inspect the retained source artifact.')
    sensitivity = _unique_research_row(register, 'iowa_forecast_further_2025',
                      'iowa_forecast_further_2025/annual_errors_private.csv', 2025,
                      'annual_error_row')
    sensitivity_ok = False
    if sensitivity:
        d, source = sensitivity
        values = [_nonnegative_number(d.get(k)) for k in
                  ('previous_weather_sqrt_volume','half_weather_sqrt_volume','price_sqrt_volume')]
        sensitivity_ok = all(v is not None for v in values)
        if sensitivity_ok:
            lines.append(f'- In the 2025 sensitivity investigation, absolute errors were USD {values[0]:,.1f}m with the previous weather and shipment adjustment, {values[1]:,.1f}m with the halved weather adjustment, and {values[2]:,.1f}m for price plus shipments. Source: `{_text(source)}`. These variants were chosen after inspecting historical results.')
    if not sensitivity_ok:
        lines.append('- The 2025 sensitivity investigation lacks an unambiguous checked annual-error row in the register; inspect the retained source artifact.')
    lines += ['', 'Other retained model scores, including the original observed-weather and earlier full-shipment investigations, are indexed in the experiment register. Their distinct cohorts and historical versions must be read from their source artifacts; they should not be compared as if they used these exact folds.', '',
              '## Replay and source provenance', '',
              'The strict replay records eligible, missing, after-cutoff and unverified inputs rather than filling unknown availability with later information. Counts below are input requirements, not independent observations or proof of publication.', '',
              '| Decision date | Replay status | Required inputs |', '|---|---|---:|']
    groups=rr.groupby(['decision_date','status'],dropna=False).size().reset_index(name='count')
    for row in groups.sort_values(['decision_date','status']).itertuples(index=False):
        lines.append(f'| {_text(row.decision_date)} | {_text(row.status)} | {int(row.count)} |')
    lines += ['', 'The [National Weather Service MOS glossary](https://forecast.weather.gov/glossary.php?word=model+output+statistics) describes its short-range MOS guidance package as generally available around 04:00 UTC for 00:00 forecast cycles and 16:00 UTC for 12:00 cycles. This general guidance does not establish the issuance or availability time of any specific historical IEM record.', '',
              '[FRED release-date documentation](https://fred.stlouisfed.org/docs/api/fred/releases_dates.html) says the data source’s release date does not necessarily indicate when that data became available on FRED or ALFRED. This distinction also applies when interpreting archived price-vintage evidence.', '',
              'Neither generic description establishes the input vintages or clears any unverified input in the strict replay. Generic timings are not assigned as individual publication timestamps or used to change replay statuses.', '',
              'Archived weather selections can rebuild the saved aggregates, but their original issue/publication availability and revision history are not independently established. Older price reconstructions and other saved transcriptions require source and vintage checks. A checksum checks saved bytes; it does not prove that they support the claimed value or source, or establish historical availability. Replay also depends on separately reviewed source assertions. If strict replay shows unverified or missing inputs, exact historical reconstruction remains incomplete.', '',
              '## Experiment record and interpretation', '',
              f'The register contains{len(register)} retained artifact rows spanning {register.study.nunique()} named studies. Rows include score records, individual prediction and incremental-comparison source rows, and summaries of prediction or diagnostic files; the same model may appear more than once. These counts are not the number of unique trials, and unrecorded attempts or original proposal timing may be missing. All outcomes through 2026 had been inspected by the time the register was reconstructed.', '',
              'Before recording a new proposal or result in `new_experiments_private.jsonl`, carry that log forward from saved output. The notebook does not automatically create a replacement when the previous log is absent. Its working directory is ephemeral; preserve each updated log with the notebook output and retain an independent trusted head and count when possible.', '',
              'All comparisons here are retrospective and exploratory. The historical exercise provides no fresh holdout. The pre-existing 2027 price-only prospective test remains unchanged; record its inputs and prediction at the cutoff, then evaluate only after the Q2 result is filed.', '']
    lines += ['## Sources and audit trail', '',
              '- [ALFRED real-time data documentation](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html): as-of vintages can differ from the latest revised observations. Not every saved historical price has independently verified vintage evidence.',
              '- [NOAA GFS MOS format guide](https://www.weather.gov/media/mdl/mdltpb05-03.pdf): P12 is a 12-hour precipitation probability, with the 24-hour projection ending the lead-hours 12–24 interval.',
              '- [SEC filings search](https://www.sec.gov/search-filings): original company financial and shipment disclosures and their filing dates.',
              '- [Iowa Environmental Mesonet MOS forecast data endpoint](https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py): archived model forecast records; current archive retrieval does not prove historical publication time or absence of revisions.',
              '- [Kaggle research notebook and outputs](https://www.kaggle.com/code/nicolasrster/iowa-fertilizer-revenue-price-follow-up/output): see `iowa_incremental_value`, `iowa_exploratory_ranges`, `iowa_replay`, and the experiment register for row-level evidence and results.', '']
    return '\n'.join(lines).replace(chr(36), 'USD ')


def save_research_report(report, output_dir):
    if not isinstance(report,str) or not report: raise ValueError('Empty research report')
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    path=out/'RESEARCH_REPORT.md'; path.write_text(report,encoding='utf8')
    return path
