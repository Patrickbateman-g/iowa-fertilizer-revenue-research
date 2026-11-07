from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

V3_OUTPUT = Path('/kaggle/working/iowa_v3')
V3_PRICE_SERIES = 'PCU325311325311A'
V3_FIRST_YEAR = 2016
V3_LAST_YEAR = 2026
V3_GROUPS = ('2016-2021', '2022-2026')


def _strict_date(value, label):
    try:
        parsed = pd.to_datetime(value, format='%Y-%m-%d', errors='raise')
        if pd.isna(parsed):
            raise ValueError('missing date')
        return parsed.strftime('%Y-%m-%d')
    except Exception as exc:
        raise ValueError(f'Invalid {label}: {value!r}') from exc


def _year(value):
    try:
        x = float(value)
        if isinstance(value, bool) or not np.isfinite(x) or x != int(x):
            raise ValueError
        return int(x)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'Invalid year: {value!r}') from exc


def _positive(frame, columns):
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        if (~np.isfinite(frame[column]) | (frame[column] <= 0)).any():
            raise ValueError(f'{column} must contain positive finite values')


def prepare_alt_price(extra):
    records = extra.get('alt_price', [])
    if not isinstance(records, list):
        raise ValueError('alt_price must be a list')
    if not records:
        return None
    frame = pd.DataFrame(records).copy()
    required = {'year','current_april','previous_april','as_of','vintage_label',
                'current_observation','previous_observation','source'}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError('Alternative price observations or provenance are missing')
    frame['year'] = frame.year.map(_year)
    if frame.year.duplicated().any():
        raise ValueError('Duplicate alternative price year')
    _positive(frame, ['current_april','previous_april'])
    for x in frame.itertuples():
        cutoff = f'{x.year}-05-31'
        if (_strict_date(x.as_of,'alt as_of') != cutoff or
            _strict_date(x.vintage_label,'alt vintage') != cutoff or
            _strict_date(x.current_observation,'alt current month') != f'{x.year}-04-01' or
            _strict_date(x.previous_observation,'alt previous month') != f'{x.year-1}-04-01'):
            raise ValueError(f'Invalid alternative price date in {x.year}')
        if (not str(x.source).startswith('https://alfred.stlouisfed.org/')
                or V3_PRICE_SERIES not in str(x.source)):
            raise ValueError(f'Unverified alternative price source or series in {x.year}')
        if hasattr(x, 'series_id') and x.series_id != V3_PRICE_SERIES:
            raise ValueError(f'Wrong alternative series in {x.year}')
    frame['alt_ratio'] = frame.current_april / frame.previous_april
    return frame.rename(columns={'current_april':'alt_current_april',
        'previous_april':'alt_previous_april','as_of':'alt_as_of',
        'source':'alt_source','vintage_label':'alt_vintage'}).set_index('year').sort_index()


def prepare_q1_volume(extra):
    records = extra.get('q1_volume', [])
    if not isinstance(records, list):
        raise ValueError('q1_volume must be a list')
    if not records:
        return None
    frame = pd.DataFrame(records).copy()
    required = {'year','current_q1_tons','previous_q1_tons','current_filed',
                'previous_filed','current_accession','previous_accession','units'}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError('Q1 volume observations or provenance are missing')
    frame['year'] = frame.year.map(_year)
    if frame.year.duplicated().any():
        raise ValueError('Duplicate Q1 volume year')
    _positive(frame, ['current_q1_tons','previous_q1_tons'])
    for x in frame.itertuples():
        cutoff = f'{x.year}-05-31'
        current = _strict_date(x.current_filed,'current Q1 volume disclosure')
        previous = _strict_date(x.previous_filed,'previous Q1 volume disclosure')
        if not (f'{x.year}-03-31' <= current <= cutoff):
            raise ValueError(f'Current Q1 volume unavailable by {cutoff}')
        if not (f'{x.year-1}-03-31' <= previous <= cutoff):
            raise ValueError(f'Previous Q1 volume unavailable by {cutoff}')
        if x.units not in ('product tons','thousand product tons','million product tons'):
            raise ValueError('Both volume figures must use the same product-ton units')
        if not str(x.current_accession).strip() or not str(x.previous_accession).strip():
            raise ValueError('Missing Q1 volume filing accessions')
    frame['volume_ratio'] = frame.current_q1_tons / frame.previous_q1_tons
    return frame.set_index('year').sort_index()


def evaluate_new_candidates(v2_results, extra):
    if not isinstance(v2_results, dict) or 'data' not in v2_results:
        raise ValueError('Run v2 first to create V2_RESULTS with input data')
    if not isinstance(extra, dict):
        raise ValueError('EXTRA_POINTS must be a dictionary')
    base = v2_results['data'].copy()
    needed = {'year','as_of','prior_q2_usd','target_q2_usd','price_ratio','price_as_of'}
    if not needed.issubset(base.columns):
        raise ValueError('The original price-ratio inputs are missing')
    base['year'] = base.year.map(_year)
    if base.year.duplicated().any():
        raise ValueError('Duplicate original model year')
    base = base[(base.year >= V3_FIRST_YEAR) & (base.year <= V3_LAST_YEAR) & base.price_ratio.notna()].copy()
    if base.empty:
        raise ValueError('No valid original price-ratio observations')
    _positive(base, ['prior_q2_usd','target_q2_usd','price_ratio'])
    for x in base.itertuples():
        if _strict_date(x.as_of,'base cutoff') != f'{x.year}-05-31' or _strict_date(x.price_as_of,'base vintage') != f'{x.year}-05-31':
            raise ValueError(f'Original price data lack correct as-of timing in {x.year}')
    alt, volume = prepare_alt_price(extra), prepare_q1_volume(extra)
    datasets = []
    if alt is not None:
        a = base.merge(alt[['alt_ratio','alt_as_of','alt_source','alt_vintage']],left_on='year',right_index=True,validate='one_to_one')
        datasets.append(('alternative_price', a, ['base','alt_price']))
    if volume is not None:
        b = base.merge(volume[['volume_ratio','current_filed','previous_filed','current_accession','previous_accession','units']],left_on='year',right_index=True,validate='one_to_one')
        datasets.append(('q1_volume', b, ['base','base_x_volume']))
    if alt is not None and volume is not None:
        c = base.merge(alt[['alt_ratio','alt_as_of','alt_source','alt_vintage']],left_on='year',right_index=True,validate='one_to_one').merge(
            volume[['volume_ratio','current_filed','previous_filed','current_accession','previous_accession','units']],left_on='year',right_index=True,validate='one_to_one')
        datasets.append(('both', c, ['base','alt_price','base_x_volume','alt_x_volume']))
    if not datasets or not any(not frame.empty for _,frame,_ in datasets):
        raise ValueError('No additional data overlap the established price model')
    records, context = [], []
    for cohort, frame, models in datasets:
        for x in frame.itertuples():
            prediction = {
                'base': x.prior_q2_usd*x.price_ratio,
                'alt_price': x.prior_q2_usd*getattr(x,'alt_ratio',np.nan),
                'base_x_volume': x.prior_q2_usd*x.price_ratio*getattr(x,'volume_ratio',np.nan),
                'alt_x_volume': x.prior_q2_usd*getattr(x,'alt_ratio',np.nan)*getattr(x,'volume_ratio',np.nan),
            }
            context.append({'year':x.year,'cohort':cohort,'original_price_ratio':x.price_ratio,
                'alternative_price_ratio':getattr(x,'alt_ratio',None),'q1_volume_ratio':getattr(x,'volume_ratio',None),
                'price_as_of':x.price_as_of,'alt_vintage':getattr(x,'alt_vintage',None),
                'volume_current_filed':getattr(x,'current_filed',None),'volume_previous_filed':getattr(x,'previous_filed',None),
                'volume_current_accession':getattr(x,'current_accession',None),'volume_previous_accession':getattr(x,'previous_accession',None)})
            for model in models:
                value=float(prediction[model])
                if not np.isfinite(value):
                    raise ValueError(f'Invalid {model} prediction in {x.year}')
                actual=float(x.target_q2_usd)
                records.append({'year':x.year,'cohort':cohort,'period':'2016-2021' if x.year<=2021 else '2022-2026',
                    'model':model,'actual_usd':actual,'prediction_usd':value,'signed_error_usd':value-actual,
                    'absolute_error_usd':abs(value-actual),'absolute_pct_error':100*abs(value-actual)/actual})
    predictions = pd.DataFrame(records)
    if predictions.empty:
        raise ValueError('No common years for the requested comparisons')
    for cohort, g in predictions.groupby('cohort'):
        if len({frozenset(h.year) for _,h in g.groupby('model')}) != 1:
            raise AssertionError(f'Unmatched forecast years: {cohort}')
    per = predictions.groupby(['cohort','period','model'],as_index=False).agg(
        n_years=('year','size'),mae_usd=('absolute_error_usd','mean'),mape_pct=('absolute_pct_error','mean'))
    all_years=predictions.assign(period='all').groupby(['cohort','period','model'],as_index=False).agg(
        n_years=('year','size'),mae_usd=('absolute_error_usd','mean'),mape_pct=('absolute_pct_error','mean'))
    scores=pd.concat([per,all_years],ignore_index=True)
    return predictions,scores,pd.DataFrame(context)


def render_v3_report(pred, scores, extra):
    names={'base':'Frozen nitrogen-price ratio','alt_price':'Alternative producer price ratio (PCU325311325311A)',
           'base_x_volume':'Frozen price ratio × Q1 volume ratio','alt_x_volume':'Alternative producer price ratio × Q1 volume ratio'}
    report=['# Iowa fertilizer revenue: exploratory new-data checks','',
        'All years through 2026 have already been inspected. These are post-hoc comparisons, not an untouched test; '
        'all models in each cohort use exactly the same years. The original April WPU065201 price-ratio candidate remains frozen.','',
        '## Matched historical results','','MAE is mean absolute error in USD millions. Lower is better.','']
    for cohort in ('alternative_price','q1_volume','both'):
        subset=scores[scores.cohort==cohort]
        if subset.empty:
            continue
        report += [f'### {cohort.replace("_"," ").title()}','','| Period | Model | Years | MAE (USD m) | MAPE |',
                   '|---|---|---:|---:|---:|']
        for period in (*V3_GROUPS,'all'):
            for model in names:
                row=subset[(subset.period==period)&(subset.model==model)]
                if len(row)==1:
                    s=row.iloc[0]
                    report.append(f'| {period} | {names[model]} | {int(s.n_years)} | {s.mae_usd/1e6:,.1f} | {s.mape_pct:.1f}% |')
        report += ['']
        for candidate in (x for x in subset.model.unique() if x!='base'):
            enough=True; better=True
            for period in V3_GROUPS:
                a=subset[(subset.period==period)&(subset.model=='base')]
                b=subset[(subset.period==period)&(subset.model==candidate)]
                if len(a)!=1 or len(b)!=1 or int(a.iloc[0].n_years)<3 or int(b.iloc[0].n_years)<3:
                    enough=False; break
                better &= b.iloc[0].mae_usd < a.iloc[0].mae_usd
            verdict=('lower error in both periods' if better else 'did not reduce error in both periods') if enough else 'insufficient matched years in both periods'
            report.append(f'- **{names[candidate]}:** {verdict}. This comparison is exploratory and cannot validate future performance.')
        report += ['']
    report += ['## Limitations and next steps','',
      '- Alternative price values use the same May 31 ALFRED vintage for both April values; the nitrogen index is a national proxy and is not CF’s realized price.',
      '- Q1 sales volume is for product tons, not Iowa-specific corn fertilizer sales; its use assumes Q1 volume changes are informative about Q2 volume changes.',
      '- Simple formulas have no fitted parameters, but their comparison and any selection occurred after the historical outcomes were known.',
      '- Features may cover different years. Compare only rows within the same cohort, and treat small samples and acquisitions/plant outages cautiously.',
      '- A real new test requires a future season such as 2027: archive inputs and the candidate before observing Q2 results.',
      '- Year-by-year actuals, predictions, errors and feature provenance remain in the output files.','']
    return '\n'.join(report)


def run_v3(v2_results, extra):
    pred, scores, context=evaluate_new_candidates(v2_results,extra)
    V3_OUTPUT.mkdir(parents=True,exist_ok=True)
    pred.to_csv(V3_OUTPUT/'predictions_private.csv',index=False)
    scores.to_csv(V3_OUTPUT/'scores_private.csv',index=False)
    context.to_csv(V3_OUTPUT/'provenance_private.csv',index=False)
    text=render_v3_report(pred,scores,extra)
    (V3_OUTPUT/'REPORT.md').write_text(text,encoding='utf-8')
    print(f'New-data analysis complete; outputs:{V3_OUTPUT}')
    return {'predictions':pred,'scores':scores,'context':context,'report':text}


if __name__ == '__main__':
    if 'V2_RESULTS' not in globals() or 'EXTRA_POINTS' not in globals():
        raise RuntimeError('Run the v2 notebook cells and define EXTRA_POINTS first')
    V3_RESULTS=run_v3(V2_RESULTS,EXTRA_POINTS)
    try:
        from IPython.display import display, Markdown
        display(Markdown(V3_RESULTS['report']))
    except ImportError:
        print(V3_RESULTS['report'])
