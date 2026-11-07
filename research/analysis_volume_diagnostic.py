from pathlib import Path

import numpy as np
import pandas as pd

V4_OUTPUT = Path('/kaggle/working/iowa_v4')


def diagnose_volume(v3_results):
    pred = v3_results['predictions'].copy()
    context = v3_results['context'].copy()
    pred = pred[(pred.cohort == 'q1_volume') & pred.model.isin(['base', 'base_x_volume'])].copy()
    context = context[context.cohort == 'q1_volume'].copy()
    required = {'year', 'actual_usd', 'prediction_usd', 'model'}
    if pred.empty or not required.issubset(pred.columns):
        raise ValueError('Missing v3 volume cohort predictions')
    if pred.duplicated(['year', 'model']).any() or context.year.duplicated().any():
        raise ValueError('Duplicate volume cohort year')
    years = pred.groupby('model').year.agg(lambda x: frozenset(x))
    if set(years.index) != {'base', 'base_x_volume'} or years['base'] != years['base_x_volume']:
        raise ValueError('Models do not use matching years')
    if not set(pred.year).issubset(range(2016, 2027)):
        raise ValueError('Diagnostic accepts only the already observed 2016--2026 years')
    base = pred[pred.model == 'base'][['year', 'actual_usd', 'prediction_usd']].rename(
        columns={'prediction_usd': 'base_prediction_usd'})
    full = pred[pred.model == 'base_x_volume'][['year', 'actual_usd', 'prediction_usd']].rename(
        columns={'actual_usd': 'full_actual_usd', 'prediction_usd': 'full_prediction_usd'})
    d = base.merge(full, on='year', validate='one_to_one').merge(
        context[['year', 'q1_volume_ratio']], on='year', validate='one_to_one').sort_values('year')
    if len(d) != len(years['base']) or not np.allclose(d.actual_usd, d.full_actual_usd):
        raise ValueError('Missing volume context or mismatched actual sales')
    numerical = d[['actual_usd', 'base_prediction_usd', 'full_prediction_usd', 'q1_volume_ratio']]
    if not np.isfinite(numerical.to_numpy(dtype=float)).all() or (numerical <= 0).any().any():
        raise ValueError('Inputs must be positive and finite')
    if not np.allclose(d.full_prediction_usd, d.base_prediction_usd * d.q1_volume_ratio):
        raise ValueError('V3 full-volume prediction does not match its inputs')
    d = d.drop(columns='full_actual_usd')
    d['damped_prediction_usd'] = d.base_prediction_usd * np.sqrt(d.q1_volume_ratio)
    for name in ('base', 'full', 'damped'):
        d[f'{name}_signed_error_usd'] = d[f'{name}_prediction_usd'] - d.actual_usd
        d[f'{name}_absolute_error_usd'] = d[f'{name}_signed_error_usd'].abs()
    d['volume_effect_usd'] = d.full_prediction_usd - d.base_prediction_usd
    d['full_extra_absolute_error_usd'] = d.full_absolute_error_usd - d.base_absolute_error_usd
    d['period'] = np.where(d.year <= 2021, '2016-2021', '2022-2026')
    results = []
    for period, group in [('2016-2021', d[d.period == '2016-2021']),
                          ('2022-2026', d[d.period == '2022-2026']), ('all', d)]:
        if group.empty:
            continue
        for model in ('base', 'full', 'damped'):
            results.append({'period': period, 'model': model, 'years': len(group),
                            'mae_usd': group[f'{model}_absolute_error_usd'].mean(),
                            'mape_pct': (100 * group[f'{model}_absolute_error_usd'] / group.actual_usd).mean()})
    return d, pd.DataFrame(results)


def render_volume_diagnostic(d, scores):
    labels = {'base': 'Frozen price', 'full': 'Full Q1 volume adjustment',
              'damped': '50%-damped Q1 volume (experimental)'}
    text = ['# Why Q1 shipment volume changed historical errors', '',
            'All 2016–2026 outcomes were examined before this investigation. These results are exploratory, not an untouched or prospective test. '
            'The frozen price forecast remains the candidate for a future test. The damped formula was specified as a sensitivity check after the original results were known.', '',
            'The full adjustment assumes Q2 sales volume changes in the same proportion as Q1; dampening uses the square root of that Q1 ratio. '
            'Neither relationship is guaranteed. Positive signed error means predicted revenue exceeded actual revenue.', '',
            '## Matched historical comparisons', '', '| Period | Model | Years | Mean absolute error ($m) |',
            '|---|---|---:|---:|']
    for x in scores.itertuples():
        text.append(f'| {x.period} | {labels[x.model]} | {x.years} | {x.mae_usd/1e6:,.1f} |')
    early = d[d.period == '2016-2021']
    if not early.empty:
        worse = early[early.full_extra_absolute_error_usd > 0]
        better = early[early.full_extra_absolute_error_usd < 0]
        text += ['', f'The full-volume adjustment raised absolute error in {len(worse)} of {len(early)} early years and lowered it in {len(better)}. '
                 'The yearly table below shows both signed errors and the increase or decrease in absolute error.']
    if not early.empty and (early.year == 2016).any():
        change = early.full_extra_absolute_error_usd.sum()
        y16 = early.loc[early.year == 2016, 'full_extra_absolute_error_usd'].iloc[0]
        text += ['', f'The sum of full-volume minus frozen-model absolute errors over 2016–2021 was USD {change/1e6:,.1f} million. '
                 f'The 2016 contribution was USD {y16/1e6:,.1f} million' +
                 (f' ({100*y16/change:,.1f}% of that signed net difference).' if abs(change) > 1e-9 else '; the net difference was zero, so its share is undefined.')]
        text += ['A share can exceed 100% if other years partly offset 2016; this is a descriptive decomposition, not causation.']
    text += ['', '## Interpretation and source limits', '',
             '- CF reported that adverse weather delayed the 2018 spring application season and shifted some shipments from Q1 to Q2. '
             'Q1 volume declined year over year while Q2 volume rose. This Q2 filing is an after-the-fact explanation, not a forecast input: '
             'https://www.sec.gov/Archives/edgar/data/1324404/000132440418000019/cf-06302018x10q.htm',
             '- The 2015 acquisition of the remaining UK business and 2016 capacity changes complicate year-over-year comparisons; they do not prove the size or direction of any forecast error: '
             'https://www.cfindustries.com/globalassets/cf-industries/media/documents/reports/annual-reports/cfi_2016-annual-report.pdf',
             '- Q1 shipment volumes are company-wide product tons, spanning different product mixes, geographies and customers; they are not Iowa corn use. The national price index is not CF’s realized price.',
             '- The damped factor is an arbitrary sensitivity check and was considered after inspecting results. Its historical score cannot establish that it will improve future forecasts. '
             'A real prospective comparison must fix and archive the rules before future Q2 outcomes are known.',
             '- The detailed year-by-year rows and errors remain in /kaggle/working/iowa_v4.', '']
    return '\n'.join(text)


def run_volume_diagnostic(v3_results):
    details, scores = diagnose_volume(v3_results)
    report = render_volume_diagnostic(details, scores)
    V4_OUTPUT.mkdir(parents=True, exist_ok=True)
    details.to_csv(V4_OUTPUT / 'yearly_diagnostics_private.csv', index=False)
    scores.to_csv(V4_OUTPUT / 'scores_private.csv', index=False)
    (V4_OUTPUT / 'REPORT.md').write_text(report, encoding='utf-8')
    print(f'Volume diagnostic complete; outputs:{V4_OUTPUT}')
    return {'details': details, 'scores': scores, 'report': report}


if __name__ == '__main__':
    if 'V3_RESULTS' not in globals():
        raise RuntimeError('Run the earlier notebook cells to create V3_RESULTS first')
    V4_RESULTS = run_volume_diagnostic(V3_RESULTS)
    try:
        from IPython.display import display, Markdown
        display(Markdown(V4_RESULTS['report']))
        columns = ['year', 'q1_volume_ratio', 'actual_usd', 'base_prediction_usd',
                   'full_prediction_usd', 'damped_prediction_usd', 'base_signed_error_usd',
                   'full_signed_error_usd', 'full_extra_absolute_error_usd']
        display(V4_RESULTS['details'][columns].round(3))
    except ImportError:
        print(V4_RESULTS['report'])
