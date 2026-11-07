from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

V2_OUT = Path('/kaggle/working/iowa_v2')
V2_MIN_TRAIN = 6
V2_FIRST_TEST = 2012
V2_LAST_TEST = 2026


def _date(value, label):
    try:
        parsed = pd.to_datetime(value, format='%Y-%m-%d', errors='raise')
        if pd.isna(parsed):
            raise ValueError('missing date')
        return parsed.strftime('%Y-%m-%d')
    except Exception as exc:
        raise ValueError(f'Invalid {label}: {value!r}') from exc


def _year_value(value, label):
    if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) != int(float(value)):
        raise ValueError(f'Invalid {label}: {value!r}')
    return int(value)


def prepare_q1(snapshot):
    data = snapshot.get('q1_revenue')
    if not isinstance(data, list) or not data:
        raise ValueError('Snapshot requires a nonempty q1_revenue list')
    q1 = pd.DataFrame(data).copy()
    required = {'year', 'q1_usd', 'q1_filed', 'q1_accession', 'q1_tag'}
    if not required.issubset(q1.columns) or q1[list(required)].isna().any().any():
        raise ValueError('Q1 revenue fields are missing')
    q1['year'] = q1['year'].map(lambda v: _year_value(v, 'Q1 year'))
    if q1.year.duplicated().any():
        raise ValueError('Duplicate Q1 revenue years')
    q1['q1_usd'] = pd.to_numeric(q1.q1_usd, errors='raise')
    if (~np.isfinite(q1.q1_usd) | (q1.q1_usd <= 0)).any():
        raise ValueError('Q1 sales must be finite and positive')
    q1['q1_filed'] = q1.q1_filed.map(lambda v: _date(v, 'Q1 filing'))
    for row in q1.itertuples():
        if row.q1_filed < f'{row.year}-03-31' or row.q1_filed > f'{row.year + 1}-05-31':
            raise ValueError(f'Q1 filing outside supported window for {row.year}')
        if not str(row.q1_accession).strip() or not str(row.q1_tag).strip():
            raise ValueError(f'Missing Q1 source identifiers for {row.year}')
    return q1.set_index('year').sort_index()


def prepare_price(snapshot):
    data = snapshot.get('price_index')
    if not data:
        return None
    if not isinstance(data, list):
        raise ValueError('price_index must be a list')
    df = pd.DataFrame(data).copy()
    required = {'year', 'current_april', 'previous_april', 'as_of',
                'current_observation', 'previous_observation', 'source', 'vintage_label'}
    if not required.issubset(df.columns) or df[list(required)].isna().any().any():
        raise ValueError('Price index requires observations, source and as-of vintage label')
    df['year'] = df.year.map(lambda v: _year_value(v, 'price year'))
    if df.year.duplicated().any():
        raise ValueError('Duplicate price-index years')
    for col in ('current_april', 'previous_april'):
        df[col] = pd.to_numeric(df[col], errors='raise')
        if (~np.isfinite(df[col]) | (df[col] <= 0)).any():
            raise ValueError('Price observations must be finite and positive')
    for row in df.itertuples():
        cutoff = f'{row.year}-05-31'
        if _date(row.as_of, 'price as_of') != cutoff or _date(row.vintage_label, 'vintage label') != cutoff:
            raise ValueError(f'Wrong price vintage cutoff in {row.year}')
        if not str(row.source).startswith('https://alfred.stlouisfed.org/'):
            raise ValueError(f'Unverified ALFRED source in {row.year}')
        if (_date(row.current_observation, 'current observation') != f'{row.year}-04-01'
                or _date(row.previous_observation, 'previous observation') != f'{row.year-1}-04-01'):
            raise ValueError(f'Wrong April observations in {row.year}')
    df['price_ratio'] = df.current_april / df.previous_april
    return df.set_index('year').sort_index()


def build_v2_data(snapshot):
    base_rev, weather, meta = read_snapshot(snapshot)
    base = make_dataset(base_rev, weather)
    q1 = prepare_q1(snapshot)
    price = prepare_price(snapshot)
    rows, excluded = [], []
    for row in base.to_dict('records'):
        year = int(row['year'])
        cutoff = row['as_of']
        if year not in q1.index or year - 1 not in q1.index:
            excluded.append((year, 'missing Q1 pair'))
            continue
        current, prev = q1.loc[year], q1.loc[year-1]
        if current.q1_filed > cutoff or prev.q1_filed > cutoff:
            excluded.append((year, 'Q1 not available by cutoff'))
            continue
        row.update(q1_usd=float(current.q1_usd), prior_q1_usd=float(prev.q1_usd),
                   q1_filed=current.q1_filed, prior_q1_filed=prev.q1_filed,
                   q1_accession=current.q1_accession, prior_q1_accession=prev.q1_accession)
        if price is not None and year in price.index:
            p = price.loc[year]
            row.update(price_ratio=float(p.price_ratio), current_april=float(p.current_april),
                       previous_april=float(p.previous_april), price_as_of=p.as_of,
                       price_vintage_label=p.vintage_label, price_source=p.source)
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError('No complete, point-in-time compatible annual rows')
    result = result.sort_values('year').reset_index(drop=True)
    if result.year.duplicated().any():
        raise ValueError('Duplicate model years')
    return result, excluded, price is not None, meta


def nitrogen_price_nowcast(previous_q2_usd, current_april_index, previous_april_index):
    vals = np.asarray([previous_q2_usd, current_april_index, previous_april_index], dtype=float)
    if not np.isfinite(vals).all() or (vals <= 0).any():
        raise ValueError('Sales and both nitrogen indices must be finite and positive')
    return float(vals[0] * vals[1] / vals[2])


def run_v2_predictions(data, use_price):
    rows = []
    for _, row in data.iterrows():
        year = int(row.year)
        if not V2_FIRST_TEST <= year <= V2_LAST_TEST:
            continue
        train = data[(data.year < year) & (data.target_filed <= row.as_of)]
        if len(train) < V2_MIN_TRAIN:
            continue
        financial = {
            'previous_q2': float(row.prior_q2_usd),
            'original_weather': ridge_predict(train, row, ['prior_q2_usd', 'rain_mm', 'heavy_rain_days']),
            'q1_ratio': float(row.prior_q2_usd * row.q1_usd / row.prior_q1_usd),
            'q1_ridge': ridge_predict(train, row, ['prior_q2_usd', 'q1_usd']),
        }
        collections = [('financial', financial, len(train))]
        if use_price and pd.notna(row.get('price_ratio')):
            ratio = {**financial, 'price_ratio':nitrogen_price_nowcast(row.prior_q2_usd, row.current_april, row.previous_april)}
            collections.append(('price_ratio', ratio, len(train)))
            price_train = train[train.price_ratio.notna()]
            if len(price_train) >= V2_MIN_TRAIN:
                fitted = {**ratio,
                    'q1_price_ridge':ridge_predict(price_train, row, ['prior_q2_usd', 'q1_usd', 'price_ratio']),
                    'q1_price_weather_ridge':ridge_predict(price_train, row, ['prior_q2_usd', 'q1_usd', 'price_ratio', 'rain_mm', 'heavy_rain_days'])}
                collections.append(('price_fitted', fitted, len(price_train)))
        stage = 'development_2012_2021' if year <= 2021 else 'previously_inspected_2022_2026'
        for cohort, values, price_train_n in collections:
            for model, pred in values.items():
                if not np.isfinite(pred):
                    raise ValueError(f'Non-finite prediction for {model} in {year}')
                actual = float(row.target_q2_usd)
                err = pred - actual
                rows.append({'year':year, 'cohort':cohort, 'stage':stage, 'model':model,
                    'financial_training_rows':len(train),
                    'price_training_rows':price_train_n if cohort == 'price_fitted' else None,
                    'target_usd':actual, 'prediction_usd':pred, 'signed_error_usd':err,
                    'absolute_error_usd':abs(err), 'absolute_pct_error':100*abs(err)/actual})
    pred = pd.DataFrame(rows)
    if pred.empty:
        raise ValueError('Insufficient complete past data for a rolling comparison')
    for cohort, group in pred.groupby('cohort'):
        sets = [frozenset(g.year) for _, g in group.groupby('model')]
        if len(set(sets)) != 1:
            raise AssertionError(f'Model cohorts differ: {cohort}')
    return pred


def study_report(data, pred, scores, excluded, use_price, snapshot):
    labels = {'previous_q2':'Previous Q2', 'original_weather':'Original weather ridge',
              'q1_ratio':'Q1 ratio', 'q1_ridge':'Q1 + previous Q2 ridge',
              'price_ratio':'Nitrogen price ratio', 'q1_price_ridge':'Q1 + price ridge',
              'q1_price_weather_ridge':'Q1 + price + weather ridge'}
    stages = {'development_2012_2021':'2012–2021 development',
              'previously_inspected_2022_2026':'2022–2026 previously inspected',
              'all_2012_2026':'All years in cohort'}
    lines = ['# CF Q2 revenue: follow-up investigation', '',
             '**Status:** Exploratory historical comparison. Every year through 2026 was already inspected, '
             'so no period below is an untouched final test. No candidate has been established as a reliable improvement.', '',
             '## What was tested', '',
             'Each forecast is made on May 31. The previous Q2 benchmark and original rainfall model are compared '
             'with fixed models using current Q1 company sales, and, where available with validated historical '
             'vintages, nitrogen producer prices. The ratio models scale previous Q2 sales by the year-on-year '
             'change in Q1 sales or April nitrogen prices. This is a nowcast, not a forecast made before Q2 begins.', '',
             '## Time-ordered results', '',
             'MAE is mean absolute error; MAPE is mean absolute percentage error. Lower is better. '
             'Within each cohort, all candidates use the same eligible years. Forecasts for each year use only earlier training '
             'labels filed by that year’s May 31.', '',
             ]
    cohort_names = {'financial':'Financial models, all available years',
                    'price_ratio':'Financial and price-ratio models, matched price-available years',
                    'price_fitted':'All models including fitted price models, matched later years'}
    for cohort in ('financial', 'price_ratio', 'price_fitted'):
        if cohort not in set(pred.cohort):
            continue
        lines += ['', f"### {cohort_names[cohort]}", '',
                  '| Period | Model | Years | MAE (USD m) | MAPE |', '|---|---|---:|---:|---:|']
        for stage in stages:
            for model in labels:
                score = scores[(scores.cohort == cohort) & (scores.stage == stage) & (scores.model == model)]
                if len(score) == 1:
                    r = score.iloc[0]
                    lines.append(f'| {stages[stage]} | {labels[model]} | {int(r.n_years)} | {r.mae_usd/1e6:,.1f} | {r.mape_pct:.1f}% |')
    lines += ['', '## Why the original errors were large', '',
              'The earlier baseline carried last year’s Q2 revenue into the next year. This could not '
              'anticipate abrupt company-wide shifts. CF reported that Q1 net sales rose from USD 1,048m '
              'in 2021 to USD 2,868m in 2022. In Q1 2023, CF reported net sales of USD 2,012m '
              'and its UAN segment had a 28% lower average selling price per ton than Q1 2022, reflecting higher global '
              'supply availability and lower energy costs. These company disclosures offer a plausible '
              'explanation of the large errors; this is descriptive and does not establish causation.',
              'Sources: [CF Q1 2022 filing](https://www.sec.gov/Archives/edgar/data/1324404/000132440422000016/cf-20220331.htm), '
              '[CF Q1 2023 filing](https://www.sec.gov/Archives/edgar/data/1324404/000132440423000012/cf-20230331.htm).', '',
              'Signed error below is predicted minus actual; negative means underprediction.']
    for y in (2022, 2023):
        yrows = pred[(pred.cohort == 'financial') & (pred.year == y) & pred.model.isin(['previous_q2','original_weather','q1_ratio','q1_ridge'])]
        if not yrows.empty:
            parts = [f'{labels[r.model]} {r.signed_error_usd/1e6:+,.1f}' for r in yrows.itertuples()]
            lines.append(f'- **{y} signed errors (USD m):** ' + '; '.join(parts) + '.')
    lines += ['', '## Coverage and limits', '',
              f'- Complete common feature rows: {len(data)}; time-ordered prediction years: '
              f'{min(pred.year)}–{max(pred.year)}. Omitted years/reasons: {excluded or "none"}.',
              '- Q1 is taken from an exact January–March company-wide fact filed by May 31; '
              'the company serves other regions, crops and industrial customers.',
              ('- Nitrogen producer prices use the ALFRED graph’s May 31 as-of vintage and exact current/prior April dates; '
               'individual revision dates were not exposed. The index is a national proxy, not CF’s realized prices.' if use_price else
               '- Nitrogen producer prices were excluded because a complete, validated point-in-time '
               'price snapshot was not supplied.'),
              '- Each ridge uses the existing fixed penalty and fold-only scaling; all parameters are '
              're-estimated at each historical prediction date. Changes in weather effects or Q1/Q2 '
              'seasonality can still cause errors. The price-equipped models have fewer training years; the weather-plus-price model has five predictors and can overfit badly.',
              '- Historical NASA weather may have been revised; original weather availability was not reconstructed. '
              'Small samples, structural shifts and trying several candidates limit the strength of any comparison.',
              '- The 2007 Q1 figure is the initially reported 447.7m; a later correction was unavailable at the 2007 cutoff. The 2005 Q1 value is a pre-IPO predecessor figure from a 2006 annual filing; the 2009 Q1 value is from an earnings release.\n'
              '- The simple nitrogen-price ratio is a retrospective candidate, not a proven forecasting model: its selection used these already-seen data. Freeze it before evaluating a future season. If the April index vintage is missing at the cutoff, use the previous-Q2 baseline and record the fallback.\n'
              '- The already-seen 2022–2026 outcomes are not a fresh holdout. Freeze a candidate and evaluate '
              'on a future season such as 2027 before claiming prospective improvement.',
              '- Per-year predictions and signed errors are in `predictions_private.csv`. '
              'Do not upload datasets or generated results to GitHub.', '']
    return '\n'.join(lines)


def run_followup(snapshot):
    if isinstance(snapshot, str):
        snapshot = json.loads(snapshot)
    if not isinstance(snapshot, dict):
        raise ValueError('SNAPSHOT_JSON must be a dict or JSON string')
    data, excluded, use_price, metadata = build_v2_data(snapshot)
    pred = run_v2_predictions(data, use_price)
    scores = pred.groupby(['cohort','stage','model'], as_index=False).agg(
        n_years=('year','size'), mae_usd=('absolute_error_usd','mean'),
        mape_pct=('absolute_pct_error','mean'))
    overall = pred.assign(stage='all_2012_2026').groupby(['cohort','stage','model'], as_index=False).agg(
        n_years=('year','size'), mae_usd=('absolute_error_usd','mean'),
        mape_pct=('absolute_pct_error','mean'))
    scores = pd.concat([scores, overall], ignore_index=True)
    V2_OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(V2_OUT / 'annual_features_private.csv', index=False)
    pred.to_csv(V2_OUT / 'predictions_private.csv', index=False)
    scores.to_csv(V2_OUT / 'scores_private.csv', index=False)
    notes = {'snapshot_metadata':metadata,'excluded_years':excluded,'uses_price_vintages':use_price,
             'evaluated_years':sorted(set(pred.year.astype(int))),
             'warning':'Historical outcomes to 2026 already inspected; no untouched test.',
             'fixed_models':sorted(set(pred.model)), 'minimum_training_rows':V2_MIN_TRAIN}
    (V2_OUT / 'provenance_private.json').write_text(json.dumps(notes, indent=2, default=str), encoding='utf-8')
    report = study_report(data, pred, scores, excluded, use_price, snapshot)
    (V2_OUT / 'REPORT.md').write_text(report, encoding='utf-8')
    print(f'Follow-up complete: {len(set(pred.year))} financial comparison years, {len(set(pred.model))} fixed candidates across matched cohorts. Outputs:{V2_OUT}')
    return {'data':data, 'predictions':pred, 'scores':scores, 'report':report, 'uses_price':use_price}


if __name__ == '__main__':
    if 'SNAPSHOT_JSON' not in globals():
        raise RuntimeError('Run the original two Kaggle cells first, then this follow-up cell')
    V2_RESULTS = run_followup(SNAPSHOT_JSON)
    try:
        from IPython.display import display, Markdown
        display(Markdown(V2_RESULTS['report']))
    except ImportError:
        print(V2_RESULTS['report'])
