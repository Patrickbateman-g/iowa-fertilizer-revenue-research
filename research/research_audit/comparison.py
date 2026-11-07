from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

MODELS = ('price_only', 'price_weather', 'price_shipments', 'price_weather_shipments')
REQUIRED = ('year', 'as_of', 'target_filed', 'target_q2_usd', 'prior_q2_usd',
            'price_ratio', 'forecast_mean_p12_pct')


def _date(value):
    if not isinstance(value, str):
        raise ValueError('Expected ISO date')
    try:
        result = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError('Invalid ISO date') from exc
    if result.isoformat() != value:
        raise ValueError('Expected YYYY-MM-DD')
    return result


def validate_annual(frame):
    if not isinstance(frame, pd.DataFrame) or not set(REQUIRED).issubset(frame):
        raise ValueError('Missing annual input columns')
    data = frame.loc[:, list(REQUIRED)].copy()
    years = pd.to_numeric(data.year, errors='raise')
    if years.isna().any() or not np.isfinite(years).all() or not np.equal(years, np.floor(years)).all():
        raise ValueError('Invalid year')
    data['year'] = years.astype(int)
    if data.year.duplicated().any():
        raise ValueError('Duplicate annual year')
    for col in ('target_q2_usd', 'prior_q2_usd', 'price_ratio', 'forecast_mean_p12_pct'):
        data[col] = pd.to_numeric(data[col], errors='raise')
        if not np.isfinite(data[col]).all():
            raise ValueError(f'Missing or non-finite {col}')
    if (data[['target_q2_usd', 'prior_q2_usd', 'price_ratio']] <= 0).any().any():
        raise ValueError('Nonpositive revenue or price ratio')
    if not data.forecast_mean_p12_pct.between(0, 100).all():
        raise ValueError('Forecast probability outside 0–100')
    for row in data.itertuples(index=False):
        if _date(row.as_of) != dt.date(row.year, 5, 31):
            raise ValueError('Wrong decision date')
        if _date(row.target_filed) <= dt.date(row.year, 6, 30):
            raise ValueError('Q2 revenue release before quarter end')
    return data.sort_values('year').reset_index(drop=True)


def validate_shipments(records):
    result = {}
    fields = {'year', 'current_q1_tons', 'previous_q1_tons', 'units',
              'current_filed', 'previous_filed', 'current_source', 'previous_source'}
    for item in records:
        if not isinstance(item, dict) or not fields <= set(item):
            raise ValueError('Incomplete shipment record')
        y = item['year']
        if isinstance(y, bool) or not isinstance(y, (int, np.integer)) or y in result:
            raise ValueError('Invalid or duplicate shipment year')
        y = int(y)
        if item['units'] != 'thousand product tons':
            raise ValueError('Unexpected shipment units')
        try:
            current = float(item['current_q1_tons'])
            prior = float(item['previous_q1_tons'])
        except (ValueError, TypeError) as exc:
            raise ValueError('Invalid shipment volume') from exc
        if not all(np.isfinite(x) and x > 0 for x in (current, prior)):
            raise ValueError('Invalid shipment volume')
        if not (dt.date(y, 3, 31) < _date(item['current_filed']) <= dt.date(y, 5, 31)
                and dt.date(y-1, 3, 31) < _date(item['previous_filed']) <= dt.date(y-1, 5, 31)):
            raise ValueError(f'Shipment release after cutoff or before quarter end: {y}')
        for col in ('current_source', 'previous_source'):
            u = urlsplit(item[col])
            if u.scheme != 'https' or u.hostname != 'www.sec.gov' or not u.path.startswith('/Archives/edgar/data/1324404/'):
                raise ValueError('Unexpected shipment source')
        result[y] = dict(item, shipment_ratio=current/prior)
    return result


def _forecast_prediction(train, test):
    x = train.forecast_mean_p12_pct.to_numpy(float)
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        raise ValueError('No training weather variation')
    z = (x-mu)/sd
    response = np.log(train.target_q2_usd.to_numpy(float) /
                      (train.prior_q2_usd.to_numpy(float)*train.price_ratio.to_numpy(float)))
    if not np.isfinite(response).all():
        raise ValueError('Invalid training response')
    beta = float((z @ response)/(z @ z+1.0))
    exponent = float((test.forecast_mean_p12_pct-mu)/sd*beta)
    if not np.isfinite(exponent) or abs(exponent) > 700:
        raise ValueError('Invalid forecast exponent')
    baseline = float(test.prior_q2_usd*test.price_ratio)
    value = baseline*math.exp(exponent)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Invalid weather prediction')
    return value, mu, sd, beta


def compare_incremental(annual, shipment_records, years=range(2016, 2027), min_train=6,
                        price_reference=None, weather_reference=None, shipment_reference=None):
    data = validate_annual(annual)
    volumes = validate_shipments(shipment_records)
    years = list(years)
    if not years or any(isinstance(y, bool) or not isinstance(y, (int, np.integer)) for y in years) or len(set(years)) != len(years):
        raise ValueError('Test years must be unique integers')
    if isinstance(min_train, bool) or not isinstance(min_train, int) or min_train < 1:
        raise ValueError('Invalid minimum training size')
    result = []
    for y in years:
        picked = data[data.year.eq(y)]
        if len(picked) != 1 or y not in volumes:
            raise ValueError(f'Missing matched annual or shipment input for {y}')
        row = next(picked.itertuples(index=False))
        cutoff = row.as_of
        train = data[(data.year < y) & (data.target_filed <= cutoff)].sort_values('year')
        if len(train) < min_train:
            raise ValueError(f'Fewer than {min_train} earlier filed training years for {y}')
        weather, mu, sd, beta = _forecast_prediction(train, row)
        price = float(row.prior_q2_usd*row.price_ratio)
        volume_factor = math.sqrt(volumes[y]['shipment_ratio'])
        predictions = {'price_only':price, 'price_weather':weather,
                       'price_shipments':price*volume_factor,
                       'price_weather_shipments':weather*volume_factor}
        for model, prediction in predictions.items():
            actual = float(row.target_q2_usd)
            if not np.isfinite(prediction) or prediction <= 0:
                raise ValueError('Invalid prediction')
            result.append({'year':int(y), 'decision_date':cutoff, 'model':model,
                           'target_filed':row.target_filed, 'actual_usd':actual,
                           'prediction_usd':prediction, 'signed_error_usd':prediction-actual,
                           'absolute_error_usd':abs(prediction-actual),
                           'training_years':len(train),
                           'training_year_list':','.join(map(str, train.year)),
                           'price_baseline_usd':price,
                           'forecast_mean_p12_pct':float(row.forecast_mean_p12_pct),
                           'weather_training_mean':mu, 'weather_training_sd':sd,
                           'weather_coefficient':beta,
                           'q1_shipment_ratio':volumes[y]['shipment_ratio'],
                           'current_shipment_filed':volumes[y]['current_filed'],
                           'previous_shipment_filed':volumes[y]['previous_filed']})
    pred = pd.DataFrame(result)
    for year, group in pred.groupby('year'):
        if (len(group) != 4 or set(group.model) != set(MODELS) or
            group.training_year_list.nunique() != 1 or group.actual_usd.nunique() != 1 or
            group.decision_date.nunique() != 1):
            raise AssertionError(f'Unmatched models in {year}')

    for ref, ours, refname in ((price_reference, 'price_only', 'price'),
                               (weather_reference, 'price_weather', 'weather'),
                               (shipment_reference, 'price_shipments', 'shipment')):
        if ref is None:
            continue
        if not {'year', 'prediction_usd', 'actual_usd'}.issubset(ref):
            raise ValueError(f'Invalid {refname} reference columns')
        if ref.year.duplicated().any():
            raise ValueError(f'Duplicate {refname} reference year')
        ours_df = pred[pred.model.eq(ours)]
        matched = ours_df.merge(ref[['year', 'prediction_usd', 'actual_usd']], on='year',
                                suffixes=('_new', '_reference'), validate='one_to_one')
        if len(matched) != len(years):
            raise ValueError(f'Missing {refname} reference years')
        for col in ('prediction_usd', 'actual_usd'):
            if not np.allclose(matched[col+'_new'], matched[col+'_reference'], rtol=1e-10, atol=0.01):
                raise AssertionError(f'{refname} {col} does not reproduce')

    scores = pred.groupby('model', as_index=False).agg(
        years=('year', 'size'), mae_usd=('absolute_error_usd', 'mean'))
    split = pred.assign(period=np.where(pred.year <= 2021, 'through_2021', '2022_onward'))
    period_scores = split.groupby(['period', 'model'], as_index=False).agg(
        years=('year', 'size'), mae_usd=('absolute_error_usd', 'mean'))
    annual_errors = pred.pivot(index='year', columns='model', values='absolute_error_usd')
    incremental = pd.DataFrame(index=annual_errors.index)
    incremental['weather_vs_price_usd'] = annual_errors.price_weather - annual_errors.price_only
    incremental['shipments_vs_price_usd'] = annual_errors.price_shipments - annual_errors.price_only
    incremental['both_vs_price_usd'] = annual_errors.price_weather_shipments - annual_errors.price_only
    incremental['weather_given_shipments_usd'] = annual_errors.price_weather_shipments - annual_errors.price_shipments
    incremental['shipments_given_weather_usd'] = annual_errors.price_weather_shipments - annual_errors.price_weather
    return {'predictions':pred, 'scores':scores, 'period_scores':period_scores,
            'incremental':incremental.reset_index()}


def save_comparison(result, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key in ('predictions','scores','period_scores','incremental'):
        result[key].to_csv(out / f'{key}_private.csv', index=False)
    report = ("Exploratory matched comparison: price-only baseline; weather-only addition; "
              "square-root Q1 company shipment ratio only; and both additions. Each comparison "
              "uses identical years, cutoffs and price baseline. Weather fits use only earlier "
              "targets filed by the decision date and training-only scaling. Shipment ratios "
              "are fixed rather than fit and their source filing dates are checked. A negative "
              "incremental error means the added signal reduced absolute error. All examined "
              "years informed previous exploration, so they are not untouched test years. "
              "This is a retrospective nowcast of company-wide revenue; shipments include "
              "regions and markets beyond Iowa corn. Forecast is precipitation probability, "
              "not observed rain. Historical weather availability, archive revisions and "
              "price vintages remain uncertain. This does not change the frozen 2027 test.\n")
    (out/'REPORT.txt').write_text(report, encoding='utf8')
    return report
