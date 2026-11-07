from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ('year', 'as_of', 'target_filed', 'target_q2_usd', 'prior_q2_usd',
            'price_ratio', 'forecast_mean_p12_pct')
DATE_INPUTS = ('prior_filed', 'price_available', 'forecast_available')


def iso_date(value):
    if not isinstance(value, str):
        raise ValueError('Dates must be ISO YYYY-MM-DD strings')
    try:
        result = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f'Invalid date: {value!r}') from exc
    if result.isoformat() != value:
        raise ValueError(f'Invalid date: {value!r}')
    return result


def daily_forecast_status(records, station, year):
    expected = pd.date_range(f'{year}-04-01', f'{year}-05-20', freq='D', tz='UTC')
    expected_set = set(expected)
    values, seen, nulls, invalid = (defaultdict(set), defaultdict(int),
                                     defaultdict(int), defaultdict(int))
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('Forecast record must be a dictionary')
        try:
            run = pd.to_datetime(row.get('runtime'), utc=True, errors='raise')
            valid = pd.to_datetime(row.get('ftime'), utc=True, errors='raise')
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid forecast timestamp') from exc
        if pd.isna(run) or pd.isna(valid) or run not in expected_set or valid != run + pd.Timedelta(hours=24):
            raise ValueError('Forecast row outside selected 00Z/+24h window')
        seen[run] += 1
        if row.get('station') != station or row.get('model') != 'GFS':
            invalid[run] += 1
            continue
        raw = row.get('p12')
        if raw is None or raw in ('', 'M', 'null'):
            nulls[run] += 1
            continue
        try:
            value = float(raw) if not isinstance(raw, bool) else float('nan')
        except (TypeError, ValueError):
            value = float('nan')
        if not math.isfinite(value) or not 0 <= value <= 100:
            invalid[run] += 1
            continue
        values[run].add(value)
    result = []
    for run in expected:
        choices = values[run]
        state = ('conflicting_values' if len(choices) > 1 else
                 'valid' if len(choices) == 1 else
                 'invalid_only' if invalid[run] else
                 'null_only' if nulls[run] else 'absent_in_selected_ledger')
        result.append({'date': run.date().isoformat(), 'state': state,
                       'p12_pct': next(iter(choices)) if state == 'valid' else None,
                       'selected_rows': seen[run], 'null_rows': nulls[run],
                       'invalid_rows': invalid[run], 'distinct_values': len(choices)})
    return pd.DataFrame(result)


def prepare_rows(frame):
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    data = frame.copy()
    years = pd.to_numeric(data.year, errors='raise')
    if years.isna().any() or not np.isfinite(years).all() or not np.equal(years, np.floor(years)).all():
        raise ValueError('Invalid year')
    data['year'] = years.astype(int)
    if data.year.duplicated().any():
        raise ValueError('Duplicate year')
    for name in ('target_q2_usd', 'prior_q2_usd', 'price_ratio', 'forecast_mean_p12_pct'):
        data[name] = pd.to_numeric(data[name], errors='raise')
        if not np.isfinite(data[name]).all():
            raise ValueError(f'Missing or non-finite {name}')
    if (data[['target_q2_usd', 'prior_q2_usd', 'price_ratio']] <= 0).any().any():
        raise ValueError('Revenue and price ratio must be positive')
    if not data.forecast_mean_p12_pct.between(0, 100).all():
        raise ValueError('Invalid forecast probability')
    for row in data.itertuples(index=False):
        year = int(row.year)
        cutoff = iso_date(row.as_of)
        if cutoff != dt.date(year, 5, 31):
            raise ValueError(f'Wrong cutoff for {year}')
        if iso_date(row.target_filed) <= dt.date(year, 6, 30):
            raise ValueError(f'Target filed before quarter ended for {year}')
        for name in DATE_INPUTS:
            if name in data.columns:
                value = getattr(row, name)
                if pd.isna(value) or iso_date(value) > cutoff:
                    raise ValueError(f'{name} missing or later than cutoff for {year}')
    return data.sort_values('year').reset_index(drop=True)


def fit_no_intercept(train, test_row, features=('forecast_mean_p12_pct',)):
    baseline = float(test_row.prior_q2_usd * test_row.price_ratio)
    if not features:
        return baseline
    x = train[list(features)].to_numpy(float)
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('Missing or invalid training features')
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    if not np.isfinite(sd).all() or (sd <= 1e-12).any():
        raise ValueError('No training feature variation')
    z = (x - mu) / sd
    values = np.array([getattr(test_row, c) for c in features], dtype=float)
    test = (values - mu) / sd
    response = np.log(train.target_q2_usd.to_numpy(float) /
                      (train.prior_q2_usd.to_numpy(float) * train.price_ratio.to_numpy(float)))
    beta = np.linalg.solve(z.T @ z + np.eye(len(features)), z.T @ response)
    exponent = float(test @ beta)
    if not np.isfinite(exponent) or abs(exponent) > 700:
        raise ValueError('Invalid prediction')
    return baseline * math.exp(exponent)


def fit_intercept_sensitivity(train, test_row, features, strength):
    if strength not in (0.0, 0.5, 1.0):
        raise ValueError('Expected preselected intercept strength 0, 0.5 or 1')
    if len(train) == 0:
        raise ValueError('No training rows')
    residuals = np.log(train.target_q2_usd.to_numpy(float) /
                       (train.prior_q2_usd.to_numpy(float) * train.price_ratio.to_numpy(float)))
    correction = float(strength * residuals.mean())
    if not np.isfinite(correction) or abs(correction) > 700:
        raise ValueError('Invalid intercept')
    return fit_no_intercept(train, test_row, features) * math.exp(correction)


def walk_forward(frame, test_years, min_train=6):
    data = prepare_rows(frame)
    years = list(test_years)
    if not years or len(set(years)) != len(years):
        raise ValueError('Test years must be nonempty and unique')
    records = []
    for year in years:
        selected = data[data.year.eq(year)]
        if len(selected) != 1:
            raise ValueError(f'Missing test year {year}; matched comparison cannot proceed')
        row = next(selected.itertuples(index=False))
        train = data[(data.year < year) & (data.target_filed <= row.as_of)].copy()
        if len(train) < min_train:
            raise ValueError(f'Insufficient earlier filed training years for {year}')
        models = (('price_only', (), 0.0),
                  ('price_half_intercept', (), 0.5),
                  ('price_forecast', ('forecast_mean_p12_pct',), 0.0),
                  ('forecast_half_intercept', ('forecast_mean_p12_pct',), 0.5),
                  ('forecast_full_intercept', ('forecast_mean_p12_pct',), 1.0))
        for model, features, strength in models:
            pred = fit_intercept_sensitivity(train, row, features, strength)
            records.append({'year': year, 'model': model, 'training_years': ','.join(map(str, train.year)),
                            'prediction_usd': pred, 'actual_usd': float(row.target_q2_usd),
                            'absolute_error_usd': abs(pred - float(row.target_q2_usd))})
    predictions = pd.DataFrame(records)
    scores = predictions.groupby('model', as_index=False).agg(
        years=('year', 'size'), mae_usd=('absolute_error_usd', 'mean'))
    return predictions, scores


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run the standalone forecast-only historical comparison.')
    parser.add_argument('--input', required=True, type=Path, help='annual feature CSV')
    parser.add_argument('--output', required=True, type=Path, help='output directory')
    parser.add_argument('--test-start', type=int, default=2022)
    parser.add_argument('--test-end', type=int, default=2026)
    args = parser.parse_args(argv)
    if args.test_end < args.test_start:
        parser.error('--test-end must be >= --test-start')
    predictions, scores = walk_forward(pd.read_csv(args.input), range(args.test_start, args.test_end + 1))
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / 'predictions_private.csv', index=False)
    scores.to_csv(args.output / 'scores_private.csv', index=False)
    print(scores.to_string(index=False))
    print('Exploratory: source availability and historical vintages must be independently established.')
    print('Outputs:', args.output)


if __name__ == '__main__':
    main()
