from __future__ import annotations
import datetime as dt
import math
import numpy as np
import pandas as pd

INTERVAL_REQUIRED = ('year', 'decision_date', 'target_filed', 'model', 'prediction_usd',
            'actual_usd', 'training_year_list')


def _interval_date(value, label):
    if not isinstance(value, str):
        raise ValueError(f'{label} must be an ISO date')
    try:
        d = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f'Invalid {label}') from exc
    if d.isoformat() != value:
        raise ValueError(f'{label} must be YYYY-MM-DD')
    return d


def _interval_training_years(value, prediction_year):
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == '':
        return ()
    if not isinstance(value, str):
        raise ValueError('training_year_list must be a comma-separated string')
    try:
        parts = [int(x) for x in value.split(',')]
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid training year list') from exc
    if any(str(x) != raw.strip() for x, raw in zip(parts, value.split(','))):
        raise ValueError('Invalid training year list')
    if len(parts) != len(set(parts)) or any(y >= prediction_year for y in parts):
        raise ValueError('Training years must be unique and precede prediction year')
    return tuple(parts)


def validate_prediction_rows(predictions, expected_models=None):
    missing = set(INTERVAL_REQUIRED) - set(predictions.columns)
    if missing:
        raise ValueError(f'Missing prediction columns: {sorted(missing)}')
    p = predictions.copy()
    if p.empty:
        raise ValueError('No predictions supplied')
    years = pd.to_numeric(p.year, errors='raise')
    if years.isna().any() or not np.isfinite(years).all() or (years != np.floor(years)).any():
        raise ValueError('Invalid prediction year')
    p['year'] = years.astype(int)
    if p.model.isna().any() or any(not isinstance(x, str) or not x.strip() for x in p.model):
        raise ValueError('Missing model name')
    if p.duplicated(['year', 'model']).any():
        raise ValueError('Duplicate model-year prediction')
    p['prediction_usd'] = pd.to_numeric(p.prediction_usd, errors='raise')
    p['actual_usd'] = pd.to_numeric(p.actual_usd, errors='raise')
    if not np.isfinite(p.prediction_usd).all() or (p.prediction_usd <= 0).any():
        raise ValueError('Invalid prediction')
    if (p.actual_usd.dropna() <= 0).any() or not np.isfinite(p.actual_usd.dropna()).all():
        raise ValueError('Invalid actual')
    for row in p.itertuples():
        cutoff = _interval_date(row.decision_date, 'decision_date')
        if cutoff.year != row.year or cutoff != dt.date(row.year, 5, 31):
            raise ValueError('Decision date must be the May 31 prediction date')
        _interval_training_years(row.training_year_list, row.year)
        if pd.isna(row.actual_usd):
            if not pd.isna(row.target_filed) and row.target_filed != '':
                raise ValueError('Filing date supplied without an outcome')
        else:
            filed = _interval_date(row.target_filed, 'target_filed')
            if filed <= dt.date(row.year, 6, 30):
                raise ValueError('Target filing must follow quarter end')
    models = set(expected_models) if expected_models is not None else set(p.model)
    if not models:
        raise ValueError('No expected models')
    for year, group in p.groupby('year'):
        if set(group.model) != models:
            raise ValueError(f'Unmatched model years: {year}')
        if group.decision_date.nunique() != 1:
            raise ValueError('Decision dates differ within a year')
        if group.actual_usd.isna().nunique() != 1 or group.target_filed.fillna('').nunique() != 1:
            raise ValueError('Targets or filing dates differ within a year')
        if group.actual_usd.notna().all() and not np.allclose(group.actual_usd, group.actual_usd.iloc[0], rtol=0, atol=0.001):
            raise ValueError('Targets differ within a year')
        if group.training_year_list.fillna('').nunique() != 1:
            raise ValueError('Training years are not matched')
    actual_dates = p.drop_duplicates('year').set_index('year').target_filed.to_dict()
    for row in p.itertuples():
        for train_year in _interval_training_years(row.training_year_list, row.year):
            if train_year in actual_dates:
                filing = actual_dates[train_year]
                if pd.isna(filing) or not filing or _interval_date(filing, 'target_filed') > _interval_date(row.decision_date, 'decision_date'):
                    raise ValueError('Training target not filed by prediction date')
    return p.sort_values(['year', 'model']).reset_index(drop=True)


def sequential_ranges(predictions, expected_models=None, coverage=0.8):
    if not isinstance(coverage, (float, int)) or not math.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError('coverage must be strictly between 0 and 1')
    p = validate_prediction_rows(predictions, expected_models)
    result = []
    for row in p.itertuples(index=False):
        earlier = p[(p.model == row.model) & (p.year < row.year) & p.actual_usd.notna()].copy()
        earlier = earlier[earlier.target_filed <= row.decision_date].sort_values('year')
        errs = (earlier.prediction_usd - earlier.actual_usd).abs().to_numpy(dtype=float)
        n = len(errs)
        k = math.ceil((n + 1) * float(coverage))
        width = float(np.sort(errs)[k - 1]) if 1 <= k <= n else float('nan')
        lower = max(0.0, float(row.prediction_usd) - width) if np.isfinite(width) else float('nan')
        upper = float(row.prediction_usd) + width if np.isfinite(width) else float('nan')
        included = (bool(lower <= float(row.actual_usd) <= upper)
                    if np.isfinite(width) and pd.notna(row.actual_usd) else None)
        result.append({'year': int(row.year), 'decision_date': row.decision_date, 'model': row.model,
                       'prediction_usd': float(row.prediction_usd), 'actual_usd': row.actual_usd,
                       'target_filed': row.target_filed, 'calibration_years': ','.join(map(str, earlier.year)),
                       'calibration_count': n, 'nominal_coverage': float(coverage), 'rank': k,
                       'absolute_error_radius_usd': width, 'lower_usd': lower, 'upper_usd': upper,
                       'contains_actual': included})
    out = pd.DataFrame(result)
    summaries = []
    for model, g in out.groupby('model', sort=True):
        ev = g[g.contains_actual.notna()]
        inside = int(ev.contains_actual.sum()) if len(ev) else 0
        summaries.append({'model': model, 'total_prediction_years': len(g),
                          'ranges_available': int(g.lower_usd.notna().sum()),
                          'evaluated_ranges': len(ev), 'covered': inside,
                          'empirical_coverage': inside / len(ev) if len(ev) else float('nan'),
                          'first_evaluation_year': int(ev.year.min()) if len(ev) else None,
                          'last_evaluation_year': int(ev.year.max()) if len(ev) else None})
    return out, pd.DataFrame(summaries)


def summarize_range_periods(ranges, periods):
    needed = {'year', 'model', 'lower_usd', 'contains_actual'}
    if not needed.issubset(ranges.columns) or ranges.empty or not periods:
        raise ValueError('Missing ranges or periods')
    rows = []
    for label, bounds in periods.items():
        if not isinstance(label, str) or not label or len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError('Invalid period')
        part = ranges[ranges.year.between(bounds[0], bounds[1])]
        for model in sorted(ranges.model.unique()):
            g = part[part.model == model]
            observed = g[g.contains_actual.notna()]
            wins = int(observed.contains_actual.sum()) if len(observed) else 0
            rows.append({'period': label, 'model': model, 'prediction_years': len(g),
                         'ranges_available': int(g.lower_usd.notna().sum()),
                         'evaluated_ranges': len(observed), 'covered': wins,
                         'empirical_coverage': wins / len(observed) if len(observed) else float('nan')})
    return pd.DataFrame(rows)
