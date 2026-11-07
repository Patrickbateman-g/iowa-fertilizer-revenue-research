from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

UTC = dt.timezone.utc
TEST_DATE = dt.date(2027, 5, 31)
FIRST_Q2_RELEASE_DATE = dt.date(2027, 7, 1)
ALFRED_SERIES = 'WPU065201'
MODEL_ID = 'cf-q2-2027-price-and-damped-volume-v1'
DEFAULT_DIR = Path('/kaggle/working/cf_2027_test')


def _date(value, name):
    if not isinstance(value, str):
        raise ValueError(f'{name} must be an ISO YYYY-MM-DD string')
    try:
        result = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f'{name} must be an ISO YYYY-MM-DD date') from exc
    if result.isoformat() != value:
        raise ValueError(f'{name} must be an ISO YYYY-MM-DD date')
    return result


def _timestamp(value, name):
    if not isinstance(value, str) or not value.endswith('Z'):
        raise ValueError(f'{name} must be a UTC ISO timestamp ending in Z')
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + '+00:00')
    except ValueError as exc:
        raise ValueError(f'Invalid {name}') from exc
    return parsed.astimezone(UTC)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a positive number')
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return number


def _mapping(value, name, allowed, required):
    if not isinstance(value, dict) or set(value) - set(allowed) or not set(required) <= set(value):
        raise ValueError(f'{name} has missing or unexpected fields')
    return value


def _https(value, name, hosts):
    if not isinstance(value, str):
        raise ValueError(f'{name} must be a source URL')
    u = urlsplit(value)
    if (u.scheme != 'https' or u.username or u.password or u.port not in (None, 443)
        or not u.hostname or not any(u.hostname == h or u.hostname.endswith('.' + h) for h in hosts)):
        raise ValueError(f'{name} must be an official HTTPS source URL')
    return value


def _disclosure(row, name, period_start, period_end, earliest, latest):
    allowed = {'value', 'units', 'period_start', 'period_end', 'published', 'source'}
    _mapping(row, name, allowed, allowed)
    value = _positive(row['value'], name + ' value')
    if row['period_start'] != period_start or row['period_end'] != period_end:
        raise ValueError(f'{name} period must be {period_start} to {period_end}')
    published = _date(row['published'], name + ' published')
    if not earliest <= published <= latest:
        raise ValueError(f'{name} publication date is outside the allowed interval')
    _https(row['source'], name + ' source', ('sec.gov', 'cfindustries.com'))
    return value


def _make_prediction(inputs, now):
    if now.tzinfo is None:
        raise ValueError('The capture clock must be timezone-aware')
    now = now.astimezone(UTC)
    if now.date() != TEST_DATE:
        raise ValueError('Prospective capture is allowed only on 2027-05-31 UTC')
    allowed = {'prior_q2', 'price', 'q1_volume', 'q1_customer_advances'}
    _mapping(inputs, 'inputs', allowed, {'prior_q2', 'price', 'q1_volume'})
    q2 = inputs['prior_q2']
    revenue = _disclosure(q2, '2026 Q2 revenue', '2026-04-01', '2026-06-30',
                          dt.date(2026, 7, 1), TEST_DATE)
    if q2['units'] != 'USD':
        raise ValueError('Prior Q2 net sales units must be USD')

    price = inputs['price']
    if not isinstance(price, dict) or not isinstance(price.get('available'), bool):
        raise ValueError('price.available must be true or false')
    if price['available']:
        keys = {'available', 'series', 'vintage', 'current_observation', 'previous_observation',
                'current_value', 'previous_value', 'current_first_published', 'source', 'captured_at'}
        _mapping(price, 'price', keys, keys)
        if (price['series'] != ALFRED_SERIES or price['vintage'] != '2027-05-31'
            or price['current_observation'] != '2027-04-01'
            or price['previous_observation'] != '2026-04-01'):
            raise ValueError('Price must use both April observations from the May 31 WPU065201 vintage')
        if not dt.date(2027, 4, 1) <= _date(price['current_first_published'], 'price publication') <= TEST_DATE:
            raise ValueError('Current price must first have been published by the cutoff')
        source = _https(price['source'], 'price source', ('alfred.stlouisfed.org',))
        if ALFRED_SERIES not in source:
            raise ValueError('Price source must identify WPU065201')
        captured = _timestamp(price['captured_at'], 'price captured_at')
        if captured.date() != TEST_DATE or captured > now:
            raise ValueError('Price vintage must be captured on May 31 no later than the forecast')
        price_ratio = _positive(price['current_value'], 'current index') / _positive(price['previous_value'], 'previous index')
        main = revenue * price_ratio
        main_status = 'price_ratio'
    else:
        keys = {'available', 'reason', 'checked_at', 'source'}
        _mapping(price, 'unavailable price', keys, keys)
        if not isinstance(price['reason'], str) or len(price['reason'].strip()) < 12:
            raise ValueError('Explain why the April price index is unavailable')
        checked = _timestamp(price['checked_at'], 'price checked_at')
        if checked.date() != TEST_DATE or checked > now:
            raise ValueError('Unavailability must be checked on May 31 no later than forecast')
        source = _https(price['source'], 'price source', ('alfred.stlouisfed.org',))
        if ALFRED_SERIES not in source:
            raise ValueError('Price source must identify WPU065201')
        price_ratio = None
        main = revenue
        main_status = 'previous_q2_fallback'

    volume = inputs['q1_volume']
    if not isinstance(volume, dict) or not isinstance(volume.get('available'), bool):
        raise ValueError('q1_volume.available must be true or false')
    if volume['available']:
        keys = {'available', 'current', 'previous'}
        _mapping(volume, 'q1_volume', keys, keys)
        current = _disclosure(volume['current'], '2027 Q1 volume', '2027-01-01', '2027-03-31',
                              dt.date(2027, 4, 1), TEST_DATE)
        previous = _disclosure(volume['previous'], '2026 Q1 volume', '2026-01-01', '2026-03-31',
                               dt.date(2026, 4, 1), dt.date(2026, 5, 31))
        if volume['current']['units'] != 'thousand product tons' or volume['previous']['units'] != 'thousand product tons':
            raise ValueError('Q1 volumes must both use thousand product tons')
        volume_ratio = current / previous
        sensitivity = main * math.sqrt(volume_ratio) if price['available'] else None
        sensitivity_status = 'damped_volume_on_price' if price['available'] else 'unavailable_missing_price'
    else:
        _mapping(volume, 'unavailable q1_volume', {'available', 'reason'}, {'available', 'reason'})
        if not isinstance(volume['reason'], str) or len(volume['reason'].strip()) < 12:
            raise ValueError('Explain why Q1 volume is unavailable')
        volume_ratio, sensitivity, sensitivity_status = None, None, 'unavailable'
    advances = inputs.get('q1_customer_advances')
    advances_value = None
    advances_note = 'Not supplied'
    if isinstance(advances, dict) and advances.get('available') is False:
        _mapping(advances, 'unavailable customer advances', {'available', 'reason'}, {'available', 'reason'})
        if not isinstance(advances['reason'], str) or len(advances['reason'].strip()) < 12:
            raise ValueError('Explain why customer advances are unavailable')
        advances_note = advances['reason'].strip()
    elif advances is not None:
        fields = {'value', 'units', 'balance_date', 'published', 'source'}
        _mapping(advances, 'Q1 customer advances', fields, fields)
        if advances['units'] != 'USD' or advances['balance_date'] != '2027-03-31':
            raise ValueError('Customer advances must be USD at 2027-03-31')
        amount = advances['value']
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0:
            raise ValueError('Customer advances must be a nonnegative finite number')
        if not dt.date(2027, 4, 1) <= _date(advances['published'], 'customer advances publication') <= TEST_DATE:
            raise ValueError('Customer advances must have been published by the cutoff')
        _https(advances['source'], 'customer advances source', ('sec.gov', 'cfindustries.com'))
        advances_value = float(amount)
        advances_note = 'Partial, company-wide deposits; context only'
    return {'model_id': MODEL_ID, 'target': 'CF consolidated Q2 2027 net sales',
            'target_units': 'USD', 'prediction_cutoff_utc': '2027-05-31T23:59:59.999999Z',
            'captured_at_utc': now.isoformat().replace('+00:00', 'Z'), 'inputs': inputs,
            'price_ratio': price_ratio, 'q1_volume_ratio': volume_ratio,
            'baseline_prediction_usd': revenue, 'main_status': main_status, 'main_prediction_usd': main,
            'sensitivity_status': sensitivity_status, 'sensitivity_prediction_usd': sensitivity,
            'context_customer_advances_usd': advances_value, 'context_customer_advances_note': advances_note,
            'warning': 'Inputs and dates are manually supplied and unverified. Archive a Kaggle version before Q2 disclosure.'}


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False, ensure_ascii=False).encode('utf-8')


def _exclusive_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
        f.write('\n')


def capture_2027(inputs, output_path=DEFAULT_DIR / 'prediction_2027.json'):
    result = _make_prediction(inputs, dt.datetime.now(UTC))
    envelope = {'prediction': result, 'sha256': hashlib.sha256(_canonical(result)).hexdigest()}
    _exclusive_write(output_path, envelope)
    print('Prediction recorded. Save a Kaggle version before the Q2 announcement; a working file is temporary.')
    return envelope


def _score_prediction(envelope, actual, now):
    if now.tzinfo is None:
        raise ValueError('The scoring clock must be timezone-aware')
    now = now.astimezone(UTC)
    _mapping(envelope, 'forecast envelope', {'prediction', 'sha256'}, {'prediction', 'sha256'})
    prediction = envelope['prediction']
    if not isinstance(prediction, dict) or prediction.get('model_id') != MODEL_ID:
        raise ValueError('Unknown forecast model')
    if hashlib.sha256(_canonical(prediction)).hexdigest() != envelope['sha256']:
        raise ValueError('Forecast digest mismatch; the forecast may have changed')
    captured = _timestamp(prediction.get('captured_at_utc'), 'forecast capture')
    if captured.date() != TEST_DATE or captured >= now:
        raise ValueError('Forecast must be captured on May 31 and before scoring')
    expected = _make_prediction(prediction['inputs'], captured)
    if expected != prediction:
        raise ValueError('Forecast content does not match the fixed formulas')
    if now.date() < FIRST_Q2_RELEASE_DATE:
        raise ValueError('Scoring is only permitted after Q2 ends')
    value = _disclosure(actual, '2027 Q2 actual', '2027-04-01', '2027-06-30',
                        FIRST_Q2_RELEASE_DATE, now.date())
    if actual['units'] != 'USD':
        raise ValueError('Actual Q2 net sales units must be USD')
    rows = []
    for label, field, status_field in [('baseline', 'baseline_prediction_usd', None),
                                       ('main', 'main_prediction_usd', 'main_status'),
                                       ('sensitivity', 'sensitivity_prediction_usd', 'sensitivity_status')]:
        forecast = prediction[field]
        status = prediction[status_field] if status_field else 'previous_q2'
        if forecast is None:
            rows.append({'model': label, 'status': status, 'prediction_usd': None,
                         'absolute_error_usd': None, 'absolute_pct_error': None})
        else:
            error = abs(forecast - value)
            rows.append({'model': label, 'status': status, 'prediction_usd': forecast,
                         'absolute_error_usd': error, 'absolute_pct_error': 100 * error / value})
    return {'model_id': MODEL_ID, 'forecast_sha256': envelope['sha256'], 'actual': actual,
            'scored_at_utc': now.isoformat().replace('+00:00', 'Z'), 'results': rows,
            'warning': 'One year is not enough to establish reliable improvement; verify the independent archive timestamp.'}


def score_2027(forecast_path, actual, output_path=DEFAULT_DIR / 'score_2027.json'):
    with open(forecast_path, encoding='utf-8') as f:
        envelope = json.load(f)
    result = _score_prediction(envelope, actual, dt.datetime.now(UTC))
    _exclusive_write(output_path, result)
    print('Score recorded. Save the output in a new Kaggle version.')
    return result


