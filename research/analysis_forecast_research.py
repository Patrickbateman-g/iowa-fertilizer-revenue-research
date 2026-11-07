import datetime as dt
import json
import math
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, parse_qs
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

F_OUTPUT = Path('/kaggle/working/iowa_forecast')
F_CACHE = F_OUTPUT / 'station_year_cache.json'
F_ENDPOINT = 'https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py'
F_STATIONS = ('KAMW', 'KCID', 'KSUX')
F_YEARS = tuple(range(2016, 2027))
F_MIN_DAYS = 40
F_MIN_TRAIN = 6
F_RIDGE = 1.0
F_CONFIG = 'iem-gfs-mos-00z-plus24-p12-apr01-may20-v1'


def _expected_runtimes(year):
    return pd.date_range(f'{year}-04-01', f'{year}-05-20', freq='D', tz='UTC')


def _as_utc(value):
    try:
        value = pd.Timestamp(value)
        if pd.isna(value):
            return None
        return value.tz_localize('UTC') if value.tzinfo is None else value.tz_convert('UTC')
    except (TypeError, ValueError, OverflowError):
        return None


def summarize_station_year(records, station, year, source_url=''):
    if station not in F_STATIONS or year not in F_YEARS:
        raise ValueError('Unexpected station or year')
    if not isinstance(records, list):
        raise ValueError('IEM response is not a JSON row array')
    expected = set(_expected_runtimes(year))
    values, duplicates, invalid = {}, 0, 0
    for row in records:
        if not isinstance(row, dict):
            invalid += 1
            continue
        runtime = _as_utc(row.get('runtime'))
        valid = _as_utc(row.get('ftime'))
        if runtime not in expected or valid is None or valid != runtime + pd.Timedelta(hours=24):
            continue
        if row.get('station') != station or row.get('model') != 'GFS':
            invalid += 1
            continue
        value = row.get('p12')
        if value is None or value in ('', 'M', 'null'):
            continue
        if isinstance(value, bool):
            invalid += 1
            continue
        try:
            value = float(value)
        except (ValueError, TypeError):
            invalid += 1
            continue
        if not np.isfinite(value) or not 0 <= value <= 100:
            invalid += 1
            continue
        if runtime in values:
            if value in values[runtime]:
                duplicates += 1
            values[runtime].add(value)
        else:
            values[runtime] = {value}
    conflicting_days = sum(len(items) > 1 for items in values.values())
    clean_values = [next(iter(items)) for items in values.values() if len(items) == 1]
    count = len(clean_values)
    usable = count >= F_MIN_DAYS and invalid == 0
    return {'year': int(year), 'station': station, 'expected_days': len(expected),
            'valid_days': count, 'coverage_fraction': count / len(expected),
            'duplicate_rows': duplicates, 'conflicting_days': conflicting_days, 'invalid_rows': invalid,
            'station_mean_p12_pct': float(np.mean(clean_values)) if clean_values else None,
            'usable': usable, 'source_url': source_url, 'status': 'ok' if usable else 'incomplete_or_invalid'}


def _query_url(station, year):
    query = {'station': station, 'model': 'GFS', 'sts': f'{year}-04-01T00:00Z',
             'ets': f'{year}-05-21T00:00Z', 'format': 'json'}
    return F_ENDPOINT + '?' + urlencode(query)


def _get_json(url, attempts=3, timeout=30):
    last = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={'User-Agent': 'IowaForecastResearch/1.0 contact: public research',
                                            'Accept': 'application/json'})
            with urlopen(request, timeout=timeout) as response:
                obj = json.load(response)
            if not isinstance(obj, list):
                raise ValueError('IEM returned an unexpected JSON structure')
            return obj
        except HTTPError as e:
            if e.code == 404:
                return []
            last = e
            if e.code not in (429, 500, 502, 503, 504):
                break
        except (URLError, TimeoutError, OSError, ValueError) as e:
            last = e
        if attempt + 1 < attempts:
            time.sleep(min(2 ** attempt, 4))
    raise RuntimeError(f'IEM request failed after retries: {type(last).__name__}: {last}')


def _load_cache():
    if not F_CACHE.exists():
        return {}
    data = json.loads(F_CACHE.read_text(encoding='utf-8'))
    if data.get('config') != F_CONFIG or not isinstance(data.get('station_years'), dict):
        raise ValueError('Existing station-year cache has an incompatible configuration')
    return data['station_years']


def _write_cache(cache):
    F_OUTPUT.mkdir(parents=True, exist_ok=True)
    tmp = F_OUTPUT / 'station_year_cache.tmp'
    tmp.write_text(json.dumps({'config': F_CONFIG, 'station_years': cache}, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(F_CACHE)


def snapshot_forecasts(snapshot):
    if not isinstance(snapshot, list):
        raise ValueError('FORECAST_SNAPSHOT must be a list of station-year summaries')
    req = {'year', 'station', 'expected_days', 'valid_days', 'duplicate_rows', 'conflicting_days',
           'invalid_rows', 'station_mean_p12_pct', 'source_url', 'status'}
    coverage = []
    seen = set()
    for item in snapshot:
        if not isinstance(item, dict) or not req.issubset(item):
            raise ValueError('Missing snapshot summary fields')
        year, station = item['year'], item['station']
        if isinstance(year, bool) or not isinstance(year, int) or year not in F_YEARS or station not in F_STATIONS:
            raise ValueError('Unexpected snapshot year or station')
        if (year, station) in seen:
            raise ValueError('Duplicate snapshot station-year')
        seen.add((year, station))
        for col in ('expected_days', 'valid_days', 'duplicate_rows', 'conflicting_days', 'invalid_rows'):
            if isinstance(item[col], bool) or not isinstance(item[col], int) or item[col] < 0:
                raise ValueError(f'Invalid {col} count')
        if item['expected_days'] != 50 or item['valid_days'] + item['conflicting_days'] > 50:
            raise ValueError('Expected 50 daily 00Z runs per station-year')
        url = urlsplit(item['source_url'])
        params = parse_qs(url.query)
        if (url.scheme != 'https' or url.hostname != 'mesonet.agron.iastate.edu'
                or url.path != '/cgi-bin/request/mos.py'
                or params.get('station') != [station] or params.get('model') != ['GFS']
                or params.get('sts') != [f'{year}-04-01T00:00Z']
                or params.get('ets') != [f'{year}-05-21T00:00Z']
                or params.get('format') != ['json']):
            raise ValueError(f'Invalid source parameters for {station} {year}')
        status = item['status']
        if not isinstance(status, str) or not status.strip():
            raise ValueError('Snapshot status is required')
        mean = item['station_mean_p12_pct']
        if mean is not None:
            if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not np.isfinite(mean) or not 0 <= mean <= 100:
                raise ValueError('Invalid p12 probability mean')
            mean = float(mean)
        if item['valid_days'] > 0 and mean is None:
            raise ValueError('Nonempty station-year must include a mean')
        if item['valid_days'] == 0 and mean is not None:
            raise ValueError('Empty station-year cannot have a mean')
        usable = (status == 'ok' and item['valid_days'] >= F_MIN_DAYS and
                  item['invalid_rows'] == 0 and mean is not None)
        coverage.append({'year': year, 'station': station, 'expected_days': 50,
            'valid_days': item['valid_days'], 'coverage_fraction': item['valid_days']/50,
            'duplicate_rows': item['duplicate_rows'], 'conflicting_days': item['conflicting_days'], 'invalid_rows': item['invalid_rows'],
            'station_mean_p12_pct': mean, 'usable': usable,
            'source_url': item['source_url'], 'status': status})
    expected = {(year, station) for year in F_YEARS for station in F_STATIONS}
    if seen != expected:
        raise ValueError(f'Snapshot must explicitly cover all 33 station-years; missing {len(expected-seen)}')
    cov = pd.DataFrame(coverage).sort_values(['year','station']).reset_index(drop=True)
    annual = []
    for year, group in cov.groupby('year', sort=True):
        complete = len(group) == 3 and set(group.station) == set(F_STATIONS) and group.usable.all()
        annual.append({'year': int(year), 'forecast_mean_p12_pct':
                       float(group.station_mean_p12_pct.mean()) if complete else np.nan,
                       'complete': bool(complete), 'stations_used': int(group.usable.sum())})
    annual = pd.DataFrame(annual)
    F_OUTPUT.mkdir(parents=True, exist_ok=True)
    (F_OUTPUT/'forecast_snapshot_private.json').write_text(
        json.dumps({'config':F_CONFIG, 'method': 'Browser-derived station-year aggregates; raw records not stored',
                    'station_years':snapshot}, indent=2, allow_nan=False), encoding='utf-8')
    cov.to_csv(F_OUTPUT/'coverage_private.csv', index=False)
    annual.to_csv(F_OUTPUT/'annual_forecast_private.csv', index=False)
    print(f'Forecast snapshot: {int(annual.complete.sum())}/11 complete years. '
          'Input summaries are manually derived; raw records and release times were not independently verified.')
    return annual, cov


def collect_forecasts():
    F_OUTPUT.mkdir(parents=True, exist_ok=True)
    cache, coverage = _load_cache(), []
    for year in F_YEARS:
        for station in F_STATIONS:
            key = f'{year}:{station}'
            url = _query_url(station, year)
            if key in cache:
                item = cache[key]
            else:
                try:
                    records = _get_json(url)
                    item = summarize_station_year(records, station, year, url)
                    del records
                    cache[key] = item
                    _write_cache(cache)
                except Exception as exc:
                    item = {'year': year, 'station': station, 'expected_days': 50, 'valid_days': 0,
                            'coverage_fraction': 0.0, 'duplicate_rows': 0, 'conflicting_days': 0, 'invalid_rows': 0,
                            'station_mean_p12_pct': None, 'usable': False, 'source_url': url,
                            'status': 'fetch_error: ' + str(exc)[:180]}
                time.sleep(0.15)
            coverage.append(item)
    cov = pd.DataFrame(coverage)
    annual = []
    for year, group in cov.groupby('year', sort=True):
        complete = len(group) == 3 and set(group.station) == set(F_STATIONS) and group.usable.all()
        annual.append({'year': int(year), 'forecast_mean_p12_pct':
                       float(group.station_mean_p12_pct.mean()) if complete else np.nan,
                       'complete': bool(complete), 'stations_used': int(group.usable.sum())})
    annual = pd.DataFrame(annual)
    cov.to_csv(F_OUTPUT / 'coverage_private.csv', index=False)
    annual.to_csv(F_OUTPUT / 'annual_forecast_private.csv', index=False)
    print(f'IEM forecast summaries: {int(annual.complete.sum())}/11 complete years; '
          f'{int(cov.status.astype(str).str.startswith("fetch_error").sum())} failed station-year requests. '
          f'Outputs:{F_OUTPUT}')
    return annual, cov


def evaluate_forecasts(v2_results, annual, coverage):
    if not isinstance(v2_results, dict) or 'data' not in v2_results:
        raise ValueError('Run the prior follow-up cells to create V2_RESULTS first')
    base = v2_results['data'].copy()
    req = {'year', 'as_of', 'price_as_of', 'price_ratio', 'prior_q2_usd', 'target_q2_usd', 'target_filed'}
    if not req.issubset(base.columns):
        raise ValueError('V2 data is missing time, target or price provenance')
    if not {'year', 'complete', 'forecast_mean_p12_pct'}.issubset(annual.columns):
        raise ValueError('Annual forecast summary fields are missing')
    if base.year.duplicated().any() or annual.year.duplicated().any():
        raise ValueError('Duplicate annual year')
    base = base[base.year.isin(F_YEARS) & base.price_ratio.notna()].copy()
    for x in base.itertuples():
        if str(x.as_of)[:10] != f'{int(x.year)}-05-31' or str(x.price_as_of)[:10] != f'{int(x.year)}-05-31':
            raise ValueError(f'Price vintage/cutoff mismatch in {x.year}')
        if str(x.target_filed)[:10] <= str(x.as_of)[:10]:
            raise ValueError(f'Q2 target was reported by the forecast date in {x.year}')
    annual = annual[annual.complete.eq(True)].copy()
    df = base.merge(annual[['year', 'forecast_mean_p12_pct']], on='year', validate='one_to_one').sort_values('year')
    for col in ('price_ratio', 'prior_q2_usd', 'target_q2_usd'):
        df[col] = pd.to_numeric(df[col], errors='raise')
        if (~np.isfinite(df[col]) | (df[col] <= 0)).any():
            raise ValueError(f'Invalid {col}')
    df['baseline'] = df.prior_q2_usd * df.price_ratio
    df['forecast_mean_p12_pct'] = pd.to_numeric(df.forecast_mean_p12_pct, errors='raise')
    if (~np.isfinite(df.forecast_mean_p12_pct) | ~df.forecast_mean_p12_pct.between(0, 100)).any():
        raise ValueError('Invalid annual forecast feature')
    results, skipped = [], []
    for _, row in df.iterrows():
        year = int(row.year)
        train = df[(df.year < year) & (df.target_filed.astype(str).str[:10] <= str(row.as_of)[:10])]
        if len(train) < F_MIN_TRAIN:
            skipped.append({'year': year, 'reason': 'fewer than six earlier price and forecast years', 'train_years': len(train)})
            continue
        x = train.forecast_mean_p12_pct.to_numpy(dtype=float)
        mean, sd = float(x.mean()), float(x.std(ddof=0))
        if not np.isfinite(sd) or sd <= 1e-12:
            skipped.append({'year': year, 'reason': 'earlier feature has no variation', 'train_years': len(train)})
            continue
        z = (x - mean) / sd
        residual = np.log(train.target_q2_usd.to_numpy(dtype=float) / train.baseline.to_numpy(dtype=float))
        beta = float(np.dot(z, residual) / (np.dot(z, z) + F_RIDGE))
        ztest = (float(row.forecast_mean_p12_pct) - mean) / sd
        log_factor = beta * ztest
        if not np.isfinite(log_factor) or abs(log_factor) > 700:
            skipped.append({'year': year, 'reason': 'non-finite prediction', 'train_years': len(train)})
            continue
        baseline, actual = float(row.baseline), float(row.target_q2_usd)
        forecast = baseline * math.exp(log_factor)
        results.append({'year': year, 'train_years': len(train), 'last_training_year': int(train.year.max()),
                        'feature_mean_p12_pct': float(row.forecast_mean_p12_pct),
                        'train_feature_mean_pct': mean, 'train_feature_sd_pct': sd, 'coefficient': beta,
                        'actual_usd': actual, 'price_only_usd': baseline, 'price_forecast_usd': forecast,
                        'price_abs_error_usd': abs(baseline-actual), 'forecast_abs_error_usd': abs(forecast-actual),
                        'forecast_minus_price_abs_error_usd': abs(forecast-actual)-abs(baseline-actual)})
    pred = pd.DataFrame(results)
    skipped_df = pd.DataFrame(skipped, columns=['year','reason','train_years'])
    lines = ['# Exploratory Iowa archived forecast test', '',
        'Historical results through 2026 were already inspected. These are retrospective, not an untouched test; the frozen 2027 forecast remains unchanged.', '',
        'Feature: equal-station mean GFS MOS p12 probability (%) for the 12 hours ending 24 hours after each 00Z model initialization from April 1 through May 20, in Iowa. Each station-year needs 40 of 50 days; identical duplicate values are counted once and days with conflicting values are excluded. Runtime is initialization, not verified product publication or proof a farmer saw the forecast.', '',
        'A one-coefficient ridge model is fitted on earlier years only, on log(actual / price-only forecast). A training-only mean and standard deviation and fixed ridge penalty 1 are used; the intercept is fixed at zero. Minimum six earlier complete years. Actual outcomes filed after the forecast date are never used for training that forecast.', '',
        f'Complete forecast years: {len(annual)} of 11; matched evaluated years: {len(pred)}.', '']
    if len(pred):
        price_mae = pred.price_abs_error_usd.mean()/1e6
        weather_mae = pred.forecast_abs_error_usd.mean()/1e6
        wins = int((pred.forecast_abs_error_usd < pred.price_abs_error_usd).sum())
        lines += ['| Model | Matched years | MAE (USD millions) |', '|---|---:|---:|',
                  f'| Existing price only | {len(pred)} | {price_mae:,.1f} |',
                  f'| Price + forecast feature | {len(pred)} | {weather_mae:,.1f} |', '',
                  f'Forecast feature had lower error in {wins} of {len(pred)} matched years.', '',
                  'Per-year predictions and training counts are in predictions_private.csv. The small sample, prior inspection, model and feature choices, missing coverage, and potentially revised archives limit interpretation. A lower error here does not establish future improvement or causation.', '']
    else:
        lines += ['No score can be calculated: insufficient complete, temporally eligible forecast history. Check coverage_private.csv for errors and missing years.', '']
    lines += ['IEM documentation: https://mesonet.agron.iastate.edu/mos/ ; original model details: https://vlab.noaa.gov/web/mdl/mav-card .',
              'P12 is probability of measurable precipitation, not expected rainfall amount; product issuance/public availability timing and archive revisions require verification. The compact station-year inputs were summarized via the browser; the raw records and calculations are not independently reverified by this notebook. No rain forecasts after May 20 initialization enter the feature. Station-averaged forecasts do not indicate which forecast a farmer saw or whether any order was placed.',
              'Compare only years present for both models and use a future, pre-recorded test before promoting any candidate.', '']
    F_OUTPUT.mkdir(parents=True, exist_ok=True)
    pred.to_csv(F_OUTPUT / 'predictions_private.csv', index=False)
    skipped_df.to_csv(F_OUTPUT / 'skipped_private.csv', index=False)
    (F_OUTPUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    return {'predictions': pred, 'skipped': skipped_df, 'data': df, 'report': '\n'.join(lines), 'coverage': coverage}
