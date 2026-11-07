from __future__ import annotations

import datetime as dt
import argparse
import hashlib
import json
import math
import os
import stat
import re
from pathlib import Path
import sys
from urllib.parse import urlparse


class ReplayUnavailable(ValueError):
    pass


_REALIZED_PERIODS = {
    'q2_sales': (0, 6, 30),
    'q2_phosphate_sales': (0, 6, 30),
    'q1_shipments_current': (0, 3, 31),
    'q1_shipments_previous': (-1, 3, 31),
    'nitrogen_price_current_april': (0, 4, 1),
    'nitrogen_price_previous_april': (-1, 4, 1),
}


def _date(value, field):
    if not isinstance(value, str):
        raise ValueError(f'{field} must be an ISO calendar date')
    try:
        date = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f'{field} must be an ISO calendar date') from exc
    if date.isoformat() != value:
        raise ValueError(f'{field} must use YYYY-MM-DD')
    return date


def _when(value, field):
    if value is None or value == '':
        return None, 'unknown'
    if not isinstance(value, str):
        raise ValueError(f'{field} must be an ISO date or offset timestamp')
    if len(value) == 10:
        return _date(value, field), 'day'
    try:
        instant = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError(f'Invalid {field}') from exc
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError(f'{field} timestamp needs a time zone offset')
    return instant.astimezone(dt.timezone.utc), 'instant'


def _relation(value, cutoff):
    when, precision = _when(value, 'availability')
    decision, decision_precision = _when(cutoff, 'cutoff')
    if decision is None:
        raise ValueError('Missing cutoff')
    if when is None:
        return 'missing'
    if decision_precision == 'day':
        available_day = when.date() if precision == 'instant' else when
        return 'before' if available_day <= decision else 'after'
    if precision == 'instant':
        return 'before' if when <= decision else 'after'
    if when < decision.date():
        return 'before'
    if when > decision.date():
        return 'after'
    return 'unknown_intraday'


def _key(record):
    return str(record['series']), str(record['period'])


def _definitely_before(first, second):
    a, ap = _when(first, 'first time')
    b, bp = _when(second, 'second time')
    if a is None or b is None:
        return False
    if ap == bp:
        return a < b
    a_day = a.date() if ap == 'instant' else a
    b_day = b.date() if bp == 'instant' else b
    return a_day < b_day


_FORECAST_STATIONS = ('KAMW', 'KCID', 'KSUX')

FORECAST_DEFINITION = {
    'model': 'GFS',
    'element': 'P12',
    'units': 'percent',
    'lead_start_hours': 12,
    'lead_end_hours': 24,
    'precipitation_threshold_inches': 0.01,
    'precipitation_basis': 'liquid_equivalent',
}


def _forecast_definition_matches(record):
    for name, expected in FORECAST_DEFINITION.items():
        value = record.get(name)
        if isinstance(expected, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != expected:
                return False
        elif value != expected:
            return False
    return True


def _forecast_recipe_status(record):
    series, period = record['series'], record['period']
    if (series == 'forecast_probability' or series.startswith(('forecast_station_', 'forecast_daily_'))):
        if not _forecast_definition_matches(record):
            return 'forecast_definition_unverified'
    if series == 'forecast_probability' or series.startswith('forecast_station_'):
        if not re.fullmatch(r'\d{4}', period):
            return 'forecast_recipe_unverified'
        year = int(period)
        components = record.get('constituent_keys')
        if record.get('derivation') != 'mean' or not isinstance(components, (list, tuple)):
            return 'forecast_recipe_unverified'
        if any(not isinstance(p, (list, tuple)) or len(p) != 2 or
               not isinstance(p[0], str) or not isinstance(p[1], str) for p in components):
            return 'forecast_recipe_unverified'
        keys = [tuple(p) for p in components]
        if len(set(keys)) != len(keys):
            return 'forecast_recipe_unverified'
        if series == 'forecast_probability':
            expected = {('forecast_station_' + station, period) for station in _FORECAST_STATIONS}
            if set(keys) != expected or len(keys) != 3:
                return 'forecast_recipe_unverified'
        else:
            station = series.removeprefix('forecast_station_')
            if station not in _FORECAST_STATIONS or not 40 <= len(keys) <= 50:
                return 'forecast_recipe_unverified'
            try:
                dates = [_date(p[1], 'forecast model run date') for p in keys]
                start, end = dt.date(year, 4, 1), dt.date(year, 5, 20)
            except ValueError:
                return 'forecast_recipe_unverified'
            if any(p[0] != 'forecast_daily_' + station for p in keys) or any(not start <= d <= end for d in dates):
                return 'forecast_recipe_unverified'
        if not 0 <= record['value'] <= 100:
            return 'forecast_recipe_unverified'
    elif series.startswith('forecast_daily_'):
        station = series.removeprefix('forecast_daily_')
        try:
            run_day = _date(period, 'forecast model run date')
            run, rp = _when(record.get('model_run_utc'), 'model run')
            valid, vp = _when(record.get('valid_at_utc'), 'forecast valid time')
        except ValueError:
            return 'forecast_daily_provenance_unverified'
        if (station not in _FORECAST_STATIONS or record.get('station') != station or
                not dt.date(run_day.year, 4, 1) <= run_day <= dt.date(run_day.year, 5, 20) or
                rp != 'instant' or vp != 'instant' or run.date() != run_day or
                run.time() != dt.time(0, 0) or valid != run + dt.timedelta(hours=24) or
                record['observation_date'] != period or not 0 <= record['value'] <= 100):
            return 'forecast_daily_provenance_unverified'
        if _definitely_before(record.get('available_at'), record.get('model_run_utc')):
            return 'forecast_daily_provenance_unverified'
    return None


def _project_ratio_recipe_status(record):
    recipes = {
        'nitrogen_price_ratio': ('nitrogen_price_current_april', 'nitrogen_price_previous_april'),
        'q1_shipment_ratio': ('q1_shipments_current', 'q1_shipments_previous'),
    }
    series = record['series']
    if series not in recipes:
        return None
    components = record.get('constituent_keys')
    if (record.get('derivation') != 'ratio' or not isinstance(components, (list, tuple)) or
            len(components) != 2 or
            any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in components)):
        return 'ratio_recipe_unverified'
    actual = [tuple(p) for p in components]
    expected = [(name, record['period']) for name in recipes[series]]
    if actual != expected:
        return 'ratio_recipe_unverified'
    return None


def validate_record(record):
    required = {'series', 'period', 'value', 'observation_date', 'source_url',
                'source_version', 'available_at', 'retrieved_at', 'evidence_type'}
    if not isinstance(record, dict) or not required.issubset(record):
        raise ValueError('Incomplete provenance record')
    for field in ('series', 'period', 'source_url', 'source_version', 'evidence_type'):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f'Missing {field}')
    url = urlparse(record['source_url'])
    if url.scheme != 'https' or not url.hostname:
        raise ValueError('Source URL must be HTTPS')
    observed = _date(record['observation_date'], 'observation_date')
    for field in ('available_at', 'retrieved_at', 'vintage_as_of'):
        _when(record.get(field), field)
    if record['series'] in _REALIZED_PERIODS:
        period = record['period']
        if len(period) != 4 or not period.isascii() or not period.isdigit():
            raise ValueError('Realized project series require a four-digit year period')
        offset, month, day = _REALIZED_PERIODS[record['series']]
        try:
            expected = dt.date(int(period) + offset, month, day)
        except ValueError as exc:
            raise ValueError('Invalid realized project period') from exc
        if observed != expected:
            raise ValueError('Realized project observation date does not match its period')
        if _definitely_before(record.get('available_at'), record['observation_date']):
            raise ValueError('Availability predates realized project observation')
    if isinstance(record['value'], bool) or not isinstance(record['value'], (int, float)) or not math.isfinite(record['value']):
        raise ValueError('Observation value must be finite and numeric')
    if record['series'] in {
        'nitrogen_price_current_april', 'nitrogen_price_previous_april',
        'nitrogen_price_ratio', 'q1_shipments_current',
        'q1_shipments_previous', 'q1_shipment_ratio',
    } and record['value'] <= 0:
        raise ValueError('Project price and shipment values must be positive')
    if record['evidence_type'] not in ('captured', 'verified_immutable_document', 'verified_asof_vintage', 'unverified'):
        raise ValueError('Unsupported evidence type')
    components = record.get('constituent_keys', [])
    derivation = record.get('derivation')
    if not isinstance(components, (list, tuple)):
        raise ValueError('Invalid constituent list')
    if components and derivation not in ('ratio', 'mean'):
        raise ValueError('Constituent keys require a supported derivation')
    if derivation is not None and (derivation not in ('ratio', 'mean') or not components):
        raise ValueError('A supported derivation requires constituent keys')
    if record['evidence_type'] in ('verified_immutable_document', 'verified_asof_vintage'):
        if record.get('source_verified') is not True or not str(record.get('verification_note') or '').strip():
            raise ValueError('Verified evidence requires a verification note and explicit source verification')
    digest = record.get('sha256')
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or
                               any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('Invalid SHA-256 string')
    return record


def _capture_integrity_status(record, capture_root):
    reference = record.get('capture_reference')
    if not isinstance(reference, str) or not reference.strip() or not record.get('sha256'):
        return 'capture_evidence_incomplete'
    reference = reference.strip()
    parsed = urlparse(reference)
    if parsed.scheme:
        if parsed.scheme == 'https' and parsed.hostname and parsed.username is None and parsed.password is None:
            return 'capture_external_unverified'
        return 'capture_evidence_incomplete'
    if capture_root is None:
        return 'capture_root_unconfigured'
    try:
        root_alias = Path(capture_root).absolute()
        root = root_alias.resolve(strict=True)
        if not root.is_dir():
            return 'capture_root_invalid'
        supplied = Path(reference)
        if '..' in supplied.parts:
            return 'capture_path_outside_root'
        path = supplied if supplied.is_absolute() else root / supplied
        try:
            relative = path.relative_to(root)
        except ValueError:
            try:
                relative = path.relative_to(root_alias)
            except ValueError:
                return 'capture_path_outside_root'
        if not relative.parts:
            return 'capture_unavailable'
        if not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_DIRECTORY'):
            return 'capture_platform_unsupported'
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for component in relative.parts[:-1]:
                following = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=directory)
                os.close(directory)
                directory = following
            fd = os.open(relative.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=directory)
            try:
                metadata = os.fstat(fd)
                if not stat.S_ISREG(metadata.st_mode):
                    return 'capture_unavailable'
                limit = 16 * 1024 * 1024
                if metadata.st_size > limit:
                    return 'capture_too_large'
                digest = hashlib.sha256()
                total = 0
                while True:
                    block = os.read(fd, 65536)
                    if not block:
                        break
                    total += len(block)
                    if total > limit:
                        return 'capture_too_large'
                    digest.update(block)
                return None if digest.hexdigest() == record['sha256'] else 'capture_checksum_mismatch'
            finally:
                os.close(fd)
        finally:
            os.close(directory)
    except (OSError, ValueError, RuntimeError):
        return 'capture_unavailable'


def record_status(record, cutoff, *, capture_root=None, require_capture=False):
    validate_record(record)
    _, cutoff_precision = _when(cutoff, 'cutoff')
    if cutoff_precision == 'unknown':
        raise ValueError('Missing cutoff')
    availability = _relation(record.get('available_at'), cutoff)
    if availability == 'after':
        if record['evidence_type'] == 'unverified':
            return 'unverified_late_claim'
        if record['source_version'].strip().lower() in ('unknown', 'unverified', 'none'):
            return 'version_unknown'
        if record['evidence_type'] in ('verified_immutable_document', 'verified_asof_vintage') and not record.get('retrieved_at'):
            return 'verification_retrieval_unknown'
        if _definitely_before(record.get('retrieved_at'), record.get('available_at')):
            return 'retrieval_before_availability'
        if record['evidence_type'] == 'captured':
            if not record.get('retrieved_at'):
                return 'capture_time_unknown'
            integrity = _capture_integrity_status(record, capture_root)
            if integrity:
                return integrity
        elif require_capture or record.get('capture_reference') or record.get('sha256'):
            integrity = _capture_integrity_status(record, capture_root)
            if integrity:
                return integrity
        return 'after_cutoff'
    if availability == 'unknown_intraday':
        return 'availability_time_unknown'
    if availability == 'missing':
        return 'availability_unknown'
    if record['evidence_type'] == 'unverified':
        return 'source_or_vintage_unverified'
    if record['source_version'].strip().lower() in ('unknown', 'unverified', 'none'):
        return 'version_unknown'
    if record['evidence_type'] in ('verified_immutable_document', 'verified_asof_vintage') and not record.get('retrieved_at'):
        return 'verification_retrieval_unknown'
    if _definitely_before(record.get('retrieved_at'), record.get('available_at')):
        return 'retrieval_before_availability'
    if record['evidence_type'] == 'captured':
        captured = _relation(record.get('retrieved_at'), cutoff)
        if captured != 'before':
            return 'capture_not_established_by_cutoff'
        integrity = _capture_integrity_status(record, capture_root)
        if integrity:
            return integrity
    if record['evidence_type'] == 'verified_asof_vintage':
        vintage, vp = _when(record.get('vintage_as_of'), 'vintage_as_of')
        decision, dp = _when(cutoff, 'cutoff')
        if vintage is None or vp != dp or vintage != decision:
            return 'exact_asof_vintage_unverified'
        if _definitely_before(record.get('retrieved_at'), record.get('vintage_as_of')):
            return 'retrieval_before_vintage'
        retrieved, rp = _when(record.get('retrieved_at'), 'retrieved_at')
        if vp == 'instant' and rp == 'day' and retrieved == vintage.date():
            return 'retrieval_time_unknown'
    if record['evidence_type'] in ('verified_immutable_document', 'verified_asof_vintage') and (
            require_capture or record.get('capture_reference') or record.get('sha256')):
        integrity = _capture_integrity_status(record, capture_root)
        if integrity:
            return integrity
    forecast_status = _forecast_recipe_status(record)
    if forecast_status:
        return forecast_status
    ratio_status = _project_ratio_recipe_status(record)
    if ratio_status:
        return ratio_status
    return 'eligible'


def replay(records, cutoff, required_keys, *, capture_root=None, require_capture=False):
    _, precision = _when(cutoff, 'cutoff')
    if precision == 'unknown':
        raise ValueError('Missing cutoff')
    required = [(str(a), str(b)) for a, b in required_keys]
    if not required:
        raise ValueError('At least one required input is needed')
    if len(set(required)) != len(required):
        raise ValueError('Duplicate required input key')
    groups = {}
    seen = set()
    for raw in records:
        record = validate_record(raw)
        key = _key(record)
        identity = key + (record['source_version'],)
        if identity in seen:
            raise ValueError(f'Duplicate observation version: {identity}')
        seen.add(identity)
        groups.setdefault(key, []).append(record)
    report, selected, dependencies = [], {}, {}
    visited, visiting = set(), set()
    def include(key):
        if key in visiting:
            raise ValueError(f'Cyclic provenance dependencies at {key}')
        if key in visited:
            return
        visiting.add(key)
        candidates = groups.get(key, [])
        candidate_status = [(r, record_status(r, cutoff, capture_root=capture_root, require_capture=require_capture)) for r in candidates]
        known = [r for r, status in candidate_status if status == 'eligible']
        ambiguous = [status for _, status in candidate_status if status not in ('eligible', 'after_cutoff')]
        status = 'missing' if not candidates else None
        chosen = None
        if ambiguous:
            status = 'unverified_candidates'
        elif not known and candidates:
            status = 'after_cutoff'
        elif known:
            times = [_when(r['available_at'], 'available_at') for r in known]
            if len({p for _, p in times}) != 1 or len({t for t, _ in times}) != len(times):
                status = 'ambiguous_version_order'
            else:
                chosen = max(zip(known, times), key=lambda pair: pair[1][0])[0]
                status = 'eligible'
        deps = set()
        if chosen is not None:
            for pair in chosen.get('constituent_keys', []):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(f'Invalid constituent key for {key}')
                dep = (str(pair[0]), str(pair[1]))
                if not all(dep):
                    raise ValueError(f'Empty constituent key for {key}')
                deps.add(dep)
            dependencies[key] = sorted(deps)
            for dep in dependencies[key]:
                include(dep)
            selected[key] = chosen
        else:
            dependencies[key] = []
        report.append({'series': key[0], 'period': key[1], 'status': status,
                       'candidate_statuses': [s for _, s in candidate_status],
                       'candidate_count': len(candidates),
                       'selected_version': chosen['source_version'] if chosen else None,
                       'selected_value': chosen['value'] if chosen else None})
        visiting.remove(key)
        visited.add(key)
    for key in required:
        include(key)
    required = [(r['series'], r['period']) for r in report]
    by_key = {(r['series'], r['period']): r for r in report}
    for key in required:
        item = by_key[key]
        if item['status'] == 'eligible' and any(by_key[dep]['status'] != 'eligible'
                                               for dep in dependencies.get(key, [])):
            item['status'] = 'blocked_by_constituent'
            item['selected_version'] = None
            item['selected_value'] = None
            selected.pop(key, None)
        if item['status'] == 'eligible' and any(
                _definitely_before(selected[key]['available_at'], selected[dep]['available_at'])
                for dep in dependencies.get(key, [])):
            item['status'] = 'derived_availability_mismatch'
            item['selected_version'] = None
            item['selected_value'] = None
            selected.pop(key, None)
        if item['status'] == 'eligible':
            derived_time, derived_precision = _when(selected[key]['available_at'], 'available_at')
            for dep in dependencies.get(key, []):
                component_time, component_precision = _when(selected[dep]['available_at'], 'available_at')
                if derived_precision != component_precision:
                    derived_day = derived_time.date() if derived_precision == 'instant' else derived_time
                    component_day = component_time.date() if component_precision == 'instant' else component_time
                    if derived_day == component_day:
                        item['status'] = 'derived_availability_order_unknown'
                        item['selected_version'] = None
                        item['selected_value'] = None
                        selected.pop(key, None)
                        break
        if item['status'] == 'eligible' and selected[key].get('derivation') is not None:
            record = selected[key]
            formula = record['derivation']
            components = [(str(p[0]), str(p[1])) for p in record.get('constituent_keys', [])]
            if formula not in ('ratio', 'mean') or not components:
                raise ValueError(f'Unsupported or incomplete derivation for {key}')
            values = [float(selected[component]['value']) for component in components]
            if formula == 'ratio':
                if len(values) != 2 or values[1] == 0:
                    raise ValueError(f'Invalid ratio components for {key}')
                rebuilt = values[0] / values[1]
                tolerance = 1e-12
            else:
                rebuilt = sum(values) / len(values)
                tolerance = 1e-5
            if not math.isclose(float(record['value']), rebuilt, rel_tol=1e-10, abs_tol=tolerance):
                item['status'] = 'derived_value_mismatch'
                item['selected_version'] = None
                item['selected_value'] = None
                selected.pop(key, None)
    return {'cutoff': cutoff, 'cutoff_precision': precision, 'records': selected,
            'report': report, 'complete': len(selected) == len(required),
            'limitation': ('Conditional on caller-supplied metadata and candidate completeness; '
                           'not independent proof of historical publication, source contents or absence of later revisions.')}


def require_replay(records, cutoff, required_keys, *, capture_root=None, require_capture=False):
    result = replay(records, cutoff, required_keys, capture_root=capture_root, require_capture=require_capture)
    if not result['complete']:
        reasons = ', '.join(f"{r['series']}:{r['period']}={r['status']}" for r in result['report'] if r['status'] != 'eligible')
        raise ReplayUnavailable('Point-in-time replay incomplete: ' + reasons)
    return result


def _entry(series, period, value, observation_date, source_url, source_version,
           available_at=None, retrieved_at=None, **metadata):
    record = {'series': series, 'period': str(period), 'value': float(value),
              'observation_date': observation_date, 'source_url': source_url,
              'source_version': source_version or 'unknown', 'available_at': available_at,
              'retrieved_at': retrieved_at, 'evidence_type': 'unverified', **metadata}
    return validate_record(record)


def _present_finite(value):
    try:
        return value is not None and not isinstance(value, bool) and math.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        return False


def _price_keys(year):
    return [('nitrogen_price_ratio', year), ('nitrogen_price_current_april', year),
            ('nitrogen_price_previous_april', year)]


def _weather_keys(year):
    return [('forecast_probability', year)] + [('forecast_station_' + station, year)
            for station in ('KAMW', 'KCID', 'KSUX')]


def _shipment_keys(year):
    return [('q1_shipment_ratio', year), ('q1_shipments_current', year),
            ('q1_shipments_previous', year)]


def project_ledger(annual_data, snapshot, volumes, forecast_manifest, old_prices, segments,
                   browser_price_evidence=None):
    def rows(obj):
        return obj.to_dict('records') if hasattr(obj, 'to_dict') else list(obj)
    annual, manifest = rows(annual_data), rows(forecast_manifest)
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get('revenue'), list):
        raise ValueError('Missing revenue snapshot')
    by_manifest = {}
    for item in manifest:
        key = (int(item['year']), str(item['station']))
        if key in by_manifest:
            raise ValueError('Duplicate forecast manifest station-year')
        by_manifest[key] = item
    price_by_year = {}
    for item in list(snapshot.get('price_index', [])) + list(old_prices):
        y = int(item['year'])
        if y in price_by_year:
            raise ValueError('Duplicate price year')
        price_by_year[y] = item
    browser_evidence = {}
    for wrapped in browser_price_evidence or []:
        if not isinstance(wrapped, dict) or not isinstance(wrapped.get('record'), dict):
            raise ValueError('Invalid browser price evidence')
        raw = wrapped['record']
        canonical = json.dumps(raw, sort_keys=True, ensure_ascii=False,
                               separators=(',', ':'), allow_nan=False)
        digest = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
        if digest != wrapped.get('canonical_sha256'):
            raise ValueError('Browser price transcription checksum mismatch')
        key = (int(raw['forecast_year']), raw['observation_date'])
        if key in browser_evidence:
            raise ValueError('Duplicate browser price observation')
        start, sp = _when(raw.get('retrieval_start_utc'), 'browser retrieval start')
        end, ep = _when(raw.get('retrieval_end_utc'), 'browser retrieval end')
        if sp != 'instant' or ep != 'instant' or end < start:
            raise ValueError('Invalid browser price retrieval interval')
        browser_evidence[key] = (raw, digest)
    ledger, seen_annual = [], set()
    for item in snapshot['revenue']:
        y = int(item['year'])
        accession = str(item.get('q2_accession') or 'unknown')
        ledger.append(_entry('q2_sales', y, item['q2_usd'], f'{y}-06-30',
            'https://www.sec.gov/edgar/browse/?CIK=1324404', accession,
            item.get('q2_filed'), None, units='USD',
            note='Snapshot transcription; accession and source contents not independently verified by replay.'))
    for item in annual:
        y = int(item['year'])
        if y in seen_annual:
            raise ValueError('Duplicate annual feature year')
        seen_annual.add(y)
        station_rows = [by_manifest.get((y, s)) for s in ('KAMW', 'KCID', 'KSUX')]
        sources = [r.get('source_url') for r in station_rows
                   if r and isinstance(r.get('source_url'), str) and r.get('source_url')]
        timestamps = [r.get('retrieved_at_utc') for r in station_rows
                      if r and isinstance(r.get('retrieved_at_utc'), str) and r.get('retrieved_at_utc')]
        means = [r.get('station_mean_p12_pct') for r in station_rows
                 if r and _present_finite(r.get('station_mean_p12_pct'))]
        if len(means) == 3 and not math.isclose(sum(float(x) for x in means) / 3,
                                                float(item['forecast_mean_p12_pct']), rel_tol=0, abs_tol=1e-5):
            raise ValueError(f'Forecast summary differs from source manifest for {y}')
        ledger.append(_entry('forecast_probability', y, item['forecast_mean_p12_pct'], f'{y}-05-20',
            'https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py', 'unknown',
            None, max(timestamps) if len(timestamps) == 3 else None,
            units='percent', constituent_sources=sources,
            constituent_keys=[['forecast_station_' + station, str(y)] for station in ('KAMW', 'KCID', 'KSUX')],
            derivation='mean',
            note='Mean of station summaries. Runtime is not publication time; current archive may be revised.'))
        for station, station_row in zip(('KAMW', 'KCID', 'KSUX'), station_rows):
            if not station_row or not _present_finite(station_row.get('station_mean_p12_pct')):
                continue
            station_source = station_row.get('source_url')
            if not isinstance(station_source, str) or not station_source:
                continue
            retrieved = station_row.get('retrieved_at_utc')
            if not isinstance(retrieved, str) or not retrieved:
                retrieved = None
            available = station_row.get('availability_time_utc')
            if not isinstance(available, str) or not available:
                available = None
            version = station_row.get('source_version')
            if not isinstance(version, str) or not version:
                version = 'unknown'
            valid_days = station_row.get('valid_days')
            ledger.append(_entry('forecast_station_' + station, y,
                station_row['station_mean_p12_pct'], f'{y}-05-20', station_source,
                version, available, retrieved, units='percent', station=station,
                valid_days=int(valid_days) if _present_finite(valid_days) else None,
                original_daily_record_reference='iowa_forecast_input_audit/forecast_daily_private.jsonl.gz',
                note='Retrieved station summary; selected daily archive rows may be incomplete or revised.'))
        p = price_by_year.get(y)
        if p:
            source = p.get('source') or p.get('current_source_url')
            if not source:
                raise ValueError('Price source missing')
            if p.get('current_april') is not None and p.get('previous_april') is not None:
                current, previous = float(p['current_april']), float(p['previous_april'])
                if min(current, previous) <= 0 or not math.isclose(float(item['price_ratio']), current / previous,
                                                                  rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f'Price ratio differs from saved observations for {y}')
            release = p.get('current_release_date')
            old_release = p.get('previous_release_date')
            if release and old_release:
                release = max(_date(release, 'price release'), _date(old_release, 'previous price release')).isoformat()
            else:
                release = None
            ledger.append(_entry('nitrogen_price_ratio', y, item['price_ratio'], f'{y}-04-01',
                source, str(p.get('vintage_label') or 'manual reconstruction'), release, None,
                constituent_keys=[['nitrogen_price_current_april', str(y)],
                                  ['nitrogen_price_previous_april', str(y)]],
                derivation='ratio',
                note='As-of label or reconstructed release dates are transcribed; version contents unverified.'))
            for prefix, observation_year in (('current', y), ('previous', y - 1)):
                value = p.get(prefix + '_april')
                if not _present_finite(value):
                    continue
                individual_source = p.get(prefix + '_source_url') or p.get('source')
                if not isinstance(individual_source, str) or not individual_source:
                    continue
                recorded_observation = p.get(prefix + '_observation') or f'{observation_year}-04-01'
                expected_observation = f'{observation_year}-04-01'
                if recorded_observation != expected_observation:
                    raise ValueError(f'Unexpected {prefix} April observation date for {y}')
                recorded_asof = p.get('as_of')
                vintage = p.get('vintage_label') or 'manual reconstruction'
                browser_item = browser_evidence.get((y, recorded_observation))
                browser = {}
                if browser_item:
                    record, transcription_sha = browser_item
                    observed = float(record['value'])
                    if not math.isfinite(observed):
                        raise ValueError('Invalid browser price transcription value')
                    browser = {'browser_observed_value': observed,
                               'browser_matches_snapshot': math.isclose(observed, float(value), rel_tol=1e-9, abs_tol=1e-7),
                               'browser_source_url': record['source_url'],
                               'browser_source_version': record['source_version'],
                               'browser_retrieval_start_utc': record['retrieval_start_utc'],
                               'browser_retrieval_end_utc': record['retrieval_end_utc'],
                               'browser_transcription_sha256': transcription_sha}
                ledger.append(_entry('nitrogen_price_' + prefix + '_april', y, value,
                    recorded_observation, individual_source, str(vintage),
                    p.get(prefix + '_release_date'), None, units='producer price index',
                    recorded_as_of=recorded_asof,
                    **browser,
                    note='As-of label and/or release date transcribed; original price version has not been independently verified.'))
    seen_volume = set()
    for item in volumes:
        y = int(item['year'])
        if y in seen_volume:
            raise ValueError('Duplicate shipment year')
        seen_volume.add(y)
        for prefix, quarter_year in (('current', y), ('previous', y - 1)):
            ledger.append(_entry('q1_shipments_' + prefix, y, item[prefix + '_q1_tons'],
                f'{quarter_year}-03-31', item[prefix + '_source'],
                item[prefix + '_accession'], item[prefix + '_filed'], None,
                units=item.get('units'),
                note='Snapshot transcription; dated filing and values not independently verified by replay.'))
        current = float(item['current_q1_tons'])
        previous = float(item['previous_q1_tons'])
        if not all(math.isfinite(v) and v > 0 for v in (current, previous)):
            raise ValueError('Invalid shipment ratio components')
        availability = max(_date(item['current_filed'], 'current shipment filing'),
                           _date(item['previous_filed'], 'previous shipment filing')).isoformat()
        ledger.append(_entry('q1_shipment_ratio', y, current / previous, f'{y}-03-31',
            item['current_source'], str(item['current_accession']) + '+' + str(item['previous_accession']),
            availability, None, units='ratio',
            constituent_sources=[item['current_source'], item['previous_source']],
            constituent_keys=[['q1_shipments_current', str(y)], ['q1_shipments_previous', str(y)]],
            derivation='ratio',
            note='Calculated from two transcribed company filing values; sources unverified by replay.'))
    seen_segment = set()
    for item in segments:
        y = int(item['year'])
        if y in seen_segment:
            raise ValueError('Duplicate phosphate segment year')
        seen_segment.add(y)
        ledger.append(_entry('q2_phosphate_sales', y, item['phosphate_q2_usd'],
            f'{y}-06-30', item['source_url'], str(item.get('basis') or 'manual transcription'),
            item.get('filed_date'), None, units='USD',
            note='Snapshot transcription; source document and original values unverified by replay.'))
    return ledger


def project_replay_report(ledger, years, *, capture_root=None, require_capture=False):
    result = []
    for y in years:
        cutoff = f'{int(y)}-05-31'
        keys = [('q2_sales', int(y)-1)] + _price_keys(int(y)) + _weather_keys(int(y)) + _shipment_keys(int(y))
        check = replay(ledger, cutoff, keys, capture_root=capture_root, require_capture=require_capture)
        for item in check['report']:
            result.append({'decision_date': cutoff, **item})
    return result


def project_fold_replay(ledger, decision_year, training_years, *, phosphate_adjusted=True, capture_root=None, require_capture=False):
    year = int(decision_year)
    training = [int(y) for y in training_years]
    if len(training) != len(set(training)) or any(y >= year for y in training):
        raise ValueError('Training years must be unique and earlier than decision year')
    report = []
    def check(cutoff, keys, role, row_year):
        output = replay(ledger, cutoff, keys, capture_root=capture_root, require_capture=require_capture)
        for item in output['report']:
            report.append({'decision_date': f'{year}-05-31', 'tested_as_of': cutoff,
                           'role': role, 'row_year': row_year, **item})
    check(f'{year}-05-31', [('q2_sales', year - 1)] + _price_keys(year) +
          _weather_keys(year) + _shipment_keys(year), 'decision_inputs', year)
    for y in training:
        historic_cutoff = f'{y}-05-31'
        check(historic_cutoff, [('q2_sales', y - 1)] + _price_keys(y) + _weather_keys(y),
              'training_features', y)
        check(f'{year}-05-31', [('q2_sales', y)], 'training_target', y)
        if phosphate_adjusted and 2010 <= y <= 2015:
            check(historic_cutoff, [('q2_phosphate_sales', y - 1)], 'training_baseline_adjustment', y)
            check(f'{year}-05-31', [('q2_phosphate_sales', y)], 'training_target_adjustment', y)
    return report


def save_private_ledger(records, report, directory):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('provenance_private.json', records), ('replay_report_private.json', report)]:
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description='Reconstruct admissible point-in-time values from a provenance ledger.')
    parser.add_argument('--ledger', required=True, type=Path, help='JSON array of provenance records')
    parser.add_argument('--requirements', required=True, type=Path, help='JSON array of [series, period] pairs')
    parser.add_argument('--cutoff', required=True, help='ISO date or timezone-aware timestamp')
    parser.add_argument('--output', required=True, type=Path, help='output JSON report')
    parser.add_argument('--capture-root', type=Path, help='directory containing captured source artifacts')
    parser.add_argument('--require-capture', action='store_true', help='Require hash-checked artifacts for every evidence type')
    args = parser.parse_args(argv)
    records = json.loads(args.ledger.read_text(encoding='utf-8'))
    requirements = json.loads(args.requirements.read_text(encoding='utf-8'))
    result = replay(records, args.cutoff, requirements, capture_root=args.capture_root, require_capture=args.require_capture)
    portable = {**result, 'records': list(result['records'].values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(portable, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(f"Replay {'complete' if result['complete'] else 'incomplete'}: {len(result['records'])}/{len(result['report'])} required inputs supported")
    if not result['complete']:
        for item in result['report']:
            if item['status'] != 'eligible':
                print(f"  {item['series']}:{item['period']} {item['status']} {item['candidate_statuses']}")
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
