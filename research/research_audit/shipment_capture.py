from copy import deepcopy
from datetime import datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import re
from urllib.parse import urlsplit

from sec_capture import _grid, _inside, _rows, _text

FIELDS = ('source_url', 'retrieved_at', 'year', 'selected_table_position',
          'matching_table_count', 'heading_context', 'header_rows', 'volume_row',
          'header_html', 'volume_row_html')
CONTEXT_FIELDS = ('units_row', 'units_row_html', 'units_row_index', 'volume_row_index')
PRODUCT_LABEL = r'(?:Tons of product sold|Sales volume by product tons)\s*\(\s*000s\s*\)'
TOTAL_LABEL = r'Total tons of product sold'


def capture_digest(capture):
    if not isinstance(capture, dict) or any(k not in capture for k in FIELDS):
        raise ValueError('Missing shipment capture field')
    data = json.dumps({k: capture.get(k) for k in FIELDS + CONTEXT_FIELDS}, sort_keys=True,
                      ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return sha256(data.encode('utf8')).hexdigest()


def inspect_capture(capture):
    if not isinstance(capture, dict) or capture.get('capture_sha256') != capture_digest(capture):
        raise ValueError('Shipment capture digest mismatch')
    year = capture['year']
    if isinstance(year, bool) or not isinstance(year, int) or not 2015 <= year <= 2026:
        raise ValueError('Unsupported shipment capture year')
    for name, minimum in (('selected_table_position', 0), ('matching_table_count', 1)):
        value = capture[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError('Invalid ' + name)
    source = _text(capture['source_url'], 'source URL')
    parsed = urlsplit(source)
    match = re.fullmatch(r'/Archives/edgar/data/1324404/(\d{18})/([^/]+\.html?)', parsed.path, re.I)
    if (parsed.scheme != 'https' or parsed.netloc != 'www.sec.gov' or parsed.query or
        parsed.fragment or not match or int(match.group(1)[10:12]) != year % 100):
        raise ValueError('Unexpected SEC source or accession year')
    retrieved = _text(capture['retrieved_at'], 'retrieval timestamp')
    try:
        stamp = datetime.fromisoformat(retrieved.replace('Z', '+00:00'))
        if stamp.tzinfo is None or stamp.utcoffset() is None or stamp.date().isoformat() < f'{year}-03-31':
            raise ValueError
    except ValueError as exc:
        raise ValueError('Invalid retrieval timestamp') from exc
    heading = _text(capture['heading_context'], 'heading context')
    if not re.search(r'\bCF INDUSTRIES\b', heading, re.I) or not re.search(r'\bCONSOLIDATED RESULTS\b', heading, re.I):
        raise ValueError('Expected CF consolidated results context')
    if re.search(r'\bNON[- ]GAAP\b', heading, re.I):
        raise ValueError('Non-GAAP context cannot identify consolidated volume table')
    rows = capture['header_rows']
    if not isinstance(rows, list) or any(not isinstance(r, str) for r in rows):
        raise ValueError('Invalid header transcription')
    rows = [_text(r, 'header row') for r in rows]
    if rows != _rows(capture['header_html']) or len(rows) != 2:
        raise ValueError('Header transcription differs or is ambiguous')
    if not re.fullmatch(r'Three months ended\s+March\s+31,?', rows[0], re.I):
        raise ValueError('Expected Q1 period header')
    if rows[1].split() != [str(year), str(year - 1)]:
        raise ValueError('Unexpected Q1 year order')
    volume = _text(capture['volume_row'], 'volume row')
    if _rows(capture['volume_row_html']) != [volume]:
        raise ValueError('Volume transcription differs from DOM')
    number = r'(?:\d+|\d{1,3}(?:,\d{3})+)'
    rowmatch = re.fullmatch('(' + PRODUCT_LABEL + '|' + TOTAL_LABEL + r')\s+(' + number + r')\s+(' + number + r')', volume, re.I)
    if not rowmatch:
        raise ValueError('Expected unambiguous product-tons row with units')
    separate_units = re.fullmatch(TOTAL_LABEL, rowmatch.group(1), re.I) is not None
    if separate_units:
        units_text = _text(capture.get('units_row'), 'shipment unit context')
        if re.fullmatch(r'Tons of product sold\s*\(\s*000s\s*\):', units_text, re.I) is None:
            raise ValueError('Missing separate thousand-product-tons heading')
        if _rows(capture.get('units_row_html')) != [units_text]:
            raise ValueError('Shipment unit transcription differs from DOM')
        units_grid = _grid(capture['units_row_html'])
        if len(units_grid) != 1 or [c['text'] for c in units_grid[0] if c['text']] != [units_text]:
            raise ValueError('Ambiguous separate shipment unit heading')
        ui, vi = capture.get('units_row_index'), capture.get('volume_row_index')
        if (isinstance(ui, bool) or isinstance(vi, bool) or not isinstance(ui, int) or
                not isinstance(vi, int) or ui < 0 or vi <= ui):
            raise ValueError('Separate unit heading must precede total row')
    elif any(capture.get(k) is not None for k in CONTEXT_FIELDS):
        raise ValueError('Unexpected extra unit context for directly labelled row')
    header = _grid(capture['header_html'])
    data = _grid(capture['volume_row_html'])
    if len(header) != 2 or len(data) != 1:
        raise ValueError('Ambiguous shipment column layout')
    periods = [c for c in header[0] if c['text']]
    years = [c for c in header[1] if c['text']]
    cells = [c for c in data[0] if c['text']]
    if len(periods) != 1 or len(years) != 2 or len(cells) != 3:
        raise ValueError('Unexpected shipment header or row layout')
    if [c['text'] for c in years] != [str(year), str(year-1)]:
        raise ValueError('Year cells do not match Q1 order')
    if any(not _inside(c, periods[0]) for c in years):
        raise ValueError('Year columns do not align with Q1 period')
    if re.fullmatch(TOTAL_LABEL if separate_units else PRODUCT_LABEL, cells[0]['text'], re.I) is None:
        raise ValueError('Unexpected product-tons label cell')
    for i, cell in enumerate(cells[1:]):
        if cell['text'].replace(',', '') != rowmatch.group(i+2).replace(',', '') or not _inside(cell, years[i]):
            raise ValueError('Shipment value and year columns do not align')
    values = [Decimal(v.replace(',', '')) for v in rowmatch.groups()[1:]]
    if any(not value.is_finite() or value <= 0 for value in values):
        raise ValueError('Invalid shipment value')
    return {'year': year, 'observation_date': f'{year}-03-31',
            'value': str(values[0]), 'units': 'thousand product tons',
            'source_url': source, 'accession': match.group(1),
            'retrieved_at': retrieved, 'capture_sha256': capture['capture_sha256'],
            'limitation': 'Selected DOM and its digest do not prove selection context, '
                          'original server bytes, historical availability or immutability.'}


def compare_captures_to_ledger(ledger, captures, expected_keys):
    if not isinstance(ledger, list) or not isinstance(captures, list):
        raise ValueError('Ledger and captures must be lists')
    expected = list(expected_keys)
    if (not expected or any(not isinstance(k, (list, tuple)) or len(k) != 2 or
        k[0] not in ('q1_shipments_current', 'q1_shipments_previous') or
        isinstance(k[1], bool) or not isinstance(k[1], int) or not 2016 <= k[1] <= 2026 for k in expected)):
        raise ValueError('Invalid expected shipment keys')
    keys = [(k[0], str(k[1])) for k in expected]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate expected shipment key')
    facts = [inspect_capture(c) for c in captures]
    by_year = {f['year']: f for f in facts}
    if len(by_year) != len(facts):
        raise ValueError('Duplicate source year')
    required_years = {int(y) - (series == 'q1_shipments_previous') for series, y in keys}
    if set(by_year) != required_years:
        raise ValueError('Capture source years differ from expected keys')
    result = deepcopy(ledger)
    for series, period in keys:
        candidates = [r for r in result if (r.get('series'), str(r.get('period'))) == (series, period)]
        if len(candidates) != 1:
            raise ValueError('Missing or duplicated shipment ledger record')
        record = candidates[0]
        fact = by_year[int(period) - (series == 'q1_shipments_previous')]
        try:
            equal = Decimal(str(record['value'])) == Decimal(fact['value'])
        except (KeyError, InvalidOperation):
            equal = False
        if (not equal or record.get('source_url') != fact['source_url'] or
            record.get('source_version') != fact['accession'] or
            record.get('observation_date') != fact['observation_date'] or
            record.get('units') != fact['units']):
            raise ValueError('Shipment value or source mismatch for ' + str((series, period)))
        if 'sec_q1_shipment_comparison' in record:
            raise ValueError('Existing shipment comparison; avoid silent overwrite')
        record['sec_q1_shipment_comparison'] = dict(fact, matched_ledger_value=True)
    return result
