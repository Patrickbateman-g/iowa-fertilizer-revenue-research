from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from html.parser import HTMLParser
import json
import re

FIELDS = ('source_url', 'retrieved_at', 'title_html', 'thead_html', 'rows_html')
TITLE = 'Producer Price Index by Commodity: Chemicals and Allied Products: Nitrogenates'
HISTORICAL_TITLE = 'Producer Price Index by Commodity for Chemicals and Allied Products: Nitrogenates'
SOURCE_URL = 'https://alfred.stlouisfed.org/series?seid=WPU065201'


def capture_digest(capture):
    if not isinstance(capture, dict) or any(not isinstance(capture.get(k), str) or not capture[k] for k in FIELDS):
        raise ValueError('Missing capture field')
    canonical = json.dumps({k: capture[k] for k in FIELDS}, sort_keys=True,
                           ensure_ascii=False, separators=(',', ':')).encode('utf8')
    return sha256(canonical).hexdigest()


def batch_digest(captures, expected_years):
    try:
        payload = json.dumps({'captures': captures, 'expected_years': expected_years},
                             sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                             allow_nan=False).encode('utf8')
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid capture batch') from exc
    return sha256(payload).hexdigest()


def require_same_batch(captures, expected_years, previous_digest):
    digest = batch_digest(captures, expected_years)
    if not isinstance(previous_digest, str) or digest != previous_digest:
        raise ValueError('ALFRED capture batch changed; rerun replay in a fresh kernel')
    return digest


class _Fragment(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None
        self.text = ''

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('script', 'style', 'table'):
            raise ValueError('Unexpected fragment structure')
        if tag == 'tr':
            if self.row is not None: raise ValueError('Nested row')
            self.row = []
        elif tag in ('th', 'td'):
            if self.row is None or self.cell is not None: raise ValueError('Unexpected cell')
            if any(k in attrs for k in ('colspan', 'rowspan')): raise ValueError('Spanning cells not supported')
            self.cell = {'tag': tag, 'scope': attrs.get('scope'), 'text': ''}
        elif tag == 'br':
            if self.cell is not None: self.cell['text'] += ' '
            self.text += ' '

    def handle_endtag(self, tag):
        if tag in ('th', 'td'):
            if self.cell is None or self.cell['tag'] != tag: raise ValueError('Unbalanced cell')
            self.cell['text'] = ' '.join(self.cell['text'].split())
            self.row.append(self.cell)
            self.cell = None
        elif tag == 'tr':
            if self.row is None or self.cell is not None: raise ValueError('Unbalanced row')
            self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        self.text += data
        if self.cell is not None: self.cell['text'] += data


def _parse(value):
    p = _Fragment()
    p.feed(value)
    p.close()
    if p.row is not None or p.cell is not None: raise ValueError('Incomplete fragment')
    return p


def _iso_date(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Expected ISO date')
    try: return date.fromisoformat(value)
    except ValueError as exc: raise ValueError('Invalid date') from exc


def _number(value):
    if not re.fullmatch(r'\d+(?:\.\d+)?', value): raise ValueError('Invalid chart value')
    try: number = Decimal(value)
    except InvalidOperation as exc: raise ValueError('Invalid chart value') from exc
    if not number.is_finite() or number <= 0: raise ValueError('Invalid chart value')
    return number


def inspect_capture(capture, decision_year):
    if type(decision_year) is not int or not 2 <= decision_year <= 9999:
        raise ValueError('Invalid decision year')
    digest = capture_digest(capture)
    if capture.get('capture_sha256') != digest:
        raise ValueError('Capture digest mismatch')
    if capture['source_url'] != SOURCE_URL:
        raise ValueError('Unexpected series URL')
    try:
        retrieved = datetime.fromisoformat(capture['retrieved_at'].replace('Z', '+00:00'))
        if retrieved.tzinfo is None or retrieved.utcoffset() is None: raise ValueError
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid retrieval timestamp') from exc
    vintage = date(decision_year, 5, 31)
    if retrieved.astimezone(timezone.utc).date() < vintage:
        raise ValueError('Retrieval precedes vintage')
    title = ' '.join(_parse(capture['title_html']).text.split())
    if title != TITLE + ' (WPU065201)': raise ValueError('Unexpected title or series ID')
    header = _parse(capture['thead_html']).rows
    if len(header) != 1 or len(header[0]) < 2:
        raise ValueError('Unexpected chart header')
    cells = header[0]
    if any(x['tag'] != 'th' for x in cells) or cells[0]['text'] != 'DateTime':
        raise ValueError('Unexpected chart columns')
    expected = {title + ' Vintage: ' + vintage.isoformat() for title in (TITLE, HISTORICAL_TITLE)}
    positions = [i for i, cell in enumerate(cells) if cell['text'] in expected]
    if len(positions) != 1: raise ValueError('Vintage column missing or ambiguous')
    col = positions[0]
    for cell in cells[1:]:
        prefixes = [title + ' Vintage: ' for title in (TITLE, HISTORICAL_TITLE)
                    if cell['text'].startswith(title + ' Vintage: ')]
        if len(prefixes) != 1: raise ValueError('Unexpected chart series')
        prefix = prefixes[0]
        _iso_date(cell['text'][len(prefix):])
    target_dates = {date(decision_year, 4, 1): 'current',
                    date(decision_year - 1, 4, 1): 'previous'}
    rows = _parse(capture['rows_html']).rows
    if len(rows) != 2: raise ValueError('Expected exactly two observation rows')
    found = {}
    for row in rows:
        if len(row) != len(cells) or row[0]['tag'] != 'th' or any(x['tag'] != 'td' for x in row[1:]):
            raise ValueError('Unexpected chart row structure')
        label = row[0]['text']
        match = re.fullmatch(r'(\d{4}-\d{2}-\d{2})\s+\d{1,2}:\d{2}:\d{2}', label)
        if not match: raise ValueError('Unexpected observation label')
        obs = _iso_date(match.group(1))
        if obs not in target_dates or obs in found: raise ValueError('Unexpected or duplicate observation')
        found[obs] = _number(row[col]['text'])
    if set(found) != set(target_dates): raise ValueError('Missing observation')
    return [{'decision_year': decision_year, 'role': target_dates[obs],
             'observation_date': obs.isoformat(), 'value': str(found[obs]),
             'as_of': vintage.isoformat(), 'source_url': SOURCE_URL,
             'retrieved_at': capture['retrieved_at'], 'capture_sha256': capture['capture_sha256'],
             'capture_kind': 'browser_dom_selected_chart_excerpt',
             'limitation': 'Later selected DOM; no proof of historical availability, original server bytes, or archive immutability.'}
            for obs in sorted(found)]


def compare_capture_to_ledger(ledger, capture, decision_year):
    if not isinstance(ledger, list) or any(not isinstance(r, dict) for r in ledger):
        raise ValueError('Ledger must be a list of records')
    facts = inspect_capture(capture, decision_year)
    out = deepcopy(ledger)
    for fact in facts:
        key = ('nitrogen_price_' + fact['role'] + '_april', str(decision_year))
        matches = [r for r in out if (r.get('series'), str(r.get('period'))) == key]
        if len(matches) != 1: raise ValueError('Missing or duplicated ledger record')
        record = matches[0]
        try: value = Decimal(str(record['value']))
        except (KeyError, InvalidOperation): raise ValueError('Invalid ledger value')
        if (not value.is_finite() or value != Decimal(fact['value']) or
                record.get('observation_date') != fact['observation_date'] or
                record.get('recorded_as_of') != fact['as_of'] or
                record.get('source_version') != fact['as_of']):
            raise ValueError('ALFRED chart and ledger mismatch for ' + str(key))
        if 'alfred_chart_comparison' in record: raise ValueError('Existing comparison')
        record['alfred_chart_comparison'] = dict(fact, matched_ledger_value=True)
    return out


def compare_captures_to_ledger(ledger, captures, expected_years):
    if not isinstance(captures, list) or not isinstance(expected_years, (list, tuple)):
        raise ValueError('Capture batch and expected years must be sequences')
    if not isinstance(ledger, list) or any(not isinstance(r, dict) for r in ledger):
        raise ValueError('Ledger must be a list of records')
    if not expected_years or any(type(y) is not int or not 2 <= y <= 9999 for y in expected_years):
        raise ValueError('Expected years must be nonempty integer years')
    if len(set(expected_years)) != len(expected_years):
        raise ValueError('Duplicate expected year')
    indexed = {}
    for item in captures:
        if not isinstance(item, dict) or set(item) != {'year', 'capture'}:
            raise ValueError('Each capture needs exactly a year and capture')
        year = item['year']
        if type(year) is not int or year not in expected_years:
            raise ValueError('Unexpected capture year')
        if year in indexed:
            raise ValueError('Duplicate capture year')
        indexed[year] = item['capture']
    parsed = {year: inspect_capture(capture, year) for year, capture in indexed.items()}
    out = deepcopy(ledger)
    diagnostics = []
    for year in sorted(expected_years):
        if year not in parsed:
            for role in ('previous', 'current'):
                diagnostics.append({'decision_year': year, 'role': role,
                                    'status': 'capture_missing', 'differences': []})
            continue
        for fact in parsed[year]:
            key = ('nitrogen_price_' + fact['role'] + '_april', str(year))
            matches = [r for r in out if (r.get('series'), str(r.get('period'))) == key]
            entry = {'decision_year': year, 'role': fact['role'], 'status': None,
                     'differences': [], 'observation_date': fact['observation_date'],
                     'source_value': fact['value'], 'capture_sha256': fact['capture_sha256']}
            if len(matches) != 1:
                entry['status'] = 'ledger_missing' if not matches else 'ledger_duplicate'
                diagnostics.append(entry)
                continue
            record = matches[0]
            if 'alfred_chart_comparison' in record:
                raise ValueError('Existing comparison')
            try:
                value = Decimal(str(record['value']))
                valid = value.is_finite()
            except (KeyError, InvalidOperation, ValueError, TypeError):
                value, valid = None, False
            if not valid or value != Decimal(fact['value']):
                entry['differences'].append('value')
            for key_name, expected in (('observation_date', fact['observation_date']),
                                       ('recorded_as_of', fact['as_of']),
                                       ('source_version', fact['as_of'])):
                if record.get(key_name) != expected:
                    entry['differences'].append(key_name)
            entry['ledger_value'] = str(value) if valid else None
            entry['status'] = 'mismatch' if entry['differences'] else 'matched'
            if entry['status'] == 'matched':
                record['alfred_chart_comparison'] = dict(fact, matched_ledger_value=True)
            diagnostics.append(entry)
    return {'ledger': out, 'diagnostics': diagnostics,
            'expected_years': sorted(expected_years), 'captured_years': sorted(indexed)}
