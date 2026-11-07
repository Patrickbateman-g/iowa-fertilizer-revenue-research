from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from html.parser import HTMLParser
import json
import re
from urllib.parse import urlsplit

FIELDS = ('source_url', 'retrieved_at', 'release_preamble_excerpt', 'table_id',
          'caption_html', 'thead_html', 'row_html')


def capture_digest(capture):
    if any(not isinstance(capture.get(k), str) or not capture[k] for k in FIELDS):
        raise ValueError('Missing or empty capture field')
    canonical = json.dumps({k: capture[k] for k in FIELDS}, ensure_ascii=False,
                           sort_keys=True, separators=(',', ':')).encode('utf-8')
    return sha256(canonical).hexdigest()


class _Cells(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.cells = []
        self.cell = None
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if self.skip:
            self.skip += 1
            return
        if tag == 'span' and 'footnoterefs' in attrs.get('class', '').lower().split():
            self.skip = 1
        elif tag in ('th', 'td'):
            if self.cell is not None:
                raise ValueError('Nested cells')
            self.cell = {'tag': tag, 'attrs': attrs, 'text': ''}
        elif tag == 'br' and self.cell is not None:
            self.cell['text'] += ' '

    def handle_endtag(self, tag):
        if self.skip:
            self.skip -= 1
        elif tag in ('th', 'td') and self.cell is not None:
            self.cell['text'] = ' '.join(self.cell['text'].split())
            self.cells.append(self.cell)
            self.cell = None

    def handle_data(self, data):
        if not self.skip and self.cell is not None:
            self.cell['text'] += data


def _cells(html):
    parser = _Cells()
    parser.feed(html)
    parser.close()
    if parser.cell is not None or parser.skip:
        raise ValueError('Incomplete HTML fragment')
    return parser.cells


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text = ''
    def handle_data(self, data):
        self.text += data
    def handle_starttag(self, tag, attrs):
        if tag == 'br': self.text += ' '


def _text(html):
    p = _Text()
    p.feed(html)
    p.close()
    return ' '.join(p.text.split())


def _publication(c):
    u = urlsplit(c['source_url'])
    match = re.fullmatch(r'/news\.release/archives/ppi_(\d{2})(\d{2})(\d{4})\.htm', u.path)
    if u.scheme != 'https' or u.netloc != 'www.bls.gov' or not match or u.query or u.fragment:
        raise ValueError('Expected canonical BLS archive URL')
    month, day, year = map(int, match.groups())
    release = date(year, month, day)
    preamble = ' '.join(c['release_preamble_excerpt'].split())
    if 'embargoed until' not in preamble.lower():
        raise ValueError('Release preamble must contain embargo notice')
    named = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})\b', preamble, re.I)
    if not named:
        raise ValueError('Release date missing from preamble')
    try:
        preamble_date = datetime.strptime(' '.join(named.groups()), '%B %d %Y').date()
    except ValueError as exc:
        raise ValueError('Invalid release date') from exc
    if preamble_date != release:
        raise ValueError('Release date and URL disagree')
    try:
        retrieved = datetime.fromisoformat(c['retrieved_at'].replace('Z', '+00:00'))
        if retrieved.tzinfo is None or retrieved.utcoffset() is None:
            raise ValueError('retrieval time must include timezone')
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid retrieval timestamp') from exc
    if retrieved.date() < release:
        raise ValueError('Capture predates release')
    return release


def inspect_capture(c, decision_year, role, cutoff):
    if c.get('capture_sha256') != capture_digest(c):
        raise ValueError('Capture digest mismatch')
    if role not in ('current', 'previous') or not isinstance(decision_year, int):
        raise ValueError('Invalid price role/year')
    if not isinstance(cutoff, date) or isinstance(cutoff, datetime):
        raise ValueError('Cutoff must be a date')
    release = _publication(c)
    observed_year = decision_year if role == 'current' else decision_year - 1
    if release.year != observed_year or release.month != (5 if role == 'current' else 9):
        raise ValueError('Unexpected first-release or September revision chronology')
    if release >= cutoff:
        raise ValueError('Release not established before decision cutoff')
    cap = _text(c['caption_html']).lower()
    if 'producer price indexes' not in cap or not re.search(r'1982\s*=\s*100', cap):
        raise ValueError('Unexpected table or index base')
    head = _cells(c['thead_html'])
    ids = [x['attrs'].get('id') for x in head if x['attrs'].get('id')]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate header id')
    groups = [x['attrs'].get('id') for x in head if x['text'].lower() == 'unadjusted index']
    if len(groups) != 1 or not groups[0]:
        raise ValueError('Unadjusted index header ambiguous')
    dates = {x['attrs'].get('id') for x in head
             if re.fullmatch(r'apr(?:il)?\.?\s*' + str(observed_year), x['text'], re.I)
             and x['attrs'].get('id')}
    if not dates:
        raise ValueError('April index date header missing')
    row = _cells(c['row_html'])
    if len(row) < 3 or row[0]['text'].lower() != 'nitrogenates' or row[1]['text'] != '06-52-01':
        raise ValueError('Unexpected commodity row')
    matches = [x for x in row[2:] if groups[0] in x['attrs'].get('headers', '').split()
               and dates.intersection(x['attrs'].get('headers', '').split())]
    if len(matches) != 1:
        raise ValueError('Target unadjusted index ambiguous or missing')
    try:
        value = Decimal(matches[0]['text'].replace(',', ''))
    except InvalidOperation as exc:
        raise ValueError('Invalid index value') from exc
    if not value.is_finite() or value <= 0:
        raise ValueError('Invalid index value')
    return {'decision_year': decision_year, 'role': role, 'value': str(value),
            'observation_date': date(observed_year, 4, 1).isoformat(),
            'release_date': release.isoformat(), 'source_url': c['source_url'],
            'retrieved_at': c['retrieved_at'], 'capture_sha256': c['capture_sha256'],
            'capture_kind': 'browser_dom_selected_excerpt',
            'limitation': 'Digest covers selected DOM only; it does not establish original server bytes, historical vintage, or absence of later revisions.'}


def compare_to_ledger(ledger, observations):
    annotated = deepcopy(ledger)
    if not isinstance(annotated, list):
        raise ValueError('Ledger must be a list')
    seen = set()
    for fact in observations:
        role, year = fact['role'], fact['decision_year']
        key = ('nitrogen_price_' + role + '_april', str(year))
        if key in seen:
            raise ValueError('Duplicate comparison')
        seen.add(key)
        matches = [r for r in annotated if (r.get('series'), str(r.get('period'))) == key]
        if len(matches) != 1:
            raise ValueError('Missing or duplicated ledger record')
        rec = matches[0]
        try:
            equivalent = Decimal(str(rec['value'])) == Decimal(fact['value'])
        except (KeyError, InvalidOperation):
            equivalent = False
        if not equivalent or rec.get('observation_date') != fact['observation_date']:
            raise ValueError('BLS/ledger mismatch for ' + str(key))
        if 'bls_comparison' in rec:
            raise ValueError('Existing BLS comparison; avoid silent overwrite')
        rec['bls_comparison'] = dict(fact, matched_ledger_value=True)
    return annotated


def compare_captures_to_ledger(ledger, captures, expected_keys):
    if not isinstance(captures, list) or not isinstance(expected_keys, (list, tuple)):
        raise ValueError('Captures and expected keys must be lists')

    def checked_key(year, role):
        if type(year) is not int or not 1 <= year <= 9999 or role not in ('current', 'previous'):
            raise ValueError('Invalid decision year or price role')
        return year, role

    expected = []
    for item in expected_keys:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError('Invalid expected capture key')
        expected.append(checked_key(*item))
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('Expected capture keys must be nonempty and unique')

    observations, seen = [], set()
    for item in captures:
        if not isinstance(item, dict):
            raise ValueError('Invalid capture item')
        if 'capture' in item:
            if not isinstance(item['capture'], dict) or any(key in item for key in FIELDS):
                raise ValueError('Ambiguous or invalid nested capture')
            capture = item['capture']
        else:
            capture = item
        key = checked_key(item.get('decision_year'), item.get('role'))
        if key in seen:
            raise ValueError('Duplicate capture key')
        seen.add(key)
        observations.append(inspect_capture(capture, key[0], key[1], date(key[0], 5, 31)))
    if seen != set(expected):
        raise ValueError('Capture keys differ from expected set')
    return compare_to_ledger(ledger, observations)
