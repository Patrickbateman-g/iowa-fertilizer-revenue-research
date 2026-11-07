from copy import deepcopy
from datetime import datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from html.parser import HTMLParser
import json
import re
from urllib.parse import urlsplit

FIELDS = ('source_url', 'retrieved_at', 'year', 'selected_table_position',
          'matching_table_count', 'heading_context', 'header_rows', 'sales_row',
          'header_html', 'sales_row_html')


def capture_digest(capture):
    if not isinstance(capture, dict) or any(k not in capture for k in FIELDS):
        raise ValueError('Missing capture field')
    canonical = json.dumps({k:capture[k] for k in FIELDS}, sort_keys=True,
        ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf8')
    return sha256(canonical).hexdigest()


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Missing ' + label)
    return ' '.join(value.split())


class _RowText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_row = False
        self.parts = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            if self.in_row:
                raise ValueError('Nested table row')
            self.in_row = True
            self.parts = []
        elif self.in_row and tag in ('td','th','br'):
            self.parts.append(' ')

    def handle_endtag(self, tag):
        if tag == 'tr':
            if not self.in_row:
                raise ValueError('Unexpected table row end')
            self.rows.append(' '.join(''.join(self.parts).split()))
            self.in_row = False
        elif self.in_row and tag in ('td','th'):
            self.parts.append(' ')

    def handle_data(self, data):
        if self.in_row:
            self.parts.append(data)


def _rows(html):
    if not isinstance(html,str) or not html.strip():
        raise ValueError('Missing HTML excerpt')
    parser = _RowText()
    parser.feed(html)
    parser.close()
    if parser.in_row:
        raise ValueError('Incomplete table row')
    return [row for row in parser.rows if row]



class _TableGrid(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_row = False
        self.cell = None
        self.cells = []
        self.position = 0
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            if self.in_row: raise ValueError('Nested row in column layout')
            self.in_row, self.cells, self.position = True, [], 0
        elif self.in_row and tag in ('td', 'th'):
            if self.cell is not None: raise ValueError('Nested cell in column layout')
            if len({name for name, _ in attrs}) != len(attrs):
                raise ValueError('Duplicate cell attribute in column layout')
            attrs = dict(attrs)
            span = attrs.get('colspan', '1')
            rowspan = attrs.get('rowspan', '1')
            if (not isinstance(span, str) or not span.isascii() or not span.isdigit() or not 1 <= int(span) <= 10000 or
                not isinstance(rowspan, str) or not rowspan.isascii() or not rowspan.isdigit() or int(rowspan) != 1):
                raise ValueError('Unsupported column span or rowspan')
            style = attrs.get('style') or ''
            if re.search(r'(?:^|;)\s*visibility\s*:\s*(?:hidden|collapse)\b', style, re.I):
                raise ValueError('Hidden table cell has ambiguous layout')
            display = re.findall(r'(?:^|;)\s*display\s*:\s*([^;]+)', style, re.I)
            if len(display) > 1:
                raise ValueError('Conflicting display declarations')
            hidden = bool(display and re.fullmatch(r'\s*none\s*(?:!\s*important\s*)?', display[0], re.I))
            self.cell = {'start': self.position, 'end': self.position + int(span), 'parts': [], 'hidden':hidden}
        elif self.cell is not None and tag == 'br':
            self.cell['parts'].append(' ')

    def handle_endtag(self, tag):
        if self.in_row and tag in ('td', 'th'):
            if self.cell is None: raise ValueError('Unexpected cell end in layout')
            c = self.cell
            c['text'] = ' '.join(''.join(c.pop('parts')).split())
            hidden = c.pop('hidden')
            if hidden:
                if c['text']:
                    raise ValueError('Hidden table cell contains text')
            else:
                self.cells.append(c)
                self.position = c['end']
            self.cell = None
        elif tag == 'tr':
            if not self.in_row or self.cell is not None: raise ValueError('Incomplete row in layout')
            self.rows.append(self.cells)
            self.in_row = False

    def handle_data(self, data):
        if self.cell is not None: self.cell['parts'].append(data)


def _grid(html):
    if not isinstance(html, str) or not html.strip(): raise ValueError('Missing HTML excerpt')
    parser = _TableGrid(); parser.feed(html); parser.close()
    if parser.in_row or parser.cell is not None: raise ValueError('Incomplete layout')
    return [row for row in parser.rows if any(cell['text'] for cell in row)]


def _inside(inner, outer):
    return outer['start'] <= inner['start'] and inner['end'] <= outer['end']


def _check_column_alignment(header_html, sales_html, year, values):
    header, sales = _grid(header_html), _grid(sales_html)
    if len(header) != 3 or len(sales) != 1:
        raise ValueError('Ambiguous column layout')
    groups = [c for c in header[0] if c['text']]
    years = [c for c in header[1] if c['text']]
    if len(groups) != 2 or len(years) != 4:
        raise ValueError('Ambiguous header cell layout')
    if [c['text'] for c in years] != [str(year), str(year-1), str(year), str(year-1)]:
        raise ValueError('Year cells do not match expected layout')
    if any(not _inside(c, groups[i//2]) for i,c in enumerate(years)):
        raise ValueError('Year cells do not align with period groups')
    sale_cells = [c for c in sales[0] if c['text']]
    if not sale_cells or sale_cells[0]['text'].lower() != 'net sales':
        raise ValueError('Unexpected sales-row label cell')
    cells = sale_cells[1:]
    if len(cells) != 8:
        raise ValueError('Expected separate dollar and value cells')
    for i in range(4):
        currency, amount = cells[2*i:2*i+2]
        if currency['text'] != '$' or amount['text'].replace(',', '') != values[i].replace(',', ''):
            raise ValueError('Unexpected sales value cell')
        if not _inside(currency, years[i]) or not _inside(amount, years[i]):
            raise ValueError('Sales cells do not align with year columns')
        if not _inside(currency, groups[i//2]) or not _inside(amount, groups[i//2]):
            raise ValueError('Sales cells do not align with period groups')


def inspect_capture(capture, year=None):
    if not isinstance(capture,dict) or capture.get('capture_sha256') != capture_digest(capture):
        raise ValueError('Capture digest mismatch')
    actual_year = capture['year']
    if isinstance(actual_year,bool) or not isinstance(actual_year,int) or not 2016 <= actual_year <= 2026:
        raise ValueError('Year outside supported CF Q2 study')
    if year is not None and (isinstance(year,bool) or year != actual_year):
        raise ValueError('Expected year differs from capture')
    for field,minimum in [('selected_table_position',0),('matching_table_count',1)]:
        value = capture[field]
        if isinstance(value,bool) or not isinstance(value,int) or value < minimum:
            raise ValueError('Invalid ' + field)
    retrieved = _text(capture['retrieved_at'], 'retrieval timestamp')
    try:
        stamp = datetime.fromisoformat(retrieved.replace('Z','+00:00'))
        if stamp.tzinfo is None or stamp.utcoffset() is None or stamp.date().isoformat() < f'{actual_year}-06-30':
            raise ValueError
    except ValueError as exc:
        raise ValueError('Invalid retrieval timestamp') from exc
    source = _text(capture['source_url'], 'source URL')
    u = urlsplit(source)
    path = re.fullmatch(r'/Archives/edgar/data/1324404/(0001324404(\d{2})\d{6})/([^/]+\.htm)', u.path, re.I)
    if (u.scheme != 'https' or u.netloc != 'www.sec.gov' or u.query or u.fragment or
        not path or int(path.group(2)) != actual_year % 100):
        raise ValueError('Unexpected CF SEC source or accession year')
    filename = path.group(3).lower()
    allowed = {f'cf-0630{actual_year}x10q.htm',f'cf-{actual_year}0630.htm'}
    if filename not in allowed:
        raise ValueError('Source is not a supported original Q2 10-Q document')
    accession = path.group(1)[:10]+'-'+path.group(1)[10:12]+'-'+path.group(1)[12:]
    heading = _text(capture['heading_context'], 'consolidated heading').upper()
    if (not re.search(r'\bCF INDUSTRIES HOLDINGS,?\s+INC\.?\b',heading) or
        not re.search(r'\bCONSOLIDATED STATEMENTS OF OPERATIONS\b',heading) or
        re.search(r'\bCONSOLIDATING STATEMENTS? OF OPERATIONS\b',heading)):
        raise ValueError('Expected CF consolidated operations heading')
    headers = capture['header_rows']
    if not isinstance(headers,list) or any(not isinstance(h,str) for h in headers):
        raise ValueError('Invalid header rows')
    headers = [_text(h,'header row') for h in headers]
    if headers != _rows(capture['header_html']):
        raise ValueError('Header transcription differs from selected DOM')
    if len(headers) != 3:
        raise ValueError('Expected three unambiguous header rows')
    if not re.fullmatch(r'three months ended\s+june\s+30,?\s+six months ended\s+june\s+30,?',headers[0],re.I):
        raise ValueError('Expected three-month then six-month headers')
    if headers[1].split() != [str(actual_year),str(actual_year-1),str(actual_year),str(actual_year-1)]:
        raise ValueError('Unexpected year order in table')
    if not re.fullmatch(r'\(\s*in millions,\s*except per share amounts\s*\)',headers[2],re.I):
        raise ValueError('Expected explicit millions and per-share units')
    sales = _text(capture['sales_row'],'net sales row')
    if _rows(capture['sales_row_html']) != [sales]:
        raise ValueError('Sales-row transcription differs from selected DOM')
    number = r'(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?'
    match = re.fullmatch(r'Net sales\s+\$\s*('+number+r')\s+\$\s*('+number+
                         r')\s+\$\s*('+number+r')\s+\$\s*('+number+r')',sales,re.I)
    if not match:
        raise ValueError('Expected four unambiguous net sales values')
    _check_column_alignment(capture['header_html'], capture['sales_row_html'], actual_year, match.groups())
    try:
        values = [Decimal(v.replace(',','')) for v in match.groups()]
    except InvalidOperation as exc:
        raise ValueError('Invalid net sales') from exc
    if any(not n.is_finite() or n <= 0 for n in values):
        raise ValueError('Net sales must be positive')
    return {'series':'q2_sales','period':str(actual_year),'observation_date':f'{actual_year}-06-30',
            'value_usd':str(values[0] * Decimal(1000000)),'source_url':source,
            'accession':accession,'retrieved_at':retrieved,'capture_sha256':capture['capture_sha256'],
            'limitation':'DOM excerpts and heading selection captured now; checksums do not prove '
                         'historical availability, immutable source bytes, or correct selection context.'}


def compare_to_ledger(ledger, facts):
    if not isinstance(ledger,list):
        raise ValueError('Ledger must be a list')
    result = deepcopy(ledger)
    seen = set()
    for fact in facts:
        key = (fact['series'],fact['period'])
        if key in seen:
            raise ValueError('Duplicate SEC comparison')
        seen.add(key)
        matches = [r for r in result if (r.get('series'),str(r.get('period'))) == key]
        if len(matches) != 1:
            raise ValueError('Missing or duplicated ledger record')
        record = matches[0]
        try:
            equal = Decimal(str(record['value'])) == Decimal(fact['value_usd'])
        except (KeyError,InvalidOperation):
            equal = False
        if not equal or record.get('observation_date') != fact['observation_date']:
            raise ValueError('SEC/ledger mismatch for '+str(key))
        if 'sec_10q_comparison' in record:
            raise ValueError('Existing SEC comparison; avoid silent overwrite')
        record['sec_10q_comparison'] = dict(fact,matched_ledger_value=True)
    return result


def compare_captures_to_ledger(ledger, captures, expected_years):
    if not isinstance(captures, list):
        raise ValueError('SEC captures must be a list')
    expected = list(expected_years)
    if (not expected or any(isinstance(y, bool) or not isinstance(y, int)
                            or not 2016 <= y <= 2026 for y in expected)
            or len(set(expected)) != len(expected)):
        raise ValueError('Expected years must be unique supported years')
    facts = [inspect_capture(c) for c in captures]
    years = [int(f['period']) for f in facts]
    if len(set(years)) != len(years) or set(years) != set(expected):
        raise ValueError('SEC capture years must exactly match expected years')
    return compare_to_ledger(ledger, facts)


def attach_index_captures(ledger, captures, expected_years):
    from sec_index_capture import inspect_index
    if not isinstance(ledger, list) or not isinstance(captures, list):
        raise ValueError('Ledger and index captures must be lists')
    expected = list(expected_years)
    if (not expected or any(isinstance(y, bool) or not isinstance(y, int)
                            or not 2016 <= y <= 2026 for y in expected)
            or len(set(expected)) != len(expected)):
        raise ValueError('Expected index years must be unique supported years')
    if any(not isinstance(c, dict) for c in captures):
        raise ValueError('Invalid index capture')
    years = [c.get('year') for c in captures]
    if (any(isinstance(y, bool) or not isinstance(y, int) for y in years)
            or len(set(years)) != len(years) or set(years) != set(expected)):
        raise ValueError('SEC index years must exactly match expected years')
    result = deepcopy(ledger)
    for capture in captures:
        year = capture['year']
        records = [r for r in result if r.get('series') == 'q2_sales'
                   and str(r.get('period')) == str(year)]
        if len(records) != 1:
            raise ValueError('Missing or duplicated ledger record')
        record = records[0]
        statement = record.get('sec_10q_comparison')
        if (not isinstance(statement, dict) or statement.get('matched_ledger_value') is not True
                or not isinstance(statement.get('source_url'), str) or not statement['source_url']):
            raise ValueError('A matched SEC statement comparison is required')
        if 'sec_10q_index_comparison' in record:
            raise ValueError('Existing SEC index comparison; avoid silent overwrite')
        fact = inspect_index(capture, expected_document_url=statement.get('source_url'))
        if fact['accession'] != statement.get('accession'):
            raise ValueError('Index and statement accessions differ')
        record['sec_10q_index_comparison'] = fact
    return result
