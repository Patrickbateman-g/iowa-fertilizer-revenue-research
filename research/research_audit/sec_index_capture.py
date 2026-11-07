from datetime import date, datetime
from hashlib import sha256
from html.parser import HTMLParser
import json
import re
from urllib.parse import urljoin, urlsplit, parse_qs

FIELDS = ('source_url', 'retrieved_at', 'year', 'form_html', 'accession_html',
          'metadata_html', 'document_row_html')
_ACC = re.compile(r'0001324404-\d{2}-\d{6}')


def capture_digest(capture):
    if not isinstance(capture, dict) or any(k not in capture for k in FIELDS):
        raise ValueError('Missing index capture field')
    payload = {k: capture[k] for k in FIELDS}
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                             separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


class _Node:
    def __init__(self, tag, attrs):
        self.tag, self.attrs, self.parts = tag, dict(attrs), []

    def text(self):
        return ' '.join(' '.join(p.text() if isinstance(p, _Node) else p
                                  for p in self.parts).split())

    def descendants(self):
        for part in self.parts:
            if isinstance(part, _Node):
                yield part
                yield from part.descendants()


class _Fragment(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node('root', [])
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        if len({name for name, _ in attrs}) != len(attrs):
            raise ValueError('Duplicate HTML attribute')
        node = _Node(tag, attrs)
        self.stack[-1].parts.append(node)
        if tag not in ('area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'):
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if self.stack[-1].tag == tag:
            self.stack.pop()

    def handle_endtag(self, tag):
        if len(self.stack) == 1 or self.stack[-1].tag != tag:
            raise ValueError('Malformed HTML fragment')
        self.stack.pop()

    def handle_data(self, text):
        self.stack[-1].parts.append(text)


def _fragment(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Missing HTML fragment')
    parser = _Fragment()
    parser.feed(text); parser.close()
    if len(parser.stack) != 1:
        raise ValueError('Incomplete HTML fragment')
    return parser.root


def _metadata(html_fragments):
    if not isinstance(html_fragments, list) or not html_fragments:
        raise ValueError('Missing metadata fragments')
    pairs = {}
    for html in html_fragments:
        root = _fragment(html)
        nodes = [n for n in root.descendants() if 'infoHead' in n.attrs.get('class', '').split()
                 or 'info' in n.attrs.get('class', '').split()]
        if len(nodes) % 2:
            raise ValueError('Incomplete metadata pair')
        for head, value in zip(nodes[::2], nodes[1::2]):
            if 'infoHead' not in head.attrs.get('class', '').split() or 'info' not in value.attrs.get('class', '').split():
                raise ValueError('Unexpected metadata order')
            key = head.text()
            if not key or key in pairs:
                raise ValueError('Duplicate or empty metadata label')
            pairs[key] = value.text()
    return pairs


def _document_url(url, base):
    if not isinstance(url, str) or not url:
        raise ValueError('Missing document URL')
    resolved = urlsplit(urljoin(base, url))
    if resolved.scheme != 'https' or resolved.hostname not in ('www.sec.gov', 'sec.gov') or resolved.username or resolved.password or resolved.fragment:
        raise ValueError('Document URL is not an SEC HTTPS URL')
    if resolved.path == '/ix' or resolved.path == '/ixviewer/doc/action':
        query = parse_qs(resolved.query, strict_parsing=True)
        if set(query) != {'doc'} or len(query['doc']) != 1:
            raise ValueError('Unexpected inline document wrapper')
        resolved = urlsplit(urljoin('https://www.sec.gov', query['doc'][0]))
        if resolved.scheme != 'https' or resolved.hostname not in ('www.sec.gov', 'sec.gov') or resolved.query or resolved.fragment:
            raise ValueError('Unexpected wrapped URL')
    elif resolved.query:
        raise ValueError('Unexpected document query')
    return resolved.path


def _primary_document(row_html, source_url, accession):
    root = _fragment(row_html)
    rows = [n for n in root.descendants() if n.tag == 'tr']
    if len(rows) != 1:
        raise ValueError('Expected exactly one selected document row')
    cells = [n for n in rows[0].parts if isinstance(n, _Node) and n.tag in ('td', 'th')]
    if len(cells) < 4 or cells[0].text() != '1' or cells[3].text() != '10-Q':
        raise ValueError('Selected document is not sequence 1, type 10-Q')
    links = [n for n in cells[2].descendants() if n.tag == 'a' and n.attrs.get('href')]
    if len(links) != 1:
        raise ValueError('Expected one primary document link')
    path = _document_url(links[0].attrs['href'], source_url)
    prefix = '/Archives/edgar/data/1324404/' + accession.replace('-', '') + '/'
    if not path.startswith(prefix) or '/' in path[len(prefix):] or not path.lower().endswith(('.htm', '.html')):
        raise ValueError('Primary document is not in selected filing accession')
    if links[0].text() != path.rsplit('/', 1)[-1]:
        raise ValueError('Primary document filename disagrees with link text')
    return 'https://www.sec.gov' + path


def inspect_index(capture, expected_document_url=None, ledger_earnings_date=None):
    if not isinstance(capture, dict) or capture.get('capture_sha256') != capture_digest(capture):
        raise ValueError('Index capture digest mismatch')
    year = capture['year']
    if isinstance(year, bool) or not isinstance(year, int) or not 2016 <= year <= 2026:
        raise ValueError('Year outside supported study')
    accession_fragment = _fragment(capture['accession_html'])
    accession_nodes = [n for n in accession_fragment.descendants() if n.attrs.get('id') == 'secNum']
    if len(accession_nodes) != 1:
        raise ValueError('Missing or ambiguous accession element')
    matches = _ACC.findall(accession_nodes[0].text())
    if len(matches) != 1:
        raise ValueError('Index accession is ambiguous')
    accession = matches[0]
    u = urlsplit(capture['source_url'])
    if u.scheme != 'https' or u.hostname not in ('www.sec.gov', 'sec.gov') or u.username or u.password or u.query or u.fragment:
        raise ValueError('Index URL must be an SEC HTTPS URL')
    path = u.path
    allowed = ('/Archives/edgar/data/1324404/' + accession + '-index',
               '/Archives/edgar/data/1324404/' + accession.replace('-', '') + '/' + accession + '-index')
    if not any(path == a + ext for a in allowed for ext in ('.htm', '.html')):
        raise ValueError('Index URL and displayed accession differ')
    form = _fragment(capture['form_html'])
    forms = [n for n in form.descendants() if n.attrs.get('id') == 'formName']
    if len(forms) != 1 or not re.match(r'^Form\s+10-Q\b', forms[0].text()):
        raise ValueError('Index is not Form 10-Q')
    pairs = _metadata(capture['metadata_html'])
    try:
        filed = date.fromisoformat(pairs['Filing Date'])
        accepted = datetime.strptime(pairs['Accepted'], '%Y-%m-%d %H:%M:%S')
        period = date.fromisoformat(pairs['Period of Report'])
        retrieved = datetime.fromisoformat(capture['retrieved_at'].replace('Z', '+00:00'))
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ValueError('Invalid filing metadata or retrieval timestamp') from None
    if period != date(year, 6, 30) or filed < period or accepted.date() != filed:
        raise ValueError('Inconsistent filing dates or report period')
    if retrieved.tzinfo is None or retrieved.utcoffset() is None or retrieved.date() < filed:
        raise ValueError('Invalid or premature retrieval timestamp')
    document_url = _primary_document(capture['document_row_html'], capture['source_url'], accession)
    if expected_document_url is not None:
        if _document_url(expected_document_url, capture['source_url']) != _document_url(document_url, capture['source_url']):
            raise ValueError('Statement excerpt source differs from index primary document')
    if ledger_earnings_date is not None:
        try:
            earnings_date = date.fromisoformat(ledger_earnings_date).isoformat()
        except (TypeError, ValueError):
            raise ValueError('Invalid ledger earnings date') from None
    else:
        earnings_date = None
    return {'year': year, 'accession': accession, 'source_url': capture['source_url'],
            'period_end': period.isoformat(), 'filing_date': filed.isoformat(),
            'acceptance_timestamp_as_displayed': accepted.isoformat(sep=' '),
            'acceptance_timezone': 'not established by retained excerpt',
            'primary_document_url': document_url, 'retrieved_at': capture['retrieved_at'],
            'capture_sha256': capture['capture_sha256'], 'ledger_earnings_date': earnings_date,
            'public_dissemination_time': None,
            'evidence_limit': 'Current selected DOM, not original response bytes or historical public availability.'}
