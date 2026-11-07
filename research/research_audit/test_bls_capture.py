import copy
from datetime import date
import unittest

from bls_capture import capture_digest, compare_captures_to_ledger, compare_to_ledger, inspect_capture


def example():
    c = {
        'source_url': 'https://www.bls.gov/news.release/archives/ppi_05142014.htm',
        'retrieved_at': '2026-11-05T12:00:00+00:00',
        'release_preamble_excerpt': 'Transmission embargoed until 8:30 a.m. (EDT), Wednesday, May 14, 2014 PRODUCER PRICE INDEXES - APRIL 2014',
        'table_id': 'table5',
        'caption_html': '<caption>Table 5. Producer price indexes [1982=100, unless otherwise indicated]</caption>',
        'thead_html': '<thead><tr><th id="g">Unadjusted index</th><th id="p">Unadjusted percent change</th></tr><tr><th id="date">Apr.<br>2014<span class="footnoteRefs"><a>(2)</a></span></th><th id="pctdate">Apr.2014</th></tr></thead>',
        'row_html': '<tr><th>Nitrogenates</th><td>06-52-01</td><td headers="g date">345.3</td><td headers="p pctdate">-99.9</td></tr>',
    }
    c['capture_sha256'] = capture_digest(c)
    return c


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.c = example()
        self.cutoff = date(2014, 5, 31)

    def inspect(self):
        return inspect_capture(self.c, 2014, 'current', self.cutoff)

    def mutate(self, key, value):
        self.c[key] = value
        self.c['capture_sha256'] = capture_digest(self.c)

    def test_selects_index_and_ignores_percentage(self):
        fact = self.inspect()
        self.assertEqual(fact['value'], '345.3')
        self.assertEqual(fact['observation_date'], '2014-04-01')
        self.assertIn('does not establish', fact['limitation'])

    def test_detects_changed_capture(self):
        self.c['row_html'] = self.c['row_html'].replace('345.3', '345.4')
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.inspect()

    def test_no_wrong_group_or_ambiguous_match(self):
        self.mutate('row_html', self.c['row_html'].replace('g date', 'p date'))
        with self.assertRaisesRegex(ValueError, 'ambiguous or missing'):
            self.inspect()
        self.c = example()
        self.mutate('row_html', self.c['row_html'].replace('</tr>', '<td headers="g date">1</td></tr>'))
        with self.assertRaisesRegex(ValueError, 'ambiguous or missing'):
            self.inspect()

    def test_wrong_code_date_base_and_host_rejected(self):
        for key, old, new in [('row_html', '06-52-01', '06-52-02'),
                              ('thead_html', '2014', '2013'),
                              ('caption_html', '1982=100', '2000=100'),
                              ('source_url', 'www.bls.gov', 'imposter.example')]:
            with self.subTest(key=key):
                self.c = example()
                self.mutate(key, self.c[key].replace(old, new))
                with self.assertRaises(ValueError): self.inspect()

    def test_chronology_and_cutoff(self):
        self.cutoff = date(2014, 5, 14)
        with self.assertRaisesRegex(ValueError, 'cutoff'): self.inspect()
        self.cutoff = date(2014, 5, 31)
        self.mutate('retrieved_at', '2014-05-13T12:00:00Z')
        with self.assertRaisesRegex(ValueError, 'predates'): self.inspect()
        self.mutate('retrieved_at', '2026-11-05T12:00:00')
        with self.assertRaisesRegex(ValueError, 'timestamp'): self.inspect()

    def test_previous_september_release(self):
        self.mutate('source_url', 'https://www.bls.gov/news.release/archives/ppi_09132013.htm')
        self.mutate('release_preamble_excerpt', 'Transmission embargoed until Friday, September 13, 2013')
        self.mutate('thead_html', self.c['thead_html'].replace('2014', '2013'))
        fact = inspect_capture(self.c, 2014, 'previous', self.cutoff)
        self.assertEqual(fact['observation_date'], '2013-04-01')
        self.assertEqual(fact['release_date'], '2013-09-13')

    def test_ledger_comparison_is_nonmutating_and_does_not_promote(self):
        ledger = [{'series': 'nitrogen_price_current_april', 'period': '2014',
                   'value': 345.3, 'observation_date': '2014-04-01',
                   'evidence_type': 'unverified'}, {'series': 'other', 'period': '2014'}]
        original = copy.deepcopy(ledger)
        out = compare_to_ledger(ledger, [self.inspect()])
        self.assertEqual(ledger, original)
        self.assertEqual(out[0]['evidence_type'], 'unverified')
        self.assertNotIn('source_verified', out[0])
        self.assertTrue(out[0]['bls_comparison']['matched_ledger_value'])
        self.assertEqual(out[1], original[1])
        ledger[0]['value'] = 345.2
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            compare_to_ledger(ledger, [self.inspect()])

    def test_capture_batch_reparses_and_attaches_without_upgrading(self):
        ledger = [{'series': 'nitrogen_price_current_april', 'period': '2014',
                   'value': 345.3, 'observation_date': '2014-04-01',
                   'evidence_type': 'unverified', 'available_at': None}]
        original = copy.deepcopy(ledger)
        batch = [{'decision_year': 2014, 'role': 'current', 'capture': self.c}]
        result = compare_captures_to_ledger(ledger, batch, [(2014, 'current')])
        self.assertEqual(ledger, original)
        self.assertEqual(result[0]['evidence_type'], 'unverified')
        self.assertIsNone(result[0]['available_at'])
        self.assertEqual(result[0]['bls_comparison']['capture_sha256'], self.c['capture_sha256'])
        self.assertEqual(result[0]['bls_comparison']['release_date'], '2014-05-14')
        self.assertTrue(result[0]['bls_comparison']['matched_ledger_value'])
        batch[0]['capture']['row_html'] = batch[0]['capture']['row_html'].replace('345.3', '345.4')
        with self.assertRaisesRegex(ValueError, 'digest'):
            compare_captures_to_ledger(ledger, batch, [(2014, 'current')])

    def test_capture_batch_fails_on_missing_duplicate_extra_or_mismatch(self):
        ledger = [{'series': 'nitrogen_price_current_april', 'period': '2014',
                   'value': 345.3, 'observation_date': '2014-04-01',
                   'evidence_type': 'unverified'}]
        item = {'decision_year': 2014, 'role': 'current', 'capture': self.c}
        with self.assertRaisesRegex(ValueError, 'expected set'):
            compare_captures_to_ledger(ledger, [item], [(2014, 'current'), (2014, 'previous')])
        with self.assertRaisesRegex(ValueError, 'Duplicate capture'):
            compare_captures_to_ledger(ledger, [item, item], [(2014, 'current')])
        with self.assertRaisesRegex(ValueError, 'expected set'):
            compare_captures_to_ledger(ledger, [item], [(2014, 'previous')])
        with self.assertRaisesRegex(ValueError, 'unique'):
            compare_captures_to_ledger(ledger, [item], [(2014, 'current'), (2014, 'current')])
        ledger[0]['value'] = 345.4
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            compare_captures_to_ledger(ledger, [item], [(2014, 'current')])

    def test_flat_saved_capture_and_ambiguous_format(self):
        ledger = [{'series': 'nitrogen_price_current_april', 'period': '2014',
                   'value': 345.3, 'observation_date': '2014-04-01',
                   'evidence_type': 'unverified'}]
        item = dict(self.c, decision_year=2014, role='current')
        result = compare_captures_to_ledger(ledger, [item], [(2014, 'current')])
        self.assertEqual(result[0]['bls_comparison']['capture_sha256'], self.c['capture_sha256'])
        item['capture'] = self.c
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            compare_captures_to_ledger(ledger, [item], [(2014, 'current')])


if __name__ == '__main__': unittest.main()
