import copy
import unittest

from alfred_chart import (TITLE, HISTORICAL_TITLE, SOURCE_URL, batch_digest, capture_digest,
                          compare_capture_to_ledger, compare_captures_to_ledger,
                          inspect_capture, require_same_batch)


def example():
    c = {
        'source_url': SOURCE_URL,
        'retrieved_at': '2030-06-01T10:00:00Z',
        'title_html': '<h1>' + TITLE + ' (WPU065201)</h1>',
        'thead_html': '<thead><tr><th scope="col">DateTime</th><th scope="col">'
            + TITLE + ' Vintage: 2025-05-31</th><th scope="col">' + TITLE
            + ' Vintage: 2029-06-01</th></tr></thead>',
        'rows_html': '<tr><th scope="row">2024-04-01 1:00:00</th><td>111.125</td><td>77</td></tr>'
                     '<tr><th scope="row">2025-04-01 1:00:00</th><td>222.250</td><td>99</td></tr>',
    }
    c['capture_sha256'] = capture_digest(c)
    return c


def ledger():
    return [dict(series='nitrogen_price_previous_april', period='2025', value=111.125,
                 observation_date='2024-04-01', recorded_as_of='2025-05-31',
                 source_version='2025-05-31', evidence_type='unverified', available_at=None),
            dict(series='nitrogen_price_current_april', period='2025', value=222.25,
                 observation_date='2025-04-01', recorded_as_of='2025-05-31',
                 source_version='2025-05-31', evidence_type='unverified', available_at=None)]


def example_for(year):
    c = example()
    c['thead_html'] = c['thead_html'].replace('2025-05-31', f'{year}-05-31')
    c['rows_html'] = c['rows_html'].replace('2024-04-01', 'PREVIOUS_DATE')
    c['rows_html'] = c['rows_html'].replace('2025-04-01', f'{year}-04-01')
    c['rows_html'] = c['rows_html'].replace('PREVIOUS_DATE', f'{year-1}-04-01')
    c['capture_sha256'] = capture_digest(c)
    return c


def ledger_for(year):
    records = ledger()
    for row in records:
        row['period'] = str(year)
        row['observation_date'] = f"{year-(row['series']=='nitrogen_price_previous_april')}-04-01"
        row['recorded_as_of'] = row['source_version'] = f'{year}-05-31'
    return records


class AlfredChartTests(unittest.TestCase):
    def setUp(self): self.c = example()

    def mutate(self, field, old, new):
        self.c[field] = self.c[field].replace(old, new)
        self.c['capture_sha256'] = capture_digest(self.c)

    def test_selects_earlier_vintage_column_and_ignores_later_revision(self):
        facts = inspect_capture(self.c, 2025)
        self.assertEqual([(f['role'], f['value']) for f in facts], [('previous', '111.125'), ('current', '222.250')])
        self.assertTrue(all(f['as_of'] == '2025-05-31' for f in facts))
        self.assertIn('no proof', facts[0]['limitation'])

    def test_digest_is_checked_before_parsing(self):
        self.c['rows_html'] = self.c['rows_html'].replace('222.250', '222.251')
        with self.assertRaisesRegex(ValueError, 'digest'): inspect_capture(self.c, 2025)

    def test_wrong_identity_vintage_or_ambiguous_column_rejected(self):
        cases = [('source_url', 'alfred.stlouisfed.org', 'example.org'),
                 ('title_html', 'WPU065201', 'WPU065202'),
                 ('thead_html', '2025-05-31', '2025-05-30'),
                 ('thead_html', '2029-06-01', '2025-05-31'),
                 ('thead_html', 'DateTime', 'Other')]
        for field, old, new in cases:
            with self.subTest(field=field, new=new):
                self.c = example(); self.mutate(field, old, new)
                with self.assertRaises(ValueError): inspect_capture(self.c, 2025)

    def test_observed_historical_series_title_is_accepted_only_as_an_exact_title(self):
        self.mutate('thead_html', TITLE + ' Vintage: 2025-05-31',
                    HISTORICAL_TITLE + ' Vintage: 2025-05-31')
        facts = inspect_capture(self.c, 2025)
        self.assertEqual([f['value'] for f in facts], ['111.125', '222.250'])
        self.mutate('thead_html', HISTORICAL_TITLE, HISTORICAL_TITLE.replace('Nitrogenates', 'Nitrate'))
        with self.assertRaises(ValueError): inspect_capture(self.c, 2025)

    def test_two_spelling_variants_for_same_vintage_are_ambiguous(self):
        self.mutate('thead_html', TITLE + ' Vintage: 2029-06-01',
                    HISTORICAL_TITLE + ' Vintage: 2025-05-31')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            inspect_capture(self.c, 2025)

    def test_rows_dates_values_and_alignment_rejected(self):
        cases = [('2024-04-01', '2023-04-01'), ('2024-04-01', '2025-04-01'),
                 ('222.250', 'NaN'), ('222.250', '-1'), ('222.250', ''),
                 ('<td>99</td>', ''), ('<td>99</td>', '<td colspan="2">99</td>')]
        for old, new in cases:
            with self.subTest(old=old, new=new):
                self.c = example(); self.mutate('rows_html', old, new)
                with self.assertRaises(ValueError): inspect_capture(self.c, 2025)

    def test_retrieval_time_requires_zone_and_cannot_predate_vintage(self):
        for value in ('2025-05-30T10:00:00Z', '2030-06-01T10:00:00', 'broken'):
            self.c = example(); self.c['retrieved_at'] = value; self.c['capture_sha256'] = capture_digest(self.c)
            with self.assertRaises(ValueError): inspect_capture(self.c, 2025)

    def test_comparison_preserves_ledger_and_does_not_promote(self):
        source = ledger(); before = copy.deepcopy(source)
        result = compare_capture_to_ledger(source, self.c, 2025)
        self.assertEqual(source, before)
        for row in result:
            self.assertEqual(row['evidence_type'], 'unverified')
            self.assertIsNone(row['available_at'])
            self.assertTrue(row['alfred_chart_comparison']['matched_ledger_value'])
        self.assertEqual(source[0]['value'], result[0]['value'])

    def test_comparison_rejects_missing_duplicates_mismatch_and_overwrite(self):
        for change in ('missing', 'duplicate', 'value', 'date', 'vintage', 'version', 'existing'):
            data = ledger()
            if change == 'missing': data.pop()
            if change == 'duplicate': data.append(copy.deepcopy(data[0]))
            if change == 'value': data[0]['value'] += 1
            if change == 'date': data[0]['observation_date'] = '2023-04-01'
            if change == 'vintage': data[0]['recorded_as_of'] = '2025-05-30'
            if change == 'version': data[0]['source_version'] = 'unknown'
            if change == 'existing': data[0]['alfred_chart_comparison'] = {}
            with self.subTest(change=change), self.assertRaises(ValueError):
                compare_capture_to_ledger(data, self.c, 2025)

    def test_batch_matches_only_exact_roles_and_reports_missing_coverage(self):
        original = ledger_for(2024) + ledger_for(2025) + ledger_for(2026)
        original[0]['value'] += 0.001
        before = copy.deepcopy(original)
        captures = [{'year': 2024, 'capture': example_for(2024)},
                    {'year': 2025, 'capture': example_for(2025)}]
        result = compare_captures_to_ledger(original, captures, [2024, 2025, 2026])
        self.assertEqual(original, before)
        self.assertEqual(result['expected_years'], [2024, 2025, 2026])
        self.assertEqual(result['captured_years'], [2024, 2025])
        self.assertEqual([d['status'] for d in result['diagnostics']],
                         ['mismatch', 'matched', 'matched', 'matched',
                          'capture_missing', 'capture_missing'])
        self.assertEqual(result['diagnostics'][0]['differences'], ['value'])
        self.assertNotIn('alfred_chart_comparison', result['ledger'][0])
        self.assertEqual(sum('alfred_chart_comparison' in r for r in result['ledger']), 3)
        self.assertTrue(all(r['evidence_type'] == 'unverified' and r['available_at'] is None
                            for r in result['ledger']))

    def test_batch_detects_missing_or_duplicate_ledger_and_metadata_mismatch(self):
        data = ledger_for(2025)
        data[0]['source_version'] = 'unknown'
        data.append(copy.deepcopy(data[1]))
        result = compare_captures_to_ledger(data, [{'year': 2025, 'capture': example()}], [2025])
        self.assertEqual([d['status'] for d in result['diagnostics']], ['mismatch', 'ledger_duplicate'])
        self.assertEqual(result['diagnostics'][0]['differences'], ['source_version'])
        self.assertFalse(any('alfred_chart_comparison' in r for r in result['ledger']))
        result = compare_captures_to_ledger([], [{'year': 2025, 'capture': example()}], [2025])
        self.assertEqual([d['status'] for d in result['diagnostics']], ['ledger_missing', 'ledger_missing'])

    def test_batch_rejects_duplicate_extra_malformed_and_corrupt_captures(self):
        c = {'year': 2025, 'capture': example()}
        for captures, years in (([c, c], [2025]), ([c], [2024]), ([c], [2025, 2025]),
                                ([{'year': True, 'capture': example()}], [2025]),
                                ([{'year': 2025}], [2025]), ([c], [])):
            with self.subTest(captures=captures, years=years), self.assertRaises(ValueError):
                compare_captures_to_ledger(ledger(), captures, years)
        bad = example(); bad['rows_html'] += 'changed'
        with self.assertRaisesRegex(ValueError, 'digest'):
            compare_captures_to_ledger(ledger(), [{'year': 2025, 'capture': bad}], [2025])

    def test_batch_digest_changes_even_when_unmatched_capture_is_resealed(self):
        c = {'year': 2025, 'capture': example()}
        changed = copy.deepcopy(c)
        changed['capture']['rows_html'] = changed['capture']['rows_html'].replace('111.125', '111.126')
        changed['capture']['capture_sha256'] = capture_digest(changed['capture'])
        digest = batch_digest([c], [2025])
        self.assertEqual(require_same_batch([c], [2025], digest), digest)
        with self.assertRaisesRegex(ValueError, 'batch changed'):
            require_same_batch([changed], [2025], digest)
        with self.assertRaisesRegex(ValueError, 'batch changed'):
            require_same_batch([c], [2025], None)
        self.assertNotEqual(digest, batch_digest([changed], [2025]))
        self.assertNotEqual(batch_digest([c], [2025]), batch_digest([c], [2025, 2026]))


if __name__ == '__main__': unittest.main()
