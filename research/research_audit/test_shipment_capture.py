from copy import deepcopy
import unittest

import shipment_capture as subject


def capture(year=2020, current='1,234', previous='1,100'):
    c = {'year': year,
         'source_url': f'https://www.sec.gov/Archives/edgar/data/1324404/0001324404{year%100:02d}000099/release.htm',
         'retrieved_at': '2026-11-05T20:00:00Z', 'heading_context': 'CF Industries Holdings, Inc. Consolidated Results',
         'selected_table_position': 2, 'matching_table_count': 3,
         'header_rows': ['Three months ended March 31,', f'{year} {year-1}'],
         'header_html': f'<tr><td></td><td colspan="2">Three months ended March 31,</td></tr>'
                        f'<tr><td></td><td>{year}</td><td>{year-1}</td></tr>',
         'volume_row': f'Tons of product sold (000s) {current} {previous}',
         'volume_row_html': f'<tr><td>Tons of product sold (000s)</td><td>{current}</td><td>{previous}</td></tr>'}
    c['capture_sha256'] = subject.capture_digest(c)
    return c


def reseal(c):
    c['capture_sha256'] = subject.capture_digest(c)
    return c


def record(c, role='current', decision_year=None):
    year = c['year'] if decision_year is None else decision_year
    return {'series':'q1_shipments_' + role, 'period':str(year),
            'observation_date':f"{c['year']}-03-31", 'value':1234,
            'source_url':c['source_url'], 'source_version':c['source_url'].split('/')[-2],
            'units':'thousand product tons', 'evidence_type':'unverified',
            'available_at':f"{c['year']}-05-05"}


class ShipmentCaptureTests(unittest.TestCase):
    def test_each_role_is_linked_to_its_own_original_source_without_promotion(self):
        c = capture()
        ledger = [record(c), record(c, 'previous', 2021)]
        before = deepcopy(ledger)
        linked = subject.compare_captures_to_ledger(ledger, [c],
                    [('q1_shipments_current', 2020), ('q1_shipments_previous', 2021)])
        self.assertEqual(ledger, before)
        self.assertEqual([r['evidence_type'] for r in linked], ['unverified', 'unverified'])
        self.assertEqual([r['available_at'] for r in linked], [r['available_at'] for r in ledger])
        self.assertTrue(all(r['sec_q1_shipment_comparison']['matched_ledger_value'] for r in linked))

    def test_later_comparative_column_does_not_replace_original_source(self):
        original, later = capture(), capture(2021, current='1,500', previous='1,234')
        ledger = [record(original, 'previous', 2021)]
        with self.assertRaisesRegex(ValueError, 'source years'):
            subject.compare_captures_to_ledger(ledger, [later], [('q1_shipments_previous', 2021)])

    def test_tampered_digest_and_transcription_rejected(self):
        c = capture(); c['volume_row'] = c['volume_row'].replace('1,234', '1,235')
        with self.assertRaisesRegex(ValueError, 'digest'): subject.inspect_capture(c)
        with self.assertRaisesRegex(ValueError, 'differs'): subject.inspect_capture(reseal(c))

    def test_wrong_unit_period_source_and_year_order_rejected(self):
        for field, old, new in [('volume_row_html','product sold','nutrient sold'),
                                ('header_html','Three months','Six months'),
                                ('source_url','www.sec.gov','other.example'),
                                ('header_rows','unused','unused')]:
            c = capture()
            if field == 'header_rows': c[field] = ['Three months ended March 31,', '2019 2020']
            else: c[field] = c[field].replace(old,new)
            with self.subTest(field=field), self.assertRaises(ValueError): subject.inspect_capture(reseal(c))

    def test_newer_consolidated_product_volume_label_is_supported(self):
        c = capture(2026, current='4,683', previous='5,004')
        for field in ('volume_row', 'volume_row_html'):
            c[field] = c[field].replace('Tons of product sold', 'Sales volume by product tons')
        fact = subject.inspect_capture(reseal(c))
        self.assertEqual(fact['value'], '4683')
        self.assertEqual(fact['units'], 'thousand product tons')

    def test_category_heading_and_non_gaap_context_are_rejected(self):
        c = capture(2015)
        c['volume_row'] = 'Tons of product sold (000s):'
        c['volume_row_html'] = '<tr><td>Tons of product sold (000s):</td></tr>'
        with self.assertRaises(ValueError): subject.inspect_capture(reseal(c))
        c = capture(); c['heading_context'] += ' Non-GAAP disclosure items'
        with self.assertRaisesRegex(ValueError, 'Non-GAAP'): subject.inspect_capture(reseal(c))

    def test_category_heading_can_support_later_total_row_only_with_context(self):
        c = capture(2015, '1,234', '1,100')
        c['volume_row'] = 'Total tons of product sold 1,234 1,100'
        c['volume_row_html'] = '<tr><td>Total tons of product sold</td><td>1,234</td><td>1,100</td></tr>'
        c.update(units_row='Tons of product sold (000s):',
                 units_row_html='<tr><td>Tons of product sold (000s):</td></tr>',
                 units_row_index=8, volume_row_index=11)
        self.assertEqual(subject.inspect_capture(reseal(c))['value'], '1234')
        for field, bad in [('units_row','Nutrient tons sold (000s):'),
                           ('units_row_index',11), ('volume_row_index',8),
                           ('units_row_html','<tr><td>Tons of product sold (000s):</td><td>5</td></tr>')]:
            changed = deepcopy(c); changed[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                subject.inspect_capture(reseal(changed))
        for label in ('Nitrogen product segments', 'Phosphate segment'):
            changed=deepcopy(c)
            changed['volume_row'] = f'{label} 1,234 1,100'
            changed['volume_row_html'] = f'<tr><td>{label}</td><td>1,234</td><td>1,100</td></tr>'
            with self.subTest(label=label), self.assertRaises(ValueError):
                subject.inspect_capture(reseal(changed))

    def test_misaligned_cells_fail_even_when_rendered_text_matches(self):
        c = capture()
        c['volume_row_html'] = '<tr><td>Tons of product sold (000s)</td><td colspan="2">1,234</td><td>1,100</td></tr>'
        with self.assertRaisesRegex(ValueError, 'align'): subject.inspect_capture(reseal(c))

    def test_missing_duplicate_or_mismatched_evidence_is_rejected(self):
        c = capture(); row = record(c)
        for captures, keys in [([], [('q1_shipments_current',2020)]),
                               ([c,c], [('q1_shipments_current',2020)]),
                               ([c], [('q1_shipments_current',2020),('q1_shipments_current',2020)])]:
            with self.subTest(keys=keys, count=len(captures)), self.assertRaises(ValueError):
                subject.compare_captures_to_ledger([row],captures,keys)
        for field, value in [('value',999),('source_url','https://www.sec.gov/other.htm'),
                             ('source_version','another'),('units','tons')]:
            bad = dict(row, **{field:value})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'mismatch'):
                subject.compare_captures_to_ledger([bad],[c],[('q1_shipments_current',2020)])


if __name__ == '__main__': unittest.main()
