from copy import deepcopy
import unittest
import sec_capture as sec


def sample():
    c = {
        'source_url':'https://www.sec.gov/Archives/edgar/data/1324404/000132440420000099/cf-20200630.htm',
        'retrieved_at':'2026-11-05T19:00:00Z',
        'year':2020, 'selected_table_position':4, 'matching_table_count':9,
        'heading_context':'CF INDUSTRIES HOLDINGS, INC. PART I FINANCIAL INFORMATION '
                          'CONSOLIDATED STATEMENTS OF OPERATIONS (Unaudited)',
        'header_rows':['Three months ended June 30, Six months ended June 30,',
                       '2020 2019 2020 2019','(in millions, except per share amounts)'],
        'sales_row':'Net sales $ 1,234 $ 1,100 $ 2,345 $ 2,222',
        'header_html':'<tr><td> </td></tr><tr><th></th><th colspan="4">Three months ended<br>June 30,</th>'
          '<th colspan="4">Six months ended<br>June 30,</th></tr>'
          '<tr><th></th><th colspan="2">2020</th><th colspan="2">2019</th>'
          '<th colspan="2">2020</th><th colspan="2">2019</th></tr>'
          '<tr><th colspan="9">(in millions, except per share amounts)</th></tr>',
        'sales_row_html':'<tr><td>Net sales</td><td>$</td><td>1,234</td><td>$</td>'
          '<td>1,100</td><td>$</td><td>2,345</td><td>$</td><td>2,222</td></tr>',
    }
    c['capture_sha256']=sec.capture_digest(c)
    return c


def reseal(c):
    c['capture_sha256']=sec.capture_digest(c)
    return c


class SECExtractTests(unittest.TestCase):
    def test_valid_current_quarter_and_no_promotion(self):
        c=sample(); fact=sec.inspect_capture(c,2020)
        self.assertEqual(fact['value_usd'],'1234000000')
        self.assertEqual(fact['accession'],'0001324404-20-000099')
        ledger=[{'series':'q2_sales','period':'2020','value':1234000000,
                 'observation_date':'2020-06-30','evidence_type':'unverified',
                 'available_at':'2020-08-05','source_version':'earlier-release'}]
        before=deepcopy(ledger)
        annotated=sec.compare_to_ledger(ledger,[fact])
        self.assertEqual(ledger,before)
        self.assertEqual(annotated[0]['evidence_type'],'unverified')
        self.assertEqual(annotated[0]['available_at'],'2020-08-05')
        self.assertEqual(annotated[0]['source_version'],'earlier-release')
        self.assertTrue(annotated[0]['sec_10q_comparison']['matched_ledger_value'])

    def test_hash_covers_each_excerpt_and_metadata(self):
        for field in sec.FIELDS:
            with self.subTest(field=field):
                c=sample(); c[field]=None
                with self.assertRaises(ValueError): sec.inspect_capture(c)

    def test_selected_dom_must_match_both_text_transcriptions(self):
        c=sample();c['sales_row']=c['sales_row'].replace('1,234','4,321')
        with self.assertRaisesRegex(ValueError,'Sales-row transcription'): sec.inspect_capture(reseal(c))
        c=sample();c['header_rows'][1]='2020 2019 2019 2020'
        with self.assertRaisesRegex(ValueError,'Header transcription'): sec.inspect_capture(reseal(c))

    def test_ytd_first_cannot_be_misread_as_quarterly(self):
        c=sample()
        c['header_rows'][0]='Six months ended June 30, Three months ended June 30,'
        c['header_html']=c['header_html'].replace('Three months ended','TEMP').replace('Six months ended','Three months ended').replace('TEMP','Six months ended')
        with self.assertRaisesRegex(ValueError,'three-month'): sec.inspect_capture(reseal(c))

    def test_ambiguous_columns_and_wrong_years_rejected(self):
        c=sample(); c['header_rows'][1]='2019 2020 2020 2019';c['header_html']=c['header_html'].replace('<th colspan="2">2020</th><th colspan="2">2019</th>','<th colspan="2">2019</th><th colspan="2">2020</th>',1)
        with self.assertRaisesRegex(ValueError,'year order'): sec.inspect_capture(reseal(c))
        c=sample(); c['header_rows'].append('additional header');c['header_html']+='<tr><th>additional header</th></tr>'
        with self.assertRaisesRegex(ValueError,'three unambiguous'): sec.inspect_capture(reseal(c))

    def test_wrong_units_and_nonconsolidated_heading_rejected(self):
        c=sample();c['header_rows'][2]='(in thousands, except per share amounts)';c['header_html']=c['header_html'].replace('millions','thousands')
        with self.assertRaisesRegex(ValueError,'millions'): sec.inspect_capture(reseal(c))
        c=sample();c['heading_context']=c['heading_context'].replace('CONSOLIDATED','CONSOLIDATING')
        with self.assertRaisesRegex(ValueError,'consolidated'): sec.inspect_capture(reseal(c))

    def test_reject_other_company_or_nonquarterly_document(self):
        for url in ['https://www.sec.gov/Archives/edgar/data/1234/000132440420000099/cf-20200630.htm',
                    'https://www.sec.gov/Archives/edgar/data/1324404/000132440420000099/cf-20201231.htm',
                    'https://www.sec.gov/Archives/edgar/data/1324404/000132440420000099/ex991.htm',
                    'https://example.com/Archives/edgar/data/1324404/000132440420000099/cf-20200630.htm']:
            c=sample();c['source_url']=url
            with self.subTest(url=url), self.assertRaises(ValueError): sec.inspect_capture(reseal(c))

    def test_gapped_sixteen_column_sec_layout(self):
        c=sample()
        c['header_html']=(
            '<tr><td colspan="16"></td></tr>' +
            '<tr>'+('<td></td>'*16)+'</tr>' +
            '<tr><td></td><td colspan="7">Three months ended<br>June 30,</td>'
            '<td></td><td colspan="7">Six months ended<br>June 30,</td></tr>' +
            '<tr><td></td><td colspan="3">2020</td><td></td><td colspan="3">2019</td>'
            '<td></td><td colspan="3">2020</td><td></td><td colspan="3">2019</td></tr>' +
            '<tr><td></td><td colspan="15">(in millions, except per share amounts)</td></tr>')
        c['sales_row_html']=(
            '<tr><td>Net sales</td><td>$</td><td>1,234</td><td></td><td></td>'
            '<td>$</td><td>1,100</td><td></td><td></td>'
            '<td>$</td><td>2,345</td><td></td><td></td>'
            '<td>$</td><td>2,222</td><td></td></tr>')
        self.assertEqual(sec.inspect_capture(reseal(c))['value_usd'], '1234000000')

    def test_gapped_twenty_four_column_sec_layout(self):
        c=sample()
        c['header_html']=(
            '<tr>'+('<td></td>'*24)+'</tr>' +
            '<tr><td colspan="3"></td><td colspan="9">Three months ended<br>June 30,</td>'
            '<td colspan="3"></td><td colspan="9">Six months ended<br>June 30,</td></tr>' +
            '<tr><td colspan="3"></td><td colspan="3">2020</td><td colspan="3"></td>'
            '<td colspan="3">2019</td><td colspan="3"></td><td colspan="3">2020</td>'
            '<td colspan="3"></td><td colspan="3">2019</td></tr>' +
            '<tr><td colspan="3"></td><td colspan="21">(in millions, except per share amounts)</td></tr>')
        c['sales_row_html']=(
            '<tr><td colspan="3">Net sales</td><td>$</td><td>1,234</td><td></td><td colspan="3"></td>'
            '<td>$</td><td>1,100</td><td></td><td colspan="3"></td>'
            '<td>$</td><td>2,345</td><td></td><td colspan="3"></td>'
            '<td>$</td><td>2,222</td><td></td></tr>')
        self.assertEqual(sec.inspect_capture(reseal(c))['value_usd'], '1234000000')

    def test_column_ranges_match_each_quarter_year_and_sales_amount(self):
        c=sample()
        c['header_html']=c['header_html'].replace('colspan="4">Three', 'colspan="2">Three', 1)
        with self.assertRaisesRegex(ValueError, 'align with period groups'): sec.inspect_capture(reseal(c))
        c=sample()
        c['sales_row_html']=c['sales_row_html'].replace('<td>1,234</td>', '<td colspan="3">1,234</td>', 1)
        with self.assertRaisesRegex(ValueError, 'align with year columns'): sec.inspect_capture(reseal(c))

    def test_unsupported_spans_and_hidden_cells_fail_closed(self):
        for attrs in ('rowspan="2"', 'colspan="0"', 'colspan="-1"', 'style="display:none"'):
            c=sample();c['sales_row_html']=c['sales_row_html'].replace('<td>Net sales</td>', f'<td {attrs}>Net sales</td>', 1)
            with self.subTest(attrs=attrs), self.assertRaises(ValueError): sec.inspect_capture(reseal(c))

    def test_empty_inline_display_none_cells_do_not_shift_columns(self):
        c=sample()
        c['header_html']=c['header_html'].replace('<tr><th></th><th colspan="4">Three',
            '<tr><th style="DISPLAY: none !important" colspan="6"></th><th></th><th colspan="4">Three',1)
        c['sales_row_html']=c['sales_row_html'].replace('<td>Net sales</td>',
            '<td style="display:none" colspan="9"></td><td>Net sales</td>',1)
        self.assertEqual(sec.inspect_capture(reseal(c))['value_usd'], '1234000000')
        c['sales_row_html']=c['sales_row_html'].replace('colspan="9"></td>', 'colspan="9">unexpected</td>',1)
        with self.assertRaisesRegex(ValueError,'Hidden table cell contains text'):
            sec._grid(c['sales_row_html'])

    def test_ambiguous_inline_visibility_is_rejected(self):
        for style in ('visibility:hidden', 'visibility:collapse', 'display:none; display:table-cell'):
            c=sample(); c['sales_row_html']=c['sales_row_html'].replace('<td>Net sales</td>',
                f'<td style="{style}"></td><td>Net sales</td>',1)
            with self.subTest(style=style),self.assertRaises(ValueError): sec.inspect_capture(reseal(c))

    def test_bad_or_duplicated_rows_rejected(self):
        c=sample();c['sales_row']+=' 1';c['sales_row_html']=c['sales_row_html'].replace('</tr>','<td>1</td></tr>')
        with self.assertRaisesRegex(ValueError,'four unambiguous'): sec.inspect_capture(reseal(c))
        c=sample();c['sales_row_html']+=c['sales_row_html']
        with self.assertRaisesRegex(ValueError,'Sales-row transcription'): sec.inspect_capture(reseal(c))

    def test_missing_timezone_and_bad_selection_metadata_rejected(self):
        for field,value in [('retrieved_at','2026-11-05T19:00:00'),('selected_table_position',-1),('matching_table_count',0),('year',True)]:
            c=sample();c[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): sec.inspect_capture(reseal(c))

    def test_ledger_mismatch_and_duplicates_fail(self):
        fact=sec.inspect_capture(sample());ledger=[{'series':'q2_sales','period':'2020','value':9,'observation_date':'2020-06-30'}]
        with self.assertRaisesRegex(ValueError,'mismatch'):sec.compare_to_ledger(ledger,[fact])
        ledger[0]['value']=1234000000
        with self.assertRaisesRegex(ValueError,'Duplicate'):sec.compare_to_ledger(ledger,[fact,fact])
        with self.assertRaisesRegex(ValueError,'duplicated'):sec.compare_to_ledger(ledger+ledger,[fact])

    def test_batch_captures_require_exact_years_and_do_not_promote(self):
        ledger=[{'series':'q2_sales','period':'2020','value':1234000000,
                 'observation_date':'2020-06-30','evidence_type':'unverified',
                 'available_at':'2020-08-05','source_version':'earlier-release'}]
        original=deepcopy(ledger)
        checked=sec.compare_captures_to_ledger(ledger,[sample()],[2020])
        self.assertEqual(ledger,original)
        self.assertEqual(checked[0]['available_at'],'2020-08-05')
        self.assertEqual(checked[0]['source_version'],'earlier-release')
        self.assertEqual(checked[0]['evidence_type'],'unverified')
        self.assertEqual(checked[0]['sec_10q_comparison']['capture_sha256'],sample()['capture_sha256'])
        for captures,years in [([], [2020]),([sample(),sample()],[2020]),
                               ([sample()],[2019]),([sample()],[2020,2020])]:
            with self.subTest(years=years,count=len(captures)),self.assertRaises(ValueError):
                sec.compare_captures_to_ledger(ledger,captures,years)

    def test_index_facts_are_separate_and_bound_to_matching_statement(self):
        from sec_index_capture import capture_digest
        ledger=[{'series':'q2_sales','period':'2020','value':1234000000,
                 'observation_date':'2020-06-30','evidence_type':'unverified',
                 'available_at':'2020-08-05','source_version':'earlier-release'}]
        checked=sec.compare_captures_to_ledger(ledger,[sample()],[2020])
        index={
          'source_url':'https://www.sec.gov/Archives/edgar/data/1324404/000132440420000099/0001324404-20-000099-index.htm',
          'retrieved_at':'2026-11-05T20:15:00Z','year':2020,
          'accession_html':'<div id="secNum">SEC Accession No. 0001324404-20-000099</div>',
          'form_html':'<div id="formName">Form 10-Q - Quarterly report</div>',
          'metadata_html':['<div><div class="infoHead">Filing Date</div><div class="info">2020-08-06</div><div class="infoHead">Accepted</div><div class="info">2020-08-06 12:00:00</div></div>',
                           '<div><div class="infoHead">Period of Report</div><div class="info">2020-06-30</div></div>'],
          'document_row_html':'<tr><td>1</td><td>10-Q</td><td><a href="/Archives/edgar/data/1324404/000132440420000099/cf-20200630.htm">cf-20200630.htm</a></td><td>10-Q</td></tr>',
        }
        index['capture_sha256']=capture_digest(index)
        before=deepcopy(checked)
        linked=sec.attach_index_captures(checked,[index],[2020])
        self.assertEqual(checked,before)
        self.assertEqual(linked[0]['available_at'],'2020-08-05')
        self.assertEqual(linked[0]['evidence_type'],'unverified')
        self.assertEqual(linked[0]['sec_10q_index_comparison']['filing_date'],'2020-08-06')
        self.assertIsNone(linked[0]['sec_10q_index_comparison']['public_dissemination_time'])
        bad=deepcopy(index);bad['document_row_html']=bad['document_row_html'].replace('cf-20200630.htm','another.htm');bad['capture_sha256']=capture_digest(bad)
        with self.assertRaisesRegex(ValueError,'differs'):sec.attach_index_captures(checked,[bad],[2020])
        with self.assertRaises(ValueError):sec.attach_index_captures(checked,[index,index],[2020])
        with self.assertRaises(ValueError):sec.attach_index_captures(ledger,[index],[2020])

if __name__=='__main__': unittest.main()
