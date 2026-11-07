import copy
import unittest
from sec_index_capture import capture_digest, inspect_index


class IndexCaptureTest(unittest.TestCase):
    def setUp(self):
        self.cap = {
            'source_url': 'https://www.sec.gov/Archives/edgar/data/1324404/000132440422000001/0001324404-22-000001-index.htm',
            'retrieved_at': '2026-11-05T20:15:00Z', 'year': 2022,
            'accession_html': '<div id="secNum"><strong><acronym>SEC</acronym> Accession <acronym>No.</acronym></strong> 0001324404-22-000001</div>',
            'form_html': '<div id="formName">Form 10-Q - Quarterly report</div>',
            'metadata_html': [
                '<div class="formGrouping"><div class="infoHead">Filing Date</div><div class="info">2022-08-03</div><div class="infoHead">Accepted</div><div class="info">2022-08-03 12:00:00</div><div class="infoHead">Documents</div><div class="info">50</div></div>',
                '<div class="formGrouping"><div class="infoHead">Period of Report</div><div class="info">2022-06-30</div></div>'],
            'document_row_html': '<tr><td>1</td><td>10-Q</td><td><a href="/ix?doc=/Archives/edgar/data/1324404/000132440422000001/cf-q2.htm">cf-q2.htm</a> iXBRL</td><td>10-Q</td><td>1000</td></tr>',
        }
        self._seal()

    def _seal(self): self.cap['capture_sha256'] = capture_digest(self.cap)

    def test_valid_with_separate_earlier_earnings_date(self):
        actual = inspect_index(self.cap,
            'https://www.sec.gov/Archives/edgar/data/1324404/000132440422000001/cf-q2.htm',
            '2022-08-02')
        self.assertEqual(actual['filing_date'], '2022-08-03')
        self.assertEqual(actual['ledger_earnings_date'], '2022-08-02')
        self.assertIsNone(actual['public_dissemination_time'])
        self.assertIn('not established', actual['acceptance_timezone'])

    def test_mutation_rejects_digest(self):
        self.cap['year'] = 2021
        with self.assertRaisesRegex(ValueError, 'digest'): inspect_index(self.cap)

    def test_title_and_index_url_must_agree(self):
        self.cap['accession_html'] = self.cap['accession_html'].replace('000001', '000002'); self._seal()
        with self.assertRaisesRegex(ValueError, 'accession'): inspect_index(self.cap)

    def test_wrong_period_rejected(self):
        self.cap['metadata_html'][1] = self.cap['metadata_html'][1].replace('2022-06-30', '2022-03-31'); self._seal()
        with self.assertRaisesRegex(ValueError, 'period'): inspect_index(self.cap)

    def test_bad_accepted_date_rejected(self):
        self.cap['metadata_html'][0] = self.cap['metadata_html'][0].replace('2022-08-03 12', '2022-08-04 12'); self._seal()
        with self.assertRaisesRegex(ValueError, 'dates'): inspect_index(self.cap)

    def test_selected_document_must_match_statement(self):
        with self.assertRaisesRegex(ValueError, 'differs'): inspect_index(self.cap, 'https://www.sec.gov/Archives/edgar/data/1324404/000132440422000001/another.htm')

    def test_selected_document_must_have_same_accession(self):
        self.cap['document_row_html'] = self.cap['document_row_html'].replace('000132440422000001/', '000132440422000002/'); self._seal()
        with self.assertRaisesRegex(ValueError, 'accession'): inspect_index(self.cap)

    def test_reject_wrong_type_and_missing_timezone(self):
        self.cap['document_row_html'] = self.cap['document_row_html'].replace('<td>10-Q</td><td>1000', '<td>8-K</td><td>1000'); self._seal()
        with self.assertRaisesRegex(ValueError, 'sequence'): inspect_index(self.cap)
        self.setUp(); self.cap['retrieved_at'] = '2022-08-04T20:00:00'; self._seal()
        with self.assertRaisesRegex(ValueError, 'retrieval'): inspect_index(self.cap)

    def test_older_index_location_and_direct_document(self):
        self.cap['source_url'] = 'https://www.sec.gov/Archives/edgar/data/1324404/0001324404-22-000001-index.html'
        self.cap['document_row_html'] = self.cap['document_row_html'].replace('/ix?doc=', '')
        self._seal()
        self.assertEqual(inspect_index(self.cap)['year'], 2022)

    def test_refuse_spoofed_wrapped_url_and_duplicate_metadata(self):
        self.cap['document_row_html'] = self.cap['document_row_html'].replace('/ix?doc=/Archives', '/ix?doc=https://evil.example/Archives'); self._seal()
        with self.assertRaisesRegex(ValueError, 'wrapped'): inspect_index(self.cap)
        self.setUp(); self.cap['metadata_html'].append(self.cap['metadata_html'][1]); self._seal()
        with self.assertRaisesRegex(ValueError, 'Duplicate'): inspect_index(self.cap)


if __name__ == '__main__': unittest.main()
