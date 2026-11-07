import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from replay import (FORECAST_DEFINITION, ReplayUnavailable, project_fold_replay, project_ledger, project_replay_report,
                    record_status, replay, require_replay)


def example(**changes):
    record = dict(series='price', period='2022-04', value=10.0,
                  observation_date='2022-04-01',
                  source_url='https://example.org/original-filing',
                  source_version='version-1', available_at='2022-05-15',
                  retrieved_at='2026-01-01', evidence_type='verified_immutable_document',
                  source_verified=True, verification_note='Checked original dated document.')
    record.update(changes)
    return record


def synthetic_forecast_recipe(year=2022, value=30):
    stations = ('KAMW', 'KCID', 'KSUX')
    y = str(year)
    records = []
    for station in stations:
        days = [dt.date(year, 4, 1) + dt.timedelta(days=i) for i in range(40)]
        keys = []
        for day in days:
            run = dt.datetime.combine(day, dt.time(), tzinfo=dt.timezone.utc)
            valid = run + dt.timedelta(hours=24)
            key = ['forecast_daily_' + station, day.isoformat()]
            keys.append(key)
            records.append(example(**FORECAST_DEFINITION, series=key[0], period=key[1], observation_date=key[1],
                                   station=station, model_run_utc=run.isoformat(),
                                   valid_at_utc=valid.isoformat(), value=value,
                                   available_at=(run + dt.timedelta(hours=1)).isoformat()))
        records.append(example(**FORECAST_DEFINITION, series='forecast_station_' + station, period=y,
                               observation_date=f'{y}-05-20', value=value,
                               available_at=f'{y}-05-21', derivation='mean',
                               constituent_keys=keys))
    records.append(example(**FORECAST_DEFINITION, series='forecast_probability', period=y,
                           observation_date=f'{y}-05-20', value=value,
                           available_at=f'{y}-05-21', derivation='mean',
                           constituent_keys=[['forecast_station_' + s, y] for s in stations]))
    return records


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp_capture = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_capture.cleanup)
        self.capture_root = Path(self.temp_capture.name)
        self.capture_file = self.capture_root / 'synthetic.json'
        self.capture_file.write_bytes(b'synthetic original artifact')
        self.capture_sha = hashlib.sha256(self.capture_file.read_bytes()).hexdigest()

    def test_forecast_definition_is_required_at_every_level(self):
        records = synthetic_forecast_recipe()
        key = ('forecast_probability', '2022')
        self.assertTrue(replay(records, '2022-05-31', [key])['complete'])
        levels = ('forecast_daily_KAMW', 'forecast_station_KAMW', 'forecast_probability')
        wrong = {'model': 'NAM', 'element': 'QPF', 'units': 'mm',
                 'lead_start_hours': 0, 'lead_end_hours': 12,
                 'precipitation_threshold_inches': 0.1,
                 'precipitation_basis': 'snowfall'}
        for level in levels:
            i = next(i for i,r in enumerate(records) if r['series']==level)
            for field, value in wrong.items():
                for substitute in (value, None):
                    with self.subTest(level=level, field=field, replacement=substitute):
                        altered = copy.deepcopy(records)
                        if substitute is None:
                            del altered[i][field]
                        else:
                            altered[i][field] = substitute
                        result = replay(altered, '2022-05-31', [key])
                        self.assertFalse(result['complete'])
                        self.assertIn('forecast_definition_unverified',
                                      [s for row in result['report'] for s in row['candidate_statuses']])

    def test_forecast_feature_requires_valid_station_and_daily_recipes(self):
        records = synthetic_forecast_recipe()
        key = ('forecast_probability', '2022')
        result = require_replay(records, '2022-05-31', [key])
        self.assertEqual(len(result['records']), 124)
        annual = dict(records[-1], constituent_keys=[], derivation=None)
        self.assertFalse(replay(records[:-1] + [annual], '2022-05-31', [key])['complete'])
        station_index = next(i for i,r in enumerate(records) if r['series']=='forecast_station_KAMW')
        station = records[station_index]
        for bad in (dict(station, derivation=None, constituent_keys=[]),
                    dict(station, constituent_keys=station['constituent_keys'][:39]),
                    dict(station, constituent_keys=station['constituent_keys'][:39] + [station['constituent_keys'][0]]),
                    dict(station, constituent_keys=[['forecast_daily_KCID', '2022-04-01']] + station['constituent_keys'][1:]),
                    dict(station, constituent_keys=[['forecast_daily_KAMW', '2022-06-01']] + station['constituent_keys'][1:])):
            with self.subTest(bad=bad['constituent_keys'][:1]):
                changed = records.copy(); changed[station_index] = bad
                self.assertFalse(replay(changed, '2022-05-31', [key])['complete'])
        wrong_stations = dict(records[-1], constituent_keys=[['forecast_station_KAMW','2022'],
                               ['forecast_station_KAMW','2022'],['forecast_station_KSUX','2022']])
        self.assertFalse(replay(records[:-1]+[wrong_stations], '2022-05-31', [key])['complete'])

    def test_daily_forecast_runs_and_cutoffs_are_not_inferred(self):
        records = synthetic_forecast_recipe()
        key = ('forecast_probability','2022')
        daily = records[0]
        cases = [dict(daily, station='KCID'),
                 dict(daily, model_run_utc='2022-04-01T06:00:00Z'),
                 dict(daily, valid_at_utc='2022-04-02T12:00:00Z'),
                 dict(daily, observation_date='2022-04-02'),
                 dict(daily, value=101),
                 dict(daily, available_at='2022-06-01'),
                 dict(daily, available_at='2022-03-31'),
                 dict(daily, model_run_utc=None),
                 dict(daily, evidence_type='unverified')]
        for bad in cases:
            with self.subTest(bad=bad):
                self.assertFalse(replay([bad]+records[1:], '2022-05-31', [key])['complete'])
        self.assertFalse(replay(records[1:], '2022-05-31', [key])['complete'])
        may20 = dict(daily, period='2022-05-20', observation_date='2022-05-20',
                     model_run_utc='2022-05-20T00:00:00Z',
                     valid_at_utc='2022-05-21T00:00:00Z', available_at='2022-05-20T01:00:00Z')
        self.assertEqual(record_status(may20, '2022-05-31'), 'eligible')

    def test_project_ratios_need_ordered_named_constituents(self):
        for ratio, numerator, denominator in [
            ('nitrogen_price_ratio','nitrogen_price_current_april','nitrogen_price_previous_april'),
            ('q1_shipment_ratio','q1_shipments_current','q1_shipments_previous')]:
            with self.subTest(ratio=ratio):
                current_date = '2022-04-01' if ratio == 'nitrogen_price_ratio' else '2022-03-31'
                previous_date = '2021-04-01' if ratio == 'nitrogen_price_ratio' else '2021-03-31'
                a = example(series=numerator, period='2022', value=20, observation_date=current_date)
                b = example(series=denominator, period='2022', value=10, observation_date=previous_date)
                combined = example(series=ratio,period='2022',value=2,
                                   derivation='ratio', constituent_keys=[[numerator,'2022'],[denominator,'2022']])
                require_replay([a,b,combined], '2022-05-31', [(ratio,'2022')])
                for bad in (dict(combined, derivation=None, constituent_keys=[]),
                            dict(combined, constituent_keys=[[denominator,'2022'],[numerator,'2022']]),
                            dict(combined, constituent_keys=[[numerator,'2021'],[denominator,'2022']])):
                    self.assertFalse(replay([a,b,bad], '2022-05-31', [(ratio,'2022')])['complete'])

    def test_project_price_and_shipment_replay_rejects_nonpositive_values(self):
        for ratio, numerator, denominator, current_date, previous_date in [
            ('nitrogen_price_ratio','nitrogen_price_current_april','nitrogen_price_previous_april',
             '2022-04-01','2021-04-01'),
            ('q1_shipment_ratio','q1_shipments_current','q1_shipments_previous',
             '2022-03-31','2021-03-31')]:
            with self.subTest(ratio=ratio):
                a = example(series=numerator, period='2022', value=20, observation_date=current_date)
                b = example(series=denominator, period='2022', value=10, observation_date=previous_date)
                combined = example(series=ratio, period='2022', value=2, derivation='ratio',
                                   constituent_keys=[[numerator,'2022'],[denominator,'2022']])
                self.assertTrue(replay([a,b,combined], '2022-05-31', [(ratio,'2022')])['complete'])
                for records in ([dict(a,value=-20),dict(b,value=-10),combined],
                                [dict(a,value=0),b,combined],
                                [a,dict(b,value=0),combined],
                                [a,b,dict(combined,value=0)],
                                [a,b,dict(combined,value=-2)]):
                    with self.assertRaisesRegex(ValueError, 'must be positive'):
                        replay(records, '2022-05-31', [(ratio,'2022')])

    def test_derived_availability_must_not_predate_any_selected_component(self):
        current = example(series='current', period='2022', value=20,
                          available_at='2022-05-22T15:00:00Z')
        prior = example(series='prior', period='2022', value=10, available_at='2022-05-15')
        aggregate = example(series='ratio', period='2022', value=2, derivation='ratio',
                            constituent_keys=[['current','2022'],['prior','2022']],
                            available_at='2022-05-22T14:59:59Z')
        result = replay([current,prior,aggregate], '2022-05-31', [('ratio','2022')])
        self.assertFalse(result['complete'])
        self.assertEqual(result['report'][-1]['status'], 'derived_availability_mismatch')
        aggregate['available_at'] = '2022-05-22T15:00:00Z'
        self.assertTrue(replay([current,prior,aggregate], '2022-05-31', [('ratio','2022')])['complete'])
        aggregate['available_at'] = '2022-05-22'
        result = replay([current,prior,aggregate], '2022-05-31', [('ratio','2022')])
        self.assertFalse(result['complete'])
        self.assertEqual(result['report'][-1]['status'], 'derived_availability_order_unknown')

    def test_derived_mixed_precision_order_is_not_invented(self):
        current = example(series='current', period='2022', value=20,
                          available_at='2022-05-20')
        prior = example(series='prior', period='2022', value=10, available_at='2022-05-19')
        aggregate = example(series='ratio', period='2022', value=2, derivation='ratio',
                            constituent_keys=[['current','2022'],['prior','2022']],
                            available_at='2022-05-20T08:00:00Z')
        for cutoff in ('2022-05-31', '2022-05-21T00:00:00Z'):
            result = replay([current, prior, aggregate], cutoff, [('ratio','2022')])
            self.assertFalse(result['complete'])
            self.assertEqual(result['report'][-1]['status'], 'derived_availability_order_unknown')
            self.assertNotIn(('ratio','2022'), result['records'])
        for component_time, expected in [('2022-05-20T07:59:59Z','eligible'),
                                         ('2022-05-20T08:00:00Z','eligible'),
                                         ('2022-05-20T08:00:01Z','derived_availability_mismatch')]:
            result = replay([dict(current,available_at=component_time),prior,aggregate],
                            '2022-05-21T00:00:00Z',[('ratio','2022')])
            self.assertEqual(result['report'][-1]['status'], expected)
            self.assertEqual(result['complete'], expected=='eligible')
        daily = dict(aggregate, available_at='2022-05-20')
        self.assertTrue(replay([current,prior,daily], '2022-05-20', [('ratio','2022')])['complete'])
        self.assertTrue(replay([dict(current,available_at='2022-05-19'),prior,aggregate],
                               '2022-05-21T00:00:00Z',[('ratio','2022')])['complete'])

    def test_known_earlier_and_late_reports(self):
        r = example()
        found = require_replay([r], '2022-05-31', [('price', '2022-04')])
        self.assertEqual(found['records'][('price', '2022-04')]['value'], 10.0)
        late = example(available_at='2022-06-01')
        self.assertEqual(record_status(late, '2022-05-31'), 'after_cutoff')
        with self.assertRaises(ReplayUnavailable):
            require_replay([late], '2022-05-31', [('price', '2022-04')])

    def test_realized_period_dates_and_publication_order(self):
        cases = {
            'q2_sales': '2022-06-30',
            'q2_phosphate_sales': '2022-06-30',
            'q1_shipments_current': '2022-03-31',
            'q1_shipments_previous': '2021-03-31',
            'nitrogen_price_current_april': '2022-04-01',
            'nitrogen_price_previous_april': '2021-04-01',
        }
        for series, observed in cases.items():
            with self.subTest(series=series):
                rec = example(series=series, period='2022', observation_date=observed,
                              available_at='2022-08-15')
                self.assertEqual(record_status(rec, '2022-09-01'), 'eligible')
                with self.assertRaisesRegex(ValueError, 'does not match its period'):
                    record_status(dict(rec, observation_date='2025-06-30'), '2022-09-01')
                with self.assertRaisesRegex(ValueError, 'Availability predates'):
                    record_status(dict(rec, available_at='2020-01-01'), '2022-09-01')
                with self.assertRaisesRegex(ValueError, 'four-digit year'):
                    record_status(dict(rec, period='2022-01'), '2022-09-01')
        forecast = example(series='generic_future_forecast', period='2022',
                           observation_date='2022-06-01', available_at='2022-05-15')
        self.assertEqual(record_status(forecast, '2022-05-31'), 'eligible')

    def test_missing_date_and_unverified_are_rejected(self):
        r = example(available_at=None)
        self.assertEqual(record_status(r, '2022-05-31'), 'availability_unknown')
        u = example(evidence_type='unverified')
        self.assertEqual(record_status(u, '2022-05-31'), 'source_or_vintage_unverified')
        with self.assertRaises(ReplayUnavailable):
            require_replay([u], '2022-05-31', [('price', '2022-04')])
        result = replay([], '2022-05-31', [('price', '2022-04')])
        self.assertEqual(result['report'][0]['status'], 'missing')

    def test_unverified_late_claim_does_not_hide_competing_candidate(self):
        early = example()
        late = example(source_version='unverified-revision', value=20,
                       available_at='2022-06-01', evidence_type='unverified')
        self.assertEqual(record_status(late, '2022-05-31'), 'unverified_late_claim')
        result = replay([early, late], '2022-05-31', [('price', '2022-04')])
        self.assertFalse(result['complete'])
        self.assertEqual(result['report'][0]['status'], 'unverified_candidates')
        self.assertIn('unverified_late_claim', result['report'][0]['candidate_statuses'])
        with self.assertRaises(ReplayUnavailable):
            require_replay([early, late], '2022-05-31', [('price', '2022-04')])
        verified_late = dict(late, evidence_type='verified_immutable_document')
        self.assertTrue(replay([early, verified_late], '2022-05-31', [('price','2022-04')])['complete'])
        captured_late = dict(late, evidence_type='captured', retrieved_at=None)
        self.assertEqual(record_status(captured_late, '2022-05-31'), 'capture_time_unknown')
        captured_late.update(retrieved_at='2022-06-02', sha256=self.capture_sha,
                             capture_reference=self.capture_file.name)
        self.assertEqual(record_status(captured_late, '2022-05-31', capture_root=self.capture_root), 'after_cutoff')
        self.assertTrue(replay([early, captured_late], '2022-05-31', [('price','2022-04')], capture_root=self.capture_root)['complete'])
        with self.assertRaisesRegex(ValueError, 'At least one'):
            replay([], '2022-05-31', [])

    def test_duplicate_and_revisions(self):
        r = example()
        with self.assertRaisesRegex(ValueError, 'Duplicate observation version'):
            replay([r, r], '2022-05-31', [('price', '2022-04')])
        revised = example(source_version='version-2', value=20, available_at='2022-06-01')
        self.assertEqual(require_replay([r, revised], '2022-05-31', [('price', '2022-04')])['records'][('price', '2022-04')]['value'], 10)
        newer = example(source_version='version-2', value=11, available_at='2022-05-20')
        self.assertEqual(require_replay([r, newer], '2022-05-31', [('price', '2022-04')])['records'][('price', '2022-04')]['value'], 11)
        unknown = example(source_version='version-2', value=20, available_at=None)
        with self.assertRaises(ReplayUnavailable):
            require_replay([r, unknown], '2022-05-31', [('price', '2022-04')])
        tied = example(source_version='version-2', value=20, available_at='2022-05-15')
        with self.assertRaises(ReplayUnavailable):
            require_replay([r, tied], '2022-05-31', [('price', '2022-04')])

    def test_derived_ratio_cannot_bypass_missing_components(self):
        ratio = example(series='ratio', period='2022', value=2,
                        constituent_keys=[['current', '2022'], ['previous', '2022']], derivation='ratio')
        current = example(series='current', period='2022', value=20)
        previous = example(series='previous', period='2022', value=10)
        result = replay([ratio, current], '2022-05-31', [('ratio', '2022')])
        statuses = {r['series']: r['status'] for r in result['report']}
        self.assertEqual(statuses['previous'], 'missing')
        self.assertEqual(statuses['ratio'], 'blocked_by_constituent')
        self.assertNotIn(('ratio', '2022'), result['records'])
        self.assertFalse(result['complete'])
        complete = require_replay([ratio, current, previous], '2022-05-31', [('ratio', '2022')])
        self.assertEqual(len(complete['records']), 3)
        wrong_ratio = dict(ratio, value=3)
        bad = replay([wrong_ratio, current, previous], '2022-05-31', [('ratio', '2022')])
        self.assertEqual(next(r for r in bad['report'] if r['series'] == 'ratio')['status'], 'derived_value_mismatch')
        self.assertFalse(bad['complete'])
        cyclic = example(series='current', period='2022', constituent_keys=[['ratio', '2022']],
                         derivation='mean')
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            replay([ratio, cyclic, previous], '2022-05-31', [('ratio', '2022')])

    def test_derived_metadata_cannot_omit_formula_or_components(self):
        current = example(series='current', period='2022', value=20)
        previous = example(series='previous', period='2022', value=10)
        bad = example(series='ratio', period='2022', value=999,
                      constituent_keys=[['current', '2022'], ['previous', '2022']])
        with self.assertRaisesRegex(ValueError, 'require a supported derivation'):
            replay([bad, current, previous], '2022-05-31', [('ratio', '2022')])
        for formula in ('', 'unknown'):
            with self.subTest(formula=formula), self.assertRaises(ValueError):
                replay([dict(bad, derivation=formula), current, previous],
                       '2022-05-31', [('ratio', '2022')])
        for components in ([], None):
            with self.subTest(components=components), self.assertRaises(ValueError):
                replay([dict(bad, derivation='ratio', constituent_keys=components)],
                       '2022-05-31', [('ratio', '2022')])
        require_replay([current], '2022-05-31', [('current', '2022')])
        average = dict(bad, value=15, derivation='mean')
        require_replay([average, current, previous], '2022-05-31', [('ratio', '2022')])

    def test_only_the_selected_version_recipe_supplies_dependencies(self):
        old = example(series='aggregate', period='2022', value=5,
                      source_version='old', available_at='2022-05-10',
                      derivation='mean', constituent_keys=[['obsolete', '2022']])
        new = example(series='aggregate', period='2022', value=10,
                      source_version='new', available_at='2022-05-20',
                      derivation='mean', constituent_keys=[['current', '2022']])
        current = example(series='current', period='2022', value=10,
                          available_at='2022-05-19')
        result = require_replay([old, new, current], '2022-05-31', [('aggregate', '2022')])
        self.assertEqual(result['records'][('aggregate', '2022')]['source_version'], 'new')
        self.assertEqual({(r['series'], r['status']) for r in result['report']},
                         {('current', 'eligible'), ('aggregate', 'eligible')})
        obsolete = example(series='obsolete', period='2022', value=5)
        result = replay([old, new, obsolete], '2022-05-31', [('aggregate', '2022')])
        self.assertFalse(result['complete'])
        self.assertEqual(next(r['status'] for r in result['report'] if r['series']=='aggregate'),
                         'blocked_by_constituent')
        unknown = dict(old, source_version='unknown-version', available_at=None)
        result = replay([unknown, new, current], '2022-05-31', [('aggregate', '2022')])
        self.assertFalse(result['complete'])
        self.assertEqual(result['report'][-1]['status'], 'unverified_candidates')

    def test_timestamp_precision_and_timezone(self):
        r = example(available_at='2022-05-31')
        self.assertEqual(record_status(r, '2022-05-31T12:00:00Z'), 'availability_time_unknown')
        timed = example(available_at='2022-05-31T14:00:00+02:00')
        self.assertEqual(record_status(timed, '2022-05-31T12:00:00Z'), 'eligible')
        self.assertEqual(record_status(timed, '2022-05-31T11:59:59Z'), 'after_cutoff')
        with self.assertRaises(ValueError):
            record_status(example(available_at='2022-05-31T12:00:00'), '2022-05-31')

    def test_retrieval_cannot_provably_precede_availability(self):
        for kind in ('captured', 'verified_immutable_document', 'verified_asof_vintage'):
            with self.subTest(kind=kind):
                r = example(evidence_type=kind, retrieved_at='2022-05-14',
                            vintage_as_of='2022-05-31')
                self.assertEqual(record_status(r, '2022-05-31'), 'retrieval_before_availability')
                with self.assertRaises(ReplayUnavailable):
                    require_replay([r], '2022-05-31', [('price', '2022-04')])
        timed = example(available_at='2022-05-15T01:00:00+02:00',
                        retrieved_at='2022-05-14T22:59:59Z')
        self.assertEqual(record_status(timed, '2022-05-31'), 'retrieval_before_availability')
        self.assertEqual(record_status(dict(timed, retrieved_at='2022-05-14T23:00:00Z'),
                                       '2022-05-31'), 'eligible')
        self.assertEqual(record_status(dict(timed, retrieved_at='2022-05-14'),
                                       '2022-05-31'), 'eligible')

    def test_vintage_and_capture_are_not_invented(self):
        v = example(evidence_type='verified_asof_vintage', vintage_as_of='2022-05-31')
        self.assertEqual(record_status(v, '2022-05-31'), 'eligible')
        self.assertEqual(record_status(dict(v, retrieved_at=None), '2022-05-31'), 'verification_retrieval_unknown')
        self.assertEqual(record_status(dict(v, vintage_as_of='2022-06-01'), '2022-05-31'), 'exact_asof_vintage_unverified')
        self.assertEqual(record_status(dict(v, vintage_as_of='2022-05-30'), '2022-05-31'), 'exact_asof_vintage_unverified')
        capture = example(evidence_type='captured', retrieved_at='2022-06-01',
                          sha256=self.capture_sha, capture_reference=self.capture_file.name)
        self.assertEqual(record_status(capture, '2022-05-31'), 'capture_not_established_by_cutoff')
        self.assertEqual(record_status(dict(capture, retrieved_at='2022-05-30'), '2022-05-31', capture_root=self.capture_root), 'eligible')
        with self.assertRaises(ValueError):
            record_status(example(source_verified=False), '2022-05-31')

    def test_capture_requires_digest_and_inspectable_reference(self):
        capture = example(evidence_type='captured', retrieved_at='2022-05-30',
                          sha256=self.capture_sha, capture_reference=self.capture_file.name)
        for change in ({'sha256': None}, {'capture_reference': None},
                       {'capture_reference': ' '},
                       {'capture_reference': 'http://example.org/capture'},
                       {'capture_reference': 'https://user:pass@example.org/capture'}):
            with self.subTest(change=change):
                record = dict(capture, **change)
                self.assertEqual(record_status(record, '2022-05-31'), 'capture_evidence_incomplete')
                with self.assertRaises(ReplayUnavailable):
                    require_replay([record], '2022-05-31', [('price', '2022-04')])
        with self.assertRaises(ValueError):
            record_status(dict(capture, sha256='not-a-sha256'), '2022-05-31')
        self.assertEqual(record_status(dict(capture, capture_reference='https://example.org/captured.json'),
                                       '2022-05-31'), 'capture_external_unverified')
        self.assertEqual(record_status(dict(capture, retrieved_at='2022-06-01'),
                                       '2022-05-31'), 'capture_not_established_by_cutoff')

    def test_capture_bytes_missing_changed_and_outside_root_fail_closed(self):
        capture = example(evidence_type='captured', retrieved_at='2022-05-30',
                          sha256=self.capture_sha, capture_reference='synthetic.json')
        status = lambda r: record_status(r, '2022-05-31', capture_root=self.capture_root)
        self.assertEqual(status(capture), 'eligible')
        self.assertTrue(replay([capture], '2022-05-31', [('price', '2022-04')],
                               capture_root=self.capture_root)['complete'])
        self.assertEqual(record_status(capture, '2022-05-31'), 'capture_root_unconfigured')
        self.assertEqual(status(dict(capture, capture_reference='missing.json')), 'capture_unavailable')
        late = dict(capture, source_version='late', available_at='2022-06-01',
                    retrieved_at='2022-06-02', capture_reference='missing.json')
        self.assertEqual(status(late), 'capture_unavailable')
        self.assertFalse(replay([example(), late], '2022-05-31', [('price', '2022-04')],
                                capture_root=self.capture_root)['complete'])
        with tempfile.TemporaryDirectory() as outside:
            other = Path(outside) / 'outside.json'; other.write_bytes(self.capture_file.read_bytes())
            self.assertEqual(status(dict(capture, capture_reference=str(other))), 'capture_path_outside_root')
            link = self.capture_root / 'escape.json'; link.symlink_to(other)
            self.assertEqual(status(dict(capture, capture_reference=link.name)), 'capture_unavailable')
        self.assertEqual(status(dict(capture, capture_reference='../outside.json')), 'capture_path_outside_root')
        self.capture_file.write_bytes(b'changed bytes')
        self.assertEqual(status(capture), 'capture_checksum_mismatch')
        self.assertFalse(replay([capture], '2022-05-31', [('price', '2022-04')],
                                capture_root=self.capture_root)['complete'])
        self.capture_file.unlink()
        self.assertEqual(status(capture), 'capture_unavailable')

    def test_optional_verified_source_artifact_cannot_be_ignored(self):
        for kind in ('verified_immutable_document', 'verified_asof_vintage'):
            record = example(evidence_type=kind, vintage_as_of='2022-05-31',
                             capture_reference=self.capture_file.name, sha256=self.capture_sha)
            self.assertEqual(record_status(record, '2022-05-31', capture_root=self.capture_root), 'eligible')
            changed = dict(record, sha256='f'*64)
            self.assertEqual(record_status(changed, '2022-05-31', capture_root=self.capture_root), 'capture_checksum_mismatch')
            self.assertFalse(replay([changed], '2022-05-31', [('price','2022-04')],
                                    capture_root=self.capture_root)['complete'])
            late = dict(changed, available_at='2022-06-01', retrieved_at='2022-06-02')
            self.assertEqual(record_status(late, '2022-05-31', capture_root=self.capture_root), 'capture_checksum_mismatch')

    def test_stronger_gate_requires_artifact_for_verified_documents_and_vintages(self):
        for kind in ('verified_immutable_document', 'verified_asof_vintage'):
            rec = example(evidence_type=kind, vintage_as_of='2022-05-31')
            self.assertEqual(record_status(rec, '2022-05-31'), 'eligible')
            self.assertEqual(record_status(rec, '2022-05-31', require_capture=True), 'capture_evidence_incomplete')
            self.assertFalse(replay([rec], '2022-05-31', [('price','2022-04')],
                                    require_capture=True)['complete'])
            rec.update(capture_reference=self.capture_file.name, sha256=self.capture_sha)
            self.assertTrue(replay([rec], '2022-05-31', [('price','2022-04')],
                                   capture_root=self.capture_root, require_capture=True)['complete'])

    def test_vintage_retrieval_cannot_predate_the_asof_date(self):
        vintage = example(evidence_type='verified_asof_vintage', vintage_as_of='2022-05-31',
                          retrieved_at='2022-05-30')
        self.assertEqual(record_status(vintage, '2022-05-31'), 'retrieval_before_vintage')
        with self.assertRaises(ReplayUnavailable):
            require_replay([vintage], '2022-05-31', [('price', '2022-04')])
        self.assertEqual(record_status(dict(vintage, retrieved_at='2022-05-31'), '2022-05-31'), 'eligible')
        timed = dict(vintage, vintage_as_of='2022-05-31T12:00:00Z',
                     retrieved_at='2022-05-31T11:59:59Z')
        self.assertEqual(record_status(timed, '2022-05-31T12:00:00Z'), 'retrieval_before_vintage')
        self.assertEqual(record_status(dict(timed, retrieved_at='2022-05-31'),
                                       '2022-05-31T12:00:00Z'), 'retrieval_time_unknown')

    def test_actual_adapter_keeps_unverified_data_unverified(self):
        annual = [dict(year=2022, forecast_mean_p12_pct=30.0, price_ratio=2.0)]
        snapshot = {'revenue': [dict(year=2021, q2_usd=100, q2_filed='2021-08-05', q2_accession='abc')],
                    'price_index': [dict(year=2022, source='https://alfred.stlouisfed.org/series?seid=WPU065201',
                                         vintage_label='2022-05-31', as_of='2022-05-31',
                                         current_april=20.0, previous_april=10.0,
                                         current_observation='2022-04-01', previous_observation='2021-04-01')]}
        volume = [dict(year=2022, current_q1_tons=20, previous_q1_tons=19,
                       current_source='https://www.sec.gov/test', previous_source='https://www.sec.gov/test2',
                       current_accession='curr', previous_accession='prev',
                       current_filed='2022-05-01', previous_filed='2021-05-01', units='thousand product tons')]
        manifest = [dict(year=2022, station=s, source_url='https://example.org/source', station_mean_p12_pct=30,
                         retrieved_at_utc='2026-01-01T00:00:00Z') for s in ('KAMW', 'KCID', 'KSUX')]
        ledger = project_ledger(annual, snapshot, volume, manifest, [], [])
        report = project_replay_report(ledger, [2022])
        self.assertEqual(len(report), 11)
        self.assertTrue(all(r['status'] != 'eligible' for r in report))
        forecast = next(r for r in ledger if r['series'] == 'forecast_probability')
        self.assertEqual(forecast['value'], 30.0)
        self.assertIsNone(forecast['available_at'])
        self.assertEqual(len(forecast['constituent_sources']), 3)
        prices = {r['series']: r for r in ledger if r['series'].startswith('nitrogen_price_')}
        self.assertEqual(prices['nitrogen_price_current_april']['value'], 20.0)
        self.assertEqual(prices['nitrogen_price_previous_april']['value'], 10.0)
        self.assertEqual(prices['nitrogen_price_previous_april']['observation_date'], '2021-04-01')
        self.assertEqual(prices['nitrogen_price_previous_april']['recorded_as_of'], '2022-05-31')
        self.assertEqual(prices['nitrogen_price_current_april']['source_version'], '2022-05-31')
        self.assertIsNone(prices['nitrogen_price_current_april']['available_at'])
        self.assertEqual(len([r for r in ledger if r['series'].startswith('forecast_station_')]), 3)
        self.assertEqual(next(r for r in ledger if r['series'] == 'q1_shipment_ratio')['value'], 20/19)
        manifest[0]['station_mean_p12_pct'] = 20
        manifest[1]['station_mean_p12_pct'] = 20
        manifest[2]['station_mean_p12_pct'] = 20
        with self.assertRaisesRegex(ValueError, 'Forecast summary differs'):
            project_ledger(annual, snapshot, volume, manifest, [], [])
        for m in manifest:
            m['station_mean_p12_pct'] = 30
        snapshot['price_index'][0].update(current_april=20.0, previous_april=8.0)
        with self.assertRaisesRegex(ValueError, 'Price ratio differs'):
            project_ledger(annual, snapshot, volume, manifest, [], [])
        snapshot['price_index'][0].update(previous_april=10.0)
        old = copy.deepcopy(snapshot)
        old['revenue'].append(old['revenue'][0])
        duplicate_ledger = project_ledger(annual, old, volume, manifest, [], [])
        with self.assertRaisesRegex(ValueError, 'Duplicate observation version'):
            project_replay_report(duplicate_ledger, [2022])

    def test_original_and_previous_april_release_metadata_and_missing_constituent(self):
        annual = [dict(year=2010, forecast_mean_p12_pct=30.0, price_ratio=2.0)]
        snapshot = {'revenue': [], 'price_index': []}
        old = [dict(year=2010, current_april=20.0, previous_april=10.0,
                    current_release_date='2010-05-15', previous_release_date='2009-09-15',
                    current_source_url='https://www.bls.gov/current-release',
                    previous_source_url='https://www.bls.gov/prior-release')]
        manifest = [dict(year=2010, station=s, source_url='https://example.org/source', station_mean_p12_pct=30)
                    for s in ('KAMW', 'KCID', 'KSUX')]
        ledger = project_ledger(annual, snapshot, [], manifest, old, [])
        current = next(r for r in ledger if r['series'] == 'nitrogen_price_current_april')
        prior = next(r for r in ledger if r['series'] == 'nitrogen_price_previous_april')
        self.assertEqual(current['available_at'], '2010-05-15')
        self.assertEqual(prior['available_at'], '2009-09-15')
        self.assertEqual(current['source_url'], 'https://www.bls.gov/current-release')
        self.assertEqual(prior['source_url'], 'https://www.bls.gov/prior-release')
        self.assertEqual(prior['observation_date'], '2009-04-01')
        self.assertEqual(current['evidence_type'], 'unverified')
        del old[0]['previous_april']
        missing = project_ledger(annual, snapshot, [], manifest, old, [])
        report = project_replay_report(missing, [2010])
        missing_prior = next(r for r in report if r['series'] == 'nitrogen_price_previous_april')
        self.assertEqual(missing_prior['status'], 'missing')

    def test_browser_price_evidence_preserves_retrieval_bounds_and_hash(self):
        annual = [dict(year=2022, forecast_mean_p12_pct=30.0, price_ratio=2.0)]
        snapshot = {'revenue': [], 'price_index': [dict(year=2022,
                    source='https://alfred.stlouisfed.org/source', vintage_label='2022-05-31',
                    as_of='2022-05-31', current_april=20, previous_april=10)]}
        observed = dict(forecast_year=2022, observation_date='2021-04-01', value=10,
                        source_url='https://alfred.stlouisfed.org/graph/', source_version='ALFRED as-of 2022-05-31',
                        retrieval_start_utc='2026-11-05T14:00:00Z', retrieval_end_utc='2026-11-05T14:03:00Z')
        encoded = json.dumps(observed, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        evidence = [{'record': observed, 'canonical_sha256': hashlib.sha256(encoded.encode('utf-8')).hexdigest()}]
        ledger = project_ledger(annual, snapshot, [], [], [], [], browser_price_evidence=evidence)
        prev = next(r for r in ledger if r['series'] == 'nitrogen_price_previous_april')
        self.assertIsNone(prev['retrieved_at'])
        self.assertEqual(prev['browser_retrieval_start_utc'], '2026-11-05T14:00:00Z')
        self.assertEqual(prev['browser_retrieval_end_utc'], '2026-11-05T14:03:00Z')
        self.assertTrue(prev['browser_matches_snapshot'])
        self.assertEqual(prev['browser_transcription_sha256'], evidence[0]['canonical_sha256'])
        self.assertEqual(prev['evidence_type'], 'unverified')
        evidence[0]['canonical_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            project_ledger(annual, snapshot, [], [], [], [], browser_price_evidence=evidence)

    def test_fold_checks_historical_features_and_current_filing_dates(self):
        records = [example(series='q2_sales', period='2019', observation_date='2019-06-30',
                           available_at='2019-08-01'),
                   example(series='q2_sales', period='2020', observation_date='2020-06-30',
                           available_at='2022-06-01'),
                   example(series='nitrogen_price_ratio', period='2020', observation_date='2020-04-01',
                           available_at='2020-06-01')]
        fold = project_fold_replay(records, 2022, [2020])
        training_target = next(x for x in fold if x['role'] == 'training_target')
        self.assertEqual(training_target['status'], 'after_cutoff')
        price = next(x for x in fold if x['role'] == 'training_features' and x['series'] == 'nitrogen_price_ratio')
        self.assertEqual(price['status'], 'after_cutoff')
        with self.assertRaises(ValueError):
            project_fold_replay(records, 2022, [2022])

    def test_cli_writes_complete_and_incomplete_private_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'records.json').write_text(json.dumps([example()]))
            (root / 'needs.json').write_text(json.dumps([['price', '2022-04']]))
            command = [sys.executable, str(Path(__file__).with_name('replay.py')),
                       '--ledger', str(root / 'records.json'), '--requirements', str(root / 'needs.json'),
                       '--cutoff', '2022-05-31', '--output', str(root / 'result.json')]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json.loads((root / 'result.json').read_text())['complete'])
            (root / 'records.json').write_text(json.dumps([example(available_at='2022-06-01')]))
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertFalse(json.loads((root / 'result.json').read_text())['complete'])


if __name__ == '__main__':
    unittest.main()
