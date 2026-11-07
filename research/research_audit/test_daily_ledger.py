import copy
from datetime import date, datetime, timedelta, timezone
import unittest
import daily_ledger as d
import replay


def fixture():
    year=2022; ledger=[]; manifest=[]; rows=[]
    for station in d.STATIONS:
        source='https://example.org/'+station
        retrieved='2026-11-05T10:00:00+00:00'
        for i in range(40):
            run=datetime(2022,4,1,tzinfo=timezone.utc)+timedelta(days=i)
            raw={'station':station,'model':'GFS','runtime':run.isoformat(),
                 'ftime':(run+timedelta(hours=24)).isoformat(),'p12':30}
            rows.append({'year':year,'station':station,'source_url':source,
                         'retrieved_at_utc':retrieved,'publication_time_utc':None,
                         'availability_time_utc':None,'source_version':'unknown',
                         'original_source_record':raw,'original_record_sha256':d._digest(raw)})
        manifest.append({'year':year,'station':station,'source_url':source,'retrieved_at_utc':retrieved,
                         'status':'retrieved','publication_time_utc':None,'availability_time_utc':None,
                         'source_version':'unknown','expected_days':50,'valid_days':40,
                         'duplicate_rows':0,'conflicting_days':0,'invalid_rows':0,
                         'station_mean_p12_pct':30.,'usable':True})
        ledger.append(dict(series='forecast_station_'+station,period=str(year),value=30.,
             observation_date=f'{year}-05-20',source_url=source,source_version='unknown',
             available_at=None,retrieved_at=retrieved,evidence_type='unverified'))
    ledger.append(dict(series='forecast_probability',period=str(year),value=30.,
             observation_date=f'{year}-05-20',source_url='https://example.org/mean',source_version='unknown',
             available_at=None,retrieved_at='2026-11-05T10:00:00+00:00',evidence_type='unverified',
             derivation='mean',constituent_keys=[['forecast_station_'+s,str(year)] for s in d.STATIONS]))
    return ledger,{'ledger':rows,'manifest':manifest}

class DailyLedgerTests(unittest.TestCase):
    def test_complete_recipe_does_not_promote_or_mutate(self):
        ledger,audit=fixture(); original=copy.deepcopy(ledger)
        built,diagnostics=d.attach_daily_forecasts(ledger,audit)
        self.assertEqual(ledger,original)
        self.assertEqual(len(built),len(ledger)+120)
        self.assertEqual({r['status'] for r in diagnostics},{'attached_unverified'})
        for r in built:
            if r['series'].startswith('forecast_daily_'):
                self.assertIsNone(r['available_at']); self.assertEqual(r['source_version'],'unknown')
                self.assertEqual(r['evidence_type'],'unverified');self.assertEqual(len(r['source_row_sha256']),1)
                self.assertTrue(replay._forecast_definition_matches(r))
        self.assertFalse(replay.replay(built,'2022-05-31',[('forecast_probability','2022')])['complete'])
        rec=next(r for r in built if r['series']=='forecast_station_KAMW')
        self.assertEqual(rec['derivation'],'mean');self.assertEqual(len(rec['constituent_keys']),40)
        self.assertTrue(replay._forecast_definition_matches(rec))
        annual=next(r for r in built if r['series']=='forecast_probability')
        self.assertTrue(replay._forecast_definition_matches(annual))

    def test_conflicting_preexisting_definition_is_not_overwritten(self):
        for series in ('forecast_station_KAMW','forecast_probability'):
            ledger,audit=fixture()
            next(r for r in ledger if r['series']==series)['model']='NAM'
            with self.subTest(series=series):
                with self.assertRaisesRegex(ValueError,'definition conflicts'):
                    d.attach_daily_forecasts(ledger,audit)

    def test_checksum_and_attribution_tampering_fail_closed(self):
        ledger,audit=fixture(); audit['ledger'][0]['original_source_record']['p12']=42
        with self.assertRaisesRegex(ValueError,'checksum'): d.attach_daily_forecasts(ledger,audit)
        ledger,audit=fixture(); audit['ledger'][0]['station']='KCID'
        with self.assertRaisesRegex(ValueError,'attribution'): d.attach_daily_forecasts(ledger,audit)

    def test_duplicate_and_conflict_follow_original_selection(self):
        ledger,audit=fixture(); one=copy.deepcopy(audit['ledger'][0]); audit['ledger'].append(one)
        m=audit['manifest'][0];m['duplicate_rows']=1
        built,diag=d.attach_daily_forecasts(ledger,audit)
        item=next(r for r in built if r['series']=='forecast_daily_KAMW' and r['period']=='2022-04-01')
        self.assertEqual(len(item['source_row_sha256']),2)
        self.assertEqual(diag[0]['status'],'attached_unverified')
        changed=copy.deepcopy(one); changed['original_source_record']['p12']=60
        changed['original_record_sha256']=d._digest(changed['original_source_record'])
        audit['ledger'].append(changed);m['conflicting_days']=1;m['valid_days']=39;m['usable']=False
        built,diag=d.attach_daily_forecasts(ledger,audit)
        self.assertEqual(diag[0]['status'],'unusable_station')
        self.assertEqual(diag[0]['conflicting_dates'],['2022-04-01'])
        self.assertFalse(any(r['series']=='forecast_daily_KAMW' for r in built))

    def test_mismatch_missing_invalid_and_unverified_are_not_promoted(self):
        ledger,audit=fixture();audit['manifest'][0]['station_mean_p12_pct']=40
        built,diag=d.attach_daily_forecasts(ledger,audit)
        self.assertEqual(diag[0]['status'],'manifest_mismatch')
        self.assertFalse(any(r['series']=='forecast_daily_KAMW' for r in built))
        ledger,audit=fixture(); audit['manifest'][0]['status']='not_attempted'
        audit['ledger']=[r for r in audit['ledger'] if r['station']!='KAMW']
        self.assertEqual(d.attach_daily_forecasts(ledger,audit)[1][0]['status'],'source_not_retrieved')
        ledger,audit=fixture();audit['ledger'][0]['availability_time_utc']='2022-04-01T04:00:00Z'
        with self.assertRaisesRegex(ValueError,'unknown'): d.attach_daily_forecasts(ledger,audit)

    def test_missing_manifest_is_explicit(self):
        ledger,audit=fixture()
        audit['manifest']=[m for m in audit['manifest'] if m['station']!='KAMW']
        audit['ledger']=[r for r in audit['ledger'] if r['station']!='KAMW']
        _,diag=d.attach_daily_forecasts(ledger,audit)
        missing=next(x for x in diag if x['station']=='KAMW')
        self.assertEqual(missing['status'],'manifest_missing')

    def test_wrong_station_model_and_runtime_are_detected(self):
        ledger,audit=fixture();r=audit['ledger'][0]['original_source_record'];r['model']='AVN'
        audit['ledger'][0]['original_record_sha256']=d._digest(r)
        diag=d.attach_daily_forecasts(ledger,audit)[1]
        self.assertEqual(diag[0]['status'],'manifest_mismatch')
        ledger,audit=fixture();r=audit['ledger'][0]['original_source_record'];r['ftime']='2022-04-03T00:00:00Z'
        audit['ledger'][0]['original_record_sha256']=d._digest(r)
        self.assertEqual(d.attach_daily_forecasts(ledger,audit)[1][0]['status'],'manifest_mismatch')

    def test_annual_mismatch_and_existing_forecast_verification_rejected(self):
        ledger,audit=fixture();ledger[-1]['value']=90
        with self.assertRaisesRegex(ValueError,'annual'): d.attach_daily_forecasts(ledger,audit)
        ledger,audit=fixture();ledger[0].update(evidence_type='captured',source_version='known')
        with self.assertRaisesRegex(ValueError,'not unverified'):d.attach_daily_forecasts(ledger,audit)
        ledger,audit=fixture();ledger[0]['available_at']='2022-05-20'
        with self.assertRaisesRegex(ValueError,'unknown timing/version'):d.attach_daily_forecasts(ledger,audit)

if __name__=='__main__': unittest.main()
