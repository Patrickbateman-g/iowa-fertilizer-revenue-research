import json
from pathlib import Path
import unittest
import pandas as pd
import replay
import workflow
import test_workflow
from replay_tests import synthetic_forecast_recipe

class CalibrationWorkflowTests(unittest.TestCase):
    setUp=test_workflow.WorkflowTests.setUp
    tearDown=test_workflow.WorkflowTests.tearDown
    args=test_workflow.WorkflowTests.args
    run_workflow=test_workflow.WorkflowTests.run_workflow

    def test_full_range_saves_an_empty_calibration_file_with_schema(self):
        manifest=self.run_workflow(*self.args(),first_year=2016,last_year=2016)
        self.assertEqual(manifest['calibration_years_used'],[])
        data=pd.read_csv(self.out/'uncertainty'/'calibration_predictions_private.csv')
        self.assertTrue(data.empty)
        self.assertIn('year',data.columns)
        self.assertIn('model',data.columns)

    def test_subset_matches_full_range_and_preserves_only_requested_outputs(self):
        self.run_workflow(*self.args(),first_year=2016,last_year=2020)
        full=pd.read_csv(self.out/'uncertainty'/'ranges_private.csv')
        subset_out=self.root/'subset'
        manifest=self.run_workflow(self.annual,self.shipments,self.ledger,self.artifacts,subset_out,
                                      first_year=2020,last_year=2020)
        subset=pd.read_csv(subset_out/'uncertainty'/'ranges_private.csv')
        expected=full[full.year.eq(2020)].sort_values('model').reset_index(drop=True)
        actual=subset.sort_values('model').reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False)
        self.assertEqual(manifest['retrospective_years'],[2020])
        self.assertEqual(manifest['calibration_years_used'],[2016,2017,2018,2019])
        self.assertEqual(len(pd.read_csv(subset_out/'comparison'/'predictions_private.csv')),4)
        text=(subset_out/'report'/'RESEARCH_REPORT.md').read_text()
        self.assertIn('2016',text)
        self.assertIn('calibration',text.lower())

    def test_missing_calibration_year_is_explicit(self):
        shipments=json.loads(self.shipments.read_text())
        self.shipments.write_text(json.dumps([v for v in shipments if v['year']!=2018]))
        manifest=self.run_workflow(*self.args(),first_year=2020,last_year=2020)
        inventory=pd.read_csv(self.out/'uncertainty'/'calibration_inventory_private.csv').set_index('year')
        self.assertEqual(inventory.loc[2018,'reason'],'missing_shipments')
        self.assertNotIn(2018,manifest['calibration_years_used'])
        ranges=pd.read_csv(self.out/'uncertainty'/'ranges_private.csv')
        self.assertTrue(ranges.lower_usd.isna().all())

    def test_original_filed_value_mismatch_is_not_hidden_by_later_revision(self):
        annual=pd.read_csv(self.annual)
        annual.loc[annual.year.eq(2016),'target_q2_usd']=1200.
        shipments=json.loads(self.shipments.read_text())
        predictions=workflow.comparison.compare_incremental(annual,shipments,[2017])['predictions']
        base=dict(series='q2_sales',period='2016',observation_date='2016-06-30',
                  source_url='https://example.org/filing',evidence_type='captured',
                  sha256=self.capture_sha,capture_reference=self.capture.name)
        first=dict(base,value=1000.,source_version='original',available_at='2016-08-01',retrieved_at='2016-08-01')
        later=dict(base,value=1200.,source_version='revised',available_at='2016-09-01',retrieved_at='2016-09-01')
        audit=workflow._audit_training_cohorts([first,later],annual,predictions,capture_root=self.root)
        row=next(r for r in audit if r['row_year']==2016)
        self.assertEqual(row['outcome_status_at_cutoff'],'eligible')
        self.assertEqual(row['filing_status'],'filing_value_mismatch')

    def test_strict_requires_older_calibration_inputs_even_if_requested_fold_is_verified(self):
        records=[]; annual=[]; volumes=[]
        def add(series,y,value,observed,available,**extra):
            records.append(dict(series=series,period=str(y),value=value,
                observation_date=observed,source_url='https://example.org/source',source_version='v1',
                available_at=available,retrieved_at=available,evidence_type='captured',
                sha256=self.capture_sha,capture_reference=self.capture.name,**extra))
        for y in range(2009,2021):
            add('q2_sales',y,1000,f'{y}-06-30',f'{y}-08-01')
            if y<=2015: add('q2_phosphate_sales',y,100,f'{y}-06-30',f'{y}-08-01')
        for y in range(2010,2021):
            adjusted=y<=2015
            forecast=20+y-2010
            annual.append(dict(year=y,as_of=f'{y}-05-31',target_filed=f'{y}-08-01',
                target_q2_usd=900 if adjusted else 1000,prior_q2_usd=900 if adjusted else 1000,
                price_ratio=1,forecast_mean_p12_pct=forecast))
            for series,observed in [('nitrogen_price_current_april',f'{y}-04-01'),
                                    ('nitrogen_price_previous_april',f'{y-1}-04-01')]:
                add(series,y,1,observed,f'{y}-05-20')
            add('nitrogen_price_ratio',y,1,f'{y}-04-01',f'{y}-05-20',derivation='ratio',
                constituent_keys=[['nitrogen_price_current_april',str(y)],['nitrogen_price_previous_april',str(y)]])
            records.extend(dict(r, sha256=self.capture_sha, capture_reference=self.capture.name)
                           for r in synthetic_forecast_recipe(y,value=forecast))
            if y>=2016:
                for series,observed in [('q1_shipments_current',f'{y}-03-31'),
                                        ('q1_shipments_previous',f'{y-1}-03-31')]:
                    add(series,y,1,observed,f'{y}-05-01')
                add('q1_shipment_ratio',y,1,f'{y}-03-31',f'{y}-05-01',derivation='ratio',
                    constituent_keys=[['q1_shipments_current',str(y)],['q1_shipments_previous',str(y)]])
                volumes.append(dict(year=y,current_q1_tons=1,previous_q1_tons=1,
                    units='thousand product tons',current_filed=f'{y}-05-01',previous_filed=f'{y-1}-05-01',
                    current_source='https://www.sec.gov/Archives/edgar/data/1324404/current.htm',
                    previous_source='https://www.sec.gov/Archives/edgar/data/1324404/previous.htm'))
        pd.DataFrame(annual).to_csv(self.annual,index=False)
        self.shipments.write_text(json.dumps(volumes))
        old=next(r for r in records if r['series']=='q1_shipments_current' and r['period']=='2016')
        old['evidence_type']='unverified'
        self.ledger.write_text(json.dumps(records))
        manifest=self.run_workflow(*self.args(),first_year=2020,last_year=2020)
        self.assertEqual(set(manifest['replay_requirement_counts']),{'eligible'})
        self.assertGreater(manifest['calibration_replay_requirement_counts']['unverified_candidates'],0)
        self.assertFalse(manifest['strict_replay_complete'])
        with self.assertRaises(replay.ReplayUnavailable):
            self.run_workflow(self.annual,self.shipments,self.ledger,self.artifacts,self.root/'strict',
                                  first_year=2020,last_year=2020,strict_replay=True)
        old['evidence_type']='captured'
        self.ledger.write_text(json.dumps(records))
        verified=self.run_workflow(self.annual,self.shipments,self.ledger,self.artifacts,self.root/'verified',
                                      first_year=2020,last_year=2020,strict_replay=True)
        self.assertTrue(verified['strict_replay_complete'])

    def test_strict_audit_does_not_ignore_calibration_provenance(self):
        manifest=self.run_workflow(*self.args(),first_year=2020,last_year=2020)
        self.assertGreater(sum(manifest['calibration_replay_requirement_counts'].values()),0)
        self.assertFalse(manifest['strict_replay_complete'])
        with self.assertRaises(replay.ReplayUnavailable):
            self.run_workflow(self.annual,self.shipments,self.ledger,self.artifacts,self.root/'strict',
                                  first_year=2020,last_year=2020,strict_replay=True)
        self.assertFalse((self.root/'strict').exists())

if __name__=='__main__': unittest.main()
