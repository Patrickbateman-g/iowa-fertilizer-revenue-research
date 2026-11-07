import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
import replay
from replay_tests import synthetic_forecast_recipe
import workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.capture=self.root/'synthetic.json'
        self.capture.write_bytes(b'synthetic source')
        self.capture_sha=hashlib.sha256(self.capture.read_bytes()).hexdigest()
        rows=[dict(year=y,as_of=f'{y}-05-31',target_filed=f'{y}-08-01',
                   target_q2_usd=1000+40*(y-2010),prior_q2_usd=900+20*(y-2010),
                   price_ratio=1+.01*(y-2010),forecast_mean_p12_pct=20+2*(y-2010))
              for y in range(2010,2027)]
        self.annual=self.root/'annual.csv'; pd.DataFrame(rows).to_csv(self.annual,index=False)
        volumes=[dict(year=y,current_q1_tons=100+y-2010,previous_q1_tons=100,
                 units='thousand product tons',current_filed=f'{y}-05-01',previous_filed=f'{y-1}-05-01',
                 current_source='https://www.sec.gov/Archives/edgar/data/1324404/current.htm',
                 previous_source='https://www.sec.gov/Archives/edgar/data/1324404/previous.htm')
                 for y in range(2016,2027)]
        self.shipments=self.root/'shipment.json'; self.shipments.write_text(json.dumps(volumes))
        self.ledger_rows=[dict(series='forecast_probability',period='2016',value=32.0,
            observation_date='2016-05-20',source_url='https://example.org/source',source_version='v1',
            available_at='2016-06-01',retrieved_at='2026-11-05',evidence_type='unverified')]
        self.ledger=self.root/'ledger.json'; self.ledger.write_text(json.dumps(self.ledger_rows))
        self.artifacts=self.root/'artifacts'; (self.artifacts/'iowa_v2').mkdir(parents=True)
        pd.DataFrame([{'model':'price_only','years':2,'mae_usd':50.}]).to_csv(
            self.artifacts/'iowa_v2'/'scores_private.csv',index=False)
        self.out=self.root/'output'

    def tearDown(self): self.tmp.cleanup()

    def args(self): return (self.annual,self.shipments,self.ledger,self.artifacts,self.out)

    def run_workflow(self,*args,**kwargs):
        return workflow.run_workflow(*args,capture_root=self.root,**kwargs)

    def test_cli_produces_matched_results_ranges_report_and_replay_without_overclaim(self):
        cmd=[sys.executable,str(Path(workflow.__file__)),'--annual',str(self.annual),
             '--shipments',str(self.shipments),'--ledger',str(self.ledger),
             '--artifacts',str(self.artifacts),'--output',str(self.out)]
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1')
        run=subprocess.run(cmd,text=True,capture_output=True,env=env)
        self.assertEqual(run.returncode,0,run.stderr)
        manifest=json.loads((self.out/'manifest.json').read_text())
        self.assertEqual(manifest['requested_model_years'],44)
        self.assertFalse(manifest['strict_replay_complete'])
        self.assertGreater(manifest['replay_requirement_counts']['unverified_candidates'],0)
        self.assertGreater(manifest['replay_requirement_counts']['missing'],0)
        preds=pd.read_csv(self.out/'comparison'/'predictions_private.csv')
        self.assertEqual(len(preds),44)
        self.assertEqual(set(preds.groupby('year').size()),{4})
        self.assertEqual(preds.year.nunique(),11)
        ranges=pd.read_csv(self.out/'uncertainty'/'ranges_private.csv')
        self.assertEqual(len(ranges),44)
        self.assertTrue(ranges[ranges.year.lt(2020)].lower_usd.isna().all())
        events=json.loads((self.out/'replay'/'replay_report_private.json').read_text())
        late=[r for r in events if r['series']=='forecast_probability' and str(r['period'])=='2016']
        self.assertTrue(late)
        self.assertTrue(all('unverified_late_claim' in r['candidate_statuses'] for r in late))
        self.assertTrue((self.out/'register'/'experiment_register_private.csv').exists())
        text=(self.out/'report'/'RESEARCH_REPORT.md').read_text()
        self.assertIn('retrospective',text.lower())
        self.assertIn('2026',text)
        self.assertIn('not independently established',text)
        self.assertFalse((self.out/'annual.csv').exists())

    def test_bad_cohort_and_mismatch_fail_before_any_output(self):
        data=pd.read_csv(self.annual); data=data[data.year.ne(2020)]; data.to_csv(self.annual,index=False)
        with self.assertRaisesRegex(ValueError,'Missing matched'):
            self.run_workflow(*self.args())
        self.assertFalse(self.out.exists())
        data=pd.concat([data,pd.DataFrame([dict(year=2020,as_of='2020-05-31',target_filed='2020-08-01',
            target_q2_usd=1400,prior_q2_usd=1100,price_ratio=1.1,forecast_mean_p12_pct=40)])],ignore_index=True)
        data.to_csv(self.annual,index=False)
        self.ledger_rows[0]['value']=32.1
        self.ledger.write_text(json.dumps(self.ledger_rows))
        with self.assertRaisesRegex(ValueError,'conflict'):
            self.run_workflow(*self.args())
        self.assertFalse(self.out.exists())

    def test_strict_replay_rejects_unverified_and_after_cutoff(self):
        with self.assertRaises(replay.ReplayUnavailable):
            self.run_workflow(*self.args(),strict_replay=True)
        self.assertFalse(self.out.exists())

    def test_strict_replay_requires_verified_evaluation_target_and_filing_date(self):
        records=[]; annual=[]
        def add(series,year,value,available,evidence='captured',**extra):
            observed = {
                'q2_sales':f'{year}-06-30', 'q2_phosphate_sales':f'{year}-06-30',
                'q1_shipments_current':f'{year}-03-31',
                'q1_shipments_previous':f'{year-1}-03-31',
                'nitrogen_price_current_april':f'{year}-04-01',
                'nitrogen_price_previous_april':f'{year-1}-04-01',
            }.get(series, available)
            records.append(dict(series=series,period=str(year),value=value,
                observation_date=observed,source_url='https://example.org/source',source_version='v1',
                available_at=available,retrieved_at=available,evidence_type=evidence,
                sha256=self.capture_sha,capture_reference=self.capture.name,**extra))
        for y in range(2009,2017):
            add('q2_sales',y,1000,'2027-01-01' if y==2016 else f'{y}-08-01',
                'unverified' if y==2016 else 'captured')
            if y<=2015: add('q2_phosphate_sales',y,100,f'{y}-08-01')
        for y in range(2010,2017):
            forecast=20+y-2010
            annual.append(dict(year=y,as_of=f'{y}-05-31',target_filed=f'{y}-08-01',
                target_q2_usd=900 if y<=2015 else 1000,
                prior_q2_usd=900 if y<=2015 else 1000,
                price_ratio=1,forecast_mean_p12_pct=forecast))
            for series in ('nitrogen_price_current_april','nitrogen_price_previous_april'):
                add(series,y,1,f'{y}-05-20')
            add('nitrogen_price_ratio',y,1,f'{y}-05-20',derivation='ratio',
                constituent_keys=[['nitrogen_price_current_april',str(y)],['nitrogen_price_previous_april',str(y)]])
            records.extend(dict(r, sha256=self.capture_sha, capture_reference=self.capture.name)
                           for r in synthetic_forecast_recipe(y,value=forecast))
        for series in ('q1_shipments_current','q1_shipments_previous'):
            add(series,2016,1,'2016-05-01')
        add('q1_shipment_ratio',2016,1,'2016-05-01',derivation='ratio',
            constituent_keys=[['q1_shipments_current','2016'],['q1_shipments_previous','2016']])
        pd.DataFrame(annual).to_csv(self.annual,index=False)
        volumes=json.loads(self.shipments.read_text())
        volumes[0]['current_q1_tons']=volumes[0]['previous_q1_tons']=1
        self.shipments.write_text(json.dumps(volumes[:1]))
        self.ledger.write_text(json.dumps(records))
        with self.assertRaises(replay.ReplayUnavailable):
            self.run_workflow(*self.args(),first_year=2016,last_year=2016,strict_replay=True)
        self.assertFalse(self.out.exists())
        exploratory=self.run_workflow(*self.args(),first_year=2016,last_year=2016)
        self.assertFalse(exploratory['strict_replay_complete'])
        self.assertEqual(exploratory['replay_requirement_counts'],{'eligible':917})
        self.assertEqual(exploratory['evaluation_outcome_requirement_counts'],
                         {'unverified_candidates':1,'blocked_by_outcome':1})
        saved_report=(self.out/'report'/'RESEARCH_REPORT.md').read_text()
        self.assertIn('## Verification of evaluated outcomes',saved_report)
        self.assertIn('| blocked_by_outcome | 1 |',saved_report)
        target=next(r for r in records if r['series']=='q2_sales' and r['period']=='2016')
        target.update(available_at='2016-07-30',retrieved_at='2016-07-30',
                      evidence_type='captured')
        self.ledger.write_text(json.dumps(records))
        with self.assertRaises(replay.ReplayUnavailable):
            self.run_workflow(self.annual,self.shipments,self.ledger,self.artifacts,self.root/'early',
                                  first_year=2016,last_year=2016,strict_replay=True)
        target.update(available_at='2016-08-01',retrieved_at='2016-08-01')
        self.ledger.write_text(json.dumps(records))
        accepted=self.run_workflow(self.annual,self.shipments,self.ledger,self.artifacts,self.root/'verified',
                                      first_year=2016,last_year=2016,strict_replay=True)
        self.assertTrue(accepted['strict_replay_complete'])
        self.assertEqual(accepted['evaluation_outcome_requirement_counts'],{'eligible':2})
        self.assertIn('| eligible | 2 |',(self.root/'verified'/'report'/'RESEARCH_REPORT.md').read_text())
        source = next(r for r in records if r['series'].startswith('forecast_daily_'))
        source.pop('capture_reference'); source.pop('sha256')
        self.ledger.write_text(json.dumps(records))
        with self.assertRaises(replay.ReplayUnavailable):
            self.run_workflow(self.annual,self.shipments,self.ledger,self.artifacts,self.root/'no-artifact',
                              first_year=2016,last_year=2016,strict_replay=True)
        self.assertFalse((self.root/'no-artifact').exists())
        source.update(capture_reference=self.capture.name, sha256=self.capture_sha)
        self.ledger.write_text(json.dumps(records))
        self.capture.write_bytes(b'tampered source')
        with self.assertRaises(replay.ReplayUnavailable):
            self.run_workflow(self.annual,self.shipments,self.ledger,self.artifacts,self.root/'tampered',
                              first_year=2016,last_year=2016,strict_replay=True)
        self.assertFalse((self.root/'tampered').exists())
        self.capture.unlink()
        with self.assertRaises(replay.ReplayUnavailable):
            self.run_workflow(self.annual,self.shipments,self.ledger,self.artifacts,self.root/'missing-capture',
                              first_year=2016,last_year=2016,strict_replay=True)
        self.assertFalse((self.root/'missing-capture').exists())

    def test_excluded_earlier_outcome_cannot_pass_strict_cohort_audit(self):
        records=[]; annual=[]
        def add(series,y,value,observed,available,**extra):
            records.append(dict(series=series,period=str(y),value=value,
                observation_date=observed,source_url='https://example.org/source',source_version='v1',
                available_at=available,retrieved_at=available,evidence_type='captured',
                sha256=self.capture_sha,capture_reference=self.capture.name,**extra))
        for y in range(2008,2018):
            add('q2_sales',y,1000,f'{y}-06-30',f'{y}-08-01')
            if 2009<=y<=2015:
                add('q2_phosphate_sales',y,100,f'{y}-06-30',f'{y}-08-01')
        for y in range(2009,2018):
            adjusted=2010<=y<=2015; forecast=20+y-2009
            annual.append(dict(year=y,as_of=f'{y}-05-31',
                target_filed='2025-08-01' if y==2013 else f'{y}-08-01',
                target_q2_usd=900 if adjusted else 1000,
                prior_q2_usd=900 if adjusted else 1000,price_ratio=1,forecast_mean_p12_pct=forecast))
            for series,value,observed in [
                ('nitrogen_price_current_april',1,f'{y}-04-01'),
                ('nitrogen_price_previous_april',1,f'{y-1}-04-01')]:
                add(series,y,value,observed,f'{y}-05-20')
            add('nitrogen_price_ratio',y,1,f'{y}-04-01',f'{y}-05-20',derivation='ratio',
                constituent_keys=[['nitrogen_price_current_april',str(y)],['nitrogen_price_previous_april',str(y)]])
            records.extend(dict(r, sha256=self.capture_sha, capture_reference=self.capture.name)
                           for r in synthetic_forecast_recipe(y,value=forecast))
        for series,observed in [('q1_shipments_current','2017-03-31'),
                                ('q1_shipments_previous','2016-03-31')]:
            add(series,2017,1,observed,'2017-05-01')
        add('q1_shipment_ratio',2017,1,'2017-03-31','2017-05-01',derivation='ratio',
            constituent_keys=[['q1_shipments_current','2017'],['q1_shipments_previous','2017']])
        pd.DataFrame(annual).to_csv(self.annual,index=False)
        volumes=json.loads(self.shipments.read_text())
        volume=next(v for v in volumes if v['year']==2017)
        volume.update(current_q1_tons=1,previous_q1_tons=1)
        self.shipments.write_text(json.dumps([volume])); self.ledger.write_text(json.dumps(records))
        with self.assertRaises(replay.ReplayUnavailable):
            self.run_workflow(*self.args(),first_year=2017,last_year=2017,strict_replay=True)
        self.assertFalse(self.out.exists())
        manifest=self.run_workflow(*self.args(),first_year=2017,last_year=2017)
        self.assertFalse(manifest['strict_replay_complete'])
        self.assertEqual(manifest['training_cohort_status_counts']['excluded_outcome_available'],1)
        self.assertEqual(manifest['training_filing_status_counts']['filing_date_mismatch'],1)
        details=json.loads((self.out/'replay'/'training_cohorts_private.json').read_text())
        row=next(r for r in details if r['row_year']==2013)
        self.assertFalse(row['included'])
        self.assertEqual(row['cohort_status'],'excluded_outcome_available')
        source=next(r for r in records if r['series']=='q2_sales' and r['period']=='2013')
        source.update(available_at='2025-08-01',retrieved_at='2025-08-01')
        predictions=workflow.comparison.compare_incremental(pd.DataFrame(annual),[volume],[2017])['predictions']
        audit=workflow._audit_training_cohorts(records,pd.DataFrame(annual),predictions,capture_root=self.root)
        late=next(r for r in audit if r['row_year']==2013)
        self.assertEqual(late['filing_status'],'eligible')
        self.assertEqual(late['cohort_status'],'eligible')
        source['evidence_type']='unverified'
        audit=workflow._audit_training_cohorts(records,pd.DataFrame(annual),predictions,capture_root=self.root)
        unknown=next(r for r in audit if r['row_year']==2013)
        self.assertEqual(unknown['filing_status'],'filing_unverified')
        self.assertEqual(unknown['cohort_status'],'excluded_outcome_unverified')

    def test_matching_old_candidate_does_not_hide_newer_selected_version(self):
        records=[dict(r, sha256=self.capture_sha, capture_reference=self.capture.name)
                 for r in synthetic_forecast_recipe(2016,value=33.0)]
        aggregate=next(r for r in records if r['series']=='forecast_probability')
        older=dict(aggregate,value=32.0,source_version='v1',available_at='2016-05-21')
        aggregate.update(source_version='v2',available_at='2016-05-22')
        self.ledger.write_text(json.dumps(records+[older]))
        with self.assertRaisesRegex(ValueError,'Eligible replay value differs'):
            self.run_workflow(*self.args())
        self.assertFalse(self.out.exists())

    def test_unrelated_future_candidate_does_not_block_earlier_cohort(self):
        later=dict(self.ledger_rows[0],period='2025',value=99.0,
                   source_version='future',observation_date='2025-05-20',available_at=None)
        self.ledger.write_text(json.dumps([self.ledger_rows[0],later]))
        result=self.run_workflow(*self.args(),first_year=2016,last_year=2016)
        self.assertEqual(result['requested_model_years'],4)

    def test_additional_notebook_artifacts_are_prefixed_and_complete(self):
        original=self.root/'original_notebook'
        (original/'iowa_corn_revenue').mkdir(parents=True)
        pd.DataFrame([{'model':'ridge_weather','stage':'holdout_2026','mae_usd':80.}]).to_csv(
            original/'iowa_corn_revenue'/'scores_private.csv',index=False)
        (original/'iowa_v2').mkdir()
        pd.DataFrame([{'model':'price_only','years':2,'mae_usd':60.}]).to_csv(
            original/'iowa_v2'/'scores_private.csv',index=False)
        result=self.run_workflow(*self.args(),first_year=2016,last_year=2016,
                                     additional_artifacts=[('original',original)])
        self.assertEqual(result['artifact_source_names'],['primary','original'])
        reg=pd.read_csv(self.out/'register'/'experiment_register_private.csv')
        self.assertIn('iowa_corn_revenue',set(reg.study))
        self.assertEqual(set(reg.artifact_source),{'primary','original'})
        self.assertEqual(reg.experiment_id.nunique(),len(reg))
        self.assertTrue(reg.result_source.str.startswith(('primary/','original/')).all())
        self.assertEqual(len(reg[reg.result_source.str.endswith('iowa_v2/scores_private.csv')]),2)

    def test_duplicate_names_and_overlapping_artifact_roots_fail(self):
        other=self.root/'other'; other.mkdir()
        with self.assertRaisesRegex(ValueError,'Duplicate artifact source name'):
            self.run_workflow(*self.args(),additional_artifacts=[('primary',other)])
        with self.assertRaisesRegex(ValueError,'must not overlap'):
            self.run_workflow(*self.args(),additional_artifacts=[('same',self.artifacts/'iowa_v2')])
        with self.assertRaisesRegex(ValueError,'must not overlap'):
            self.run_workflow(*self.args(),additional_artifacts=[('same',self.artifacts)])
        self.assertFalse(self.out.exists())

    def test_prevent_output_in_git_worktree_and_prevent_overwriting(self):
        checkout=self.root/'checkout'; (checkout/'.git').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'Git checkout'):
            self.run_workflow(self.annual,self.shipments,self.ledger,self.artifacts,checkout/'results')
        self.out.mkdir(); (self.out/'existing').write_text('preserve')
        with self.assertRaisesRegex(ValueError,'empty directory'):
            self.run_workflow(*self.args())
        self.assertEqual((self.out/'existing').read_text(),'preserve')


if __name__=='__main__': unittest.main()
