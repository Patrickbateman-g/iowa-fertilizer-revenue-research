import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from experiment_log import append_event, canonical, read_log, verify

class LogTests(unittest.TestCase):
    def test_prospective_year_cannot_reuse_a_prior_result_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'private.jsonl'
            proposal={'model':'m','why':'reason','inputs':['price'],
                      'seen_years':[],'stage':'prospective',
                      'planned_tests':[{'year':2027,'cutoff_utc':'2027-05-31T00:00:00Z'}]}
            result={'status':'error','test_years':[2027],
                    'metrics':{'diagnostic':'unavailable'},'artifact':'private.csv'}
            append_event(path,'proposal','first',proposal,'2026-01-01T00:00:00Z')
            append_event(path,'result','first',result,'2027-08-01T00:00:00Z')
            before=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'precede every planned decision cutoff'):
                append_event(path,'proposal','second',proposal,'2027-08-02T00:00:00Z')
            self.assertEqual(path.read_bytes(),before)
            append_event(path,'proposal','third',{k:v for k,v in dict(proposal,stage='retrospective').items() if k!='planned_tests'},
                         '2027-08-03T00:00:00Z')
            append_event(path,'result','third',result,'2027-08-03T00:00:00Z')
            self.assertEqual(verify(read_log(path))['events'],4)

    def test_proposals_before_any_result_can_share_the_same_future_test(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'private.jsonl'
            proposal={'model':'m','why':'reason','inputs':['price'],
                      'seen_years':[],'stage':'prospective',
                      'planned_tests':[{'year':2027,'cutoff_utc':'2027-05-31T00:00:00Z'}]}
            result={'status':'error','test_years':[2027],
                    'metrics':{'error':1},'artifact':'private.csv'}
            append_event(path,'proposal','first',proposal,'2026-01-01T00:00:00Z')
            append_event(path,'proposal','second',proposal,'2026-01-02T00:00:00Z')
            append_event(path,'result','first',result,'2027-08-01T00:00:00Z')
            append_event(path,'result','second',result,'2027-08-02T00:00:00Z')
            self.assertEqual(verify(read_log(path))['events'],4)

    def test_whole_chain_timestamps_cannot_go_backwards(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'private.jsonl'
            proposal={'model':'m','why':'reason','inputs':['price'],
                      'seen_years':[],'stage':'prospective',
                      'planned_tests':[{'year':2027,'cutoff_utc':'2027-05-31T00:00:00Z'}]}
            result={'status':'failure','test_years':[2027],
                    'metrics':{'error':1},'artifact':'private.csv'}
            append_event(path,'proposal','first',proposal,'2026-01-01T00:00:00Z')
            append_event(path,'result','first',result,'2027-08-01T02:00:00+02:00')
            before=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'precedes previous event'):
                append_event(path,'proposal','second',proposal,'2026-05-01T00:00:00Z')
            self.assertEqual(path.read_bytes(),before)
            next_proposal=dict(proposal,planned_tests=[{'year':2028,'cutoff_utc':'2028-05-31T00:00:00Z'}])
            append_event(path,'proposal','second',next_proposal,'2027-07-31T20:00:00-04:00')
            self.assertEqual(verify(read_log(path))['events'],3)
            changed=read_log(path)
            changed[2]['recorded_utc']='2027-07-31T23:59:59Z'
            body={k:v for k,v in changed[2].items() if k!='sha256'}
            changed[2]['sha256']=hashlib.sha256(canonical(body).encode('utf8')).hexdigest()
            with self.assertRaisesRegex(ValueError,'precedes previous event'):
                verify(changed)

    def test_result_timestamp_cannot_precede_proposal(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'private.jsonl'
            proposal={'model':'m','why':'reason','inputs':['price'],
                      'seen_years':[],'stage':'prospective',
                      'planned_tests':[{'year':2027,'cutoff_utc':'2027-05-31T00:00:00Z'}]}
            result={'status':'error','test_years':[2027],
                    'metrics':{'error':1},'artifact':'private.csv'}
            append_event(path,'proposal','m',proposal,'2026-01-02T02:00:00+02:00')
            with self.assertRaisesRegex(ValueError,'precedes proposal'):
                append_event(path,'result','m',result,'2026-01-01T23:59:59Z')
            self.assertEqual(len(read_log(path)),1)
            append_event(path,'result','m',result,'2026-01-02T00:00:00Z')
            self.assertEqual(verify(read_log(path))['events'],2)

    def test_append_validation_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'private.jsonl'
            proposal={'model':'price+weather','why':'test weather','inputs':['price','weather'],
                      'seen_years':[2022,2023],'stage':'retrospective'}
            result={'status':'failure','test_years':[2022,2023],
                    'metrics':{'mae_usd':500.0},'artifact':'predictions_private.csv'}
            with self.assertRaisesRegex(ValueError,'no registered proposal'):
                append_event(path,'result','a',result)
            append_event(path,'proposal','a',proposal)
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                append_event(path,'proposal','a',proposal)
            append_event(path,'result','a',result)
            events=read_log(path)
            self.assertEqual(verify(events)['events'],2)
            events[1]['payload']['metrics']['mae_usd']=1.0
            with self.assertRaisesRegex(ValueError,'hash mismatch'): verify(events)
    def test_prospective_cannot_relabel_a_seen_year(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'l'
            proposal={'model':'m','why':'reason','inputs':['price'],'seen_years':[2026],'stage':'prospective',
                      'planned_tests':[{'year':2026,'cutoff_utc':'2026-05-31T00:00:00Z'}]}
            append_event(path,'proposal','m',proposal,'2025-12-01T00:00:00Z')
            result={'status':'success','test_years':[2026],'metrics':{'error':1},'artifact':'private.csv'}
            with self.assertRaisesRegex(ValueError,'seen years'): append_event(path,'result','m',result)
            self.assertEqual(len(read_log(path)),1)

    def test_planned_cutoff_cannot_be_retroactive_or_unspecified(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'private.jsonl'
            base={'model':'m','why':'reason','inputs':['price'],'seen_years':[],'stage':'prospective'}
            with self.assertRaisesRegex(ValueError,'require planned test cutoffs'):
                append_event(path,'proposal','missing',base,'2026-01-01T00:00:00Z')
            for raw in ('2027-05-31','2027-05-31T00:00:00'):
                with self.subTest(raw=raw), self.assertRaisesRegex(ValueError,'timezone'):
                    append_event(path,'proposal','invalid',dict(base,planned_tests=[{'year':2027,'cutoff_utc':raw}]),'2026-01-01T00:00:00Z')
            for invalid in ([{'year':2027,'cutoff_utc':'2027-06-01T00:00:00Z'}],
                            [{'year':2027,'cutoff_utc':'2028-01-01T00:00:00Z'}],
                            [{'year':2027,'cutoff_utc':'2027-05-31T00:00:00Z'}]*2):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    append_event(path,'proposal','invalid',dict(base,planned_tests=invalid),'2026-01-01T00:00:00Z')
            proposal=dict(base,planned_tests=[{'year':2027,'cutoff_utc':'2027-05-31T00:00:00Z'}])
            for when in ('2027-05-31T00:00:00Z','2027-06-01T00:00:00Z'):
                with self.subTest(when=when), self.assertRaisesRegex(ValueError,'precede every'):
                    append_event(path,'proposal','late',proposal,when)
            self.assertFalse(path.exists())
            append_event(path,'proposal','valid',proposal,'2027-05-30T23:59:59+00:00')
            too_early={'status':'success','test_years':[2027],'metrics':{'error':1},'artifact':'private.csv'}
            with self.assertRaisesRegex(ValueError,'cannot precede quarter end'):
                append_event(path,'result','valid',too_early,'2027-06-30T23:59:59Z')
            result={'status':'success','test_years':[2028],'metrics':{'error':1},'artifact':'private.csv'}
            with self.assertRaisesRegex(ValueError,'unplanned test years'):
                append_event(path,'result','valid',result,'2028-08-01T00:00:00Z')
            self.assertEqual(len(read_log(path)),1)
            tampered=read_log(path)
            tampered[0]['recorded_utc']='2027-06-01T00:00:00Z'
            body={k:v for k,v in tampered[0].items() if k!='sha256'}
            tampered[0]['sha256']=hashlib.sha256(canonical(body).encode('utf8')).hexdigest()
            with self.assertRaisesRegex(ValueError,'precede every'):
                verify(tampered)

    def test_legacy_chain_is_readable_but_cannot_gain_new_prospective_results(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'private.jsonl'
            payload={'model':'m','why':'reason','inputs':['price'],'seen_years':[],'stage':'prospective'}
            base={'sequence':1,'previous_sha256':'0'*64,'recorded_utc':'2026-01-01T00:00:00Z',
                  'kind':'proposal','experiment_id':'legacy','payload':payload}
            event=dict(base,sha256=hashlib.sha256(canonical(base).encode('utf8')).hexdigest())
            path.write_text(canonical(event)+'\n',encoding='utf8')
            self.assertEqual(verify(read_log(path))['legacy_prospective_without_cutoffs'],1)
            before=path.read_bytes()
            result={'status':'success','test_years':[2027],'metrics':{'error':1},'artifact':'private.csv'}
            with self.assertRaisesRegex(ValueError,'legacy proposal without planned cutoffs'):
                append_event(path,'result','legacy',result,'2027-08-01T00:00:00Z')
            self.assertEqual(path.read_bytes(),before)

    def test_anchor_rejects_legacy_prospective_events_in_suffix(self):
        def event(n, previous, kind, ident, payload, when):
            base={'sequence':n,'previous_sha256':previous,'recorded_utc':when,
                  'kind':kind,'experiment_id':ident,'payload':payload}
            return dict(base,sha256=hashlib.sha256(canonical(base).encode('utf8')).hexdigest())
        legacy={'model':'old','why':'reason','inputs':['price'],'seen_years':[],
                'stage':'prospective'}
        first=event(1,'0'*64,'proposal','old',legacy,'2026-01-01T00:00:00Z')
        anchor=verify([first])
        result={'status':'success','test_years':[2027],'metrics':{'error':1},'artifact':'private.csv'}
        suffix_result=event(2,first['sha256'],'result','old',result,'2027-08-01T00:00:00Z')
        self.assertEqual(verify([first,suffix_result])['legacy_prospective_without_cutoffs'],1)
        with self.assertRaisesRegex(ValueError,'Legacy prospective result in post-anchor'):
            verify([first,suffix_result],anchor)
        suffix_proposal=event(2,first['sha256'],'proposal','new_legacy',legacy,'2026-01-02T00:00:00Z')
        with self.assertRaisesRegex(ValueError,'Legacy prospective proposal in post-anchor'):
            verify([first,suffix_proposal],anchor)
        new=dict(legacy,planned_tests=[{'year':2027,'cutoff_utc':'2027-05-31T00:00:00Z'}])
        modern=event(2,first['sha256'],'proposal','modern',new,'2026-01-02T00:00:00Z')
        self.assertEqual(verify([first,modern],anchor)['events'],2)

    def test_reject_invalid_or_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'l'
            with self.assertRaises(ValueError):
                append_event(path,'proposal','x',{'model':'x'})
            self.assertFalse(path.exists())
            with self.assertRaises(ValueError):
                append_event(path,'proposal','x',{'model':'x','why':'x','inputs':['a'],'seen_years':[True],'stage':'prospective'})

    def test_external_anchor_detects_rechaining_and_truncation_but_allows_append(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'private.jsonl'
            p={'model':'m','why':'original reason','inputs':['price'],
               'seen_years':[2026],'stage':'retrospective'}
            r={'status':'failure','test_years':[2026],'metrics':{'error':2},'artifact':'result.csv'}
            append_event(path,'proposal','a',p)
            append_event(path,'result','a',r)
            saved=read_log(path)
            anchor=verify(saved)
            append_event(path,'proposal','b',dict(p,model='another'))
            self.assertEqual(verify(read_log(path),anchor)['events'],3)
            with self.assertRaisesRegex(ValueError,'truncated'):
                verify(saved[:1],anchor)
            rewritten=copy.deepcopy(saved)
            rewritten[0]['payload']['why']='changed reason'
            head='0'*64
            for event in rewritten:
                event['previous_sha256']=head
                body={k:v for k,v in event.items() if k!='sha256'}
                head=hashlib.sha256(canonical(body).encode('utf8')).hexdigest()
                event['sha256']=head
            self.assertEqual(verify(rewritten)['events'],2)
            with self.assertRaisesRegex(ValueError,'anchor does not match'):
                verify(rewritten,anchor)

    def test_reject_malformed_anchor(self):
        for anchor in ({}, {'events':True,'head':'0'*64},
                       {'events':0,'head':'0'*64}, {'events':1,'head':'bad'}):
            with self.subTest(anchor=anchor), self.assertRaisesRegex(ValueError,'Invalid trusted anchor'):
                verify([],anchor)

    def test_cli_checks_a_separately_saved_anchor(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            log=folder/'private.jsonl'
            anchor=folder/'anchor.json'
            p={'model':'m','why':'reason','inputs':['price'],
               'seen_years':[2026],'stage':'retrospective'}
            append_event(log,'proposal','a',p)
            anchor.write_text(json.dumps(verify(read_log(log))),encoding='utf8')
            command=[sys.executable,str(Path(__file__).with_name('experiment_log.py')),
                     '--log',str(log),'--kind','verify','--anchor',str(anchor)]
            good=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(good.returncode,0,good.stderr)
            event=read_log(log)[0]
            event['payload']['why']='edited'
            body={k:v for k,v in event.items() if k!='sha256'}
            event['sha256']=hashlib.sha256(canonical(body).encode('utf8')).hexdigest()
            log.write_text(canonical(event)+'\n',encoding='utf8')
            bad=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0)
            self.assertIn('Trusted anchor does not match',bad.stderr)

if __name__=='__main__': unittest.main()
