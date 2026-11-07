import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from experiment_log import append_event, canonical, read_log, verify
from notebook_log_runner import prepare_prospective_log


def proposal(model):
    return {'model':model,'why':'research question','inputs':['price'],
            'seen_years':[2026],'stage':'retrospective'}


class NotebookLogRunnerTests(unittest.TestCase):
    def test_carry_forward_rejects_legacy_proposal_after_prior_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            prior=Path(d)/'prior.jsonl'; target=Path(d)/'working.jsonl'
            append_event(prior,'proposal','old',proposal('old'),'2026-01-01T00:00:00Z')
            first=read_log(prior)[0]
            legacy={'model':'legacy','why':'reason','inputs':['price'],'seen_years':[],
                    'stage':'prospective'}
            body={'sequence':2,'previous_sha256':first['sha256'],
                  'recorded_utc':'2026-01-02T00:00:00Z','kind':'proposal',
                  'experiment_id':'legacy','payload':legacy}
            second=dict(body,sha256=hashlib.sha256(canonical(body).encode('utf8')).hexdigest())
            target.write_text(prior.read_text(encoding='utf8')+canonical(second)+'\n',encoding='utf8')
            before=target.read_bytes()
            with self.assertRaisesRegex(ValueError,'Legacy prospective proposal in post-anchor'):
                prepare_prospective_log(target,prior)
            self.assertEqual(target.read_bytes(),before)

    def test_clean_reproduction_creates_no_prospective_log(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'working'/'new_experiments_private.jsonl'
            result=prepare_prospective_log(target)
            self.assertEqual(result['status'],'no_prospective_log_carried')
            self.assertFalse(target.exists())
            self.assertFalse(target.parent.exists())

    def test_carries_exact_prior_bytes_and_accepts_a_valid_extension(self):
        with tempfile.TemporaryDirectory() as d:
            prior=Path(d)/'mounted'/'new_experiments_private.jsonl'
            target=Path(d)/'working'/'new_experiments_private.jsonl'
            append_event(prior,'proposal','a',proposal('a'))
            anchor=verify(read_log(prior))
            result=prepare_prospective_log(target,prior,anchor)
            self.assertEqual(result['status'],'previous_log_copied')
            self.assertTrue(result['trusted_anchor_checked'])
            self.assertEqual(target.read_bytes(),prior.read_bytes())
            append_event(target,'proposal','b',proposal('b'))
            before=target.read_bytes()
            result=prepare_prospective_log(target,prior,anchor)
            self.assertEqual(result['status'],'existing_verified')
            self.assertEqual(result['events'],2)
            self.assertEqual(target.read_bytes(),before)

    def test_missing_mount_and_bad_anchor_fail_without_creating_log(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'working'/'log.jsonl'
            missing=Path(d)/'missing.jsonl'
            with self.assertRaises(FileNotFoundError):
                prepare_prospective_log(target,missing)
            self.assertFalse(target.exists())
            prior=Path(d)/'prior.jsonl'
            append_event(prior,'proposal','a',proposal('a'))
            anchor={'events':1,'head':'f'*64}
            with self.assertRaisesRegex(ValueError,'anchor does not match'):
                prepare_prospective_log(target,prior,anchor)
            self.assertFalse(target.exists())
            with self.assertRaisesRegex(ValueError,'without an experiment log'):
                prepare_prospective_log(target,anchor=anchor)

    def test_existing_divergent_or_truncated_log_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            prior=Path(d)/'prior.jsonl'; target=Path(d)/'working.jsonl'
            append_event(prior,'proposal','a',proposal('a'))
            append_event(prior,'proposal','b',proposal('b'))
            append_event(target,'proposal','other',proposal('other'))
            before=target.read_bytes()
            with self.assertRaisesRegex(ValueError,'truncated'):
                prepare_prospective_log(target,prior)
            self.assertEqual(target.read_bytes(),before)
            append_event(target,'proposal','other2',proposal('other2'))
            before=target.read_bytes()
            with self.assertRaisesRegex(ValueError,'anchor does not match'):
                prepare_prospective_log(target,prior)
            self.assertEqual(target.read_bytes(),before)

    def test_corrupt_empty_and_partial_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            prior=Path(d)/'prior.jsonl'; target=Path(d)/'target.jsonl'
            for contents in (b'',b'{}\n',b'{not json}\n'):
                prior.write_bytes(contents)
                with self.assertRaises(ValueError):
                    prepare_prospective_log(target,prior)
                self.assertFalse(target.exists())
            prior.unlink()
            append_event(prior,'proposal','a',proposal('a'))
            prior.write_bytes(prior.read_bytes().rstrip(b'\n'))
            with self.assertRaisesRegex(ValueError,'incomplete final line'):
                prepare_prospective_log(target,prior)
            self.assertFalse(target.exists())


if __name__=='__main__': unittest.main()
