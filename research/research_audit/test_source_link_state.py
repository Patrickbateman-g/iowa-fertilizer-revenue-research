from copy import deepcopy
import unittest
from source_link_state import checkpoint, prepare, require_same_links


class SourceLinkStateTests(unittest.TestCase):
    def setUp(self):
        self.base = [{'series': 'price', 'period': '2020', 'value': 10,
                      'evidence_type': 'unverified', 'available_at': None}]
        self.linked = deepcopy(self.base)
        self.linked[0]['bls_comparison'] = {'capture_sha256': 'a' * 64}

    def test_first_run_and_repeat_return_detached_unchanged_base(self):
        original = deepcopy(self.base)
        first = prepare(self.base)
        state = checkpoint(first, self.linked)
        self.assertEqual(prepare(self.linked, state), original)
        self.assertEqual(prepare(self.base, state), original)
        require_same_links(self.linked, state)
        altered = deepcopy(self.linked)
        altered[0]['bls_comparison']['capture_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'links differ'):
            require_same_links(altered, state)
        self.assertEqual(self.base, original)
        first[0]['value'] = -1
        self.assertEqual(prepare(self.linked, state), original)

    def test_changed_base_and_annotation_are_rejected(self):
        state = checkpoint(self.base, self.linked)
        for changed in (deepcopy(self.base), deepcopy(self.linked)):
            changed[0]['value'] += 1
            with self.assertRaisesRegex(ValueError, 'ledger changed'):
                prepare(changed, state)
        changed = deepcopy(self.linked)
        changed[0]['bls_comparison']['capture_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'ledger changed'):
            prepare(changed, state)
        state['base'][0]['value'] = 12
        with self.assertRaisesRegex(ValueError, 'base changed'):
            prepare(self.linked, state)

    def test_orphaned_or_partial_links_are_never_silently_removed(self):
        with self.assertRaisesRegex(ValueError, 'no checkpoint'):
            prepare(self.linked)
        bad = deepcopy(self.linked)
        bad[0]['value'] += 1
        with self.assertRaisesRegex(ValueError, 'changed the base'):
            checkpoint(self.base, bad)
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            prepare(self.base, {'base': self.base})

    def test_rerun_reparses_a_capture_and_rejects_resealed_changed_source(self):
        from test_bls_capture import example
        from bls_capture import capture_digest, compare_captures_to_ledger
        base = [{'series': 'nitrogen_price_current_april', 'period': '2014',
                 'value': 345.3, 'observation_date': '2014-04-01',
                 'evidence_type': 'unverified'}]
        capture = {**example(), 'decision_year': 2014, 'role': 'current'}
        first = compare_captures_to_ledger(prepare(base), [capture], [(2014, 'current')])
        state = checkpoint(base, first)
        repeat = compare_captures_to_ledger(prepare(first, state), [capture], [(2014, 'current')])
        require_same_links(repeat, state)
        self.assertEqual(first, repeat)
        changed = deepcopy(capture)
        changed['retrieved_at'] = '2026-11-06T12:00:00Z'
        changed['capture_sha256'] = capture_digest(changed)
        different = compare_captures_to_ledger(prepare(first, state), [changed], [(2014, 'current')])
        with self.assertRaisesRegex(ValueError, 'links differ'):
            require_same_links(different, state)

    def test_shipment_links_rerun_only_from_clean_checkpoint(self):
        from test_shipment_capture import capture, record
        from shipment_capture import capture_digest, compare_captures_to_ledger
        c = capture()
        base = [record(c), record(c, 'previous', 2021)]
        keys = [('q1_shipments_current', 2020), ('q1_shipments_previous', 2021)]
        first = compare_captures_to_ledger(prepare(base), [c], keys)
        state = checkpoint(base, first)
        second = compare_captures_to_ledger(prepare(first, state), [c], keys)
        require_same_links(second, state)
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, 'no checkpoint'):
            prepare(first)
        changed = deepcopy(c)
        changed['retrieved_at'] = '2026-11-06T20:00:00Z'
        changed['capture_sha256'] = capture_digest(changed)
        different = compare_captures_to_ledger(prepare(first, state), [changed], keys)
        with self.assertRaisesRegex(ValueError, 'links differ'):
            require_same_links(different, state)

    def test_alfred_links_reparse_and_detect_resealed_capture(self):
        from alfred_chart import capture_digest, compare_capture_to_ledger
        from test_alfred_chart import example, ledger
        capture = example()
        base = ledger()
        first = compare_capture_to_ledger(prepare(base), capture, 2025)
        state = checkpoint(base, first)
        second = compare_capture_to_ledger(prepare(first, state), capture, 2025)
        require_same_links(second, state)
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, 'no checkpoint'):
            prepare(first)
        changed = deepcopy(capture)
        changed['retrieved_at'] = '2030-06-02T10:00:00Z'
        changed['capture_sha256'] = capture_digest(changed)
        different = compare_capture_to_ledger(prepare(first, state), changed, 2025)
        with self.assertRaisesRegex(ValueError, 'links differ'):
            require_same_links(different, state)

    def test_unmatched_alfred_batch_change_is_detected_even_without_a_new_link(self):
        from alfred_chart import (batch_digest, capture_digest, compare_captures_to_ledger,
                                  require_same_batch)
        from test_alfred_chart import example, ledger
        base = ledger()
        base[0]['value'] += 10
        base[1]['value'] += 10
        first_capture = {'year': 2025, 'capture': example()}
        first = compare_captures_to_ledger(prepare(base), [first_capture], [2025])
        self.assertEqual(first['ledger'], base)
        state = checkpoint(base, first['ledger'])
        first_digest = batch_digest([first_capture], [2025])
        changed = deepcopy(first_capture)
        changed['capture']['retrieved_at'] = '2030-06-02T10:00:00Z'
        changed['capture']['capture_sha256'] = capture_digest(changed['capture'])
        repeat = compare_captures_to_ledger(prepare(first['ledger'], state), [changed], [2025])
        require_same_links(repeat['ledger'], state)
        with self.assertRaisesRegex(ValueError, 'batch changed'):
            require_same_batch([changed], [2025], first_digest)


if __name__ == '__main__':
    unittest.main()
