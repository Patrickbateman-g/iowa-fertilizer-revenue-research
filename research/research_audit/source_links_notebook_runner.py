import json, os, subprocess, sys
from pathlib import Path
import alfred_chart, bls_capture, sec_capture, sec_index_capture, shipment_capture, source_link_state
_source_tools = Path('/kaggle/working/iowa_research_tools')
_prior_link_state = globals().get('_SOURCE_LINK_STATE')
_base_ledger = source_link_state.prepare(REPLAY_LEDGER, _prior_link_state)
_original_report = REPLAY_MODULE['project_replay_report'](REPLAY_LEDGER, range(2022, 2027))
_bls = json.loads(Path('/kaggle/working/iowa_price_audit/bls_dom_captures_private.json').read_text())
_sec = json.loads(Path('/kaggle/working/iowa_sec_audit/sec_q2_dom_captures_private.json').read_text())
_idx = json.loads(Path('/kaggle/working/iowa_sec_audit/sec_index_dom_captures_private.json').read_text())
_q1 = json.loads(Path('/kaggle/working/iowa_sec_audit/sec_q1_shipments_dom_captures_private.json').read_text())
_linked = bls_capture.compare_captures_to_ledger(_base_ledger, _bls,
             [(year, role) for year in range(2010, 2016) for role in ('current', 'previous')])
_linked = sec_capture.compare_captures_to_ledger(_linked, _sec, list(range(2016, 2027)))
_linked = sec_capture.attach_index_captures(_linked, _idx, [2017, 2018, *range(2020, 2027)])
_linked = shipment_capture.compare_captures_to_ledger(_linked, _q1,
             [('q1_shipments_' + role, year) for year in range(2016, 2027)
              for role in ('current', 'previous')])
if 'ALFRED_CAPTURES' in globals():
    if 'ALFRED_EXPECTED_YEARS' not in globals():
        raise ValueError('Declare expected ALFRED capture years')
    _alfred_captures = ALFRED_CAPTURES
    _alfred_years = ALFRED_EXPECTED_YEARS
elif 'ALFRED_CAPTURE' in globals():
    _alfred_captures = [{'year': 2025, 'capture': ALFRED_CAPTURE}]
    _alfred_years = [2025]
else:
    raise ValueError('Supply ALFRED chart evidence')
_alfred = alfred_chart.compare_captures_to_ledger(_linked, _alfred_captures, _alfred_years)
_linked = _alfred['ledger']
_alfred_digest = alfred_chart.batch_digest(_alfred_captures, _alfred_years)
if _prior_link_state is not None:
    alfred_chart.require_same_batch(_alfred_captures, _alfred_years,
                                    globals().get('_ALFRED_AUDIT_DIGEST'))
_alfred_matched = sum(d['status'] == 'matched' for d in _alfred['diagnostics'])
_counts = {key: sum(key in r for r in _linked) for key in
           ('bls_comparison', 'sec_10q_comparison', 'sec_10q_index_comparison',
            'sec_q1_shipment_comparison', 'alfred_chart_comparison')}
assert _counts == {'bls_comparison': 12, 'sec_10q_comparison': 11,
                   'sec_10q_index_comparison': 9, 'sec_q1_shipment_comparison': 22,
                   'alfred_chart_comparison': _alfred_matched}
_next_link_state = source_link_state.checkpoint(_base_ledger, _linked)
source_link_state.require_same_links(_linked, _prior_link_state)
_new_report = REPLAY_MODULE['project_replay_report'](_linked, range(2022, 2027))
assert _new_report == _original_report
_test_env = dict(os.environ, OPENBLAS_NUM_THREADS='1')
_test = subprocess.run([sys.executable, '-m', 'unittest', '-q', 'test_alfred_chart', 'test_bls_capture',
                        'test_sec_capture', 'test_sec_index_capture', 'test_shipment_capture',
                        'test_source_link_state'], cwd=_source_tools,
                       env=_test_env, text=True, capture_output=True, check=True)
REPLAY_LEDGER, REPLAY_REPORT = _linked, _new_report
_output = REPLAY_MODULE['save_private_ledger'](REPLAY_LEDGER, REPLAY_REPORT,
                                                '/kaggle/working/iowa_replay')
(_output / 'alfred_chart_captures_private.json').write_text(
    json.dumps(_alfred_captures, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
(_output / 'alfred_chart_diagnostics_private.json').write_text(
    json.dumps(_alfred['diagnostics'], indent=2, ensure_ascii=False) + '\n', encoding='utf8')
if len(_alfred_captures) == 1 and _alfred_years == [2025]:
    (_output / 'alfred_chart_capture_private.json').write_text(
        json.dumps(_alfred_captures[0]['capture'], indent=2, ensure_ascii=False) + '\n', encoding='utf8')
(_output / 'source_link_summary_private.json').write_text(json.dumps({
    'counts': _counts, 'replay_eligibility_unchanged': True,
    'alfred_expected_years': _alfred['expected_years'],
    'alfred_captured_years': _alfred['captured_years'],
    'alfred_diagnostic_counts': {status: sum(d['status'] == status for d in _alfred['diagnostics'])
                                 for status in ('matched', 'mismatch', 'capture_missing',
                                                'ledger_missing', 'ledger_duplicate')},
    'limitation': 'Later selected DOM excerpts corroborate values, links, and an archived '
                  'ALFRED chart vintage; they do not establish historical availability, '
                  'original server bytes, or absence of later archive changes.'
}, indent=2) + '\n', encoding='utf8')
_SOURCE_LINK_STATE = _next_link_state
_ALFRED_AUDIT_DIGEST = _alfred_digest
print('Source links:', _counts)
print('ALFRED chart comparison statuses:', {status: sum(d['status'] == status for d in _alfred['diagnostics'])
      for status in ('matched', 'mismatch', 'capture_missing', 'ledger_missing', 'ledger_duplicate')})
print('Replay eligibility unchanged; historical availability and archive immutability remain unverified.')
print(_test.stderr)
