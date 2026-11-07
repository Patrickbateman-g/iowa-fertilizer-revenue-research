import datetime as dt
import hashlib
import tempfile
from pathlib import Path
import protocol_2027 as p

def disc(value, start, end, published, units='USD'):
    return dict(value=value, units=units, period_start=start, period_end=end,
                published=published, source='https://www.sec.gov/synthetic')

def fails(fn, fragment):
    try:
        fn()
    except (ValueError, FileExistsError) as e:
        assert fragment.lower() in str(e).lower(), (fragment, e)
    else:
        raise AssertionError('Expected failure: ' + fragment)

now = dt.datetime(2027, 5, 31, 20, 0, tzinfo=p.UTC)
price = dict(available=True, series='WPU065201', vintage='2027-05-31',
    current_observation='2027-04-01', previous_observation='2026-04-01',
    current_value=120, previous_value=100, current_first_published='2027-05-15',
    source='https://alfred.stlouisfed.org/series?seid=WPU065201', captured_at='2027-05-31T19:00:00Z')
inputs = dict(prior_q2=disc(100, '2026-04-01', '2026-06-30', '2026-08-01'),
    price=price, q1_volume=dict(available=True,
      current=disc(144, '2027-01-01', '2027-03-31', '2027-05-01', 'thousand product tons'),
      previous=disc(100, '2026-01-01', '2026-03-31', '2026-05-01', 'thousand product tons')))
r = p._make_prediction(inputs, now)
assert r['baseline_prediction_usd'] == 100 and r['main_prediction_usd'] == 120
assert abs(r['sensitivity_prediction_usd'] - 144) < 1e-10
fails(lambda: p._make_prediction(inputs, now + dt.timedelta(days=1)), 'only on')
fails(lambda: p._make_prediction(inputs, now - dt.timedelta(days=1)), 'only on')
fails(lambda: p._make_prediction(dict(inputs, price=dict(price, captured_at='2027-05-31T21:00:00Z')), now), 'later than')
fails(lambda: p._make_prediction(dict(inputs, price=dict(price, vintage='2027-06-01')), now), 'may 31')
fails(lambda: p._make_prediction(dict(inputs, actual_q2=200), now), 'unexpected')
badvolume = dict(inputs['q1_volume'], current=dict(inputs['q1_volume']['current'], published='2027-06-01'))
fails(lambda: p._make_prediction(dict(inputs, q1_volume=badvolume), now), 'publication')
missing = dict(available=False, reason='The value was not published yet',
               checked_at='2027-05-31T19:00:00Z', source='https://alfred.stlouisfed.org/series?seid=WPU065201')
fallback = p._make_prediction(dict(inputs, price=missing), now)
assert fallback['main_prediction_usd'] == 100 and fallback['sensitivity_prediction_usd'] is None
no_volume = p._make_prediction(dict(inputs, q1_volume={'available': False, 'reason': 'Quarterly figures not available'}), now)
assert no_volume['sensitivity_prediction_usd'] is None
advances = dict(value=42, units='USD', balance_date='2027-03-31', published='2027-05-02', source='https://www.sec.gov/synthetic')
with_advances = p._make_prediction(dict(inputs, q1_customer_advances=advances), now)
assert with_advances['context_customer_advances_usd'] == 42
assert with_advances['main_prediction_usd'] == r['main_prediction_usd']
assert with_advances['sensitivity_prediction_usd'] == r['sensitivity_prediction_usd']
fails(lambda: p._make_prediction(dict(inputs, q1_customer_advances=dict(advances, published='2027-06-01')), now), 'cutoff')
without_advances = p._make_prediction(dict(inputs, q1_customer_advances={'available': False, 'reason': 'CF did not disclose this separately'}), now)
assert without_advances['context_customer_advances_usd'] is None
assert without_advances['main_prediction_usd'] == r['main_prediction_usd']
envelope = dict(prediction=r, sha256=hashlib.sha256(p._canonical(r)).hexdigest())
actual = disc(150, '2027-04-01', '2027-06-30', '2027-08-01')
scores = p._score_prediction(envelope, actual, dt.datetime(2027, 8, 2, tzinfo=p.UTC))
assert [round(x['absolute_error_usd']) for x in scores['results']] == [50, 30, 6]
fails(lambda: p._score_prediction(envelope, dict(actual, published='2027-06-01'), dt.datetime(2027, 8, 2, tzinfo=p.UTC)), 'publication')
fails(lambda: p._score_prediction(envelope, actual, dt.datetime(2027, 7, 2, tzinfo=p.UTC)), 'publication')
tampered = dict(envelope, prediction=dict(r, main_prediction_usd=1))
fails(lambda: p._score_prediction(tampered, actual, dt.datetime(2027, 8, 2, tzinfo=p.UTC)), 'digest')
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / 'prediction.json'
    p._exclusive_write(path, envelope)
    fails(lambda: p._exclusive_write(path, envelope), 'exist')
print('Synthetic 2027 Q2 protocol checks passed')
