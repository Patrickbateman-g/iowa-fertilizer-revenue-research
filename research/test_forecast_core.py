import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from forecast_core import (daily_forecast_status, fit_no_intercept,
                           fit_intercept_sensitivity, prepare_rows, walk_forward)


def rows():
    return pd.DataFrame([
        dict(year=y, as_of=f'{y}-05-31', target_filed=f'{y}-08-01',
             target_q2_usd=1000 + 35*(y-2010), prior_q2_usd=900 + 20*(y-2010),
             price_ratio=1 + .01*(y-2010), forecast_mean_p12_pct=20 + 2*(y-2010),
             prior_filed=f'{y-1}-08-01', price_available=f'{y}-05-15',
             forecast_available=f'{y}-05-21') for y in range(2010, 2019)
    ])


def weather(value=25):
    return dict(station='KAMW', model='GFS', runtime='2017-04-01T00:00:00Z',
                ftime='2017-04-02T00:00:00Z', p12=value)


class CoreTests(unittest.TestCase):
    def test_release_dates_and_target_exclusion(self):
        data = rows()
        data.loc[data.year.eq(2016), 'target_filed'] = '2018-06-01'
        pred, _ = walk_forward(data, [2018])
        self.assertNotIn('2016', pred.iloc[0].training_years.split(','))
        data.loc[data.year.eq(2018), 'price_available'] = '2018-06-01'
        with self.assertRaisesRegex(ValueError, 'later than cutoff'):
            prepare_rows(data)

    def test_missing_and_duplicate_annual_rows(self):
        data = rows()
        with self.assertRaisesRegex(ValueError, 'Duplicate year'):
            prepare_rows(pd.concat([data, data.iloc[[0]]]))
        data.loc[0, 'forecast_mean_p12_pct'] = np.nan
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            prepare_rows(data)
        with self.assertRaisesRegex(ValueError, 'Missing test year'):
            walk_forward(rows(), [2019])

    def test_daily_duplicates_conflicts_and_absences(self):
        d = daily_forecast_status([weather(), weather()], 'KAMW', 2017)
        self.assertEqual(d.iloc[0].state, 'valid')
        self.assertEqual(d.iloc[0].selected_rows, 2)
        self.assertEqual(d.iloc[0].distinct_values, 1)
        self.assertEqual(d.iloc[1].state, 'absent_in_selected_ledger')
        d = daily_forecast_status([weather(), weather(30)], 'KAMW', 2017)
        self.assertEqual(d.iloc[0].state, 'conflicting_values')
        self.assertTrue(pd.isna(d.iloc[0].p12_pct))
        d = daily_forecast_status([weather(None)], 'KAMW', 2017)
        self.assertEqual(d.iloc[0].state, 'null_only')

    def test_matched_years_and_reference_formula(self):
        data = rows()
        pred, score = walk_forward(data, [2017, 2018])
        self.assertEqual(set(pred.year), {2017, 2018})
        self.assertEqual(set(score.years), {2})
        for _, group in pred.groupby('year'):
            self.assertEqual(group.training_years.nunique(), 1)
        train = data[data.year < 2018]
        row = SimpleNamespace(**data[data.year.eq(2018)].iloc[0].to_dict())
        x = train.forecast_mean_p12_pct.to_numpy()
        z = (x - x.mean()) / x.std()
        y = np.log(train.target_q2_usd / (train.prior_q2_usd * train.price_ratio)).to_numpy()
        expected = row.prior_q2_usd * row.price_ratio * np.exp(((row.forecast_mean_p12_pct - x.mean()) / x.std()) * (z @ y) / (z @ z + 1))
        self.assertAlmostEqual(fit_no_intercept(train, row), expected)

    def test_intercept_strength_and_future_target_isolation(self):
        data = rows()
        train = data[data.year < 2018]
        row = SimpleNamespace(**data[data.year.eq(2018)].iloc[0].to_dict())
        residual_mean = np.log(train.target_q2_usd / (train.prior_q2_usd * train.price_ratio)).mean()
        old = fit_no_intercept(train, row)
        self.assertAlmostEqual(fit_intercept_sensitivity(train, row, ('forecast_mean_p12_pct',), 0), old)
        self.assertAlmostEqual(fit_intercept_sensitivity(train, row, ('forecast_mean_p12_pct',), .5), old * np.exp(.5 * residual_mean))
        self.assertAlmostEqual(fit_intercept_sensitivity(train, row, ('forecast_mean_p12_pct',), 1), old * np.exp(residual_mean))
        base = walk_forward(data, [2017])[0].set_index('model').prediction_usd
        data.loc[data.year.ge(2017), 'target_q2_usd'] *= 10
        changed = walk_forward(data, [2017])[0].set_index('model').prediction_usd
        pd.testing.assert_series_equal(base, changed)


if __name__ == '__main__':
    unittest.main()
