import unittest
import numpy as np
import pandas as pd
from intervals import sequential_ranges, validate_prediction_rows, summarize_range_periods

MODELS = ('price_only', 'price_weather', 'price_shipments', 'price_weather_shipments')

def sample():
    return pd.DataFrame([
        {'year': y, 'decision_date': f'{y}-05-31', 'target_filed': f'{y}-08-01',
         'model': model, 'prediction_usd': 1000 + (y-2015)*10 + 3*j,
         'actual_usd': 1000., 'training_year_list': ''}
        for y in range(2016,2024) for j,model in enumerate(MODELS)])

class RangeTests(unittest.TestCase):
    def test_finite_rank_calculated_from_earlier_errors(self):
        frame,summary=sequential_ranges(sample(),MODELS)
        row=frame[(frame.model=='price_only') & frame.year.eq(2020)].iloc[0]
        self.assertEqual(row.calibration_count,4)
        self.assertEqual(row.calibration_years,'2016,2017,2018,2019')
        self.assertEqual(row['rank'],4)
        self.assertEqual(row.absolute_error_radius_usd,40)
        self.assertEqual(row.lower_usd,1010)
        self.assertEqual(row.upper_usd,1090)
        self.assertEqual(bool(row.contains_actual),False)
        self.assertTrue(pd.isna(frame[(frame.model=='price_only') & frame.year.eq(2019)].iloc[0].lower_usd))
        self.assertEqual(int(summary[summary.model=='price_only'].iloc[0].evaluated_ranges),4)

    def test_period_summary_preserves_unavailable_denominator(self):
        frame,_=sequential_ranges(sample(),MODELS)
        summary=summarize_range_periods(frame,{'early': (2016,2019), 'late': (2020,2023)})
        early=summary[(summary.model=='price_only') & (summary.period=='early')].iloc[0]
        late=summary[(summary.model=='price_only') & (summary.period=='late')].iloc[0]
        self.assertEqual(early.evaluated_ranges,0)
        self.assertTrue(pd.isna(early.empirical_coverage))
        self.assertEqual(late.evaluated_ranges,4)
        self.assertEqual(late.covered,0)

    def test_late_and_missing_labels_are_not_used(self):
        p=sample()
        p.loc[p.year.eq(2016),'target_filed']='2021-06-01'
        p.loc[p.year.eq(2017),['target_filed','actual_usd']]=[None,np.nan]
        out,_=sequential_ranges(p,MODELS)
        row=out[(out.year==2021) & (out.model=='price_only')].iloc[0]
        self.assertEqual(row.calibration_years,'2018,2019,2020')
        self.assertTrue(pd.isna(row.lower_usd))
        row=out[(out.year==2022) & (out.model=='price_only')].iloc[0]
        self.assertEqual(row.calibration_years,'2016,2018,2019,2020,2021')

    def test_current_and_future_outcomes_cannot_change_earlier_ranges(self):
        p=sample()
        a,_=sequential_ranges(p,MODELS)
        p.loc[p.year.ge(2020),'actual_usd']=500.
        b,_=sequential_ranges(p,MODELS)
        a=a[a.year.le(2020)].reset_index(drop=True)
        b=b[b.year.le(2020)].reset_index(drop=True)
        pd.testing.assert_series_equal(a.lower_usd,b.lower_usd)
        pd.testing.assert_series_equal(a.upper_usd,b.upper_usd)

    def test_matched_and_valid_folds(self):
        p=sample()
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            validate_prediction_rows(pd.concat([p,p.iloc[[0]]]),MODELS)
        with self.assertRaisesRegex(ValueError,'Unmatched'):
            validate_prediction_rows(p.drop(index=0),MODELS)
        p=sample(); p.loc[0,'target_filed']='2016-06-15'
        with self.assertRaisesRegex(ValueError,'quarter end'):
            validate_prediction_rows(p,MODELS)
        p=sample(); p.loc[p.year.eq(2020),'training_year_list']='2016,2020'
        with self.assertRaisesRegex(ValueError,'precede'):
            validate_prediction_rows(p,MODELS)
        p=sample(); p.loc[p.year.eq(2016),'target_filed']='2020-06-01'; p.loc[p.year.eq(2020),'training_year_list']='2016'
        with self.assertRaisesRegex(ValueError,'not filed'):
            validate_prediction_rows(p,MODELS)

    def test_inconsistent_or_missing_values_fail_closed(self):
        p=sample(); p.loc[0,'actual_usd']=1234
        with self.assertRaisesRegex(ValueError,'Targets differ'):
            validate_prediction_rows(p,MODELS)
        p=sample(); p.loc[0,'prediction_usd']=np.nan
        with self.assertRaisesRegex(ValueError,'Invalid prediction'):
            validate_prediction_rows(p,MODELS)
        p=sample(); p.loc[0,'target_filed']=None
        with self.assertRaisesRegex(ValueError,'target_filed'):
            validate_prediction_rows(p,MODELS)

if __name__=='__main__': unittest.main()
