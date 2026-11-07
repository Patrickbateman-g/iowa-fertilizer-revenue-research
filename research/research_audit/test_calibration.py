import unittest
import pandas as pd
from comparison import MODELS, compare_incremental
from intervals import sequential_ranges
from calibration import prepare_calibration_predictions, used_calibration_years
from test_comparison import inputs

class CalibrationTests(unittest.TestCase):
    def test_subset_reproduces_earlier_full_run(self):
        annual,ship=inputs()
        full=compare_incremental(annual,ship,range(2016,2021))['predictions']
        all_ranges,_=sequential_ranges(full,MODELS)
        before,inventory=prepare_calibration_predictions(annual,ship,2020)
        selected=compare_incremental(annual,ship,[2020])['predictions']
        subset_ranges,_=sequential_ranges(pd.concat([before,selected],ignore_index=True),MODELS)
        got=subset_ranges[subset_ranges.year.eq(2020)].reset_index(drop=True)
        expected=all_ranges[all_ranges.year.eq(2020)].reset_index(drop=True)
        pd.testing.assert_frame_equal(got,expected)
        self.assertEqual(inventory.status.tolist(),['prepared']*4)
        self.assertEqual(used_calibration_years(got),[2016,2017,2018,2019])

    def test_missing_fold_is_recorded(self):
        annual,ship=inputs()
        annual=annual[annual.year.ne(2017)]
        ship=[x for x in ship if x['year']!=2018]
        before,inventory=prepare_calibration_predictions(annual,ship,2020)
        reasons=inventory.set_index('year').reason.to_dict()
        self.assertEqual(reasons[2017],'missing_annual')
        self.assertEqual(reasons[2018],'missing_shipments')
        self.assertEqual(set(before.year),{2016,2019})

    def test_training_limit_and_zero_variation_are_recorded(self):
        annual,ship=inputs()
        annual.loc[annual.year.eq(2010),'target_filed']='2018-06-01'
        _,inventory=prepare_calibration_predictions(annual,ship,2017)
        self.assertEqual(inventory.iloc[0].reason,'insufficient_training')
        annual,ship=inputs()
        annual.loc[annual.year.lt(2016),'forecast_mean_p12_pct']=25.
        _,inventory=prepare_calibration_predictions(annual,ship,2017)
        self.assertEqual(inventory.iloc[0].reason,'no_training_weather_variation')

    def test_late_outcome_and_future_perturbations_do_not_leak(self):
        annual,ship=inputs()
        annual.loc[annual.year.eq(2018),'target_filed']='2020-06-01'
        before,_=prepare_calibration_predictions(annual,ship,2020)
        selected=compare_incremental(annual,ship,[2020])['predictions']
        ranges,_=sequential_ranges(pd.concat([before,selected],ignore_index=True),MODELS)
        subset=ranges[ranges.year.eq(2020)]
        self.assertNotIn(2018,used_calibration_years(subset))
        later=annual.copy(); later.loc[later.year.ge(2020),'target_q2_usd']*=10
        before2,_=prepare_calibration_predictions(later,ship,2020)
        selected2=compare_incremental(later,ship,[2020])['predictions']
        ranges2,_=sequential_ranges(pd.concat([before2,selected2],ignore_index=True),MODELS)
        p1=subset[['model','prediction_usd','lower_usd','upper_usd']].reset_index(drop=True)
        p2=ranges2[ranges2.year.eq(2020)][['model','prediction_usd','lower_usd','upper_usd']].reset_index(drop=True)
        pd.testing.assert_frame_equal(p1,p2)

    def test_candidate_provenance_claim_not_implicit(self):
        annual,ship=inputs()
        _,inventory=prepare_calibration_predictions(annual,ship,2017)
        self.assertEqual(inventory.iloc[0].reason,'requires_separate_replay_audit')
        with self.assertRaises(ValueError):
            used_calibration_years(pd.DataFrame([{'year':2020,'calibration_years':'2019,2020','calibration_count':2}]))
        with self.assertRaisesRegex(ValueError,'Calibration count'):
            used_calibration_years(pd.DataFrame([{'year':2020,'calibration_years':'2019','calibration_count':2}]))

if __name__=='__main__': unittest.main()
