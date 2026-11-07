import unittest
import numpy as np
import pandas as pd
from comparison import compare_incremental, validate_annual, validate_shipments


def inputs():
    data = pd.DataFrame([dict(year=y, as_of=f'{y}-05-31', target_filed=f'{y}-08-01',
        target_q2_usd=1000+40*(y-2010), prior_q2_usd=900+20*(y-2010),
        price_ratio=1+.01*(y-2010), forecast_mean_p12_pct=20+2*(y-2010))
        for y in range(2010,2021)])
    volumes = [dict(year=y, current_q1_tons=100+y-2010, previous_q1_tons=100,
        units='thousand product tons', current_filed=f'{y}-05-01',
        previous_filed=f'{y-1}-05-01',
        current_source='https://www.sec.gov/Archives/edgar/data/1324404/current.htm',
        previous_source='https://www.sec.gov/Archives/edgar/data/1324404/previous.htm')
        for y in range(2016,2021)]
    return data, volumes


class ComparisonTests(unittest.TestCase):
    def test_same_years_cutoffs_folds_and_factors(self):
        data, volumes = inputs()
        result=compare_incremental(data, volumes, [2018,2019])
        pred=result['predictions']
        self.assertEqual(len(pred),8)
        for y,g in pred.groupby('year'):
            self.assertEqual(g.training_year_list.nunique(),1)
            self.assertEqual(g.actual_usd.nunique(),1)
            self.assertEqual(g.decision_date.nunique(),1)
            p=g.set_index('model').prediction_usd
            factor=((100+y-2010)/100)**.5
            self.assertAlmostEqual(p.price_shipments,p.price_only*factor)
            self.assertAlmostEqual(p.price_weather_shipments,p.price_weather*factor)
            self.assertTrue(all(int(z)<y for z in g.iloc[0].training_year_list.split(',')))

    def test_future_target_independence_and_filing(self):
        data, vols=inputs()
        data.loc[data.year.eq(2016),'target_filed']='2019-06-01'
        a=compare_incremental(data,vols,[2019])['predictions'].set_index('model')
        self.assertNotIn('2016',a.iloc[0].training_year_list.split(','))
        data.loc[data.year.ge(2019),'target_q2_usd']*=10
        b=compare_incremental(data,vols,[2019])['predictions'].set_index('model')
        pd.testing.assert_series_equal(a.prediction_usd,b.prediction_usd)

    def test_missing_duplicate_and_late_shipments(self):
        data, vols=inputs()
        with self.assertRaisesRegex(ValueError,'Missing matched'):
            compare_incremental(data,vols,[2021])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            validate_shipments(vols+[vols[0]])
        late=[dict(x) for x in vols]
        late[0]['current_filed']='2016-06-01'
        with self.assertRaisesRegex(ValueError,'after cutoff'):
            validate_shipments(late)
        with self.assertRaisesRegex(ValueError,'Duplicate annual'):
            validate_annual(pd.concat([data,data.iloc[[0]]]))
        data.loc[0,'forecast_mean_p12_pct']=np.nan
        with self.assertRaisesRegex(ValueError,'non-finite'):
            validate_annual(data)

    def test_reference_reproduction_rejects_wrong_value(self):
        data, vols=inputs()
        p=compare_incremental(data,vols,[2018])['predictions']
        ref=p[p.model.eq('price_only')][['year','prediction_usd','actual_usd']].copy()
        compare_incremental(data,vols,[2018],price_reference=ref)
        ref.loc[:,'prediction_usd']*=1.1
        with self.assertRaisesRegex(AssertionError,'does not reproduce'):
            compare_incremental(data,vols,[2018],price_reference=ref)


if __name__ == '__main__':
    unittest.main()
