import json
import unittest
import pandas as pd
from comparison import compare_incremental, MODELS
from test_comparison import inputs
from intervals import sequential_ranges, summarize_range_periods
from report import build_research_report


def data():
    annual, shipments=inputs()
    result=compare_incremental(annual,shipments,range(2016,2021))
    ranges,_=sequential_ranges(result['predictions'],MODELS)
    coverage=summarize_range_periods(ranges,{'2016-2021':(2016,2021),'2022-2026':(2022,2026)})
    replay=[{'decision_date':'2020-05-31','series':'weather','status':'unverified_candidates'},
            {'decision_date':'2020-05-31','series':'price','status':'eligible'}]
    scores=[{'cohort':'full','period':'2022-2026','variant':'alpha_0','mae_musd':148.715},
            {'cohort':'full','period':'2022-2026','variant':'alpha_0.5','mae_musd':188.329}]
    reg=pd.DataFrame([{'study':'iowa_intercept_research','result_kind':'reported_score',
                       'result_source':'iowa_intercept_research/scores_private.csv',
                       'result_row_json':json.dumps(x)} for x in scores])
    return result,ranges,coverage,replay,reg


class ReportTests(unittest.TestCase):
    def test_range_bounds_radius_and_coverage_are_rederived_from_prior_errors(self):
        comparison,ranges,coverage,replay,register=data()
        key=(ranges.year==2020)&ranges.model.eq('price_only')
        forged=ranges.copy()
        forged.loc[key,'lower_usd']=0.0
        forged.loc[key,'upper_usd']=1e15
        forged.loc[key,'contains_actual']=True
        forged_coverage=summarize_range_periods(forged,{'2016-2021':(2016,2021),'2022-2026':(2022,2026)})
        with self.assertRaisesRegex(ValueError,'bounds or radius'):
            build_research_report(comparison,forged,forged_coverage,replay,register)
        forged.loc[key,'absolute_error_radius_usd']=1e15
        forged.loc[key,'upper_usd']=forged.loc[key,'prediction_usd']+1e15
        with self.assertRaisesRegex(ValueError,'bounds or radius'):
            build_research_report(comparison,forged,forged_coverage,replay,register)
        for col,value in [('calibration_count',3),('rank',2),('nominal_coverage',.5),('calibration_years','2017,2018,2019')]:
            bad=ranges.copy(); bad.loc[key,col]=value
            with self.subTest(col=col), self.assertRaisesRegex(ValueError,'calibration years, count or rank'):
                build_research_report(comparison,bad,coverage,replay,register)
        for col,value in [('decision_date','2020-06-01'),('target_filed','2020-08-02')]:
            bad=ranges.copy(); bad.loc[key,col]=value
            with self.subTest(col=col), self.assertRaisesRegex(ValueError,'Ranges and predictions do not match'):
                build_research_report(comparison,bad,coverage,replay,register)

    def test_report_rederives_subset_ranges_using_explicit_earlier_folds(self):
        annual, shipments=inputs()
        full=compare_incremental(annual,shipments,range(2016,2021))
        subset=compare_incremental(annual,shipments,range(2018,2021))
        all_ranges,_=sequential_ranges(full['predictions'],MODELS)
        selected=all_ranges[all_ranges.year.ge(2018)].copy()
        coverage=summarize_range_periods(selected,{'2016-2021':(2016,2021),'2022-2026':(2022,2026)})
        _,_,_,replay,register=data()
        text=build_research_report(subset,selected,coverage,replay,register,calibration_context=True,
                                   calibration_predictions=full['predictions'])
        self.assertIn('| 2018–2020 |',text)
        with self.assertRaisesRegex(ValueError,'requires earlier prediction history'):
            build_research_report(subset,selected,coverage,replay,register,calibration_context=True)
        with self.assertRaisesRegex(ValueError,'Pre-window calibration requires'):
            build_research_report(subset,selected,coverage,replay,register,
                                  calibration_predictions=full['predictions'])
        changed=full['predictions'].copy()
        changed.loc[changed.year==2016,'actual_usd']*=2000
        with self.assertRaisesRegex(ValueError,'bounds or radius'):
            build_research_report(subset,selected,coverage,replay,register,calibration_context=True,
                                  calibration_predictions=changed)
        missing=full['predictions'][full['predictions'].year.ne(2016)]
        with self.assertRaisesRegex(ValueError,'calibration years, count or rank'):
            build_research_report(subset,selected,coverage,replay,register,calibration_context=True,
                                  calibration_predictions=missing)
        inconsistent=full['predictions'].copy()
        inconsistent.loc[(inconsistent.year==2016)&inconsistent.model.eq('price_only'),'actual_usd']*=2
        with self.assertRaisesRegex(ValueError,'matched models and outcomes'):
            build_research_report(subset,selected,coverage,replay,register,calibration_context=True,
                                  calibration_predictions=inconsistent)

    def test_report_contains_estimates_failures_uncertainty_and_provenance(self):
        text=build_research_report(*data())
        for fragment in ('2016 |', '2020 |', 'Price + both', 'unverified_candidates',
                         '188.3m', '148.7m', 'not the number of unique trials',
                         'ceil((n+1)', '80%', '2027', 'ALFRED', 'SEC filings',
                         'Iowa Environmental Mesonet', 'source artifacts'):
            self.assertIn(fragment,text)

    def test_weather_window_coverage_is_described_honestly(self):
        text=build_research_report(*data())
        self.assertIn('a window of up to 50 dates, requiring at least 40 available dates per station',text)
        self.assertNotIn('summarizes 50 forecast initialization dates',text)

    def test_generic_availability_context_does_not_clear_replay(self):
        text=build_research_report(*data())
        self.assertIn('https://forecast.weather.gov/glossary.php?word=model+output+statistics',text)
        self.assertIn('04:00 UTC for 00:00 forecast cycles',text)
        self.assertIn('16:00 UTC for 12:00 cycles',text)
        self.assertIn('https://fred.stlouisfed.org/docs/api/fred/releases_dates.html',text)
        self.assertIn('does not necessarily indicate when that data became available on FRED or ALFRED',text)
        self.assertIn('does not establish the issuance or availability time of any specific historical IEM record',text)
        self.assertIn('Neither generic description establishes the input vintages or clears any unverified input',text)
        self.assertIn('not assigned as individual publication timestamps or used to change replay statuses',text)
        self.assertIn('| 2020-05-31 | unverified_candidates | 1 |',text)

    def test_holdout_warning_and_private_log_persistence(self):
        text=build_research_report(*data())
        pos=text.index('## Recorded negative and mixed findings')
        warning=text.index('Labels such as holdout_2026',pos)
        first_finding=text.find('\n- ',pos)
        self.assertTrue(first_finding == -1 or warning < first_finding)
        self.assertIn('are no longer untouched holdouts',text)
        self.assertIn('`new_experiments_private.jsonl`',text)
        self.assertIn('carry that log forward from saved output',text)
        self.assertNotIn('$',text)
        self.assertIn('| 2016–2020 |',text)
        self.assertNotIn('through_2021',text)

    def test_coverage_labels_show_only_requested_years_and_calibration_scope(self):
        annual, shipments=inputs()
        comparison=compare_incremental(annual,shipments,range(2018,2021))
        ranges,_=sequential_ranges(comparison['predictions'],MODELS)
        coverage=summarize_range_periods(ranges,{'2016-2021':(2016,2021),'2022-2026':(2022,2026)})
        _,_,_,replay,register=data()
        text=build_research_report(comparison,ranges,coverage,replay,register)
        section=text.split('## Uncertainty and observed coverage',1)[1].split('## Recorded negative',1)[0]
        self.assertIn('| 2018–2020 | Price |',section)
        self.assertNotIn('| 2016-2021 |',section)
        self.assertNotIn('| 2022-2026 |',section)
        self.assertIn('only prior prediction rows included in the selected reporting window',section)

    def test_late_subset_coverage_has_accurate_label(self):
        annual, shipments=inputs()
        extra_annual=[]
        for y in range(2021,2027):
            row=annual.iloc[-1].to_dict()
            row.update(year=y,as_of=f'{y}-05-31',target_filed=f'{y}-08-01',
                       target_q2_usd=1000+40*(y-2010),prior_q2_usd=900+20*(y-2010),
                       price_ratio=1+.01*(y-2010),forecast_mean_p12_pct=20+2*(y-2010))
            extra_annual.append(row)
            vol=dict(shipments[-1])
            vol.update(year=y,current_filed=f'{y}-05-01',previous_filed=f'{y-1}-05-01')
            shipments.append(vol)
        annual=pd.concat([annual,pd.DataFrame(extra_annual)],ignore_index=True)
        comparison=compare_incremental(annual,shipments,[2025,2026])
        ranges,_=sequential_ranges(comparison['predictions'],MODELS)
        coverage=summarize_range_periods(ranges,{'2016-2021':(2016,2021),'2022-2026':(2022,2026)})
        _,_,_,replay,register=data()
        text=build_research_report(comparison,ranges,coverage,replay,register)
        section=text.split('## Uncertainty and observed coverage',1)[1].split('## Recorded negative',1)[0]
        self.assertIn('| 2025–2026 | Price |',section)
        self.assertNotIn('| 2022-2026 |',section)
        self.assertNotIn('| 2016-2021 |',section)

    def test_calibration_depends_on_requested_prediction_rows(self):
        comparison,_,_,_,_=data()
        full,_=sequential_ranges(comparison['predictions'],MODELS)
        selected=comparison['predictions'].query('year >= 2018')
        short,_=sequential_ranges(selected,MODELS)
        full_2020=full.query("year == 2020 and model == 'price_only'").iloc[0]
        short_2020=short.query("year == 2020 and model == 'price_only'").iloc[0]
        self.assertEqual(full_2020.calibration_count,4)
        self.assertEqual(short_2020.calibration_count,2)
        self.assertTrue(pd.notna(full_2020.lower_usd))
        self.assertTrue(pd.isna(short_2020.lower_usd))

    def test_previous_failures_use_matched_same_source_evidence(self):
        comparison,ranges,cov,replay,reg=data()
        records=[{'model':'last_year','stage':'holdout_2026','n_years':1,'mae_usd':3e6},
                 {'model':'ridge_prior','stage':'holdout_2026','n_years':1,'mae_usd':2e6},
                 {'model':'ridge_weather','stage':'holdout_2026','n_years':1,'mae_usd':7e6}]
        def entry(study, file, kind, row):
            return {'study':study,'result_kind':kind,'result_source':'original/'+study+'/'+file,
                    'result_row_json':json.dumps(row)}
        rows=[entry('iowa_corn_revenue','scores_private.csv','reported_score',r) for r in records]
        rows += [entry('iowa_corn_revenue','predictions_private.csv','derived_from_prediction_rows',
                       {'model':r['model'],'stage':r['stage'],'rows':1,'years':'2026',
                        'error_count':1,'missing_or_invalid_error_count':0,'mean_absolute_error':r['mae_usd']})
                 for r in records]
        scores=[{'model':m,'period':'2016-2021','years':1,'mae_usd':v}
                for m,v in [('base',1e6),('full',3e6),('damped',2e6)]]
        rows += [entry('iowa_v4','scores_private.csv','reported_score',r) for r in scores]
        rows.append(entry('iowa_v4','yearly_diagnostics_private.csv','diagnostic_row',
                           {'year':2016,'period':'2016-2021','actual_usd':10e6,
                            'base_prediction_usd':11e6,'base_absolute_error_usd':1e6,
                            'full_prediction_usd':13e6,'full_absolute_error_usd':3e6,
                            'damped_prediction_usd':12e6,'damped_absolute_error_usd':2e6}))
        reg=pd.concat([reg,pd.DataFrame(rows)],ignore_index=True)
        text=build_research_report(comparison,ranges,cov,replay,reg)
        self.assertIn('Original observed-weather ridge (holdout_2026; years 2026): USD 7.0m',text)
        self.assertIn('Earlier shipment investigation, matched years 2016:',text)
        self.assertIn('full USD 3.0m',text)

    def test_original_and_shipment_duplicates_cannot_be_mixed(self):
        comparison,ranges,cov,replay,reg=data()
        def entry(study,root,file,kind,row):
            return {'study':study,'result_kind':kind,'result_source':root+'/'+study+'/'+file,
                    'result_row_json':json.dumps(row)}
        scores=[{'model':m,'stage':'development','n_years':1,'mae_usd':v}
                for m,v in [('last_year',3e6),('ridge_prior',2e6),('ridge_weather',7e6)]]
        rows=[entry('iowa_corn_revenue','a','scores_private.csv','reported_score',r) for r in scores]
        rows += [entry('iowa_corn_revenue','a','predictions_private.csv','derived_from_prediction_rows',
                       {'model':r['model'],'stage':r['stage'],'rows':1,
                        'years':'2016' if r['model']=='ridge_weather' else '2017',
                        'error_count':1,'missing_or_invalid_error_count':0,'mean_absolute_error':r['mae_usd']}) for r in scores]
        rows += [entry('iowa_corn_revenue','b','scores_private.csv','reported_score',scores[0])]
        v4=[{'model':m,'period':'2016-2021','years':1,'mae_usd':v}
            for m,v in [('base',1e6),('full',3e6),('damped',2e6)]]
        rows += [entry('iowa_v4','a','scores_private.csv','reported_score',r) for r in v4]
        detail={'year':2016,'period':'2016-2021','actual_usd':10e6,
                'base_prediction_usd':11e6,'base_absolute_error_usd':1e6,
                'full_prediction_usd':13e6,'full_absolute_error_usd':3e6,
                'damped_prediction_usd':12e6,'damped_absolute_error_usd':2e6}
        rows += [entry('iowa_v4','a','yearly_diagnostics_private.csv','diagnostic_row',detail)]*2
        reg=pd.concat([reg,pd.DataFrame(rows)],ignore_index=True)
        text=build_research_report(comparison,ranges,cov,replay,reg)
        self.assertIn('lack unique, matching year-level prediction summaries',text)
        self.assertNotIn('Original observed-weather ridge (development;',text)
        self.assertIn('lack unique, matching annual diagnostic rows',text)
        self.assertNotIn('Earlier shipment investigation, matched years',text)

    def test_duplicate_original_scores_never_silently_win(self):
        comparison,ranges,cov,replay,reg=data()
        rows=[]
        for model,mae in [('last_year',3e6),('ridge_prior',2e6),('ridge_weather',7e6)]:
            rows.append({'study':'iowa_corn_revenue','result_kind':'reported_score',
                         'result_source':'a/iowa_corn_revenue/scores_private.csv',
                         'result_row_json':json.dumps({'model':model,'stage':'holdout_2026','n_years':1,'mae_usd':mae})})
            rows.append({'study':'iowa_corn_revenue','result_kind':'derived_from_prediction_rows',
                         'result_source':'a/iowa_corn_revenue/predictions_private.csv',
                         'result_row_json':json.dumps({'model':model,'stage':'holdout_2026','rows':1,'years':'2026',
                             'error_count':1,'missing_or_invalid_error_count':0,'mean_absolute_error':mae})})
        rows.append(dict(rows[0]))
        reg=pd.concat([reg,pd.DataFrame(rows)],ignore_index=True)
        text=build_research_report(comparison,ranges,cov,replay,reg)
        self.assertIn('lack unique, matching year-level prediction summaries',text)
        self.assertNotIn('Original observed-weather ridge (holdout_2026;',text)

    def test_shipment_score_requires_reproducible_annual_errors(self):
        comparison,ranges,cov,replay,reg=data()
        def entry(file,kind,row):
            return {'study':'iowa_v4','result_kind':kind,'result_source':'x/iowa_v4/'+file,
                    'result_row_json':json.dumps(row)}
        rows=[entry('scores_private.csv','reported_score',{'period':'2016-2021','model':m,'years':1,'mae_usd':v})
              for m,v in [('base',1e6),('full',3e6),('damped',2e6)]]
        rows.append(entry('yearly_diagnostics_private.csv','diagnostic_row',
                          {'year':2016,'period':'2016-2021','actual_usd':10e6,
                           'base_prediction_usd':11e6,'base_absolute_error_usd':1e6,
                           'full_prediction_usd':19e6,'full_absolute_error_usd':3e6,
                           'damped_prediction_usd':12e6,'damped_absolute_error_usd':2e6}))
        reg=pd.concat([reg,pd.DataFrame(rows)],ignore_index=True)
        text=build_research_report(comparison,ranges,cov,replay,reg)
        self.assertIn('lack unique, matching annual diagnostic rows',text)
        self.assertNotIn('Earlier shipment investigation, matched years',text)

    def test_unrelated_period_names_and_different_sources_are_not_compared(self):
        comparison,ranges,cov,replay,reg=data()
        rows=[{'study':'iowa_v4','result_kind':'reported_score',
               'result_source':'x/iowa_v4/scores_private.csv',
               'result_row_json':json.dumps({'period':'2016-2021 excluding 2018',
                                              'model':'base','years':5,'mae_usd':3e6})}]
        reg=pd.concat([reg,pd.DataFrame(rows)],ignore_index=True)
        text=build_research_report(comparison,ranges,cov,replay,reg)
        self.assertNotIn('Earlier shipment investigation, matched years',text)
        self.assertNotIn('Earlier shipment scores from `x/iowa_v4/scores_private.csv`',text)

    def test_specific_failed_investigations_use_matching_retained_rows(self):
        comparison,ranges,cov,replay,reg=data()
        def entry(study,filename,kind,record):
            return {'study':study,'result_kind':kind,
                    'result_source':'primary/'+study+'/'+filename,
                    'result_row_json':json.dumps(record)}
        rows=[entry('iowa_fieldwork_2022','annual_private.csv','annual_error_row',
                    {'year':2022,'actual_usd':100e6,'price_forecast_fieldwork':160e6,
                     'full_six_week_absolute_error_usd':60e6,'one_week':130e6,
                     'one_week_absolute_error_usd':30e6,'price_forecast':120e6,
                     'forecast_only_absolute_error_usd':20e6}),
              entry('iowa_missing_fieldwork','paired_private.csv','paired_comparison',
                    {'year':2023,'model':'price_forecast_fieldwork','prior':5e6,
                     'new':12e6,'new_minus_prior_usd':7e6}),
              entry('iowa_missing_fieldwork','paired_private.csv','paired_comparison',
                    {'year':2023,'model':'price_forecast','prior':100e6,
                     'new':1e6,'new_minus_prior_usd':-99e6}),
              entry('iowa_missing_fieldwork','paired_private.csv','paired_comparison',
                    {'year':2026,'model':'price_forecast_fieldwork','prior':9e6,
                     'new':7e6,'new_minus_prior_usd':-2e6}),
              entry('iowa_forecast_further_2025','annual_errors_private.csv','annual_error_row',
                    {'year':2025,'previous_weather_sqrt_volume':90.,
                     'half_weather_sqrt_volume':70.,'price_sqrt_volume':80.})]
        reg=pd.concat([reg,pd.DataFrame(rows)],ignore_index=True)
        text=build_research_report(comparison,ranges,cov,replay,reg)
        self.assertIn('USD 60.0m for the six-week model, 30.0m for the matched one-week model, and 20.0m',text)
        self.assertIn('In 2023, the longer-history fieldwork comparison had USD 5.0m absolute error',text)
        self.assertIn('change was USD +7.0m',text)
        self.assertIn('change was USD -2.0m',text)
        self.assertIn('USD 90.0m with the previous weather and shipment adjustment, 70.0m',text)
        self.assertNotIn('lacks an unambiguous checked', text)

    def test_ambiguous_or_invalid_source_rows_are_not_presented_as_facts(self):
        comparison,ranges,cov,replay,reg=data()
        row={'study':'iowa_missing_fieldwork','result_kind':'paired_comparison',
             'result_source':'iowa_missing_fieldwork/paired_private.csv',
             'result_row_json':json.dumps({'year':2023,'model':'price_forecast_fieldwork',
                'prior':5e6,'new':12e6,'new_minus_prior_usd':7e6})}
        bad2022={'study':'iowa_fieldwork_2022','result_kind':'annual_error_row',
                 'result_source':'iowa_fieldwork_2022/annual_private.csv',
                 'result_row_json':json.dumps({'year':2022,'actual_usd':100e6,
                     'price_forecast_fieldwork':160e6,'full_six_week_absolute_error_usd':1,
                     'one_week':130e6,'one_week_absolute_error_usd':30e6,
                     'price_forecast':120e6,'forecast_only_absolute_error_usd':20e6})}
        reg=pd.concat([reg,pd.DataFrame([row,row,bad2022])],ignore_index=True)
        text=build_research_report(comparison,ranges,cov,replay,reg)
        self.assertIn('2022 fieldwork investigation cannot be summarized by year',text)
        self.assertIn('2023 longer-history fieldwork comparison lacks an unambiguous',text)
        self.assertIn('2025 sensitivity investigation lacks an unambiguous',text)
        self.assertNotIn('In 2023, the longer-history fieldwork comparison had',text)

    def test_rejects_stale_period_score_and_missing_model(self):
        comparison,ranges,cov,replay,reg=data()
        comparison['period_scores'].loc[0,'mae_usd']+=100
        with self.assertRaisesRegex(ValueError,'scores do not match'):
            build_research_report(comparison,ranges,cov,replay,reg)
        comparison,ranges,cov,replay,reg=data()
        comparison['predictions']=comparison['predictions'].iloc[1:].copy()
        with self.assertRaisesRegex(ValueError,'Unmatched'):
            build_research_report(comparison,ranges,cov,replay,reg)

    def test_rejects_false_coverage_and_missing_range(self):
        comparison,ranges,cov,replay,reg=data()
        cov.loc[0,'evaluated_ranges']+=1
        with self.assertRaisesRegex(ValueError,'Coverage counts'):
            build_research_report(comparison,ranges,cov,replay,reg)
        comparison,ranges,cov,replay,reg=data()
        ranges=ranges.iloc[1:].copy()
        with self.assertRaisesRegex(ValueError,'Ranges and predictions'):
            build_research_report(comparison,ranges,cov,replay,reg)

if __name__=='__main__': unittest.main()
