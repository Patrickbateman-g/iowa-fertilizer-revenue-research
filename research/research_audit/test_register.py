import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from register import (build_register, save_register, SOURCE_REVISION, SOURCE_PREFIX, PRIVATE_NOTEBOOK_SOURCE,
                      MODEL_INPUTS, STUDY_SEEN_EVIDENCE, _candidate_inputs, _seen_evidence)


class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, relative, data):
        path=self.root/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(data).to_csv(path,index=False)
        return path

    def test_original_score_keeps_failure_and_source_result_hashes(self):
        p=self.put('iowa_corn_revenue/scores_private.csv',[
            {'model':'last_year','stage':'holdout_2026','n_years':1,'mae_usd':3},
            {'model':'ridge_weather','stage':'holdout_2026','n_years':1,'mae_usd':7}])
        out=build_register(self.root)
        self.assertEqual(len(out),2)
        self.assertEqual(set(out.study),{'iowa_corn_revenue'})
        self.assertIn('original company-revenue',out.why_proposed.iloc[0])
        row=out[out.model.eq('ridge_weather')].iloc[0]
        self.assertEqual(json.loads(row.result_row_json)['mae_usd'],7)
        self.assertEqual(json.loads(row.result_row_json)['stage'],'holdout_2026')
        self.assertEqual(row.source_sha256,hashlib.sha256(p.read_bytes()).hexdigest())
        self.assertEqual(row.result_sha256,hashlib.sha256(row.result_row_json.encode('utf8')).hexdigest())
        self.assertIn('not an untouched',row.status)
        self.assertIn('Not reliably reconstructed',row.seen_at_original_proposal)

    def test_candidate_specific_inputs_are_scoped_to_study_and_model(self):
        self.put('iowa_forecast_further_2025/scores_private.csv',[
            {'candidate':'price_only','mae_usd':10},
            {'candidate':'q1_revenue_ratio','mae_usd':20},
            {'candidate':'price_sqrt_volume','mae_usd':30},
            {'candidate':'previous_weather_sqrt_volume','mae_usd':40},
            {'candidate':'unknown_candidate','mae_usd':50}])
        self.put('iowa_v2/scores_private.csv',[{'candidate':'price_only','mae_usd':12}])
        out=build_register(self.root)
        def row(study, candidate):
            return out[(out.study==study)&(out.model==candidate)].iloc[0]
        price=row('iowa_forecast_further_2025','price_only')
        self.assertNotIn('shipments',price.model_inputs)
        self.assertNotIn('forecast',price.model_inputs.lower())
        q1=row('iowa_forecast_further_2025','q1_revenue_ratio')
        self.assertIn('Q1 company revenue',q1.model_inputs)
        self.assertNotIn('nitrogenates',q1.model_inputs)
        self.assertIn('analysis_followup.py',q1.model_inputs_source)
        volume=row('iowa_forecast_further_2025','price_sqrt_volume')
        self.assertIn('shipments',volume.model_inputs)
        self.assertNotIn('forecast',volume.model_inputs.lower())
        weather=row('iowa_forecast_further_2025','previous_weather_sqrt_volume')
        self.assertIn('forecast',weather.model_inputs)
        self.assertIn('shipments',weather.model_inputs)
        self.assertIn('phosphate',weather.model_inputs)
        self.assertIn('analysis_phosphate_forecast_2025.py',weather.model_inputs_source)
        self.assertEqual(row('iowa_forecast_further_2025','unknown_candidate').model_inputs_scope,'not recovered')
        self.assertEqual(row('iowa_v2','price_only').model_inputs_scope,'not recovered')
        self.assertEqual(row('iowa_v2','price_only').model_inputs_source,'')

    def test_reconstructed_source_links_are_pinned_to_verified_snapshot(self):
        self.put('iowa_forecast_further_2025/predictions_private.csv',[
            {'year':2025,'candidate':'previous_weather_sqrt_volume','prediction_usd':100}])
        out=build_register(self.root)
        self.assertEqual(SOURCE_REVISION, '434c8808e4fe77cf8c59e9de0c6cca152737cb92')
        for row in out.itertuples():
            self.assertIn('/blob/'+SOURCE_REVISION+'/', row.known_seen_years_source)
            links=row.model_inputs_source.split(' | ')
            self.assertEqual(len(links), 2)
            self.assertTrue(all('/blob/'+SOURCE_REVISION+'/' in link for link in links))
            self.assertTrue(row.known_seen_years_source.endswith('/analysis_forecast_further_2025.py'))
        paths=set()
        for (study, model), (_, filenames) in MODEL_INPUTS.items():
            links=_candidate_inputs(study, model)[1].split(' | ')
            self.assertEqual(links, [name if name == PRIVATE_NOTEBOOK_SOURCE else SOURCE_PREFIX + name
                                     for name in filenames])
            paths.update(filenames)
        for study, (_, filename, _) in STUDY_SEEN_EVIDENCE.items():
            self.assertEqual(_seen_evidence(study)[1], SOURCE_PREFIX + filename)
            paths.add(filename)
        self.assertEqual(paths, {
            'analysis_decoupled_forecast.py', 'analysis_fieldwork_2022_diagnostic.py',
            'analysis_fieldwork_research.py', 'analysis_fieldwork_six_weeks.py',
            'analysis_followup.py', 'analysis_forecast_2025_diagnostic.py',
            'analysis_forecast_further_2025.py', 'analysis_forecast_research.py',
            'analysis_longer_fieldwork_history.py', 'analysis_missing_fieldwork.py',
            'analysis_missing_forecast_days.py', PRIVATE_NOTEBOOK_SOURCE,
            'analysis_older_years_diagnosis.py',
            'analysis_phosphate_forecast.py', 'analysis_phosphate_forecast_2025.py',
            'research_audit/comparison.py',
        })

    def test_matched_comparison_candidate_inputs_are_specific_to_each_model(self):
        candidates = ('price_only', 'price_weather', 'price_shipments',
                      'price_weather_shipments')
        self.put('iowa_incremental_value/predictions_private.csv', [
            {'year': 2022, 'model': model, 'absolute_error_usd': 1}
            for model in candidates])
        self.put('other_study/scores_private.csv', [{'model': 'price_weather', 'mae_usd': 2}])
        out = build_register(self.root)
        for model in candidates:
            rows = out[(out.study == 'iowa_incremental_value') & (out.model == model)]
            self.assertEqual(set(rows.result_kind), {'prediction_row', 'derived_from_prediction_rows'})
            for row in rows.itertuples():
                self.assertIn('price ratio', row.model_inputs)
                self.assertEqual('forecast' in row.model_inputs.lower(), 'weather' in model)
                self.assertEqual('shipments' in row.model_inputs.lower(), 'shipments' in model)
                self.assertEqual(row.model_inputs_source, SOURCE_PREFIX + 'research_audit/comparison.py')
                self.assertIn('Not reliably reconstructed', row.seen_at_original_proposal)
        other = out[out.study.eq('other_study')].iloc[0]
        self.assertEqual(other.model_inputs_scope, 'not recovered')

    def test_phosphate_forecast_variants_distinguish_adjustment_and_cohort(self):
        variants = ('prior_unadjusted', 'prior_adjusted', 'full_unadjusted', 'full_adjusted')
        self.put('iowa_phosphate_forecast/predictions_private.csv', [
            {'year': 2022, 'variant': name, 'absolute_error_usd': 1} for name in variants])
        self.put('other_study/scores_private.csv', [{'variant': 'full_adjusted', 'mae_usd': 2}])
        out = build_register(self.root)
        for name in variants:
            rows = out[(out.study == 'iowa_phosphate_forecast') & (out.model == name)]
            self.assertEqual(set(rows.result_kind), {'prediction_row', 'derived_from_prediction_rows'})
            for row in rows.itertuples():
                self.assertIn('forecast', row.model_inputs.lower())
                self.assertIn('fieldwork is not a forecast feature', row.model_inputs)
                self.assertEqual('phosphate segment sales adjust' in row.model_inputs,
                                 name.endswith('_adjusted'))
                self.assertEqual('full cohort with recovered older years' in row.model_inputs,
                                 name.startswith('full_'))
                self.assertEqual(row.model_inputs_source, SOURCE_PREFIX + 'analysis_phosphate_forecast.py')
                self.assertEqual(row.known_seen_years_at_study, 'unknown')
        self.assertEqual(out[out.study == 'other_study'].iloc[0].model_inputs_scope, 'not recovered')

    def test_decoupled_model_cohorts_are_described_without_inventing_fieldwork_features(self):
        variants = [cohort + suffix for cohort in ('prior', 'full', 'decoupled')
                    for suffix in ('_adjusted', '_unadjusted')]
        self.put('iowa_decoupled_forecast/predictions_private.csv', [
            {'year':2023, 'variant':name, 'absolute_error_usd':1} for name in variants])
        out = build_register(self.root)
        for name in variants:
            for row in out[out.model.eq(name)].itertuples():
                self.assertIn('fieldwork is not a forecast feature', row.model_inputs)
                self.assertEqual('no fieldwork eligibility' in row.model_inputs, name.startswith('decoupled_'))
                self.assertEqual('phosphate segment sales adjust' in row.model_inputs, name.endswith('_adjusted'))
                self.assertEqual(row.model_inputs_source, SOURCE_PREFIX + 'analysis_decoupled_forecast.py')

    def test_other_retained_candidate_definitions_use_actual_sources(self):
        studies = {'iowa_recency_sensitivity': ('prior_long', 'all_full', 'all_half_weight_added',
                    'all_half_adjust', 'all_forecast_only', 'prior_forecast_only', 'price_only'),
                   'iowa_phosphate_sensitivity': ('unadjusted', 'segment_adjusted'),
                   'iowa_phosphate_alt_fieldwork': ('unadjusted', 'segment_adjusted'),
                   'iowa_missing_forecast_days': ('original', 'common_day')}
        for study, names in studies.items():
            self.put(study + '/predictions_private.csv', [
                {'year':2023, 'candidate':name, 'absolute_error_usd':1} for name in names])
        out=build_register(self.root)
        for study, names in studies.items():
            for name in names:
                row=out[(out.study==study)&(out.model==name)].iloc[0]
                self.assertTrue(row.model_inputs_scope.startswith('recovered from study code'))
                expected=(SOURCE_PREFIX+'analysis_missing_forecast_days.py'
                          if study=='iowa_missing_forecast_days' else PRIVATE_NOTEBOOK_SOURCE)
                self.assertEqual(row.model_inputs_source, expected)
        price=out[(out.study=='iowa_recency_sensitivity')&(out.model=='price_only')].iloc[0]
        self.assertIn('without fitted forecast or fieldwork',price.model_inputs)
        self.assertIn('these do not enter the price-only calculation',price.model_inputs)
        common=out[(out.study=='iowa_missing_forecast_days')&(out.model=='common_day')].iloc[0]
        self.assertIn('common available dates',common.model_inputs)

    def test_fieldwork_model_inputs_distinguish_features_and_window_scope(self):
        models = ('price_only', 'price_forecast', 'price_fieldwork',
                  'price_forecast_fieldwork')
        for study in ('iowa_fieldwork', 'iowa_fieldwork_six'):
            self.put(study + '/predictions_private.csv', [
                {'year': 2022, 'model': label, 'absolute_error_usd': 1}
                for label in models])
        for study in ('iowa_long_history', 'iowa_missing_fieldwork',
                      'iowa_missing_fieldwork_sensitivity'):
            self.put(study + '/predictions_private.csv', [
                {'year': 2022, 'model': 'price_forecast_fieldwork', 'absolute_error_usd': 1}])
        out = build_register(self.root)
        for row in out.itertuples():
            if row.study not in ('iowa_fieldwork', 'iowa_fieldwork_six',
                                 'iowa_long_history', 'iowa_missing_fieldwork',
                                 'iowa_missing_fieldwork_sensitivity'):
                continue
            self.assertIn('prior q2 revenue', row.model_inputs.lower())
            self.assertEqual('GFS MOS P12' in row.model_inputs, 'forecast' in row.model)
            self.assertEqual('USDA fieldwork days' in row.model_inputs, 'fieldwork' in row.model)
            self.assertIn('cohort requires complete forecast and fieldwork data', row.model_inputs)
            if 'fieldwork' in row.model:
                if row.study == 'iowa_fieldwork':
                    self.assertIn('window used for this artifact not established from the label', row.model_inputs)
                elif row.study == 'iowa_fieldwork_six':
                    self.assertIn('one- or six-week feature; inspect source path or window field', row.model_inputs)
                else:
                    self.assertIn('six-week average', row.model_inputs)
            self.assertIn('/blob/' + SOURCE_REVISION + '/', row.model_inputs_source)
            self.assertIn('Not reliably reconstructed', row.seen_at_original_proposal)
        self.assertEqual(_candidate_inputs('unrelated', 'price_forecast')[2], 'not recovered')

    def test_matched_incremental_rows_are_scoped_and_preserve_sources(self):
        path = self.put('iowa_incremental_value/incremental_private.csv', [
            {'year': 2022, 'weather_vs_price_usd': -3, 'shipments_vs_price_usd': 2},
            {'year': 2023, 'weather_vs_price_usd': 5, 'shipments_vs_price_usd': -4}])
        self.put('other_study/incremental_private.csv', [{'private_input': 99}])
        out = build_register(self.root)
        self.assertEqual(len(out), 2)
        self.assertEqual(set(out.result_kind), {'incremental_comparison_row'})
        self.assertEqual(set(out.study), {'iowa_incremental_value'})
        self.assertEqual(set(out.source_sha256), {hashlib.sha256(path.read_bytes()).hexdigest()})
        rows = [json.loads(item) for item in out.result_row_json]
        self.assertEqual(rows[0]['weather_vs_price_usd'], -3)
        self.assertEqual(rows[1]['shipments_vs_price_usd'], -4)
        self.assertTrue(out.model_inputs_scope.eq('not recovered').all())

    def test_candidate_input_mapping_applies_to_prediction_summaries(self):
        self.put('iowa_phosphate_forecast_2025_sensitivities/predictions_private.csv',[
            {'year':2022,'candidate':'adjusted_full','absolute_error_usd':4},
            {'year':2023,'candidate':'adjusted_full','absolute_error_usd':6},
            {'year':2022,'candidate':'adjusted_sqrt_q1_volume','absolute_error_usd':7},
            {'year':2023,'candidate':'adjusted_sqrt_q1_volume','absolute_error_usd':8}])
        out=build_register(self.root)
        out=out[out.result_kind.eq('derived_from_prediction_rows')].set_index('model')
        self.assertIn('forecast',out.loc['adjusted_full','model_inputs'])
        self.assertNotIn('shipments',out.loc['adjusted_full','model_inputs'])
        self.assertIn('shipments',out.loc['adjusted_sqrt_q1_volume','model_inputs'])
        self.assertTrue(out.model_inputs_scope.str.startswith('recovered from study code').all())
        self.assertTrue(out.model_inputs_source.str.contains('analysis_phosphate_forecast_2025.py').all())

    def test_prediction_only_wide_forecast_models(self):
        self.put('iowa_forecast/predictions_private.csv',[
            {'year':2022,'price_abs_error_usd':4.,'forecast_abs_error_usd':5.},
            {'year':2023,'price_abs_error_usd':8.,'forecast_abs_error_usd':2.}])
        out=build_register(self.root)
        out=out[out.result_kind.eq('derived_from_prediction_rows')].set_index('model')
        self.assertEqual(set(out.index),{'price_only','price_forecast'})
        self.assertEqual(json.loads(out.loc['price_only','result_row_json'])['mae_usd'],6.)
        self.assertEqual(json.loads(out.loc['price_forecast','result_row_json'])['years'],'2022,2023')
        self.assertEqual(set(out.result_kind),{'derived_from_prediction_rows'})

    def test_only_named_fieldwork_annual_results_are_indexed(self):
        p=self.put('iowa_fieldwork_2022/annual_private.csv',[
            {'year':2022,'actual_usd':100,'full_six_week_absolute_error_usd':30,
             'one_week_absolute_error_usd':20,'forecast_only_absolute_error_usd':10}])
        self.put('other_study/annual_private.csv',[{'year':2022,'private_input':7}])
        out=build_register(self.root)
        self.assertEqual(len(out),1)
        row=out.iloc[0]
        self.assertEqual(row.study,'iowa_fieldwork_2022')
        self.assertEqual(row.result_kind,'annual_error_row')
        self.assertEqual(json.loads(row.result_row_json)['full_six_week_absolute_error_usd'],30)
        self.assertEqual(row.source_sha256,hashlib.sha256(p.read_bytes()).hexdigest())

    def test_prediction_only_older_year_diagnostic_keeps_groups(self):
        self.put('iowa_older_year_diagnostic/predictions_private.csv',[
            {'year':2023,'training':'prior','absolute_error_usd':5},
            {'year':2023,'training':'new','absolute_error_usd':9},
            {'year':2026,'training':'prior','absolute_error_usd':7},
            {'year':2026,'training':'new','absolute_error_usd':8}])
        out=build_register(self.root)
        out=out[out.result_kind.eq('derived_from_prediction_rows')]
        self.assertEqual(len(out),2)
        self.assertEqual(set(out.model),{'price_forecast_fieldwork'})
        records=[json.loads(x) for x in out.result_row_json]
        self.assertEqual({d['training'] for d in records},{'prior','new'})
        self.assertEqual({d['years'] for d in records},{'2023,2026'})
        self.assertEqual({d['mean_absolute_error'] for d in records},{6.,8.5})

    def test_older_year_decomposition_is_indexed_only_for_its_study(self):
        terms=self.put('iowa_older_year_diagnostic/feature_terms_private.csv',[
            {'year':2023,'training':'prior','feature':'forecast','log_contribution':-0.2},
            {'year':2023,'training':'new','feature':'forecast','log_contribution':0.1}])
        years=self.put('iowa_older_year_diagnostic/year_contributions_private.csv',[
            {'year':2023,'added_year':2010,'average_prediction_change_usd':-3}])
        self.put('unrelated/feature_terms_private.csv',[{'secret_input':1}])
        self.put('iowa_older_year_diagnostic/added_year_profiles_private.csv',[{'training_input':2}])
        out=build_register(self.root)
        self.assertEqual(len(out),3)
        self.assertEqual(set(out.model),{'price_forecast_fieldwork'})
        self.assertEqual(set(out.result_kind),{'diagnostic_row'})
        self.assertEqual(set(out.source_sha256),{hashlib.sha256(terms.read_bytes()).hexdigest(),
                                                 hashlib.sha256(years.read_bytes()).hexdigest()})
        values=[json.loads(s) for s in out.result_row_json]
        self.assertEqual({x['training'] for x in values if 'training' in x},{'prior','new'})
        self.assertEqual(next(x for x in values if 'added_year' in x)['average_prediction_change_usd'],-3)

    def test_more_seen_years_are_attributed_to_the_specific_studies(self):
        sources={
            'iowa_fieldwork':'analysis_fieldwork_research.py',
            'iowa_fieldwork_six':'analysis_fieldwork_six_weeks.py',
            'iowa_long_history':'analysis_longer_fieldwork_history.py',
            'iowa_older_year_diagnostic':'analysis_older_years_diagnosis.py',
            'iowa_decoupled_forecast':'analysis_decoupled_forecast.py',
            'iowa_missing_fieldwork':'analysis_missing_fieldwork.py',
            'iowa_missing_fieldwork_sensitivity':'analysis_missing_fieldwork.py',
        }
        for study in sources:
            self.put(f'{study}/scores_private.csv',[{'model':'forecast','mae_usd':2}])
        self.put('iowa_unknown/scores_private.csv',[{'model':'baseline','mae_usd':3}])
        out=build_register(self.root).set_index('study')
        for study,source in sources.items():
            row=out.loc[study]
            self.assertEqual(row.known_seen_years_at_study,'2022,2023,2024,2025,2026')
            self.assertTrue(row.known_seen_years_source.endswith('/'+source))
            self.assertIn('not a complete history',row.known_seen_years_scope)
            self.assertIn('Not reliably reconstructed',row.seen_at_original_proposal)
        self.assertEqual(out.loc['iowa_unknown'].known_seen_years_at_study,'unknown')

    def test_matched_fieldwork_prediction_windows_remain_distinct(self):
        self.put('iowa_fieldwork_six/matched_predictions_private.csv',[
            {'year':2022,'model':'price_forecast_fieldwork',
             'prediction_usd_one':100,'prediction_usd_six':110,
             'absolute_error_usd_one':10,'absolute_error_usd_six':20},
            {'year':2023,'model':'price_forecast_fieldwork',
             'prediction_usd_one':120,'prediction_usd_six':130,
             'absolute_error_usd_one':30,'absolute_error_usd_six':40}])
        out=build_register(self.root)
        out=out[out.result_kind.eq('derived_from_prediction_rows')]
        self.assertEqual(len(out),2)
        summaries={r['fieldwork_window']:r for r in map(json.loads,out.result_row_json)}
        self.assertEqual(set(summaries),{'one','six'})
        self.assertEqual(summaries['one']['mean_absolute_error'],20)
        self.assertEqual(summaries['six']['mean_absolute_error'],30)
        self.assertEqual(summaries['one']['error_column'],'absolute_error_usd_one')
        self.assertEqual(summaries['six']['error_column'],'absolute_error_usd_six')
        self.assertEqual(summaries['six']['years'],'2022,2023')
        self.assertEqual(summaries['six']['missing_or_invalid_error_count'],0)

    def test_known_seen_years_are_study_scoped_and_other_years_remain_unknown(self):
        self.put('iowa_fieldwork_2022/scores_private.csv',[
            {'model':'half_increment','years':2,'mae_usd':3}])
        self.put('iowa_phosphate_forecast_2025/scores_private.csv',[
            {'model':'diagnostic','years':2,'mae_usd':4}])
        self.put('iowa_unknown/scores_private.csv',[
            {'model':'baseline','years':2,'mae_usd':5}])
        out=build_register(self.root).set_index('study')
        fieldwork=out.loc['iowa_fieldwork_2022']
        self.assertEqual(fieldwork.known_seen_years_at_study,'2022')
        self.assertTrue(fieldwork.known_seen_years_source.endswith('/analysis_fieldwork_2022_diagnostic.py'))
        self.assertIn('not a complete history',fieldwork.known_seen_years_scope)
        self.assertIn('Not reliably reconstructed',fieldwork.seen_at_original_proposal)
        phosphate=out.loc['iowa_phosphate_forecast_2025']
        self.assertEqual(phosphate.known_seen_years_at_study,'2025')
        self.assertTrue(phosphate.known_seen_years_source.endswith('/analysis_phosphate_forecast_2025.py'))
        unknown=out.loc['iowa_unknown']
        self.assertEqual(unknown.known_seen_years_at_study,'unknown')
        self.assertEqual(unknown.known_seen_years_source,'')

    def test_original_weather_point_sensitivity_preserved(self):
        self.put('iowa_corn_revenue/sensitivity_private.csv',[
            {'excluded_point':'Ames','stage':'holdout_2026','n_years':1,'mae_usd':2},
            {'excluded_point':'Sioux City','stage':'holdout_2026','n_years':1,'mae_usd':3}])
        out=build_register(self.root)
        self.assertEqual(len(out),2)
        self.assertEqual(set(out.result_kind),{'diagnostic_row'})
        self.assertEqual({json.loads(x)['excluded_point'] for x in out.result_row_json},{'Ames','Sioux City'})

    def test_missing_error_cannot_silently_become_complete_mean(self):
        self.put('iowa_forecast/predictions_private.csv',[
            {'year':2022,'price_abs_error_usd':None,'forecast_abs_error_usd':1.},
            {'year':2023,'price_abs_error_usd':6.,'forecast_abs_error_usd':2.}])
        out=build_register(self.root)
        out=out[out.result_kind.eq('derived_from_prediction_rows')].set_index('model')
        row=json.loads(out.loc['price_only','result_row_json'])
        self.assertEqual(row['error_count'],1)
        self.assertEqual(row['missing_or_invalid_error_count'],1)

    def test_prediction_rows_preserve_order_duplicates_hashes_and_model_inputs(self):
        path=self.put('iowa_forecast_further_2025/predictions_private.csv',[
            {'year':2022,'candidate':'price_sqrt_volume','prediction_usd':100,
             'actual_usd':110,'absolute_error_usd':10},
            {'year':2022,'candidate':'price_sqrt_volume','prediction_usd':100,
             'actual_usd':110,'absolute_error_usd':10},
            {'year':2023,'candidate':'q1_revenue_ratio','prediction_usd':120,
             'actual_usd':100,'absolute_error_usd':20}])
        out=build_register(self.root)
        rows=out[out.result_kind.eq('prediction_row')].reset_index(drop=True)
        self.assertEqual(len(rows),3)
        self.assertEqual([json.loads(x)['year'] for x in rows.result_row_json],[2022,2022,2023])
        self.assertEqual([x.rsplit('-',1)[-1] for x in rows.experiment_id],['1','2','3'])
        self.assertEqual(rows.result_sha256.iloc[0],rows.result_sha256.iloc[1])
        self.assertNotEqual(rows.experiment_id.iloc[0],rows.experiment_id.iloc[1])
        self.assertTrue((rows.source_sha256==hashlib.sha256(path.read_bytes()).hexdigest()).all())
        self.assertIn('shipments',rows.model_inputs.iloc[0])
        self.assertIn('Q1 company revenue',rows.model_inputs.iloc[2])
        self.assertTrue(rows.status.str.contains('not necessarily a unique trial').all())
        self.assertEqual(len(out[out.result_kind.eq('derived_from_prediction_rows')]),2)

    def test_wide_prediction_row_stays_one_unlabelled_source_record(self):
        self.put('iowa_forecast/predictions_private.csv',[
            {'year':2022,'price_abs_error_usd':4,'forecast_abs_error_usd':5}])
        out=build_register(self.root)
        row=out[out.result_kind.eq('prediction_row')].iloc[0]
        self.assertEqual(row.model,'model not named in prediction row')
        self.assertEqual(row.model_inputs_scope,'not recovered')
        payload=json.loads(row.result_row_json)
        self.assertEqual((payload['price_abs_error_usd'],payload['forecast_abs_error_usd']),(4,5))
        self.assertEqual(len(out[out.result_kind.eq('derived_from_prediction_rows')]),2)

    def test_empty_artifact_is_reported_and_saved_manifest_is_honest(self):
        p=self.root/'iowa_unknown'/'scores_private.csv'
        p.parent.mkdir()
        p.write_text('',encoding='utf8')
        out=build_register(self.root)
        self.assertEqual(out.iloc[0].result_kind,'empty')
        self.assertIn('no result recovered',out.iloc[0].status)
        notes=save_register(out,self.root/'registry')
        self.assertIn('Other filenames',notes)
        self.assertIn('not evidence of original motives',notes)
        self.assertIn('Neither establishes historical availability',notes)

    def test_retained_skips_and_paired_results_are_discoverable(self):
        self.put('iowa_fieldwork/skipped_private.csv',[
            {'year':2019,'model':'fieldwork','reason':'missing prior data'},
            {'year':2020,'model':'forecast','reason':'too few training years'}])
        self.put('iowa_missing_forecast_days/omitted_private.csv',[
            {'year':2021,'model':'forecast','reason':'insufficient valid days'}])
        self.put('iowa_long_history/paired_private.csv',[
            {'year':2022,'model':'forecast','extended_minus_original_usd':7}])
        self.put('iowa_long_history/wins_private.csv',[
            {'model':'forecast','years':3,'extended_wins':1}])
        self.put('iowa_phosphate_forecast/annual_errors_private.csv',[
            {'year':2023,'price_only':8,'forecast':9}])
        out=build_register(self.root)
        self.assertEqual(len(out),6)
        skips=out[out.result_kind.eq('skipped_or_omitted')]
        self.assertEqual(len(skips),3)
        self.assertEqual({json.loads(x)['reason'] for x in skips.result_row_json},
                         {'missing prior data','too few training years','insufficient valid days'})
        self.assertTrue(skips.status.str.contains('not necessarily a failed model').all())
        paired=out[out.result_kind.eq('paired_comparison')]
        self.assertEqual(len(paired),2)
        self.assertEqual(len(out[out.result_kind.eq('annual_error_row')]),1)
        self.assertTrue(out.source_sha256.str.fullmatch('[0-9a-f]{64}').all())

    def test_empty_skipped_files_do_not_claim_a_model_failed(self):
        for relative, content in [('iowa_fieldwork/skipped_private.csv','\n'),
                                  ('iowa_missing_forecast_days/omitted_private.csv','year,reason\n')]:
            p=self.root/relative; p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(content,encoding='utf8')
        out=build_register(self.root)
        self.assertEqual(list(out.result_kind),['empty','empty'])
        self.assertTrue(out.status.str.contains('no result recovered').all())

    def test_hashes_change_when_result_changes(self):
        p=self.put('iowa_v2/scores_private.csv',[{'model':'baseline','mae_usd':1}])
        first=build_register(self.root).iloc[0]
        self.put('iowa_v2/scores_private.csv',[{'model':'baseline','mae_usd':2}])
        second=build_register(self.root).iloc[0]
        self.assertNotEqual(first.source_sha256,second.source_sha256)
        self.assertNotEqual(first.result_sha256,second.result_sha256)
        self.assertEqual(p.name,'scores_private.csv')


if __name__=='__main__':
    unittest.main()
