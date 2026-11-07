from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import pandas as pd

import comparison
import calibration
import intervals
import register
import replay
import report


def _read_json_array(path, label):
    obj=json.loads(Path(path).read_text(encoding='utf8'))
    if not isinstance(obj,list):
        raise ValueError(f'{label} must be a JSON array')
    return obj


def _is_repository_path(path):
    p=Path(path).resolve()
    return any((ancestor/'.git').exists() for ancestor in (p, *p.parents))


def _output_safety(output, inputs, artifacts):
    out=Path(output).resolve()
    if _is_repository_path(out):
        raise ValueError('Output must be outside any Git checkout or worktree')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('Output must be an empty directory or a new path')
    for p in inputs:
        source=Path(p).resolve()
        if source == out or out in source.parents:
            raise ValueError('Input cannot be located inside output')
    art=Path(artifacts).resolve()
    if not art.is_dir(): raise ValueError('Artifact root must exist as a directory')
    if art == out or out in art.parents:
        raise ValueError('Output cannot equal or contain the artifact root')
    return out


def _values(ledger, series, period):
    return [float(r['value']) for r in ledger if r['series']==series and str(r['period'])==str(period)]


def _status_combinations(actual, components, atol):
    if any(not component for component in components):
        return 'missing_ledger_value'
    if math.prod(len(x) for x in components)>10000:
        return 'too_many_versions_to_match'
    matches=any(math.isclose(actual, float(sum(v)), rel_tol=1e-10, abs_tol=atol)
                for v in itertools.product(*components))
    return 'matches_a_recorded_candidate' if matches else 'mismatch'


def compare_ledger_values(annual, shipments, ledger, shipment_years=None):
    for item in ledger: replay.validate_record(item)
    vol=comparison.validate_shipments(shipments)
    selected_years=set(vol) if shipment_years is None else set(shipment_years)
    frame=comparison.validate_annual(annual)
    rows=[]
    def check(year, field, value, components):
        tolerance=(0.01 if field.endswith('_usd') else 1e-5 if field=='forecast_mean_p12_pct' else 1e-9)
        rows.append({'year':int(year),'input':field,'status':_status_combinations(float(value),components,tolerance)})
    for row in frame.itertuples(index=False):
        y=int(row.year)
        check(y,'forecast_mean_p12_pct',row.forecast_mean_p12_pct,[_values(ledger,'forecast_probability',y)])
        check(y,'price_ratio',row.price_ratio,[_values(ledger,'nitrogen_price_ratio',y)])
        for field,year in [('target_q2_usd',y),('prior_q2_usd',y-1)]:
            pieces=[_values(ledger,'q2_sales',year)]
            if 2010 <= y <= 2015:
                pieces.append([-x for x in _values(ledger,'q2_phosphate_sales',year)])
            check(y,field,getattr(row,field),pieces)
        if y in vol and y in selected_years:
            item=vol[y]
            for prefix in ('current','previous'):
                check(y,'q1_shipments_'+prefix,item[prefix+'_q1_tons'],
                      [_values(ledger,'q1_shipments_'+prefix,y)])
            check(y,'q1_shipment_ratio',item['shipment_ratio'],[_values(ledger,'q1_shipment_ratio',y)])
    return pd.DataFrame(rows)


def _assert_selected_values_match(fold_reports, annual, shipments):
    data=comparison.validate_annual(annual).set_index('year')
    vols=comparison.validate_shipments(shipments)
    frame=pd.DataFrame(fold_reports)
    frame=frame[~frame.series.str.startswith('forecast_daily_')].copy()
    keycols=['decision_date','role','row_year','series']
    if frame.duplicated(keycols).any(): raise ValueError('Duplicate replay requirement')
    lookup={tuple(row[k] for k in keycols):row for row in frame.to_dict('records')}
    def selected(decision,role,year,series):
        record=lookup.get((decision,role,year,series))
        return float(record['selected_value']) if record and record['status']=='eligible' else None
    def check(actual,expected,label,atol=1e-9):
        if actual is not None and not math.isclose(float(actual),float(expected),rel_tol=1e-10,abs_tol=atol):
            raise ValueError(f'Eligible replay value differs from model input: {label}')
    for (decision,role,y),g in frame.groupby(['decision_date','role','row_year']):
        y=int(y)
        if role not in ('decision_inputs','training_features','training_target'): continue
        if y not in data.index: raise ValueError('Replay refers to missing annual row')
        annual_row=data.loc[y]
        def v(series): return selected(decision,role,y,series)
        if role in ('decision_inputs','training_features'):
            check(v('forecast_probability'),annual_row.forecast_mean_p12_pct,'forecast',atol=1e-5)
            check(v('nitrogen_price_ratio'),annual_row.price_ratio,'price ratio')
            stations=[v('forecast_station_'+x) for x in ('KAMW','KCID','KSUX')]
            if all(x is not None for x in stations):
                check(sum(stations)/3,annual_row.forecast_mean_p12_pct,'forecast station constituents',atol=1e-5)
            now,prev=v('nitrogen_price_current_april'),v('nitrogen_price_previous_april')
            if now is not None and prev is not None:
                if prev<=0: raise ValueError('Eligible price denominator is nonpositive')
                check(now/prev,annual_row.price_ratio,'price constituents')
            prior=v('q2_sales')
            if prior is not None and role=='training_features' and 2010<=y<=2015:
                phosphate=selected(decision,'training_baseline_adjustment',y,'q2_phosphate_sales')
                prior=prior-phosphate if phosphate is not None else None
            check(prior,annual_row.prior_q2_usd,'prior revenue',atol=.01)
            if role=='decision_inputs':
                item=vols[y]
                check(v('q1_shipments_current'),item['current_q1_tons'],'current shipments')
                check(v('q1_shipments_previous'),item['previous_q1_tons'],'previous shipments')
                check(v('q1_shipment_ratio'),item['shipment_ratio'],'shipment ratio')
                current,previous=v('q1_shipments_current'),v('q1_shipments_previous')
                if current is not None and previous is not None:
                    if previous<=0: raise ValueError('Eligible shipment denominator is nonpositive')
                    check(current/previous,item['shipment_ratio'],'shipment constituents')
        else:
            target=v('q2_sales')
            if target is not None and 2010<=y<=2015:
                phosphate=selected(decision,'training_target_adjustment',y,'q2_phosphate_sales')
                target=target-phosphate if phosphate is not None else None
            check(target,annual_row.target_q2_usd,'training target',atol=.01)


def _audit_evaluation_targets(ledger, annual, years, *, capture_root=None, require_capture=False):
    data = comparison.validate_annual(annual).set_index('year')
    result = []
    for year in years:
        row = data.loc[int(year)]
        filed = row.target_filed
        key = ('q2_sales', str(year))
        evaluated = replay.replay(ledger, filed, [key], capture_root=capture_root, require_capture=require_capture)
        for item in evaluated['report']:
            result.append({'year':int(year), 'tested_as_of':filed,
                           'role':'evaluation_outcome', **item})
        selected = evaluated['records'].get(key)
        if selected is None:
            filing_status = 'blocked_by_outcome'
            version = None
            value = None
        else:
            if not math.isclose(float(selected['value']), float(row.target_q2_usd),
                                rel_tol=1e-10, abs_tol=.01):
                raise ValueError(f'Eligible evaluation outcome differs from model target: {year}')
            availability, precision = replay._when(selected['available_at'], 'available_at')
            availability_day = availability.date() if precision == 'instant' else availability
            filing_status = ('eligible' if availability_day == replay._date(filed, 'target_filed')
                             else 'filing_date_mismatch')
            version = selected['source_version'] if filing_status == 'eligible' else None
            value = selected['value'] if filing_status == 'eligible' else None
        result.append({'year':int(year), 'tested_as_of':filed, 'role':'evaluation_filing_date',
                       'series':'q2_sales', 'period':str(year), 'status':filing_status,
                       'candidate_statuses':[], 'candidate_count':0,
                       'selected_version':version, 'selected_value':value})
    return result


def _audit_training_cohorts(ledger, annual, predictions, *, capture_root=None, require_capture=False):
    data = comparison.validate_annual(annual)
    folds = predictions[predictions.model.eq('price_weather')]
    records = []
    for fold in folds.itertuples(index=False):
        cutoff = fold.decision_date
        included = {int(y) for y in fold.training_year_list.split(',') if y}
        candidates = data[data.year.lt(fold.year)]
        for row in candidates.itertuples(index=False):
            y = int(row.year)
            key = ('q2_sales', str(y))
            on_filing = replay.replay(ledger, row.target_filed, [key], capture_root=capture_root, require_capture=require_capture)
            filing_record = on_filing['records'].get(key)
            if filing_record is None:
                filing_status = 'filing_unverified'
            else:
                available, precision = replay._when(filing_record['available_at'], 'available_at')
                available_day = available.date() if precision == 'instant' else available
                filing_status = ('eligible' if available_day == replay._date(row.target_filed, 'target_filed')
                                 else 'filing_date_mismatch')
                if filing_status == 'eligible':
                    filed_value=float(filing_record['value'])
                    if 2010 <= y <= 2015:
                        component_key=('q2_phosphate_sales',str(y))
                        component=replay.replay(ledger,row.target_filed,[component_key],capture_root=capture_root, require_capture=require_capture)['records'].get(component_key)
                        if component is None:
                            filing_status='filing_adjustment_unverified'
                        else:
                            filed_value-=float(component['value'])
                    if filing_status == 'eligible' and not math.isclose(filed_value,float(row.target_q2_usd),rel_tol=1e-10,abs_tol=.01):
                        filing_status='filing_value_mismatch'
            decision = replay.replay(ledger, cutoff, [key], capture_root=capture_root, require_capture=require_capture)
            q2 = decision['records'].get(key)
            outcome_status = decision['report'][-1]['status']
            if q2 is not None:
                value = float(q2['value'])
                if 2010 <= y <= 2015:
                    segment_key = ('q2_phosphate_sales', str(y))
                    segment = replay.replay(ledger, cutoff, [segment_key], capture_root=capture_root, require_capture=require_capture)
                    component = segment['records'].get(segment_key)
                    if component is None:
                        outcome_status = 'adjustment_unverified'
                    else:
                        value -= float(component['value'])
                if outcome_status == 'eligible' and not math.isclose(value, float(row.target_q2_usd),
                                                                      rel_tol=1e-10, abs_tol=.01):
                    outcome_status = 'outcome_value_mismatch'
            if y in included:
                cohort_status = 'eligible' if outcome_status == 'eligible' else 'included_outcome_unverified'
            elif q2 is not None:
                cohort_status = 'excluded_outcome_available'
            elif outcome_status == 'after_cutoff' and filing_status == 'eligible':
                cohort_status = 'eligible'
            else:
                cohort_status = 'excluded_outcome_unverified'
            records.append({'decision_date':cutoff, 'row_year':y, 'target_filed':row.target_filed,
                            'included':y in included, 'filing_status':filing_status,
                            'outcome_status_at_cutoff':outcome_status, 'cohort_status':cohort_status})
    return records


def _checksum(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as inp:
        for block in iter(lambda:inp.read(65536),b''): digest.update(block)
    return digest.hexdigest()


def _artifact_sources(primary, additional):
    sources=[('primary',Path(primary).resolve())]
    for item in additional or ():
        if not isinstance(item,(list,tuple)) or len(item)!=2:
            raise ValueError('Additional artifact sources must be (name, directory) pairs')
        name,root=item
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',name):
            raise ValueError('Artifact source name must contain only letters, numbers, - or _')
        sources.append((name,Path(root).resolve()))
    names=[name for name,_ in sources]
    if len(set(names))!=len(names): raise ValueError('Duplicate artifact source name')
    for i,(_,a) in enumerate(sources):
        if not a.is_dir(): raise ValueError('Artifact source directory must exist')
        for _,b in sources[:i]:
            if a==b or a in b.parents or b in a.parents:
                raise ValueError('Artifact source directories must not overlap')
    return sources


def _source_arg(value):
    if '=' not in value: raise argparse.ArgumentTypeError('Expected NAME=DIRECTORY')
    name,path=value.split('=',1)
    if not name or not path: raise argparse.ArgumentTypeError('Expected NAME=DIRECTORY')
    return name,Path(path)


def run_workflow(annual_path, shipments_path, ledger_path, artifacts_root, output_dir,
                 first_year=2016, last_year=2026, strict_replay=False, additional_artifacts=None, capture_root=None):
    if isinstance(first_year,bool) or isinstance(last_year,bool) or not 2016<=first_year<=last_year<=2026:
        raise ValueError('Retrospective test years must lie within 2016–2026')
    paths=[Path(annual_path),Path(shipments_path),Path(ledger_path)]
    for path in paths:
        if not path.is_file(): raise ValueError(f'Required input is missing:{path.name}')
    sources=_artifact_sources(artifacts_root,additional_artifacts)
    out=_output_safety(output_dir,paths,artifacts_root)
    for _,source in sources[1:]: _output_safety(out,paths,source)
    annual=pd.read_csv(paths[0])
    volumes=_read_json_array(paths[1],'Shipments')
    ledger=_read_json_array(paths[2],'Ledger')
    result=comparison.compare_incremental(annual,volumes,range(first_year,last_year+1))
    expected={(y,m) for y in range(first_year,last_year+1) for m in comparison.MODELS}
    present={(int(r.year),r.model) for r in result['predictions'].itertuples()}
    if present!=expected or len(result['predictions'])!=len(expected):
        raise AssertionError('Requested model/year cohort is incomplete')
    historical_folds,calibration_inventory=calibration.prepare_calibration_predictions(
        annual,volumes,first_year)
    full_predictions=pd.concat([historical_folds,result['predictions']],ignore_index=True)
    full_ranges,_=intervals.sequential_ranges(full_predictions,comparison.MODELS,coverage=.8)
    ranges=full_ranges[full_ranges.year.between(first_year,last_year)].copy().reset_index(drop=True)
    all_used=calibration.used_calibration_years(ranges)
    used_calibration=sorted(y for y in all_used if y<first_year)
    if not calibration_inventory.empty:
        calibration_inventory.loc[calibration_inventory.year.isin(used_calibration),'status']='used_in_range'
        calibration_inventory.loc[(calibration_inventory.status=='prepared'),'status']='prepared_not_used'
    historical_used=(historical_folds[historical_folds.year.isin(used_calibration)].copy()
                     if not historical_folds.empty else result['predictions'].iloc[0:0].copy())
    used_years=set(range(first_year,last_year+1)) | set(used_calibration)
    for text in historical_used.training_year_list if not historical_used.empty else []:
        used_years.update(int(y) for y in text.split(',') if y)
    for text in result['predictions'].training_year_list:
        used_years.update(int(y) for y in text.split(',') if y)
    used_annual=annual[annual.year.isin(used_years)]
    value_check=compare_ledger_values(used_annual,volumes,ledger,
                                      shipment_years=set(range(first_year,last_year+1)) | set(used_calibration))
    if (value_check.status=='mismatch').any():
        raise ValueError('Ledger candidate values conflict with one or more model inputs')
    coverage_rows=[]
    for model, group in ranges.groupby('model',sort=True):
        evaluated=group[group.contains_actual.notna()]
        hits=int(evaluated.contains_actual.sum()) if len(evaluated) else 0
        coverage_rows.append({'model':model,'total_prediction_years':len(group),
                              'ranges_available':int(group.lower_usd.notna().sum()),
                              'evaluated_ranges':len(evaluated),'covered':hits,
                              'empirical_coverage':hits/len(evaluated) if len(evaluated) else float('nan'),
                              'first_evaluation_year':int(evaluated.year.min()) if len(evaluated) else None,
                              'last_evaluation_year':int(evaluated.year.max()) if len(evaluated) else None})
    coverage=pd.DataFrame(coverage_rows)
    coverage_period=intervals.summarize_range_periods(ranges,{'2016-2021':(2016,2021),'2022-2026':(2022,2026)})
    fold_reports=[]
    for row in result['predictions'].query("model == 'price_weather'").itertuples():
        training=[int(y) for y in row.training_year_list.split(',') if y]
        fold_reports.extend(replay.project_fold_replay(ledger,int(row.year),training,capture_root=capture_root,require_capture=True))
    _assert_selected_values_match(fold_reports,annual,volumes)
    status_counts=pd.DataFrame(fold_reports).groupby('status').size().to_dict()
    incomplete=sum(value for key,value in status_counts.items() if key!='eligible')
    evaluation_reports=_audit_evaluation_targets(ledger, annual, range(first_year,last_year+1),capture_root=capture_root,require_capture=True)
    evaluation_counts=pd.DataFrame(evaluation_reports).groupby('status').size().to_dict()
    evaluation_incomplete=sum(value for key,value in evaluation_counts.items() if key!='eligible')
    cohort_reports=_audit_training_cohorts(ledger, annual, result['predictions'],capture_root=capture_root,require_capture=True)
    cohort_frame=pd.DataFrame(cohort_reports)
    cohort_counts=cohort_frame.cohort_status.value_counts().to_dict()
    filing_counts=cohort_frame.filing_status.value_counts().to_dict()
    cohort_incomplete=(sum(v for k,v in cohort_counts.items() if k!='eligible') +
                       sum(v for k,v in filing_counts.items() if k!='eligible'))
    calibration_reports=[]
    for row in historical_used[historical_used.model.eq('price_weather')].itertuples() if not historical_used.empty else []:
        training=[int(y) for y in row.training_year_list.split(',') if y]
        calibration_reports.extend(replay.project_fold_replay(ledger,int(row.year),training,capture_root=capture_root,require_capture=True))
    if calibration_reports:
        _assert_selected_values_match(calibration_reports,annual,volumes)
    cal_status_counts=(pd.DataFrame(calibration_reports).groupby('status').size().to_dict()
                       if calibration_reports else {})
    cal_evaluation=_audit_evaluation_targets(ledger,annual,used_calibration,capture_root=capture_root,require_capture=True) if used_calibration else []
    cal_eval_counts=(pd.DataFrame(cal_evaluation).groupby('status').size().to_dict()
                     if cal_evaluation else {})
    cal_cohorts=_audit_training_cohorts(ledger,annual,historical_used,capture_root=capture_root,require_capture=True) if used_calibration else []
    cal_cohort_counts=(pd.DataFrame(cal_cohorts).cohort_status.value_counts().to_dict() if cal_cohorts else {})
    cal_filing_counts=(pd.DataFrame(cal_cohorts).filing_status.value_counts().to_dict() if cal_cohorts else {})
    cal_incomplete=sum(v for counts in (cal_status_counts,cal_eval_counts,cal_cohort_counts,cal_filing_counts)
                       for k,v in counts.items() if k!='eligible')
    if strict_replay and (incomplete or evaluation_incomplete or cohort_incomplete or cal_incomplete or
                          not value_check.status.eq('matches_a_recorded_candidate').all()):
        raise replay.ReplayUnavailable('Strict replay incomplete; inspect provenance and constituent availability')
    inventories=[]
    for name,root in sources:
        part=register.build_register(root).copy()
        part['artifact_source']=name
        part['result_source']=name+'/'+part['result_source']
        part['experiment_id']=name+'/'+part['experiment_id']
        inventories.append(part)
    reg=pd.concat(inventories,ignore_index=True)
    if reg.experiment_id.duplicated().any():
        raise ValueError('Artifact source prefixes did not produce unique register IDs')
    markdown=report.build_research_report(result,ranges,coverage_period,fold_reports,reg,calibration_context=True,
                                          calibration_predictions=full_predictions)
    markdown += ('\n## Capture integrity requirement\n\n'
                 'The workflow requires a local SHA-256-matched artifact for every evidence type '
                 'before strict replay can be complete. A match checks current bytes only; '
                 'it does not prove that those bytes support a value, that a source is genuine, '
                 'or that the artifact existed or was public by the historical cutoff. '
                 'Those claims require independent source- and time-specific review.\n')
    outcome_lines = ['## Verification of evaluated outcomes', '',
        'Each scored Q2 outcome and its stated filing date are checked separately from prediction-time inputs. '
        'The counts below are evidence requirements, not independent observations. '
        'The filing-date check assumes the recorded source availability is the claimed filing day; '
        'a separate earlier earnings release needs a separately modeled date.', '',
        '| Outcome or filing status | Requirements |', '|---|---:|']
    for status,count in sorted(evaluation_counts.items()):
        outcome_lines.append(f'| {status} | {int(count)} |')
    outcome_lines.extend(['', 'Unverified or missing outcomes prevent strict replay even when the prediction-time inputs pass.', ''])
    markdown=markdown.replace('## Experiment record and interpretation',
                              '\n'.join(outcome_lines)+'\n## Experiment record and interpretation',1)
    markdown += ('\n## Training cohort completeness\n\n'
                 'The audit checks each earlier annual candidate, including excluded years, '
                 'against its declared filing date and the fold cutoff. Later evidence is used '
                 'only to audit the historical cohort, not as a prediction input. Counts are '
                 'conditional on a complete supplied ledger.\n\n'
                 + '\n'.join(f'- Cohort {k}: {int(v)}' for k,v in sorted(cohort_counts.items()))
                 + '\n' + '\n'.join(f'- Filing {k}: {int(v)}' for k,v in sorted(filing_counts.items()))
                 + '\n')
    skipped=calibration_inventory[calibration_inventory.status.eq('skipped')].to_dict('records')
    markdown += ('\n## Calibration context and verification\n\n'
                 'Earlier matching folds from 2016 onward contribute filed errors where available. '
                 'Only requested years are shown as predictions above. Preparation alone does not '
                 'verify provenance; strict replay also audits every used earlier fold, its '
                 'training cohort and its outcome. The inventory records missing or unbuildable folds.\n\n'
                 + 'Earlier calibration years used: ' + (', '.join(map(str,used_calibration)) or 'none')
                 + '. Skipped candidate folds: ' + str(len(skipped)) + '.\n')
    manifest={
        'retrospective_years':list(range(first_year,last_year+1)),
        'models':list(comparison.MODELS), 'decision_date':'May 31 per year',
        'calibration_years_used':used_calibration,
        'calibration_skipped_folds':skipped,
        'calibration_replay_requirement_counts':{str(k):int(v) for k,v in cal_status_counts.items()},
        'calibration_outcome_requirement_counts':{str(k):int(v) for k,v in cal_eval_counts.items()},
        'calibration_cohort_status_counts':{str(k):int(v) for k,v in cal_cohort_counts.items()},
        'calibration_filing_status_counts':{str(k):int(v) for k,v in cal_filing_counts.items()},
        'training_minimum':6,'nominal_range_coverage':0.8,
        'requested_model_years':len(expected),'register_rows':len(reg),
        'artifact_source_names':[name for name,_ in sources],
        'strict_replay_complete':bool(incomplete==0 and evaluation_incomplete==0 and cohort_incomplete==0 and cal_incomplete==0 and value_check.status.eq('matches_a_recorded_candidate').all()),
        'replay_requirement_counts':{str(k):int(v) for k,v in status_counts.items()},
        'evaluation_outcome_requirement_counts':{str(k):int(v) for k,v in evaluation_counts.items()},
        'training_cohort_status_counts':{str(k):int(v) for k,v in cohort_counts.items()},
        'training_filing_status_counts':{str(k):int(v) for k,v in filing_counts.items()},
        'ledger_value_comparison_counts':{str(k):int(v) for k,v in value_check.status.value_counts().items()},
        'input_sha256':{name:_checksum(path) for name,path in zip(('annual','shipments','ledger'),paths)},
        'limitations':('Strict replay requires hash-matched artifacts for every evidence type, but matches only current bytes, not source meaning, historic capture, or publication. Matches are against supplied values, not independent proof of source data or availability. '
                       'Unknown/missing candidate values and unverified replay records remain unknown. '
                       'Historical model selection and vintages limit the conclusions. '
                       'The frozen 2027 test is unchanged.')}
    out.mkdir(parents=True,mode=0o700,exist_ok=True)
    comparison.save_comparison(result,out/'comparison')
    interval_dir=out/'uncertainty'; interval_dir.mkdir()
    ranges.to_csv(interval_dir/'ranges_private.csv',index=False)
    historical_used.to_csv(interval_dir/'calibration_predictions_private.csv',index=False)
    calibration_inventory.to_csv(interval_dir/'calibration_inventory_private.csv',index=False)
    coverage.to_csv(interval_dir/'coverage_private.csv',index=False)
    coverage_period.to_csv(interval_dir/'coverage_by_period_private.csv',index=False)
    replay_dir=out/'replay'
    replay.save_private_ledger(ledger,fold_reports,replay_dir)
    (replay_dir/'evaluation_outcomes_private.json').write_text(
        json.dumps(evaluation_reports,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    (replay_dir/'training_cohorts_private.json').write_text(
        json.dumps(cohort_reports,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    (replay_dir/'calibration_replay_private.json').write_text(json.dumps(calibration_reports,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    (replay_dir/'calibration_outcomes_private.json').write_text(json.dumps(cal_evaluation,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    (replay_dir/'calibration_cohorts_private.json').write_text(json.dumps(cal_cohorts,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    value_check.to_csv(replay_dir/'value_matching_private.csv',index=False)
    register.save_register(reg,out/'register')
    report.save_research_report(markdown,out/'report')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf8')
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description='Run the retrospective research workflow.')
    parser.add_argument('--annual',required=True,type=Path,help='already-adjusted annual CSV')
    parser.add_argument('--shipments',required=True,type=Path,help='Q1 shipment JSON array')
    parser.add_argument('--ledger',required=True,type=Path,help='provenance ledger JSON array')
    parser.add_argument('--artifacts',required=True,type=Path,help='folder containing retained study outputs')
    parser.add_argument('--additional-artifacts',action='append',default=[],type=_source_arg,metavar='NAME=DIRECTORY',
                        help='Additional artifact root; may be repeated with unique names')
    parser.add_argument('--output',required=True,type=Path,help='New or empty folder outside any Git worktree')
    parser.add_argument('--first-year',type=int,default=2016)
    parser.add_argument('--last-year',type=int,default=2026)
    parser.add_argument('--strict-replay',action='store_true',help='Fail without writing if any provenance is unavailable')
    parser.add_argument('--capture-root',type=Path,help='directory containing captured source artifacts')
    args=parser.parse_args(argv)
    status=run_workflow(args.annual,args.shipments,args.ledger,args.artifacts,args.output,
                        args.first_year,args.last_year,args.strict_replay,args.additional_artifacts,args.capture_root)
    print(f"Matched predictions: {status['requested_model_years']}; register rows: {status['register_rows']}")
    print('Strict historical replay complete:',status['strict_replay_complete'])
    print('Replay statuses:',status['replay_requirement_counts'])
    print('Candidate value matching:',status['ledger_value_comparison_counts'])
    print('Outputs:',args.output)
    print(status['limitations'])


if __name__=='__main__':
    main()
