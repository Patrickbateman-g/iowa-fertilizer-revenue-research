from pathlib import Path
import json
import hashlib
import math
import pandas as pd

SOURCE_REVISION = '434c8808e4fe77cf8c59e9de0c6cca152737cb92'
SOURCE_PREFIX = ('https://github.com/Patrickbateman-g/iowa-corn-fertilizer-revenue/'
                 'blob/' + SOURCE_REVISION + '/')
PRIVATE_NOTEBOOK_SOURCE = ('https://www.kaggle.com/code/nicolasrster/'
                           'iowa-fertilizer-revenue-price-follow-up?scriptVersionId=366095225')

STUDIES = {
 'iowa_corn_revenue': ('Test original company-revenue and observed-weather baselines', 'Prior Q2 company revenue and/or NASA observed-weather proxies'),
 'iowa_v2': ('Financial and weather baselines; test incremental Q1 and nitrogen price information', 'Prior Q2, weather, Q1 sales and/or national nitrogen PPI'),
 'iowa_v3': ('Test alternative nitrogen PPI and Q1 shipment scaling', 'Prior Q2, price indices and/or Q1 company shipment volumes'),
 'iowa_v4': ('Diagnose shipment scaling across earlier and later periods', 'Prior Q2, nitrogen PPI, Q1 shipments'),
 'iowa_forecast': ('Test whether archived forecasts add to the nitrogen price baseline', 'Price baseline and Iowa GFS MOS precipitation probabilities'),
 'iowa_forecast_diagnostic': ('Investigate forecast errors by omitting evaluation years', 'Price baseline and Iowa forecast probabilities'),
 'iowa_fieldwork': ('Test whether the most recent fieldwork conditions add information', 'Price, forecast probability, USDA fieldwork'),
 'iowa_fieldwork_2022': ('Diagnose sensitivity to an omitted year in fieldwork comparison', 'Price, forecast probability, USDA fieldwork'),
 'iowa_fieldwork_six': ('Compare one versus six recent weeks of fieldwork', 'Price, forecast probability, weekly USDA fieldwork'),
 'iowa_long_history': ('Test additional older years of training', 'Price, forecast probability, six-week USDA fieldwork'),
 'iowa_missing_fieldwork': ('Recover older fieldwork weeks and evaluate additional training rows', 'Price, forecast probability, reconstructed USDA fieldwork'),
 'iowa_missing_fieldwork_sensitivity': ('Test alternative reconstruction of an older fieldwork report', 'Price, forecast probability, alternative fieldwork reconstruction'),
 'iowa_older_year_diagnostic': ('Diagnose the effect of recovered older training years', 'Price, forecast probability, six-week fieldwork'),
 'iowa_phosphate_sensitivity': ('Test adjustment of older revenue for phosphate', 'Price, forecast probability, fieldwork and historic segment disclosures'),
 'iowa_phosphate_alt_fieldwork': ('Test phosphate adjustment with alternate 2013 fieldwork', 'Price, forecast probability, fieldwork and segment disclosures'),
 'iowa_phosphate_forecast': ('Test phosphate adjustment for the simpler forecast-only model', 'Price, forecast probability and historic segment disclosures'),
 'iowa_phosphate_forecast_2025': ('Diagnose 2025 forecast error', 'Price, forecast probability and phosphate-adjusted history'),
 'iowa_phosphate_forecast_2025_sensitivities': ('Compare fixed sensitivities after inspecting 2025', 'Price, forecast probability, phosphate history and/or Q1 shipments'),
 'iowa_recency_sensitivity': ('Examine training-window and older-year sensitivities', 'Price, forecast probability, fieldwork and historical filings'),
 'iowa_forecast_further_2025': ('Test damped weather and Q1-based sensitivities after examining 2025', 'Price, forecast, Q1 shipment volumes and/or Q1 sales'),
 'iowa_decoupled_forecast': ('Remove unrelated fieldwork eligibility from forecast-only training', 'Price, forecast probability, historic segment disclosures'),
 'iowa_missing_forecast_days': ('Test common-day and complete-coverage forecast windows', 'Archived daily forecast probabilities, price and prior revenue'),
 'iowa_intercept_research': ('Test zero versus estimated intercept and price-only controls', 'Price baseline and forecast probability'),
 'iowa_forecast_core': ('Reproduce the forecast and intercept sensitivities in a standalone runner', 'Price baseline and forecast probability'),
 'iowa_incremental_value': ('Compare price, weather and Q1 shipments on matched years', 'Price baseline, Iowa forecast probabilities and Q1 company shipment volumes'),
}

ADDITIONAL_RESULTS = {
    'skipped_private.csv': 'skipped_or_omitted',
    'omitted_private.csv': 'skipped_or_omitted',
    'paired_private.csv': 'paired_comparison',
    'wins_private.csv': 'paired_comparison',
    'annual_errors_private.csv': 'annual_error_row',
}

STUDY_SEEN_EVIDENCE = {
    'iowa_forecast': (
        '2022,2023,2024,2025,2026', 'analysis_forecast_research.py',
        'Forecast study states historical results through 2026 were already inspected'),
    'iowa_forecast_diagnostic': (
        '2025', 'analysis_forecast_2025_diagnostic.py',
        'Post-hoc investigation of 2025 forecast error'),
    'iowa_fieldwork_2022': (
        '2022', 'analysis_fieldwork_2022_diagnostic.py',
        'Investigation of the already-inspected 2022 fieldwork error'),
    'iowa_phosphate_forecast_2025': (
        '2025', 'analysis_phosphate_forecast_2025.py',
        'Retrospective diagnosis states 2025 outcomes were already inspected'),
    'iowa_forecast_further_2025': (
        '2025', 'analysis_forecast_further_2025.py',
        'Further 2025 diagnosis using already-seen outcomes'),
    'iowa_fieldwork': (
        '2022,2023,2024,2025,2026', 'analysis_fieldwork_research.py',
        'Fieldwork research states outcomes through 2026 were already inspected'),
    'iowa_fieldwork_six': (
        '2022,2023,2024,2025,2026', 'analysis_fieldwork_six_weeks.py',
        'One- versus six-week comparison states years through 2026 were inspected'),
    'iowa_long_history': (
        '2022,2023,2024,2025,2026', 'analysis_longer_fieldwork_history.py',
        'Long-history study states tests through 2026 were already inspected'),
    'iowa_older_year_diagnostic': (
        '2022,2023,2024,2025,2026', 'analysis_older_years_diagnosis.py',
        'Older-year diagnosis states outcomes through 2026 were inspected'),
    'iowa_missing_fieldwork': (
        '2022,2023,2024,2025,2026', 'analysis_missing_fieldwork.py',
        'Recovered-fieldwork study states that its 2022–2026 test years were already examined'),
    'iowa_missing_fieldwork_sensitivity': (
        '2022,2023,2024,2025,2026', 'analysis_missing_fieldwork.py',
        'The alternate 2013 fieldwork study uses the same already-examined 2022–2026 test years'),
    'iowa_decoupled_forecast': (
        '2022,2023,2024,2025,2026', 'analysis_decoupled_forecast.py',
        'Decoupled forecast comparison states all 2022–2026 results were inspected'),
}

_PRICE = 'Prior Q2 revenue and current/prior April nitrogenates price index'
_Q1_REVENUE = 'Prior Q2 revenue and current/prior Q1 company revenue'
_FORECAST_ADJUSTED = (_PRICE + '; Iowa GFS MOS P12 forecast and earlier filed Q2 '
                      'revenue outcomes; older training revenue adjusted for phosphate')
_COMPARISON_PRICE = 'Prior Q2 revenue and the price ratio supplied in the annual input'
_COMPARISON_WEATHER = (_COMPARISON_PRICE + '; Iowa GFS MOS P12 forecast and earlier filed '
                       'Q2 revenue outcomes supplied in the annual input')
_VOLUME = '; current/prior Q1 company product shipments'
_FURTHER_SOURCE = ('analysis_forecast_further_2025.py',)
_PHOSPHATE_SOURCE = ('analysis_phosphate_forecast_2025.py',)
_COMPARISON_SOURCE = ('research_audit/comparison.py',)
MODEL_INPUTS = {
    ('iowa_incremental_value', 'price_only'): (_COMPARISON_PRICE, _COMPARISON_SOURCE),
    ('iowa_incremental_value', 'price_weather'): (_COMPARISON_WEATHER, _COMPARISON_SOURCE),
    ('iowa_incremental_value', 'price_shipments'): (_COMPARISON_PRICE + _VOLUME, _COMPARISON_SOURCE),
    ('iowa_incremental_value', 'price_weather_shipments'): (_COMPARISON_WEATHER + _VOLUME, _COMPARISON_SOURCE),
    ('iowa_forecast_further_2025', 'price_only'): (_PRICE, _FURTHER_SOURCE),
    ('iowa_forecast_further_2025', 'price_sqrt_volume'): (_PRICE + _VOLUME, _FURTHER_SOURCE),
    ('iowa_forecast_further_2025', 'q1_revenue_ratio'): (_Q1_REVENUE, _FURTHER_SOURCE + ('analysis_followup.py',)),
    ('iowa_forecast_further_2025', 'geometric_price_q1_revenue'): (_PRICE + '; current/prior Q1 company revenue', _FURTHER_SOURCE + ('analysis_followup.py',)),
    ('iowa_forecast_further_2025', 'previous_weather_sqrt_volume'): (_FORECAST_ADJUSTED + _VOLUME, _FURTHER_SOURCE + _PHOSPHATE_SOURCE),
    ('iowa_forecast_further_2025', 'half_weather_sqrt_volume'): (_FORECAST_ADJUSTED + _VOLUME, _FURTHER_SOURCE + _PHOSPHATE_SOURCE),
    ('iowa_phosphate_forecast_2025_sensitivities', 'price_only'): (_PRICE, _PHOSPHATE_SOURCE),
    ('iowa_phosphate_forecast_2025_sensitivities', 'adjusted_full'): (_FORECAST_ADJUSTED, _PHOSPHATE_SOURCE),
    ('iowa_phosphate_forecast_2025_sensitivities', 'adjusted_half_log_correction'): (_FORECAST_ADJUSTED, _PHOSPHATE_SOURCE),
    ('iowa_phosphate_forecast_2025_sensitivities', 'adjusted_training_range_clip'): (_FORECAST_ADJUSTED, _PHOSPHATE_SOURCE),
    ('iowa_phosphate_forecast_2025_sensitivities', 'adjusted_sqrt_q1_volume'): (_FORECAST_ADJUSTED + _VOLUME, _PHOSPHATE_SOURCE),
    ('iowa_phosphate_forecast_2025_sensitivities', 'unadjusted_forecast'): (
        _PRICE + '; Iowa GFS MOS P12 forecast and earlier filed Q2 revenue outcomes; unadjusted training revenue',
        _PHOSPHATE_SOURCE + ('analysis_phosphate_forecast.py',)),
}

for _variant, _history, _adjusted in (
    ('prior_unadjusted', 'prior longer-history cohort', False),
    ('prior_adjusted', 'prior longer-history cohort', True),
    ('full_unadjusted', 'full cohort with recovered older years', False),
    ('full_adjusted', 'full cohort with recovered older years', True),
):
    _description = (_PRICE + '; Iowa GFS MOS P12 forecast and earlier filed Q2 revenue outcomes; '
                    + _history + '; fieldwork completeness affects training cohort selection, '
                    'but fieldwork is not a forecast feature')
    if _adjusted:
        _description += '; phosphate segment sales adjust training revenue and prior revenue for 2010–2015'
    else:
        _description += '; training revenue is unadjusted for phosphate'
    MODEL_INPUTS[('iowa_phosphate_forecast', _variant)] = (
        _description, ('analysis_phosphate_forecast.py',))

for _cohort, _cohort_note in (
    ('prior', 'prior fieldwork-complete matched cohort'),
    ('full', 'full fieldwork-complete matched cohort with recovered older years'),
    ('decoupled', 'forecast, price and revenue completeness only; no fieldwork eligibility'),
):
    for _adjusted in (False, True):
        _name = _cohort + ('_adjusted' if _adjusted else '_unadjusted')
        _description = (_PRICE + '; Iowa GFS MOS P12 forecast and earlier filed Q2 revenue outcomes; '
                        + _cohort_note + '; fieldwork is not a forecast feature')
        _description += ('; phosphate segment sales adjust historical training target and prior revenue'
                         if _adjusted else '; training revenue is unadjusted for phosphate')
        MODEL_INPUTS[('iowa_decoupled_forecast', _name)] = (
            _description, ('analysis_decoupled_forecast.py',))

for _label, _feature in (('original', 'original station-level daily forecast means'),
                         ('common_day', 'station forecast means on common available dates')):
    MODEL_INPUTS[('iowa_missing_forecast_days', _label)] = (
        _PRICE + '; ' + _feature + '; earlier filed Q2 outcomes; historical training '
        'revenue adjusted for phosphate; coverage cohort is identified by the result row',
        ('analysis_missing_forecast_days.py',))

_RECENCY_BASE = (_PRICE + '; earlier filed Q2 outcomes; matched training cohort '
                 'requires complete forecast and fieldwork')
for _label, _detail in (
    ('prior_long', 'prior longer-history cohort; forecast and six-week fieldwork features'),
    ('all_full', 'full recovered-history cohort; forecast and six-week fieldwork features'),
    ('all_half_weight_added', 'full recovered-history cohort; forecast and six-week fieldwork; '
                              'half weight on the three recovered years'),
    ('all_half_adjust', 'full recovered-history cohort; forecast and six-week fieldwork; '
                        'half the fitted log adjustment'),
    ('all_forecast_only', 'full recovered-history cohort; forecast feature only'),
    ('prior_forecast_only', 'prior longer-history cohort; forecast feature only'),
    ('price_only', 'price baseline without fitted forecast or fieldwork features'),
):
    _inputs = (_PRICE + '; without fitted forecast or fieldwork features; matched evaluation uses '
               'a cohort requiring complete forecast and fieldwork and earlier filed Q2 outcomes; '
               'these do not enter the price-only calculation'
               if _label == 'price_only' else _RECENCY_BASE + '; ' + _detail)
    MODEL_INPUTS[('iowa_recency_sensitivity', _label)] = (
        _inputs, (PRIVATE_NOTEBOOK_SOURCE,))

for _study in ('iowa_phosphate_sensitivity', 'iowa_phosphate_alt_fieldwork'):
    for _label, _detail in (
        ('unadjusted', 'training revenue unadjusted for phosphate'),
        ('segment_adjusted', 'phosphate segment sales adjust historical training target and prior revenue'),
    ):
        MODEL_INPUTS[(_study, _label)] = (
            _PRICE + '; forecast probability and six-week fieldwork; earlier filed Q2 outcomes; '
            'matched full recovered-history cohort; ' + _detail, (PRIVATE_NOTEBOOK_SOURCE,))

_FIELDWORK_FITTED = _PRICE + '; earlier filed Q2 revenue outcomes'
_FIELDWORK_FORECAST = _FIELDWORK_FITTED + '; Iowa GFS MOS P12 forecast'
_FIELDWORK_MODELS = {
    'price_only': _PRICE,
    'price_forecast': _FIELDWORK_FORECAST,
    'price_fieldwork': _FIELDWORK_FITTED + '; USDA fieldwork days',
    'price_forecast_fieldwork': _FIELDWORK_FORECAST + '; USDA fieldwork days',
}
for _study, _source, _labels, _window in (
    ('iowa_fieldwork', 'analysis_fieldwork_research.py', tuple(_FIELDWORK_MODELS),
     'one- or six-week feature; window used for this artifact not established from the label'),
    ('iowa_fieldwork_six', 'analysis_fieldwork_six_weeks.py', tuple(_FIELDWORK_MODELS),
     'one- or six-week feature; inspect source path or window field'),
    ('iowa_long_history', 'analysis_longer_fieldwork_history.py',
     ('price_only', 'price_forecast', 'price_forecast_fieldwork'), 'six-week average'),
    ('iowa_missing_fieldwork', 'analysis_missing_fieldwork.py',
     ('price_forecast', 'price_forecast_fieldwork'), 'six-week average'),
    ('iowa_missing_fieldwork_sensitivity', 'analysis_missing_fieldwork.py',
     ('price_forecast', 'price_forecast_fieldwork'), 'six-week average'),
):
    for _label in _labels:
        _description = _FIELDWORK_MODELS[_label]
        if 'fieldwork' in _label:
            _description += ' (' + _window + ')'
        _description += ('; matched training and evaluation cohort requires complete '
                         'forecast and fieldwork data even when those are not prediction features')
        MODEL_INPUTS[(_study, _label)] = (_description, (_source,))

def _candidate_inputs(study, model):
    entry = MODEL_INPUTS.get((study, model))
    if entry is None:
        return ('Unknown; broad study inputs are listed separately', '', 'not recovered')
    inputs, filenames = entry
    return (inputs, ' | '.join(name if name == PRIVATE_NOTEBOOK_SOURCE else SOURCE_PREFIX + name
                                for name in filenames),
            'recovered from study code; not evidence of original motivation or availability')

def _seen_evidence(study):
    entry = STUDY_SEEN_EVIDENCE.get(study)
    if entry is None:
        return ('unknown', '', 'No study-specific seen-year evidence recorded')
    years, filename, note = entry
    url = SOURCE_PREFIX + filename
    return (years, url, note + '; source assertion about this investigation, '
            'not a complete history or evidence about the original baseline proposal')

def study_for(path, root):
    rel=Path(path).relative_to(root)
    name=next((p for p in rel.parts if p in STUDIES), rel.parts[0])
    return name, STUDIES.get(name, ('Purpose not recovered; inspect source artifact', 'Inputs not recovered; inspect source artifact'))


def _hash_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(65536),b''):
            digest.update(block)
    return digest.hexdigest()


def _clean(value):
    if pd.isna(value): return None
    value=value.item() if hasattr(value,'item') else value
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def _canonical(row):
    return json.dumps({str(k):_clean(v) for k,v in row.items()},sort_keys=True,
                      ensure_ascii=False,separators=(',',':'),allow_nan=False)


def _prediction_summaries(frame, study=None):
    if frame.empty: return []
    if {'price_abs_error_usd','forecast_abs_error_usd'}.issubset(frame.columns):
        out=[]
        for model,col in [('price_only','price_abs_error_usd'),('price_forecast','forecast_abs_error_usd')]:
            values=pd.to_numeric(frame[col],errors='coerce')
            finite=values.map(lambda x: pd.notna(x) and math.isfinite(float(x)))
            years=','.join(map(str, sorted(set(frame.year.dropna().astype(int))))) if 'year' in frame else None
            out.append((model,{'model':model,'rows':len(frame),'years':years,
                               'error_count':int(finite.sum()),'missing_or_invalid_error_count':int((~finite).sum()),
                               'mae_usd':float(values[finite].mean()) if finite.any() else None}))
        return out
    keys=[k for k in ('model','candidate','variant','cohort','training','period','stage','window') if k in frame]
    if keys:
        grouper=keys[0] if len(keys)==1 else keys
        groups=frame.groupby(grouper,dropna=False,sort=True)
    else:
        groups=[((),frame)]
    out=[]
    for _,part in groups:
        fallback=('price_forecast_fieldwork' if study=='iowa_older_year_diagnostic'
                  else 'model not named in prediction artifact')
        label=next((_clean(part[k].iloc[0]) for k in ('model','candidate','variant') if k in part and _clean(part[k].iloc[0]) is not None),fallback)
        summary={k:_clean(part[k].iloc[0]) for k in keys}
        summary['rows']=len(part)
        if 'year' in part:
            valid_years=pd.to_numeric(part.year,errors='coerce').dropna()
            summary['years']=','.join(map(str,sorted(set(int(y) for y in valid_years if float(y).is_integer()))))
        paired_columns = {f'{name}_usd_{window}' for name in ('prediction', 'absolute_error')
                          for window in ('one', 'six')}
        if paired_columns.issubset(part.columns):
            for window in ('one', 'six'):
                column = f'absolute_error_usd_{window}'
                vals = pd.to_numeric(part[column], errors='coerce')
                finite = vals.map(lambda x: pd.notna(x) and math.isfinite(float(x)))
                evidence = dict(summary, fieldwork_window=window, error_column=column,
                                error_count=int(finite.sum()),
                                missing_or_invalid_error_count=int((~finite).sum()),
                                mean_absolute_error=float(vals[finite].mean()) if finite.any() else None)
                out.append((str(label), evidence))
            continue
        for col in ('absolute_error_usd','absolute_error_m','absolute_error_musd'):
            if col in part:
                vals=pd.to_numeric(part[col],errors='coerce')
                finite=vals.map(lambda x: pd.notna(x) and math.isfinite(float(x)))
                summary['error_column']=col
                summary['error_count']=int(finite.sum())
                summary['missing_or_invalid_error_count']=int((~finite).sum())
                summary['mean_absolute_error']=float(vals[finite].mean()) if finite.any() else None
                break
        out.append((str(label),summary))
    return out


def build_register(root):
    root=Path(root)
    rows=[]
    paths=(set(root.glob('**/*scores*.csv')) | set(root.glob('**/*predictions*.csv')) |
           set(root.glob('**/*diagnostic*.csv')) | set(root.glob('**/*sensitivity*.csv')) |
           set(root.glob('**/leave_one_training_year_out*.csv')))
    for filename in ADDITIONAL_RESULTS:
        paths.update(root.glob('**/' + filename))
    paths.update(root.glob('**/iowa_fieldwork_2022/annual_private.csv'))
    for filename in ('feature_terms_private.csv', 'year_contributions_private.csv'):
        paths.update(root.glob('**/iowa_older_year_diagnostic/' + filename))
    paths.update(root.glob('**/iowa_incremental_value/incremental_private.csv'))
    for p in sorted(paths):
        if not p.is_file(): continue
        relative=str(p.relative_to(root)); digest=_hash_file(p)
        study,(why,inputs)=study_for(p,root)
        known_seen, seen_source, seen_scope = _seen_evidence(study)
        def add(index,model,kind,payload,status='retrospective; not an untouched evaluation'):
            canonical=_canonical(payload)
            model_inputs, model_source, model_scope = _candidate_inputs(study, model)
            rows.append({'experiment_id':f'{relative}#{kind}-{index}', 'study':study,
                'model':model,'why_proposed':why,'inputs':inputs,
                'model_inputs':model_inputs,'model_inputs_source':model_source,
                'model_inputs_scope':model_scope,
                'seen_at_original_proposal':'Not reliably reconstructed; consult dated source and history',
                'known_seen_years_at_study':known_seen,
                'known_seen_years_source':seen_source,
                'known_seen_years_scope':seen_scope,
                'seen_at_registration':'All historical outcomes through 2026 have been inspected',
                'result_source':relative,'result_kind':kind,'source_sha256':digest,
                'result_row_json':canonical,'result_sha256':hashlib.sha256(canonical.encode('utf8')).hexdigest(),
                'status':status})
        try:
            frame=pd.read_csv(p)
        except pd.errors.EmptyDataError as exc:
            if not p.read_bytes().strip():
                add(1,'not recoverable','empty',{'rows':0},'empty artifact; no result recovered')
            else:
                add(1,'not recoverable','unreadable',{'error_type':type(exc).__name__},'artifact unreadable; results not recovered')
            continue
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            add(1,'not recoverable','unreadable',{'error_type':type(exc).__name__},'artifact unreadable; results not recovered')
            continue
        if frame.empty:
            add(1,'not recoverable','empty',{'rows':0},'empty artifact; no result recovered')
            continue
        if 'scores' in p.name:
            for i,record in frame.iterrows():
                d={k:_clean(v) for k,v in record.to_dict().items()}
                model=next((str(d[k]) for k in ('model','candidate','variant') if d.get(k) is not None),
                           'model not named in score row')
                add(i+1,model,'reported_score',d)
        elif 'predictions' in p.name:
            summaries=_prediction_summaries(frame,study)
            for i,(model,summary) in enumerate(summaries,1):
                add(i,model,'derived_from_prediction_rows',summary)
            for ordinal, (_, record) in enumerate(frame.iterrows(), 1):
                d={k:_clean(v) for k,v in record.to_dict().items()}
                fallback=('price_forecast_fieldwork' if study=='iowa_older_year_diagnostic'
                          else 'model not named in prediction row')
                model=next((str(d[k]) for k in ('model','candidate','variant')
                            if d.get(k) is not None), fallback)
                add(ordinal,model,'prediction_row',d,
                    'retained prediction source row; retrospective; not necessarily a unique trial')
        else:
            for i,record in frame.iterrows():
                d={k:_clean(v) for k,v in record.to_dict().items()}
                if study == 'iowa_older_year_diagnostic' and p.name in ('feature_terms_private.csv', 'year_contributions_private.csv'):
                    model = 'price_forecast_fieldwork'
                else:
                    model=next((str(d[k]) for k in ('model','candidate','variant','training') if d.get(k) is not None),
                               'diagnostic; model not named in row')
                kind=('annual_error_row' if p.name=='annual_private.csv' and p.parent.name=='iowa_fieldwork_2022'
                      else 'incremental_comparison_row' if p.name=='incremental_private.csv' and p.parent.name=='iowa_incremental_value'
                      else ADDITIONAL_RESULTS.get(p.name,'diagnostic_row'))
                status=('skipped or omitted record; reason is in the saved row; not necessarily a failed model'
                        if kind=='skipped_or_omitted' else 'retrospective; not an untouched evaluation')
                add(i+1,model,kind,d,status)
    result=pd.DataFrame(rows)
    if result.empty: raise ValueError('No retained score or prediction artifacts available')
    if result.experiment_id.duplicated().any(): raise ValueError('Duplicate register ID')
    return result


def save_register(register, output):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    register.to_csv(out/'experiment_register_private.csv',index=False)
    notes=('This register indexes retained score, prediction, diagnostic, sensitivity '
       'and leave-one-training-year-out CSV filenames matching its defined patterns, '
       'plus specifically named skipped, omitted, paired, win-count, annual-error and '
       'older-year feature-term and year-contribution files, and the matched '
       'comparison annual incremental-error file. '
       'A row can be a reported score, a derived summary, or an individual '
       'retained prediction row. Prediction-row IDs use parsed-record order, '
       'not physical line numbers. Wide rows may describe multiple models and '
       'are not split into invented trials. A row is not '
       'necessarily a unique model. The same model can appear in both forms. '
       'Repeated evaluations and excluded-year sensitivities are kept, not collapsed. '
       'Skipped or omitted rows preserve their stated reason and do not necessarily indicate a failed model; '
       'an empty file is not evidence of failure. '
       'Purpose and inputs are reconstructed broad descriptions, not evidence of '
       'original motives; original reasons and exactly which outcomes were known at proposal time are '
       'not reliably recoverable; do not infer them from the registration date. '
       'Known seen years are minimum years asserted in linked study descriptions; '
       'they describe the retrospective investigation, not necessarily the original '
       'proposal of a baseline model. Unknown does not mean no years were seen, '
       'and a source assertion does not independently prove its timing. '
       'Other filenames, unrecorded attempts and unpublished decisions may be missing. '
       'Source SHA-256 checks the saved artifact bytes; result SHA-256 checks the '
       'canonical registered row. Neither establishes historical availability or '
       'when a model was proposed. All outcomes through 2026 were already inspected '
       'by the time of this reconstruction; none is a new untouched holdout. '
       '\n')
    (out/'REGISTER_README.txt').write_text(notes,encoding='utf-8')
    return notes
