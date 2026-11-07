import datetime as dt
import math
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

FW_YEARS = range(2016, 2027)
FW_MIN_TRAIN = 6
FW_RIDGE = 1.0
FW_OUT = Path('/kaggle/working/iowa_fieldwork')


def _fw_date(value):
    if not isinstance(value, str):
        raise ValueError('Use ISO YYYY-MM-DD date strings')
    result = dt.date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError('Use ISO YYYY-MM-DD date strings')
    return result


def _fw_expected(year, weeks):
    if weeks not in (1, 6):
        raise ValueError('Only the predefined one-week or six-week scopes are supported')
    may20 = dt.date(year, 5, 20)
    latest_sunday = may20 - dt.timedelta(days=(may20.weekday() + 1) % 7)
    return [latest_sunday - dt.timedelta(days=7*i) for i in range(weeks-1, -1, -1)]


def summarize_fieldwork(snapshot, weeks=6):
    if weeks not in (1, 6) or not isinstance(snapshot, list):
        raise ValueError('Need a list of one-week or six-week summaries')
    records = {}
    for item in snapshot:
        if not isinstance(item, dict) or not {'year','week_end','release_date','days_suitable','source_url'}.issubset(item):
            raise ValueError('Weekly snapshot missing required fields')
        year = item['year']
        if isinstance(year, bool) or not isinstance(year, int) or year not in FW_YEARS:
            raise ValueError('Unexpected fieldwork year')
        end = _fw_date(item['week_end'])
        if end not in _fw_expected(year, weeks):
            raise ValueError('Week end does not match fixed annual window')
        key = (year, end)
        if key in records:
            raise ValueError('Duplicate weekly report')
        val, release, url = item['days_suitable'], item['release_date'], item['source_url']
        if val is None:
            if release is not None or url is not None:
                raise ValueError('An unavailable report needs null date and source')
            records[key] = None
            continue
        if isinstance(val, bool) or not isinstance(val, (int,float)) or not np.isfinite(val) or not 0 <= val <= 7:
            raise ValueError('Weekly days suitable must be finite and in [0,7]')
        release = _fw_date(release)
        if not end < release <= dt.date(year,5,31):
            raise ValueError('Report date must be after week end and by cutoff')
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or not parsed.hostname or not (parsed.hostname == 'usda.gov' or parsed.hostname.endswith('.usda.gov') or parsed.hostname in ('iowaagriculture.gov', 'publications.iowa.gov', 'k923.fm', 'whoradio.iheart.com', 'news.utcrops.com')):
            raise ValueError('Use an official USDA/Iowa or explicitly recognized secondary HTTPS source URL')
        records[key] = {'year':year, 'week_end':end.isoformat(), 'release_date':release.isoformat(),
                        'days_suitable':float(val), 'source_url':url, 'basis':item.get('basis', 'not_stated')}
    coverage, annual = [], []
    for year in FW_YEARS:
        expected = _fw_expected(year,weeks)
        selected = [records.get((year,end)) for end in expected]
        ok = all(x is not None for x in selected)
        for end,item in zip(expected,selected):
            coverage.append(item if item else {'year':year,'week_end':end.isoformat(),
                 'release_date':None,'days_suitable':None,'source_url':None})
        annual.append({'year':year, 'fieldwork_complete':ok, 'weeks_expected':weeks,
                       'weeks_present':sum(x is not None for x in selected),
                       'fieldwork_mean_days':float(np.mean([x['days_suitable'] for x in selected])) if ok else np.nan,
                       'first_week_end':expected[0].isoformat(),'last_week_end':expected[-1].isoformat()})
    return pd.DataFrame(annual), pd.DataFrame(coverage)


def run_fieldwork_research(forecast_results, snapshot, weeks=6):
    annual, coverage = summarize_fieldwork(snapshot, weeks=weeks)
    if not isinstance(forecast_results,dict) or 'data' not in forecast_results:
        raise ValueError('Run the existing forecast notebook cells first')
    base = forecast_results['data'].copy()
    fields = {'year','as_of','price_as_of','target_filed','target_q2_usd','prior_q2_usd',
              'price_ratio','forecast_mean_p12_pct'}
    if not fields.issubset(base.columns) or base.year.duplicated().any():
        raise ValueError('Expected unique forecast research base rows')
    for row in base.itertuples():
        cutoff = f'{int(row.year)}-05-31'
        if str(row.as_of)[:10] != cutoff or str(row.price_as_of)[:10] != cutoff or str(row.target_filed)[:10] <= cutoff:
            raise ValueError('Cutoff, price vintage, or target release inconsistency')
    data = base.merge(annual[annual.fieldwork_complete], on='year', how='inner', validate='one_to_one').sort_values('year')
    required = ['target_q2_usd','prior_q2_usd','price_ratio','forecast_mean_p12_pct','fieldwork_mean_days']
    for field in required:
        data[field] = pd.to_numeric(data[field], errors='raise')
        if not np.isfinite(data[field]).all():
            raise ValueError(f'Nonfinite data for {field}')
    if (data[['target_q2_usd','prior_q2_usd','price_ratio']] <= 0).any().any():
        raise ValueError('Sales and price inputs must be positive')
    data['baseline'] = data.prior_q2_usd * data.price_ratio
    models = {'price_only':[], 'price_forecast':['forecast_mean_p12_pct'],
              'price_fieldwork':['fieldwork_mean_days'],
              'price_forecast_fieldwork':['forecast_mean_p12_pct','fieldwork_mean_days']}
    results, skipped = [], []
    for row in data.itertuples():
        train = data[(data.year < row.year) & (data.target_filed.astype(str).str[:10] <= str(row.as_of)[:10])]
        if len(train) < FW_MIN_TRAIN:
            skipped.append({'year':int(row.year), 'reason':'fewer than six earlier complete years','train_years':len(train)})
            continue
        feature_names = ['forecast_mean_p12_pct','fieldwork_mean_days']
        x = train[feature_names].to_numpy(float)
        mu, sd = x.mean(axis=0), x.std(axis=0,ddof=0)
        if (sd <= 1e-12).any() or not np.isfinite(sd).all():
            skipped.append({'year':int(row.year), 'reason':'no training feature variation','train_years':len(train)})
            continue
        z = (x-mu)/sd
        z_test = (np.array([getattr(row,col) for col in feature_names],float)-mu)/sd
        response = np.log(train.target_q2_usd.to_numpy(float)/train.baseline.to_numpy(float))
        for label, cols in models.items():
            if not cols:
                pred = float(row.baseline)
            else:
                indexes = [feature_names.index(col) for col in cols]
                xx = z[:,indexes]
                beta = np.linalg.solve(xx.T@xx + FW_RIDGE*np.eye(len(cols)), xx.T@response)
                exponent = float(z_test[indexes]@beta)
                if not np.isfinite(exponent) or abs(exponent) > 700:
                    raise ValueError('Nonfinite or extreme forecast')
                pred = float(row.baseline)*math.exp(exponent)
            results.append({'year':int(row.year),'model':label,'train_years':len(train),
                            'prediction_usd':pred,'actual_usd':float(row.target_q2_usd),
                            'absolute_error_usd':abs(pred-float(row.target_q2_usd))})
    predictions = pd.DataFrame(results, columns=['year','model','train_years','prediction_usd','actual_usd','absolute_error_usd'])
    skipped = pd.DataFrame(skipped,columns=['year','reason','train_years'])
    if not predictions.empty:
        counts = predictions.groupby('model').year.apply(lambda x:tuple(sorted(x)))
        if len(set(counts)) != 1:
            raise AssertionError('Candidate test years differ')
        scores = predictions.groupby('model',as_index=False).agg(years=('year','size'),mae_usd=('absolute_error_usd','mean'))
    else:
        scores = pd.DataFrame(columns=['model','years','mae_usd'])
    fw_window = 'six-week average' if weeks == 6 else 'latest eligible single week'
    report = f'''# Iowa fieldwork: exploratory comparison

Fieldwork feature: {fw_window} of USDA weekly Iowa days suitable for fieldwork. The eligible Sunday week endings are on or before May 20; every source must be dated by May 31. All expected weeks must be present. Some input values may come from secondary reproductions or from the preceding-week column of a later report; source dates refer to the specific document used and previous-week entries may have been revised. See coverage_private.csv for provenance. A weekly opportunity for fieldwork does not establish fertilizer purchases. Source values and dates were entered from browser-visible sources and not independently verified by this notebook; original report availability and possible revisions remain uncertain.

The price-only baseline and three alternatives (forecast, fieldwork, and both) use the same eligible years. Each model uses the same earlier complete training cohort, earlier filed targets, training-only scaling, a fixed ridge penalty of one, and an intercept fixed at zero for log(actual/baseline). At least six earlier years are required. The 2-feature model may overfit such a small sample. All outcomes through 2026 were already inspected; the feature and model choices are exploratory and selected after seeing prior results. None is a new independent holdout. The frozen 2027 protocol is unchanged.

Complete fieldwork years: {int(annual.fieldwork_complete.sum())}/11. Joined forecast, price and fieldwork years: {len(data)}. Evaluation years: {predictions.year.nunique() if not predictions.empty else 0}. See coverage_private.csv, annual_private.csv, predictions_private.csv, scores_private.csv and skipped_private.csv.
'''
    FW_OUT.mkdir(parents=True,exist_ok=True)
    coverage.to_csv(FW_OUT/'coverage_private.csv',index=False)
    annual.to_csv(FW_OUT/'annual_private.csv',index=False)
    predictions.to_csv(FW_OUT/'predictions_private.csv',index=False)
    scores.to_csv(FW_OUT/'scores_private.csv',index=False)
    skipped.to_csv(FW_OUT/'skipped_private.csv',index=False)
    (FW_OUT/'REPORT.md').write_text(report,encoding='utf8')
    print(report)
    return {'annual':annual,'coverage':coverage,'data':data,'predictions':predictions,'scores':scores,'skipped':skipped,'report':report}


from pathlib import Path
import numpy as np
import pandas as pd

def compare_fieldwork_windows(forecast_results, six_week_snapshot):
    global FW_OUT
    previous_out = FW_OUT
    root = Path('/kaggle/working/iowa_fieldwork_six')
    try:
        FW_OUT = root/'six_weeks'
        six = run_fieldwork_research(forecast_results, six_week_snapshot, weeks=6)
        complete = set(six['annual'].loc[six['annual'].fieldwork_complete, 'year'].astype(int))
        one_snapshot = [x for x in six_week_snapshot if x['year'] in complete
                        and _fw_date(x['week_end']) == _fw_expected(x['year'], 1)[0]]
        FW_OUT = root/'one_week_matched'
        one = run_fieldwork_research(forecast_results, one_snapshot, weeks=1)
    finally:
        FW_OUT = previous_out
    left = one['predictions'].copy()
    right = six['predictions'].copy()
    columns = ['year','model','train_years','actual_usd']
    if left.empty or right.empty:
        print('Insufficient matched complete years for a comparison.')
        return {'one':one, 'six':six, 'comparison':pd.DataFrame(), 'scores':pd.DataFrame()}
    joined=left.merge(right,on=columns,how='outer',suffixes=('_one','_six'),indicator=True,validate='one_to_one')
    if not joined['_merge'].eq('both').all():
        raise ValueError('One- and six-week folds are not identical')
    for label in ('price_only','price_forecast'):
        q=joined[joined.model.eq(label)]
        if not np.allclose(q.prediction_usd_one,q.prediction_usd_six,rtol=1e-10,atol=0):
            raise ValueError('Matched baseline predictions disagree')
    combo=joined[joined.model.eq('price_forecast_fieldwork')].copy()
    combo['six_minus_one_abs_error_usd']=combo.absolute_error_usd_six-combo.absolute_error_usd_one
    summaries=[]
    for name, dataset, window in [('price_only',one,'none'),('price_forecast',one,'none'),
        ('price_fieldwork',one,'one'),('price_fieldwork',six,'six'),
        ('price_forecast_fieldwork',one,'one'),('price_forecast_fieldwork',six,'six')]:
        q=dataset['predictions'].query('model == @name')
        summaries.append({'model':name,'fieldwork_window':window,'years':len(q),
                          'mae_musd':q.absolute_error_usd.mean()/1e6})
    scores=pd.DataFrame(summaries)
    root.mkdir(parents=True,exist_ok=True)
    joined.to_csv(root/'matched_predictions_private.csv',index=False)
    scores.to_csv(root/'matched_scores_private.csv',index=False)
    n=len(combo)
    wins=int((combo.six_minus_one_abs_error_usd<0).sum())
    report=f'''# Six-week Iowa fieldwork comparison

The six-week feature averages statewide days suitable for fieldwork over the six consecutive weeks ending on or before May 20. One-week features use the final week of the same six. Reports must be dated by May 31. Some entries are previous-week values from a later dated report and can incorporate revisions available at that time. Some reports are hosted by a university or reproduced by news outlets; the stored source date is that of the document actually used. Report contents and release dates were manually transcribed and not independently reverified by the notebook.

Only years with all six reports and matching forecast and price data enter either model; both windows use identical expanding training folds, training-only scaling, six prior observations minimum and fixed ridge penalty. Joined years: {len(six['data'])}; comparable test years: {n}. The six-week price+forecast+fieldwork model had lower absolute error than the matched one-week model in {wins} of {n} years. Full results are stored alongside source dates, coverage, and missing weeks.

This is a retrospective research comparison after prior years were inspected, with very few annual examples. It does not establish causal effects or prove future improvement. Fieldwork suitability is not farmer orders and covers all farm activities. The frozen 2027 forecast is unchanged.
'''
    (root/'REPORT.md').write_text(report,encoding='utf8')
    print(report)
    display(scores.round(2))
    display((combo[['year','absolute_error_usd_one','absolute_error_usd_six','six_minus_one_abs_error_usd']].set_index('year')/1e6).round(1))
    return {'one':one,'six':six,'comparison':joined,'scores':scores,'report':report}

FIELDWORK_WINDOW_RESULTS = compare_fieldwork_windows(FORECAST_RESULTS, FIELDWORK_SIX_WEEKS)
