from pathlib import Path
import math
import numpy as np
import pandas as pd

FIELDWORK_2022_OUT = Path('/kaggle/working/iowa_fieldwork_2022')

def investigate_fieldwork_2022(window_results):
    six = window_results['six']
    one = window_results['one']
    data = six['data'].copy().sort_values('year')
    if data.year.duplicated().any() or 2022 not in set(data.year):
        raise ValueError('Expected one complete 2022 annual feature row')
    current = data.loc[data.year.eq(2022)].iloc[0]
    cutoff = str(current.as_of)[:10]
    train = data[(data.year < 2022) & (data.target_filed.astype(str).str[:10] <= cutoff)].copy()
    if len(train) < FW_MIN_TRAIN:
        raise ValueError('2022 lacks the required earlier complete training years')
    features=['forecast_mean_p12_pct', 'fieldwork_mean_days']
    raw=train[features].to_numpy(dtype=float)
    means=raw.mean(axis=0); sd=raw.std(axis=0, ddof=0)
    if not np.isfinite(raw).all() or (sd <= 1e-12).any():
        raise ValueError('Invalid or constant training features')
    z=(raw-means)/sd
    z_test=(current[features].to_numpy(dtype=float)-means)/sd
    if (train[['target_q2_usd','baseline']] <= 0).any().any():
        raise ValueError('Invalid revenue or baseline')
    residual=np.log(train.target_q2_usd.to_numpy(float)/train.baseline.to_numpy(float))
    coefficients=np.linalg.solve(z.T@z+FW_RIDGE*np.eye(2), z.T@residual)
    terms=coefficients*z_test
    recon=float(current.baseline)*math.exp(float(terms.sum()))
    predictions=six['predictions'].copy()
    for label in ('price_only','price_forecast','price_forecast_fieldwork'):
        if predictions.query('model == @label').year.duplicated().any():
            raise ValueError('Duplicate prediction years')
    p=predictions.pivot(index='year',columns='model',values='prediction_usd').reset_index()
    actual=predictions[['year','actual_usd']].drop_duplicates()
    if actual.year.duplicated().any():
        raise ValueError('Conflicting actuals')
    p=p.merge(actual,on='year',validate='one_to_one')
    original=one['predictions'].query("model == 'price_forecast_fieldwork'")[['year','prediction_usd']].rename(columns={'prediction_usd':'one_week'})
    p=p.merge(original,on='year',validate='one_to_one').sort_values('year')
    if not np.isclose(recon,float(p.loc[p.year.eq(2022),'price_forecast_fieldwork'].iloc[0]),rtol=1e-10):
        raise AssertionError('Reconstructed 2022 prediction differs from original')
    if (p[['price_only','price_forecast','price_forecast_fieldwork']] <= 0).any().any():
        raise ValueError('Predictions must be positive for the geometric blend')
    p['half_increment']=np.sqrt(p.price_forecast*p.price_forecast_fieldwork)
    labels={'price_only':'price_only','forecast_only':'price_forecast',
            'one_week':'one_week','full_six_week':'price_forecast_fieldwork','half_increment':'half_increment'}
    for label,col in labels.items():
        p[label+'_absolute_error_usd']=(p[col]-p.actual_usd).abs()
    out=[]
    for omit in [None]+p.year.astype(int).tolist():
        subset=p if omit is None else p[p.year.ne(omit)]
        for label in labels:
            out.append({'omitted_year':'none' if omit is None else str(omit),
                        'model':label,'years':len(subset),
                        'mae_usd':float(subset[label+'_absolute_error_usd'].mean()),
                        'wins_vs_forecast_only':int((subset[label+'_absolute_error_usd']<subset.forecast_only_absolute_error_usd).sum())})
    scores=pd.DataFrame(out)
    corr=float(np.corrcoef(z[:,0],z[:,1])[0,1])
    diag=pd.DataFrame({'feature':features,'train_mean':means,'train_sd':sd,
        'train_min':raw.min(axis=0),'train_max':raw.max(axis=0),
        'value_2022':current[features].to_numpy(dtype=float),'z_2022':z_test,
        'coefficient':coefficients,'log_contribution':terms})
    r=p[p.year.eq(2022)].iloc[0]
    report=f'''# Exploratory 2022 six-week fieldwork error investigation

The price-only prediction was ${r.price_only/1e6:,.1f}m against actual ${r.actual_usd/1e6:,.1f}m. The six-week fieldwork mean was {float(current.fieldwork_mean_days):.2f} days/week, versus training mean {means[1]:.2f} (SD {sd[1]:.2f}, z={z_test[1]:.2f}). The fitted fieldwork coefficient {coefficients[1]:+.4f} contributed {terms[1]:+.4f} to the log adjustment; the rain-forecast contribution was {terms[0]:+.4f}. The combined factor was {math.exp(float(terms.sum())):.4f}, producing ${recon/1e6:,.1f}m, absolute error ${r.full_six_week_absolute_error_usd/1e6:,.1f}m. Earlier train years: {', '.join(map(str,train.year.astype(int)))}. The two training feature correlations were {corr:.2f}; six training observations cannot support strong claims about a stable relation.

The one-week comparison predicted ${r.one_week/1e6:,.1f}m, error ${r.one_week_absolute_error_usd/1e6:,.1f}m. Forecast-only error was ${r.forecast_only_absolute_error_usd/1e6:,.1f}m. Halving the incremental log adjustment is an exploratory sensitivity, not an untouched test: in 2022 its error was ${r.half_increment_absolute_error_usd/1e6:,.1f}m. Annual errors, matched MAE, wins against forecast-only and leave-one-test-year-out sensitivity are retained in the tables. The adjustment was chosen after inspecting history; no new prospective claim follows. The frozen 2027 protocol is unchanged.

This reconstructs how the model made its prediction; it does not establish why company revenue changed or why a farmer bought fertilizer. Fieldwork is a statewide opportunity measure for all activities, not farmer orders. Some source observations were recorded from later previous-week releases or contemporaneous reproductions and may have been revised; release dates and sources are in the earlier coverage table. CF's later Q2 disclosures are contextual only and must not be used in a May 31 prediction: https://www.sec.gov/Archives/edgar/data/1324404/000132440422000022/cf-20220630.htm
'''
    FIELDWORK_2022_OUT.mkdir(parents=True,exist_ok=True)
    p.to_csv(FIELDWORK_2022_OUT/'annual_private.csv',index=False)
    scores.to_csv(FIELDWORK_2022_OUT/'scores_private.csv',index=False)
    diag.to_csv(FIELDWORK_2022_OUT/'diagnostic_private.csv',index=False)
    (FIELDWORK_2022_OUT/'REPORT.md').write_text(report,encoding='utf8')
    print(report)
    display(diag.round(4))
    display(scores[scores.omitted_year.eq('none')].assign(mae_musd=lambda d:d.mae_usd/1e6).drop(columns='mae_usd').round(2))
    display((p.set_index('year')[[f'{n}_absolute_error_usd' for n in labels]]/1e6).round(1))
    return {'diagnostic':diag,'scores':scores,'predictions':p,'report':report}

FIELDWORK_2022_RESULTS = investigate_fieldwork_2022(FIELDWORK_WINDOW_RESULTS)
