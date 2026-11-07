from pathlib import Path
import math
import numpy as np
import pandas as pd

def investigate_forecast_2025(forecast_results):
    p = forecast_results['predictions'].copy().sort_values('year')
    data = forecast_results['data'].copy()
    if p.empty or p.year.duplicated().any() or not (p.year == 2025).any():
        raise ValueError('Need unique matched forecast predictions including 2025')
    required = ['price_only_usd', 'price_forecast_usd', 'actual_usd', 'feature_mean_p12_pct',
                'train_feature_mean_pct', 'train_feature_sd_pct', 'coefficient']
    if not set(required).issubset(p.columns) or not np.isfinite(p[required].to_numpy(float)).all():
        raise ValueError('Missing or nonfinite model quantities')
    if (p[['price_only_usd', 'price_forecast_usd', 'actual_usd']] <= 0).any().any():
        raise ValueError('Predictions and actual values must be positive')
    p['half_forecast_usd'] = np.sqrt(p.price_only_usd * p.price_forecast_usd)
    cap_predictions, prior_min, prior_max = [], [], []
    for row in p.itertuples():
        train = data[(data.year < row.year) & (data.target_filed.astype(str).str[:10] <= str(row.year)+'-05-31')]
        if len(train) != row.train_years:
            raise ValueError('Training cohort differs from the original forecast')
        low, high = float(train.forecast_mean_p12_pct.min()), float(train.forecast_mean_p12_pct.max())
        x = np.clip(row.feature_mean_p12_pct, low, high)
        z = (x - row.train_feature_mean_pct) / row.train_feature_sd_pct
        cap_predictions.append(row.price_only_usd * math.exp(row.coefficient * z))
        prior_min.append(low); prior_max.append(high)
    p['capped_forecast_usd'] = cap_predictions
    p['prior_feature_min_pct'], p['prior_feature_max_pct'] = prior_min, prior_max
    p['feature_z'] = (p.feature_mean_p12_pct-p.train_feature_mean_pct)/p.train_feature_sd_pct
    models = {'price_only':'price_only_usd','full':'price_forecast_usd',
              'half':'half_forecast_usd','capped':'capped_forecast_usd'}
    for label, col in models.items():
        p[label+'_error_usd'] = (p[col]-p.actual_usd).abs()
    scores=[]
    for omitted in [None, 2023, 2026]:
        group = p if omitted is None else p[p.year != omitted]
        if omitted is not None and len(group) != len(p)-1:
            continue
        for label in models:
            scores.append({'omitted_year': 'none' if omitted is None else str(omitted),
                           'model':label,'years':len(group),'mae_musd':group[label+'_error_usd'].mean()/1e6,
                           'wins_vs_price':int((group[label+'_error_usd'] < group.price_only_error_usd).sum())})
    score = pd.DataFrame(scores)
    out = Path('/kaggle/working/iowa_forecast_diagnostic')
    out.mkdir(parents=True, exist_ok=True)
    p.to_csv(out/'yearly_diagnostic_private.csv', index=False)
    score.to_csv(out/'scores_private.csv', index=False)
    r = p[p.year == 2025].iloc[0]
    report = f'''# 2025 forecast error: exploratory investigation

The price-only prediction was ${r.price_only_usd/1e6:,.1f}m against actual revenue ${r.actual_usd/1e6:,.1f}m. The annual forecast feature was {r.feature_mean_p12_pct:.2f}%, versus prior-years' mean {r.train_feature_mean_pct:.2f}% (z={r.feature_z:.2f}, prior range {r.prior_feature_min_pct:.2f}–{r.prior_feature_max_pct:.2f}%). Its fitted coefficient was {r.coefficient:.4f}; the full model multiplied the price prediction by {r.price_forecast_usd/r.price_only_usd:.3f}, moving it to ${r.price_forecast_usd/1e6:,.1f}m and worsening absolute error from ${r.price_only_error_usd/1e6:,.1f}m to ${r.full_error_usd/1e6:,.1f}m.

Two post-hoc sensitivities are compared with the same price-only and full predictions in the same five rolling years: half the log adjustment and, separately, clipping the forecast feature to the range of the training years. They use the original coefficients and only earlier-year information, but were selected after inspecting these results. They are not independent tests or proven improvements. See scores_private.csv for MAE, wins, and sensitivity to excluding a test year; see yearly_diagnostic_private.csv for annual results and training-range diagnostics. The frozen 2027 protocol is unchanged.

The model used precipitation probability at three Iowa stations, not rain amount or actual farmer orders. This statistical diagnosis does not identify the cause of company revenue or establish that an individual farmer used the forecast. Q2 company disclosures are after the May 31 prediction cutoff and are explanations only, never inputs.

Sources: https://www.sec.gov/Archives/edgar/data/1324404/000132440425000024/cf-20250630.htm ; https://www.nass.usda.gov/Statistics_by_State/Iowa/Publications/Crop_Progress_%26_Condition/2025/IA-Crop-Progress-05-12-25.pdf ; https://www.sec.gov/Archives/edgar/data/1324404/000132440425000013/cf-05072025_ex991xearnings.htm
'''
    (out/'REPORT.md').write_text(report, encoding='utf8')
    print(report)
    display(score.round(2))
    display((p[['year','price_only_error_usd','full_error_usd','half_error_usd','capped_error_usd']].set_index('year')/1e6).round(1))
    return {'predictions':p, 'scores':score, 'report':report}

DIAGNOSTIC_2025 = investigate_forecast_2025(FORECAST_RESULTS)
