from __future__ import annotations

import numpy as np
import pandas as pd

import comparison


def prepare_calibration_predictions(annual, shipment_records, first_year, min_train=6):
    if isinstance(first_year, bool) or not isinstance(first_year, (int, np.integer)) or not 2016 <= first_year <= 2026:
        raise ValueError('first_year must be an integer within 2016–2026')
    if isinstance(min_train, bool) or not isinstance(min_train, int) or min_train < 1:
        raise ValueError('Invalid minimum training size')
    data=comparison.validate_annual(annual)
    shipment_records=list(shipment_records)
    vols=comparison.validate_shipments(shipment_records)
    rows=[]
    folds=[]
    for year in range(2016, int(first_year)):
        missing=[]
        if not data.year.eq(year).any(): missing.append('annual')
        if year not in vols: missing.append('shipments')
        cutoff=f'{year}-05-31'
        train=data[(data.year < year) & (data.target_filed <= cutoff)].sort_values('year')
        rec={'year':year,'status':None,'reason':None,'training_years':len(train)}
        if missing:
            rec.update(status='skipped',reason='missing_'+'_and_'.join(missing))
        elif len(train) < min_train:
            rec.update(status='skipped',reason='insufficient_training')
        elif float(train.forecast_mean_p12_pct.std(ddof=0)) <= 1e-12:
            rec.update(status='skipped',reason='no_training_weather_variation')
        else:
            fold=comparison.compare_incremental(data, shipment_records, [year], min_train=min_train)
            folds.append(fold['predictions'])
            rec.update(status='prepared',reason='requires_separate_replay_audit')
        rows.append(rec)
    predictions=pd.concat(folds,ignore_index=True) if folds else pd.DataFrame()
    inventory=pd.DataFrame(rows,columns=['year','status','reason','training_years'])
    return predictions, inventory


def used_calibration_years(requested_ranges):
    needed={'year','calibration_years','calibration_count'}
    if not isinstance(requested_ranges,pd.DataFrame) or requested_ranges.empty or not needed.issubset(requested_ranges):
        raise ValueError('Missing requested range calibration years')
    result=set()
    for r in requested_ranges.itertuples(index=False):
        text=r.calibration_years
        if text is None or (isinstance(text,float) and np.isnan(text)) or text=='':
            if r.calibration_count != 0:
                raise ValueError('Calibration count does not match year list')
            continue
        if not isinstance(text,str): raise ValueError('Invalid calibration years')
        try:
            years=[int(t) for t in text.split(',')]
        except ValueError as exc:
            raise ValueError('Invalid calibration years') from exc
        if len(years)!=len(set(years)) or any(str(y)!=t.strip() or y>=int(r.year) for y,t in zip(years,text.split(','))):
            raise ValueError('Invalid or future calibration year')
        if r.calibration_count != len(years):
            raise ValueError('Calibration count does not match year list')
        result.update(years)
    return sorted(result)
