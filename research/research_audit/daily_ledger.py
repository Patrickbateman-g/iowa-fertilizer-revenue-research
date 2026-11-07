from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math

import replay

STATIONS = ('KAMW', 'KCID', 'KSUX')


def _attach_definition(record):
    for key, expected in replay.FORECAST_DEFINITION.items():
        if key in record and record[key] is not None:
            value=record[key]
            if (isinstance(value, bool) or value != expected):
                raise ValueError('Existing forecast definition conflicts with GFS P12 research feature')
        record[key]=expected


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value)) or value == ''


def _utc(value):
    if not isinstance(value, str):
        return None
    try:
        v = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if v.tzinfo is None: v = v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _records(value):
    if hasattr(value, 'to_dict'):
        value = value.to_dict('records')
    if not isinstance(value, (list, tuple)):
        raise ValueError('Expected a manifest list or DataFrame')
    return list(value)


def _year(value):
    if isinstance(value, bool): raise ValueError('Invalid year')
    try:
        n = int(value)
        if float(value) != n or not 2010 <= n <= 2026: raise ValueError
        return n
    except (TypeError, ValueError, OverflowError):
        raise ValueError('Invalid forecast year') from None


def _match_manifest(rebuilt, manifest):
    for key in ('expected_days','valid_days','duplicate_rows','conflicting_days','invalid_rows'):
        value=manifest.get(key)
        if _missing(value) or isinstance(value, bool): return False
        try:
            if float(value) != rebuilt[key]: return False
        except (TypeError, ValueError): return False
    usable=manifest.get('usable')
    if not isinstance(usable, bool):
        if str(usable) not in ('True','False'): return False
        usable=str(usable)=='True'
    if usable != rebuilt['usable']: return False
    actual=manifest.get('station_mean_p12_pct')
    wanted=rebuilt['station_mean_p12_pct']
    if wanted is None: return _missing(actual)
    try: return not isinstance(actual,bool) and math.isclose(float(actual),wanted,rel_tol=1e-12,abs_tol=1e-10)
    except (TypeError,ValueError): return False


def attach_daily_forecasts(project_ledger, audit):
    if not isinstance(project_ledger, list) or not isinstance(audit, dict) or not {'ledger','manifest'} <= set(audit):
        raise ValueError('Missing project ledger or selected forecast audit')
    if not isinstance(audit['ledger'], list): raise ValueError('Invalid selected row ledger')
    result=deepcopy(project_ledger)
    yearly={}; stations={}
    for record in result:
        replay.validate_record(record)
        s=record['series']; period=str(record['period'])
        if s=='forecast_probability':
            key=_year(period)
            if key in yearly: raise ValueError('Ambiguous annual weather record')
            yearly[key]=record
        elif s.startswith('forecast_station_'):
            key=(_year(period),s.removeprefix('forecast_station_'))
            if key[1] not in STATIONS or key in stations: raise ValueError('Ambiguous station record')
            stations[key]=record
    manifests={}
    for m in _records(audit['manifest']):
        if not isinstance(m,dict): raise ValueError('Invalid forecast manifest row')
        key=(_year(m.get('year')),m.get('station'))
        if key[1] not in STATIONS or key in manifests: raise ValueError('Duplicate or invalid station manifest')
        manifests[key]=m
    groups=defaultdict(list)
    for wrapped in audit['ledger']:
        if not isinstance(wrapped,dict): raise ValueError('Invalid saved forecast row')
        key=(_year(wrapped.get('year')),wrapped.get('station'))
        m=manifests.get(key)
        if m is None or m.get('status')!='retrieved': raise ValueError('Saved row without retrieved source manifest')
        if wrapped.get('source_url')!=m.get('source_url') or wrapped.get('retrieved_at_utc')!=m.get('retrieved_at_utc'):
            raise ValueError('Saved row attribution differs from manifest')
        if not _missing(wrapped.get('publication_time_utc')) or not _missing(wrapped.get('availability_time_utc')):
            raise ValueError('This adapter accepts only unknown historical forecast times')
        if wrapped.get('source_version') not in (None,'','unknown'):
            raise ValueError('This adapter accepts only unknown forecast version')
        if 'original_source_record' in wrapped:
            raw=wrapped['original_source_record']; expected=wrapped.get('original_record_sha256')
        elif 'invalid_non_dict' in wrapped:
            raw=wrapped['invalid_non_dict']; expected=wrapped.get('invalid_non_dict_sha256')
        else: raise ValueError('Selected source record missing')
        if _digest(raw)!=expected: raise ValueError('Selected source record checksum mismatch')
        groups[key].append((raw,expected))
    diagnostics=[]; additions=[]
    expected_keys=set(stations) | {(year,station) for year in yearly for station in STATIONS}
    for year,station in sorted(expected_keys-set(manifests)):
        diagnostics.append({'year':year,'station':station,'status':'manifest_missing',
                            'selected_rows':0,'missing_dates':[], 'conflicting_dates':[],
                            'valid_days':0})
    for year,station in sorted(manifests):
        key=(year,station); m=manifests[key]
        info={'year':year,'station':station,'status':None,'selected_rows':len(groups[key]),
              'missing_dates':[], 'conflicting_dates':[], 'valid_days':0}
        if m.get('status')!='retrieved':
            info['status']='source_not_retrieved';diagnostics.append(info);continue
        if (not _missing(m.get('publication_time_utc')) or not _missing(m.get('availability_time_utc')) or
                m.get('source_version') not in (None,'','unknown')):
            raise ValueError('Historical timing/version must remain unknown')
        source=m.get('source_url'); retrieved=m.get('retrieved_at_utc')
        if not isinstance(source,str) or not source.startswith('https://') or _utc(retrieved) is None:
            raise ValueError('Missing source URL or retrieval time')
        expected=[date(year,4,1)+timedelta(days=i) for i in range(50)]
        values=defaultdict(set); hashes=defaultdict(lambda:defaultdict(list)); duplicates=invalid=0
        for raw,digest in groups[key]:
            if not isinstance(raw,dict): invalid+=1; continue
            run=_utc(raw.get('runtime')); valid=_utc(raw.get('ftime'))
            if run is None or valid is None or run.time()!=datetime.min.time() or run.date() not in expected or valid!=run+timedelta(hours=24):
                invalid+=1; continue
            if raw.get('station')!=station or raw.get('model')!='GFS': invalid+=1;continue
            v=raw.get('p12')
            if v is None or v in ('','M','null'): continue
            if isinstance(v,bool): invalid+=1;continue
            try: v=float(v)
            except (TypeError,ValueError): invalid+=1;continue
            if not math.isfinite(v) or not 0<=v<=100: invalid+=1;continue
            day=run.date()
            if v in values[day]: duplicates+=1
            values[day].add(v); hashes[day][v].append(digest)
        conflicts=[d for d in expected if len(values[d])>1]
        clean={d:next(iter(values[d])) for d in expected if len(values[d])==1}
        rebuilt={'expected_days':50,'valid_days':len(clean),'duplicate_rows':duplicates,
                 'conflicting_days':len(conflicts),'invalid_rows':invalid,
                 'station_mean_p12_pct':sum(clean.values())/len(clean) if clean else None,
                 'usable':len(clean)>=40 and invalid==0}
        info.update(valid_days=len(clean),missing_dates=[d.isoformat() for d in expected if d not in clean],
                    conflicting_dates=[d.isoformat() for d in conflicts])
        if not _match_manifest(rebuilt,m): info['status']='manifest_mismatch'
        elif not rebuilt['usable']: info['status']='unusable_station'
        elif key not in stations: info['status']='station_summary_missing'
        elif (stations[key]['evidence_type']!='unverified' or
              stations[key]['source_version']!='unknown' or not _missing(stations[key]['available_at'])):
            raise ValueError('Existing station evidence is not unverified with unknown timing/version')
        elif (stations[key]['source_url']!=source or not math.isclose(float(stations[key]['value']),rebuilt['station_mean_p12_pct'],rel_tol=0,abs_tol=1e-5)):
            info['status']='station_summary_mismatch'
        else:
            entry=stations[key]
            _attach_definition(entry)
            entry['constituent_keys']=[['forecast_daily_'+station,d.isoformat()] for d in sorted(clean)]
            entry['derivation']='mean'
            entry['note']=str(entry.get('note',''))+' Selected daily keys attached; original historical availability and source completeness remain unknown.'
            for d,v in sorted(clean.items()):
                run=datetime(d.year,d.month,d.day,tzinfo=timezone.utc)
                r={'series':'forecast_daily_'+station,'period':d.isoformat(),'value':v,
                   'observation_date':d.isoformat(),'source_url':source,'source_version':'unknown',
                   'available_at':None,'retrieved_at':retrieved,'evidence_type':'unverified',
                   'station':station,'model':'GFS','model_run_utc':run.isoformat(),
                   'valid_at_utc':(run+timedelta(hours=24)).isoformat(),
                   'source_row_sha256':sorted(hashes[d][v]),'selection_completeness':'unknown',
                   'note':'Selected later archive rows; model run is not publication time.'}
                r.update(replay.FORECAST_DEFINITION)
                replay.validate_record(r);additions.append(r)
            info['status']='attached_unverified'
        diagnostics.append(info)
    for year,annual in yearly.items():
        if (annual['evidence_type']!='unverified' or annual['source_version']!='unknown' or
                not _missing(annual['available_at'])):
            raise ValueError('Existing annual weather evidence is not unverified with unknown timing/version')
        group=[x for x in diagnostics if x['year']==year]
        if len(group)==3 and all(x['status']=='attached_unverified' for x in group):
            avg=sum(float(stations[(year,s)]['value']) for s in STATIONS)/3
            if not math.isclose(float(annual['value']),avg,rel_tol=0,abs_tol=1e-5):
                raise ValueError('Recorded annual forecast does not match station summaries')
            _attach_definition(annual)
    identities={(r['series'],str(r['period']),r['source_version']) for r in result}
    for item in additions:
        ident=(item['series'],item['period'],item['source_version'])
        if ident in identities: raise ValueError('Daily replay record already present')
        identities.add(ident);result.append(item)
    return result, diagnostics
