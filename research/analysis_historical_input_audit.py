
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import time

import pandas as pd

AUDIT_FORECAST_ENDPOINT = 'https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py'
AUDIT_FORECAST_STATIONS = ('KAMW','KCID','KSUX')
AUDIT_FORECAST_YEARS = tuple(range(2010,2027))
AUDIT_FORECAST_FIELDS = ('station','model','runtime','ftime','p12')
AUDIT_FORECAST_OUT = Path('/kaggle/working/iowa_forecast_input_audit')

def _audit_canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)

def _audit_hash(value):
    return hashlib.sha256(_audit_canonical(value).encode('utf-8')).hexdigest()

def _audit_utc(value):
    try:
        t=pd.Timestamp(value)
        if pd.isna(t): return None
        return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
    except (TypeError, ValueError, OverflowError):
        return None

def _audit_forecast_url(station,year):
    if station not in AUDIT_FORECAST_STATIONS or year not in AUDIT_FORECAST_YEARS:
        raise ValueError('Unknown station or year')
    params={'station':station,'model':'GFS','sts':f'{year}-04-01T00:00Z',
            'ets':f'{year}-05-21T00:00Z','format':'json'}
    return AUDIT_FORECAST_ENDPOINT+'?'+urlencode(params)

def audit_forecast_rows(records, station, year):
    if not isinstance(records,list): raise ValueError('Expected JSON row array')
    expected=set(pd.date_range(f'{year}-04-01',f'{year}-05-20',freq='D',tz='UTC'))
    daily=[]; values={}; invalid=0; duplicates=0
    for row in records:
        if not isinstance(row,dict):
            daily.append({'invalid_non_dict':row,'invalid_non_dict_sha256':_audit_hash(row)})
            invalid+=1
            continue
        runtime=_audit_utc(row.get('runtime')); valid=_audit_utc(row.get('ftime'))
        if runtime not in expected or valid is None or valid != runtime+pd.Timedelta(hours=24):
            continue
        daily.append({'original_source_record':row,'original_record_sha256':_audit_hash(row)})
        if row.get('station') != station or row.get('model') != 'GFS':
            invalid+=1
            continue
        value=row.get('p12')
        if value is None or value in ('','M','null'): continue
        if isinstance(value,bool): invalid+=1; continue
        try: value=float(value)
        except (ValueError,TypeError): invalid+=1; continue
        if not math.isfinite(value) or not 0<=value<=100: invalid+=1; continue
        if runtime in values:
            if value in values[runtime]: duplicates+=1
            values[runtime].add(value)
        else: values[runtime]={value}
    conflict=sum(len(x)>1 for x in values.values())
    clean=[next(iter(x)) for x in values.values() if len(x)==1]
    return daily,{'expected_days':50,'valid_days':len(clean),'duplicate_rows':duplicates,
                  'conflicting_days':conflict,'invalid_rows':invalid,
                  'station_mean_p12_pct':sum(clean)/len(clean) if clean else None,
                  'usable':len(clean)>=40 and invalid==0}

def verify_saved_forecast_ledger(path, manifests):
    groups={}
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if 'original_source_record' in row:
                raw=row['original_source_record']
                if _audit_hash(raw)!=row['original_record_sha256']:
                    raise AssertionError('Saved original-row checksum mismatch')
            else:
                raw=row['invalid_non_dict']
                if _audit_hash(raw)!=row['invalid_non_dict_sha256']:
                    raise AssertionError('Saved invalid-row checksum mismatch')
            groups.setdefault((row['year'],row['station']),[]).append(raw)
    retrieved={(m['year'],m['station']) for m in manifests if m['status']=='retrieved'}
    if not set(groups).issubset(retrieved): raise AssertionError('Unattributed saved row')
    for m in manifests:
        if m['status']!='retrieved': continue
        _, rebuilt=audit_forecast_rows(groups.get((m['year'],m['station']),[]),m['station'],m['year'])
        for key in ('expected_days','valid_days','duplicate_rows','conflicting_days','invalid_rows','usable'):
            if rebuilt[key]!=m[key]: raise AssertionError('Saved ledger failed to rebuild '+key)
        a,b=rebuilt['station_mean_p12_pct'],m['station_mean_p12_pct']
        if (a is None)!=(b is None) or (a is not None and not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)):
            raise AssertionError('Saved ledger failed to rebuild mean')
    return len(retrieved)

def audit_forecast_history(historical_snapshot, newer_snapshot, *, output_dir=AUDIT_FORECAST_OUT,
                           timeout_seconds=8, max_requests=51, max_seconds=100, fetch=None,
                           collected=None):
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    if timeout_seconds<=0 or not 0<=max_requests<=51 or max_seconds<=0:
        raise ValueError('Invalid network limits')
    snapshots={}
    for item in list(historical_snapshot)+list(newer_snapshot):
        key=(int(item['year']),str(item['station']))
        if key in snapshots: raise ValueError('Duplicate snapshot station/year')
        snapshots[key]=item
    expected={(y,s) for y in AUDIT_FORECAST_YEARS for s in AUDIT_FORECAST_STATIONS}
    if set(snapshots)!=expected:
        raise ValueError('Historical snapshot coverage differs from 2010–2026 station set')
    if collected is not None and (not isinstance(collected,dict) or not set(collected).issubset(expected)):
        raise ValueError('Invalid browser-collected station/year mapping')
    def http_fetch(url,timeout):
        req=Request(url,headers={'User-Agent':'IowaForecastResearch/1.0 public research','Accept':'application/json'})
        with urlopen(req,timeout=timeout) as response:
            return response.read()
    fetch=fetch or http_fetch
    ledger=[]; manifests=[]; start=time.monotonic(); attempts=0; stop_reason=None
    last_attempt=None
    for year in AUDIT_FORECAST_YEARS:
        for station in AUDIT_FORECAST_STATIONS:
            url=_audit_forecast_url(station,year); snapshot=snapshots[(year,station)]
            m={'year':year,'station':station,'source_url':url,'source_version':'unknown',
               'publication_time_utc':None,'availability_time_utc':None,
               'historical_availability':'unknown','retrieved_at_utc':None,
               'response_sha256':None,'parsed_response_canonical_sha256':None,
               'selected_payload_canonical_sha256':None,'records_scope':None,
               'selection_completeness':'unknown','retrieval_method':None,
               'status':'not_attempted','details':None,
               'prior_valid_days':snapshot.get('valid_days'),
               'prior_mean_p12_pct':snapshot.get('station_mean_p12_pct')}
            if collected is not None and (year,station) not in collected:
                m['details']='not present in browser-collected mapping'; manifests.append(m); continue
            if stop_reason is not None or attempts>=max_requests or time.monotonic()-start>=max_seconds:
                m['details']=stop_reason or 'request/time budget exhausted'; manifests.append(m); continue
            if collected is None and last_attempt is not None:
                pause=1.1-(time.monotonic()-last_attempt)
                if pause>0: time.sleep(pause)
            if time.monotonic()-start>=max_seconds:
                m['details']='request/time budget exhausted'; manifests.append(m); continue
            last_attempt=time.monotonic()
            attempts+=1
            try:
                if collected is None:
                    payload=fetch(url,timeout_seconds)
                    if not isinstance(payload,bytes): raise ValueError('Fetch did not return bytes')
                    if len(payload)>15_000_000: raise ValueError('Response exceeded byte limit')
                    records=json.loads(payload.decode('utf-8'))
                    retrieved=datetime.now(timezone.utc).isoformat()
                    m['response_sha256']=hashlib.sha256(payload).hexdigest()
                    m['retrieval_method']='direct_http'; m['records_scope']='complete_json_array'
                else:
                    item=collected[(year,station)]
                    if not isinstance(item,dict) or item.get('source_url')!=url:
                        raise ValueError('Collected source URL missing or does not match')
                    retrieved=str(item.get('retrieved_at_utc',''))
                    timestamp=_audit_utc(retrieved)
                    if timestamp is None or not retrieved.endswith(('Z','+00:00')):
                        raise ValueError('Collected retrieval time must be an explicit UTC timestamp')
                    if timestamp>pd.Timestamp.now(tz='UTC')+pd.Timedelta(minutes=1):
                        raise ValueError('Collected retrieval time is in the future')
                    records=item.get('records')
                    if not isinstance(records,list): raise ValueError('Collected response is not an array')
                    scope=item.get('records_scope')
                    if scope not in ('complete_json_array','selection_only'):
                        raise ValueError('Browser-collected record scope must be explicit')
                    if scope=='selection_only':
                        expected_times=set(pd.date_range(f'{year}-04-01',f'{year}-05-20',freq='D',tz='UTC'))
                        for source_row in records:
                            if not isinstance(source_row,dict): raise ValueError('Selected source row must be a dict')
                            rt=_audit_utc(source_row.get('runtime')); vt=_audit_utc(source_row.get('ftime'))
                            if rt not in expected_times or vt is None or vt!=rt+pd.Timedelta(hours=24):
                                raise ValueError('Selected source row lies outside fixed selection')
                        m['selected_payload_canonical_sha256']=_audit_hash(records)
                        m['details']='Browser preselection; completeness and full-response invalid counts are not independently verified'
                    else:
                        m['parsed_response_canonical_sha256']=_audit_hash(records)
                    m['records_scope']=scope; m['retrieval_method']='browser_parsed_array'
                kept,summary=audit_forecast_rows(records,station,year)
                m.update(summary); m['retrieved_at_utc']=retrieved
                m['status']='retrieved'
                previous=snapshot.get('station_mean_p12_pct')
                m['mean_difference_pct_points']=(summary['station_mean_p12_pct']-float(previous)
                    if summary['station_mean_p12_pct'] is not None and previous is not None else None)
                m['same_count_and_mean_within_1e5_pct_points']=(summary['valid_days']==snapshot.get('valid_days') and
                    (m['mean_difference_pct_points'] is not None and abs(m['mean_difference_pct_points'])<=1e-5))
                for item in kept:
                    ledger.append({'year':year,'station':station,'source_url':url,
                        'retrieved_at_utc':retrieved,'publication_time_utc':None,
                        'availability_time_utc':None,'source_version':'unknown',
                        'retrieval_method':m['retrieval_method'],'records_scope':m['records_scope'],**item})
            except (HTTPError,URLError,TimeoutError,OSError,ValueError,UnicodeError,json.JSONDecodeError) as exc:
                m['status']='fetch_or_parse_error'; m['details']=f'{type(exc).__name__}: {str(exc)[:200]}'
                if isinstance(exc,HTTPError):
                    if exc.code in (401,403,429) or exc.code>=500:
                        stop_reason='source access/throttle/server error; remaining requests were not attempted'
                elif collected is None and (isinstance(exc,(URLError,TimeoutError,OSError)) or attempts==1):
                    stop_reason='source network/first response failed; remaining requests were not attempted'
            manifests.append(m)
    for r in ledger:
        if 'original_source_record' in r:
            if _audit_hash(r['original_source_record'])!=r['original_record_sha256']:
                raise AssertionError('Original-row checksum mismatch')
        elif _audit_hash(r['invalid_non_dict'])!=r['invalid_non_dict_sha256']:
            raise AssertionError('Invalid-row checksum mismatch')
    with gzip.open(output/'forecast_daily_private.jsonl.gz','wt',encoding='utf-8') as f:
        for row in ledger: f.write(_audit_canonical(row)+'\n')
    verified=verify_saved_forecast_ledger(output/'forecast_daily_private.jsonl.gz',manifests)
    pd.DataFrame(manifests).to_csv(output/'forecast_sources_private.csv',index=False)
    successes=sum(x['status']=='retrieved' for x in manifests)
    report=(f'Forecast station-years retrieved: {successes}/{len(manifests)}; '
        f'selected source rows preserved: {len(ledger)}; rebuilt from saved ledger: {verified}. '
        'See status and snapshot comparison per station-year. '
        'Source version, actual publication and availability times are unknown. '
        'This is the current archive response, possibly revised; it does not prove historical availability. '
        'Model initialization is not publication time. Row hashes cover canonical JSON of each saved full original daily record; '
        'Direct response hashes identify response bytes but those full bytes are not retained. '
        'Browser-collected parsed arrays or selected-row payloads have a canonical JSON hash, not a response-byte hash. '
        'Browser selection completeness and invalid counts outside selected rows cannot be verified by this code. '
        'Snapshot comparison uses an explicit 1e-5 percentage-point tolerance; differences can reflect original rounding or revisions. '
        'The GFS query can return legacy AVN records; they are retained and counted invalid rather than silently accepted. '
        'Requests not attempted and failures are not missing weather observations. Frozen 2027 test unchanged.\n')
    (output/'REPORT.txt').write_text(report,encoding='utf-8')
    print(report)
    return {'ledger':ledger,'manifest':pd.DataFrame(manifests),'report':report}


from pathlib import Path
from urllib.parse import urlencode, urlsplit, parse_qs
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import csv
import datetime as dt
import hashlib
import io
import json
import math
import time

PRICE_AUDIT_OUT = Path('/kaggle/working/iowa_price_audit')
PRICE_AUDIT_SERIES = 'WPU065201'
PRICE_AUDIT_ENDPOINT = 'https://alfred.stlouisfed.org/graph/alfredgraph.csv'

class PriceObservationUnavailable(ValueError):
    pass

def _price_audit_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')

def _price_audit_date(value):
    if not isinstance(value,str): raise ValueError('Date must be an ISO string')
    value2=dt.date.fromisoformat(value)
    if value2.isoformat()!=value: raise ValueError('Date must be canonical ISO')
    return value

def _price_audit_record(value, year, observation, source, basis,
                        recorded_as_of=None, recorded_release_date=None):
    year=int(year); observation=_price_audit_date(str(observation))
    value=float(value)
    if not math.isfinite(value) or value<=0: raise ValueError('Invalid price')
    u=urlsplit(str(source))
    if u.scheme!='https' or not (u.hostname=='alfred.stlouisfed.org' or
                                  u.hostname=='bls.gov' or (u.hostname or '').endswith('.bls.gov')):
        raise ValueError('Unexpected price source')
    if recorded_as_of is not None: _price_audit_date(recorded_as_of)
    if recorded_release_date is not None: _price_audit_date(recorded_release_date)
    return {'forecast_year':year,'series_id':PRICE_AUDIT_SERIES,
            'observation_date':observation,'value':value,'source_url':str(source),
            'recorded_as_of_date':recorded_as_of,
            'source_release_date_from_snapshot':recorded_release_date,
            'publication_or_available_at_utc':None,
            'historical_retrieved_at_utc':None,
            'source_version':('ALFRED as-of date '+recorded_as_of if recorded_as_of else
                              'BLS reconstructed from dated releases; per-value version unverified'),
            'evidence_status':basis,
            'note':'Snapshot value; source content and time-of-day were not independently verified.'}

def price_audit_snapshot(snapshot, older_prices):
    if not isinstance(snapshot,dict) or not isinstance(snapshot.get('price_index'),list):
        raise ValueError('Missing snapshot price_index')
    if not isinstance(older_prices,list): raise ValueError('Missing historical price list')
    observations=[]; seen=set()
    for item in snapshot['price_index']:
        y=int(item['year']); cutoff=f'{y}-05-31'
        if y in seen: raise ValueError('Duplicate snapshot year')
        seen.add(y)
        if item['as_of']!=cutoff or item['vintage_label']!=cutoff:
            raise ValueError('Snapshot as-of label mismatch')
        for prefix, obs_year in [('previous',y-1),('current',y)]:
            obs=item[prefix+'_observation']
            if obs!=f'{obs_year}-04-01': raise ValueError('Wrong snapshot observation date')
            observations.append(_price_audit_record(item[prefix+'_april'],y,obs,item['source'],
                                'manual_alfred_asof_snapshot_not_independently_retrieved',cutoff))
    seen=set()
    for item in older_prices:
        y=int(item['year'])
        if y in seen: raise ValueError('Duplicate historical year')
        seen.add(y)
        for prefix, obs_year in [('previous',y-1),('current',y)]:
            release=_price_audit_date(item[prefix+'_release_date'])
            if dt.date.fromisoformat(release)>dt.date(y,5,31):
                raise ValueError('Historical release later than cutoff')
            observations.append(_price_audit_record(item[prefix+'_april'],y,f'{obs_year}-04-01',
                item[prefix+'_source_url'],'manual_bls_reconstruction_not_independently_retrieved',
                recorded_release_date=release))
    return observations

def _price_audit_url(year):
    y=int(year); cutoff=f'{y}-05-31'
    return PRICE_AUDIT_ENDPOINT+'?'+urlencode({'id':PRICE_AUDIT_SERIES,
        'vintage_date':cutoff,'cosd':f'{y-1}-04-01','coed':f'{y}-04-01'})

def parse_price_alfred_csv(raw,year):
    if len(raw)>500_000: raise ValueError('Response exceeds small audit limit')
    text=raw.decode('utf-8-sig',errors='strict')
    lines=list(csv.reader(io.StringIO(text)))
    expected=f'{PRICE_AUDIT_SERIES}_{int(year)}0531'
    if not lines or len(lines[0])!=2 or lines[0][0] not in ('observation_date','DATE','date') or lines[0][1]!=expected:
        raise ValueError('ALFRED response has unexpected column/vintage header')
    found={}
    for row in lines[1:]:
        if len(row)!=2: raise ValueError('Unexpected ALFRED CSV shape')
        date=_price_audit_date(row[0])
        if date in found: raise ValueError('Duplicate observation date')
        if row[1] in ('','.'): continue
        v=float(row[1])
        if not math.isfinite(v) or v<=0: raise ValueError('Invalid source value')
        found[date]=v
    dates=(f'{int(year)-1}-04-01',f'{int(year)}-04-01')
    if any(d not in found for d in dates): raise PriceObservationUnavailable('As-of source missing an April observation')
    return {d:found[d] for d in dates}

def run_price_audit(snapshot,older_prices,fetch_alfred=False,timeout=8):
    observations=price_audit_snapshot(snapshot,older_prices)
    now=_price_audit_now(); PRICE_AUDIT_OUT.mkdir(parents=True,exist_ok=True)
    reconciled=[]; records=[]; attempted=False; network_stopped=False
    years=sorted({r['forecast_year'] for r in observations})
    for y in years:
        url=_price_audit_url(y); recorded=[r for r in observations if r['forecast_year']==y]
        result={'forecast_year':y,'source_url':url,'requested_as_of_date':f'{y}-05-31',
                'retrieved_at_utc':None,'response_sha256':None,'response_file':None,
                'status':'not_requested','comparison':None,'error':None}
        if fetch_alfred and not network_stopped:
            attempted=True
            try:
                req=Request(url,headers={'User-Agent':'IowaFertilizerResearch/1.0 (source audit)',
                                         'Accept':'text/csv'})
                with urlopen(req,timeout=timeout) as response:
                    raw=response.read(500_001)
                retrieved=_price_audit_now(); parsed=parse_price_alfred_csv(raw,y)
                digest=hashlib.sha256(raw).hexdigest()
                dest=PRICE_AUDIT_OUT/f'alfred_{y}_asof_may31.csv'
                dest.write_bytes(raw)
                discrepancies=[]
                for original in recorded:
                    source_value=parsed.get(original['observation_date'])
                    if source_value is None or not math.isclose(original['value'],source_value,rel_tol=1e-9,abs_tol=1e-7):
                        discrepancies.append({'observation_date':original['observation_date'],
                            'snapshot_value':original['value'],'source_value':source_value})
                    records.append({'forecast_year':y,'series_id':PRICE_AUDIT_SERIES,
                        'observation_date':original['observation_date'],'value':source_value,
                        'source_url':url,'source_version':f'ALFRED as-of {y}-05-31',
                        'publication_or_available_at_utc':None,'retrieved_at_utc':retrieved,
                        'response_sha256':digest,'response_file':dest.name,
                        'evidence_status':'fresh_source_response_asof_header_checked',
                        'note':'Header identifies requested as-of date; individual release and intraday availability not supplied.'})
                result.update(status='retrieved',retrieved_at_utc=retrieved,response_sha256=digest,
                              response_file=dest.name,comparison='match' if not discrepancies else 'mismatch',
                              discrepancies=discrepancies)
            except HTTPError as exc:
                if exc.code==404:
                    result.update(status='source_vintage_unavailable',error=f'HTTP {exc.code}')
                else:
                    result.update(status='fetch_failed',error=f'HTTP {exc.code}')
                    network_stopped=True
            except PriceObservationUnavailable as exc:
                result.update(status='observation_unavailable',error=str(exc))
            except (OSError,URLError,TimeoutError) as exc:
                result.update(status='fetch_failed',error=f'{type(exc).__name__}: {str(exc)[:180]}')
                network_stopped=True
            except (UnicodeError,ValueError,csv.Error) as exc:
                result.update(status='unverifiable_response',error=f'{type(exc).__name__}: {str(exc)[:180]}')
                network_stopped=True
            time.sleep(0.2)
        elif fetch_alfred and network_stopped:
            result.update(status='not_attempted_after_failure')
        reconciled.append(result)
    def encode_record(record):
        canonical=json.dumps(record,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
        return {'record':record,'canonical_sha256':hashlib.sha256(canonical.encode('utf-8')).hexdigest()}
    ledger=[encode_record(r) for r in observations+records]
    (PRICE_AUDIT_OUT/'price_observations_private.jsonl').write_text(
        ''.join(json.dumps(r,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n' for r in ledger),encoding='utf8')
    (PRICE_AUDIT_OUT/'reconciliation_private.json').write_text(json.dumps(reconciled,indent=2),encoding='utf8')
    totals={'audit_created_at_utc':now,'series_id':PRICE_AUDIT_SERIES,
            'snapshot_observations':len(observations),'fresh_observations':len(records),
            'source_vintages_retrieved':sum(r['status']=='retrieved' for r in reconciled),
            'matching_vintages':sum(r['comparison']=='match' for r in reconciled),
            'mismatching_vintages':sum(r['comparison']=='mismatch' for r in reconciled),
            'fetch_requested':bool(fetch_alfred),'fetch_attempted':attempted,
            'fetch_failure':next((r for r in reconciled if r['status'] in ('fetch_failed','unverifiable_response')),None),
            'warning':'Checksums prove byte integrity only, not publication time, archive completeness or historical availability. Earlier BLS reconstructions are not an independent ALFRED source; they may legitimately differ. No existing predictions, datasets or 2027 protocol were changed.'}
    (PRICE_AUDIT_OUT/'manifest_private.json').write_text(json.dumps(totals,indent=2),encoding='utf8')
    print('Price audit:',json.dumps(totals,indent=2))
    return {'manifest':totals,'observations':observations,'fresh_observations':records,'reconciliation':reconciled}

def ingest_browser_price_evidence(pairs,snapshot):
    if not isinstance(pairs,list) or not isinstance(snapshot,dict):
        raise ValueError('Expected browser evidence list and existing snapshot')
    existing={int(r['year']):r for r in snapshot['price_index']}
    if len(existing)!=len(snapshot['price_index']): raise ValueError('Duplicate model snapshot year')
    seen=set(); rows=[]
    for p in pairs:
        if not isinstance(p,dict): raise ValueError('Invalid browser evidence pair')
        y=int(p['year']); cutoff=f'{y}-05-31'
        if y in seen: raise ValueError('Duplicate browser evidence year')
        seen.add(y)
        if y not in existing: raise ValueError('Browser vintage has no saved model counterpart')
        if existing[y]['as_of']!=cutoff or existing[y]['vintage_label']!=cutoff:
            raise ValueError('Model snapshot vintage mismatch')
        u=urlsplit(p['source_url']); q=parse_qs(u.query)
        expected={'id':[PRICE_AUDIT_SERIES],'vintage_date':[cutoff],
                  'cosd':[f'{y-1}-04-01'],'coed':[f'{y}-04-01']}
        if u.scheme!='https' or u.hostname!='alfred.stlouisfed.org' or u.path!='/graph/' or any(q.get(k)!=v for k,v in expected.items()):
            raise ValueError('Graph source/version does not match requested year')
        start=dt.datetime.fromisoformat(p['retrieval_start_utc'].replace('Z','+00:00'))
        end=dt.datetime.fromisoformat(p['retrieval_end_utc'].replace('Z','+00:00'))
        if start.tzinfo is None or end.tzinfo is None or start.utcoffset()!=dt.timedelta(0) or end.utcoffset()!=dt.timedelta(0) or end<start:
            raise ValueError('Retrieval interval must be ordered and in UTC')
        for label,obs_year in [('previous',y-1),('current',y)]:
            observed=float(p[label+'_april']); saved=float(existing[y][label+'_april'])
            if not all(math.isfinite(x) and x>0 for x in (observed,saved)):
                raise ValueError('Invalid observation')
            if existing[y][label+'_observation']!=f'{obs_year}-04-01':
                raise ValueError('Snapshot observation period mismatch')
            row={'forecast_year':y,'series_id':PRICE_AUDIT_SERIES,
                 'observation_date':f'{obs_year}-04-01','value':observed,
                 'source_url':p['source_url'],'source_version':'ALFRED as-of '+cutoff,
                 'historical_publication_or_available_at_utc':None,
                 'retrieval_start_utc':p['retrieval_start_utc'],'retrieval_end_utc':p['retrieval_end_utc'],
                 'displayed_series_updated_text':p.get('displayed_series_updated_optional'),
                 'source_byte_sha256':None,'snapshot_value':saved,
                 'matches_snapshot':math.isclose(observed,saved,rel_tol=1e-9,abs_tol=1e-7),
                 'evidence_status':'browser_transcribed_asof_graph_not_source_bytes',
                 'note':'Requested as-of date and visible graph value; retrieval interval is current audit, not historical availability. Original source bytes and individual release times not retained.'}
            canonical=json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
            rows.append({'record':row,'canonical_sha256':hashlib.sha256(canonical.encode('utf8')).hexdigest()})
    PRICE_AUDIT_OUT.mkdir(parents=True,exist_ok=True)
    (PRICE_AUDIT_OUT/'browser_price_evidence_private.jsonl').write_text(
        ''.join(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
    summary={'browser_vintage_pairs':len(pairs),'browser_observations':len(rows),
             'matching_snapshot_observations':sum(r['record']['matches_snapshot'] for r in rows),
             'mismatches':[r['record'] for r in rows if not r['record']['matches_snapshot']],
             'missing_model_years':sorted(set(existing)-seen),
             'note':'Browser-transcribed ALFRED as-of graphs. Record hashes verify saved transcription only; they do not prove raw source bytes or exact historical availability.'}
    (PRICE_AUDIT_OUT/'browser_price_reconciliation_private.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('Browser price evidence:',json.dumps(summary,indent=2))
    return {'summary':summary,'observations':rows}

PRICE_AUDIT_RESULTS=run_price_audit(SNAPSHOT_JSON,HISTORICAL_PRICES,
    fetch_alfred=bool(globals().get('PRICE_AUDIT_FETCH_ALFRED',False)))
