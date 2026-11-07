from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re


def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def _planned_cutoffs(payload, recorded):
    planned=payload['planned_tests']
    if not isinstance(planned,list) or not planned:
        raise ValueError('Prospective proposal needs planned test cutoffs')
    result={}
    for item in planned:
        if not isinstance(item,dict) or set(item)!={'year','cutoff_utc'}:
            raise ValueError('Invalid planned test cutoff')
        year=item['year']; raw=item['cutoff_utc']
        if isinstance(year,bool) or not isinstance(year,int) or not 1 <= year <= 9999 or year in result:
            raise ValueError('Invalid or duplicate planned test year')
        if not isinstance(raw,str) or 'T' not in raw:
            raise ValueError('Planned cutoff needs an ISO timestamp and timezone')
        try: cutoff=dt.datetime.fromisoformat(raw.replace('Z','+00:00'))
        except ValueError as exc: raise ValueError('Invalid planned test cutoff') from exc
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError('Planned cutoff needs an ISO timestamp and timezone')
        cutoff=cutoff.astimezone(dt.timezone.utc)
        if cutoff.year != year:
            raise ValueError('Planned cutoff must be in its test year')
        if cutoff > dt.datetime(year,5,31,tzinfo=dt.timezone.utc):
            raise ValueError('Planned cutoff cannot be later than May 31 00:00 UTC')
        if recorded >= cutoff:
            raise ValueError('Prospective proposal must precede every planned decision cutoff')
        result[year]=cutoff
    return result


def verify(events, anchor=None):
    if anchor is not None:
        if (not isinstance(anchor,dict) or not {'events','head'} <= set(anchor)
                or isinstance(anchor['events'],bool) or not isinstance(anchor['events'],int)
                or anchor['events'] < 1 or not isinstance(anchor['head'],str)
                or re.fullmatch(r'[0-9a-f]{64}',anchor['head']) is None):
            raise ValueError('Invalid trusted anchor: expected positive events and lowercase SHA-256 head')
        if len(events)<anchor['events']:
            raise ValueError('Log truncated before trusted anchor')
    proposals={}; planned_tests={}; attempted_years=set(); prior_attempts={}; head='0'*64; previous_recorded=None
    for i,event in enumerate(events):
        required={'sequence','previous_sha256','recorded_utc','kind','experiment_id','payload','sha256'}
        if set(event)!=required or event['sequence']!=i+1 or event['previous_sha256']!=head:
            raise ValueError('Broken event chain or schema')
        base={k:v for k,v in event.items() if k!='sha256'}
        expected=hashlib.sha256(canonical(base).encode('utf8')).hexdigest()
        if event['sha256']!=expected: raise ValueError('Event hash mismatch')
        try:
            recorded=dt.datetime.fromisoformat(event['recorded_utc'].replace('Z','+00:00'))
        except (AttributeError,ValueError) as exc: raise ValueError('Invalid event timestamp') from exc
        if recorded.tzinfo is None or recorded.utcoffset() is None: raise ValueError('Event timestamp needs timezone')
        ident=event['experiment_id']; kind=event['kind']; payload=event['payload']
        if not isinstance(ident,str) or not ident.strip() or not isinstance(payload,dict): raise ValueError('Invalid event')
        if kind=='proposal':
            if ident in proposals: raise ValueError('Duplicate proposal')
            base_fields={'model','why','inputs','seen_years','stage'}
            if set(payload) not in (base_fields,base_fields|{'planned_tests'}): raise ValueError('Incomplete proposal')
            if any(not isinstance(payload[k],str) or not payload[k].strip() for k in ('model','why','stage')): raise ValueError('Missing proposal description')
            if payload['stage'] not in ('prospective','retrospective'): raise ValueError('Unknown proposal stage')
            if not isinstance(payload['inputs'],list) or not payload['inputs'] or any(not isinstance(x,str) or not x.strip() for x in payload['inputs']): raise ValueError('Missing model inputs')
            if not isinstance(payload['seen_years'],list) or any(isinstance(x,bool) or not isinstance(x,int) for x in payload['seen_years']) or len(set(payload['seen_years']))!=len(payload['seen_years']): raise ValueError('Invalid seen years')
            if 'planned_tests' in payload:
                if payload['stage']!='prospective': raise ValueError('Planned test cutoffs apply to prospective proposals')
                planned_tests[ident]=_planned_cutoffs(payload,recorded)
            else:
                if payload['stage']=='prospective' and anchor is not None and i >= anchor['events']:
                    raise ValueError('Legacy prospective proposal in post-anchor extension')
                planned_tests[ident]=None
            proposals[ident]=event
            prior_attempts[ident]=set(attempted_years)
        elif kind=='result':
            if ident not in proposals: raise ValueError('Result has no registered proposal')
            if (anchor is not None and i >= anchor['events'] and
                    proposals[ident]['payload']['stage']=='prospective' and planned_tests[ident] is None):
                raise ValueError('Legacy prospective result in post-anchor extension')
            proposal_time=dt.datetime.fromisoformat(proposals[ident]['recorded_utc'].replace('Z','+00:00'))
            if recorded < proposal_time: raise ValueError('Result timestamp precedes proposal timestamp')
            if set(payload)!={'status','test_years','metrics','artifact'}: raise ValueError('Incomplete result')
            if payload['status'] not in ('success','failure','mixed','inconclusive','error'): raise ValueError('Unknown result status')
            if not isinstance(payload['test_years'],list) or any(isinstance(x,bool) or not isinstance(x,int) for x in payload['test_years']) or len(set(payload['test_years']))!=len(payload['test_years']): raise ValueError('Invalid test years')
            if proposals[ident]['payload']['stage']=='prospective' and set(payload['test_years']) & set(proposals[ident]['payload']['seen_years']): raise ValueError('Prospective result includes seen years')
            if proposals[ident]['payload']['stage']=='prospective' and set(payload['test_years']) & prior_attempts[ident]: raise ValueError('Prospective result includes previously attempted years')
            if proposals[ident]['payload']['stage']=='prospective' and planned_tests[ident] is not None:
                if not payload['test_years'] or not set(payload['test_years']) <= set(planned_tests[ident]):
                    raise ValueError('Prospective result includes unplanned test years')
                if payload['status'] in ('success','failure','mixed'):
                    for year in payload['test_years']:
                        if recorded < dt.datetime(year,7,1,tzinfo=dt.timezone.utc):
                            raise ValueError('Scored Q2 result cannot precede quarter end')
            if not isinstance(payload['metrics'],dict) or not payload['metrics'] or not isinstance(payload['artifact'],str) or not payload['artifact'].strip(): raise ValueError('Missing result evidence')
            attempted_years.update(payload['test_years'])
        else: raise ValueError('Unknown event kind')
        if previous_recorded is not None and recorded < previous_recorded:
            raise ValueError('Event timestamp precedes previous event timestamp')
        previous_recorded=recorded
        head=event['sha256']
        if anchor is not None and i+1==anchor['events'] and head!=anchor['head']:
            raise ValueError('Trusted anchor does not match log prefix')
    legacy=sum(1 for ident,p in proposals.items() if p['payload']['stage']=='prospective' and planned_tests[ident] is None)
    return {'events':len(events),'proposals':len(proposals),'head':head,
            'legacy_prospective_without_cutoffs':legacy}


def read_log(path):
    path=Path(path)
    if not path.exists(): return []
    events=[]
    with path.open(encoding='utf8') as f:
        for line in f:
            if not line.strip(): raise ValueError('Blank event line')
            events.append(json.loads(line))
    verify(events)
    return events


def append_event(path,kind,experiment_id,payload,recorded_utc=None):
    p=Path(path); events=read_log(p); head=verify(events)['head']
    if kind=='proposal' and isinstance(payload,dict) and payload.get('stage')=='prospective' and 'planned_tests' not in payload:
        raise ValueError('New prospective proposals require planned test cutoffs')
    if kind=='result':
        prior=next((e for e in events if e['kind']=='proposal' and e['experiment_id']==experiment_id),None)
        if prior and prior['payload'].get('stage')=='prospective' and 'planned_tests' not in prior['payload']:
            raise ValueError('Cannot append a prospective result to a legacy proposal without planned cutoffs')
    if recorded_utc is None: recorded_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    try: when=dt.datetime.fromisoformat(recorded_utc.replace('Z','+00:00'))
    except (AttributeError,ValueError) as exc: raise ValueError('Timestamp must have timezone') from exc
    if when.tzinfo is None or when.utcoffset() is None: raise ValueError('Timestamp must have timezone')
    base={'sequence':len(events)+1,'previous_sha256':head,'recorded_utc':recorded_utc,
          'kind':kind,'experiment_id':experiment_id,'payload':payload}
    event={**base,'sha256':hashlib.sha256(canonical(base).encode('utf8')).hexdigest()}
    verify(events+[event]); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf8') as f:
        f.write(canonical(event)+'\n'); f.flush(); os.fsync(f.fileno())
    return event


def main(argv=None):
    parser=argparse.ArgumentParser(description='Append to or verify an experiment log.')
    parser.add_argument('--log',required=True,type=Path)
    parser.add_argument('--kind',choices=('proposal','result','verify'),required=True)
    parser.add_argument('--id',help='Stable experiment identifier')
    parser.add_argument('--payload',type=Path,help='JSON object containing required fields')
    parser.add_argument('--anchor',type=Path,help='Independently retained JSON head/count to verify against (verify only)')
    args=parser.parse_args(argv)
    if args.kind=='verify':
        if args.id or args.payload: parser.error('verify takes no id or payload')
        if not args.log.exists(): parser.error('log does not exist')
        anchor=json.loads(args.anchor.read_text(encoding='utf8')) if args.anchor else None
        print(canonical(verify(read_log(args.log),anchor))); return
    if args.anchor: parser.error('--anchor is available only with verify')
    if not args.id or not args.payload: parser.error('proposal/result requires --id and --payload')
    payload=json.loads(args.payload.read_text(encoding='utf8'))
    print(canonical(append_event(args.log,args.kind,args.id,payload)))


if __name__=='__main__': main()
