from __future__ import annotations

import json
import os
from pathlib import Path

import experiment_log


def _read_verified(path):
    raw=Path(path).read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Experiment log has an incomplete final line')
    events=[]
    try:
        for line in raw.decode('utf8').splitlines():
            if not line.strip():
                raise ValueError('Blank event line')
            events.append(json.loads(line))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid experiment log') from exc
    if not events:
        raise ValueError('Experiment log is empty')
    summary=experiment_log.verify(events)
    return raw, events, summary


def prepare_prospective_log(destination, previous_log=None, anchor=None):
    target=Path(destination)
    prior=None if previous_log is None else Path(previous_log)
    previous=None
    if prior is not None:
        if not prior.is_file():
            raise FileNotFoundError('Specified previous experiment log is unavailable')
        previous=_read_verified(prior)
    if target.exists():
        if not target.is_file():
            raise ValueError('Experiment log destination is not a file')
        _,current,summary=_read_verified(target)
        if previous is not None:
            experiment_log.verify(current,previous[2])
        if anchor is not None:
            experiment_log.verify(current,anchor)
        return {'status':'existing_verified','events':summary['events'],'head':summary['head'],
                'previous_supplied':previous is not None,'trusted_anchor_checked':anchor is not None}
    if previous is None:
        if anchor is not None:
            raise ValueError('Cannot check an anchor without an experiment log')
        return {'status':'no_prospective_log_carried','events':0,'head':None,
                'previous_supplied':False,'trusted_anchor_checked':False,
                'message':('No log was created. Mount the previous output before logging; '
                           'absence here does not prove that no previous log exists.')}
    raw,events,summary=previous
    if anchor is not None:
        experiment_log.verify(events,anchor)
    target.parent.mkdir(parents=True,mode=0o700,exist_ok=True)
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    return {'status':'previous_log_copied','events':summary['events'],'head':summary['head'],
            'previous_supplied':True,'trusted_anchor_checked':anchor is not None}
