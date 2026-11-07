from copy import deepcopy
from hashlib import sha256
import json

ANNOTATIONS = ('bls_comparison', 'sec_10q_comparison', 'sec_10q_index_comparison',
               'sec_q1_shipment_comparison', 'alfred_chart_comparison')


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def prepare(ledger, state=None):
    if not isinstance(ledger, list) or any(not isinstance(r, dict) for r in ledger):
        raise ValueError('Replay ledger must be a list of records')
    if state is None:
        if any(field in r for r in ledger for field in ANNOTATIONS):
            raise ValueError('Existing source links have no checkpoint; rerun the replay cell first')
        return deepcopy(ledger)
    if not isinstance(state, dict) or set(state) != {'base', 'base_sha256', 'linked_sha256'}:
        raise ValueError('Invalid source-link checkpoint')
    base = state['base']
    if (not isinstance(base, list) or any(not isinstance(r, dict) for r in base)
            or any(field in r for r in base for field in ANNOTATIONS)
            or _digest(base) != state['base_sha256']):
        raise ValueError('Source-link base changed; rerun the replay cell in a fresh kernel')
    if _digest(ledger) not in (state['base_sha256'], state['linked_sha256']):
        raise ValueError('Replay ledger changed since source-link checkpoint')
    return deepcopy(base)


def checkpoint(base, linked):
    if not isinstance(base, list) or not isinstance(linked, list) or len(base) != len(linked):
        raise ValueError('Incompatible source-link ledgers')
    for original, current in zip(base, linked):
        if (not isinstance(original, dict) or not isinstance(current, dict)
                or any(field in original for field in ANNOTATIONS)
                or any(k not in current or current[k] != value for k, value in original.items())):
            raise ValueError('Source linking changed the base ledger')
    saved = deepcopy(base)
    return {'base': saved, 'base_sha256': _digest(saved), 'linked_sha256': _digest(linked)}


def require_same_links(linked, state):
    if state is not None and _digest(linked) != state['linked_sha256']:
        raise ValueError('Recomputed source links differ from the checkpoint')
